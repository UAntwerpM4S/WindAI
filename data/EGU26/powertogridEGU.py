#!/usr/bin/env python3
"""
Adds four variables to the existing Zarr store:
  - mask(points): 1 where at least one farm maps to that point, NaN elsewhere
  - turbines(points): summed turbine count at that point, NaN elsewhere
  - capacity(points): summed nameplate capacity (MW) at that point, NaN elsewhere
  - power(time, points): summed power (MW) at that point

Power rules (STRICT NaN propagation for co-located farms):
  - If multiple farms map to the same point, their power is ADDED.
  - If ANY of those farms is NaN at a given time, the point value becomes NaN at that time.
  - If timestamp missing in CSV after reindex -> NaN.
  - If no turbines at a point -> NaN.

Grid:
  - Dataset uses "points" dimension. Farms are snapped to nearest (lon,lat) point via KDTree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.spatial import cKDTree


ZARR_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/cerra_EGU.zarr")
POWER_CSV = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/BE_UK_offshore_per_unit_3H_meanMW_shifted.csv")
META_CSV  = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

# Metadata columns
COL_FARM = "farm"
COL_LAT  = "lat"
COL_LON  = "lon"
COL_CAP  = "capacity_mw"
COL_TURB = "turbines"

# Variable names to write
VAR_MASK = "turbinemask"
VAR_TURB = "turbinecount"
VAR_CAP  = "capacity"
VAR_POW  = "power"

OVERWRITE_IF_EXISTS = True


def _normalize_lon(lon_series: pd.Series, lon_grid: np.ndarray) -> pd.Series:
    """Match longitude convention of the target grid (0..360 vs -180..180)."""
    lon_max = np.nanmax(lon_grid)
    if lon_max > 180:
        return (lon_series + 360) % 360
    return lon_series


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(META_CSV)
    required = {COL_FARM, COL_LAT, COL_LON, COL_CAP, COL_TURB}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    df[COL_CAP] = pd.to_numeric(df[COL_CAP], errors="coerce")
    df[COL_TURB] = pd.to_numeric(df[COL_TURB], errors="coerce")
    df = df.dropna(subset=[COL_FARM, COL_LAT, COL_LON]).copy()
    return df


def load_power_series(target_times: pd.DatetimeIndex, farms: list[str]) -> pd.DataFrame:
    df = pd.read_csv(POWER_CSV)
    if "time" not in df.columns:
        raise ValueError("Power CSV must contain a 'time' column.")

    # Parse to UTC then drop tz to match dataset time (tz-naive)
    time_parsed = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True, errors="coerce"))
    df = df.drop(columns=["time"])

    if time_parsed.tz is not None:
        idx = time_parsed.tz_convert("UTC").tz_localize(None)
    else:
        idx = time_parsed.tz_localize(None)
    df.index = idx

    df = df.apply(pd.to_numeric, errors="coerce")

    keep_cols = [c for c in df.columns if c in farms]
    if not keep_cols:
        raise ValueError("No overlapping farm columns between metadata and power CSV.")
    df = df[keep_cols]

    # Align to dataset timestamps; missing timestamps become NaN
    return df.reindex(target_times)


def map_farms_to_points(ds: xr.Dataset, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Snap each farm to nearest ds point via KDTree on lon/lat."""

    lonp = np.asarray(ds["longitude"].values)
    latp = np.asarray(ds["latitude"].values)

    meta_df = meta_df.copy()
    meta_df["lon_norm"] = _normalize_lon(meta_df[COL_LON], lonp)

    tree = cKDTree(np.c_[lonp, latp])  # (points, 2)
    dist, idx = tree.query(np.c_[meta_df["lon_norm"].to_numpy(), meta_df[COL_LAT].to_numpy()], k=1)

    meta_df["point"] = idx.astype(int)
    meta_df["deg_error"] = dist
    meta_df["km_error_approx"] = meta_df["deg_error"] * 111.0  # rough QA
    return meta_df


def build_static_fields(mapped: pd.DataFrame, npoints: int):
    mask = np.full((npoints,), np.nan, dtype="float32")
    turb = np.full((npoints,), np.nan, dtype="float32")
    cap  = np.full((npoints,), np.nan, dtype="float32")

    for p, grp in mapped.groupby("point"):
        p = int(p)
        mask[p] = 1.0
        turb[p] = grp[COL_TURB].sum(skipna=True)
        cap[p]  = grp[COL_CAP].sum(skipna=True)

    return mask, turb, cap


def build_power_field(ds: xr.Dataset, mapped: pd.DataFrame, power_df: pd.DataFrame) -> xr.DataArray:
    """
    power(time, points) with STRICT NaN propagation for co-located farms:
      - sum across farms at a point
      - if any farm at that point is NaN at time t, output is NaN at time t
    Implemented via min_count=len(cols).
    """
    ntime = ds.sizes["time"]
    npoints = ds.sizes["points"]
    out = np.full((ntime, npoints), np.nan, dtype="float32")

    farms_available = set(power_df.columns)
    mapped = mapped[mapped[COL_FARM].isin(farms_available)].copy()
    if mapped.empty:
        return xr.DataArray(
            out,
            coords={"time": ds["time"], "points": ds["points"]},
            dims=("time", "points"),
            name=VAR_POW,
        )

    farms_by_point = mapped.groupby("point")[COL_FARM].apply(list)

    for p, farms in farms_by_point.items():
        p = int(p)
        cols = [f for f in farms if f in power_df.columns]
        if not cols:
            continue

        # STRICT: requires ALL farms non-NaN to produce a sum
        summed = power_df[cols].sum(axis=1, skipna=True, min_count=len(cols))
        out[:, p] = summed.to_numpy(dtype="float32")

    power_da = xr.DataArray(
        out,
        coords={"time": ds["time"], "points": ds["points"]},
        dims=("time", "points"),
        name=VAR_POW,
    )

    # Ensure NaN where there are no turbines (no farms mapped)
    turbine_points = np.zeros((npoints,), dtype=bool)
    turbine_points[list(farms_by_point.index.astype(int))] = True
    turbine_mask = xr.DataArray(turbine_points, coords={"points": ds["points"]}, dims=("points",))
    return power_da.where(turbine_mask)


def append_to_zarr():
    print(f"[LOAD] Opening Zarr: {ZARR_PATH}")
    ds = xr.open_zarr(ZARR_PATH, consolidated=False)

    if "points" not in ds.dims or "time" not in ds.dims:
        raise ValueError(f"Expected dims ('time','points'). Found: {dict(ds.dims)}")

    existing = set(ds.data_vars)
    new_vars = {VAR_POW, VAR_MASK, VAR_TURB, VAR_CAP}
    clashes = sorted(v for v in new_vars if v in existing)

    if clashes and not OVERWRITE_IF_EXISTS:
        raise ValueError(
            f"Variables already exist in the Zarr: {clashes}. "
            f"Set OVERWRITE_IF_EXISTS=True if you want to overwrite them."
        )

    meta = load_metadata()
    mapped = map_farms_to_points(ds, meta)

    cerra_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    power_df = load_power_series(cerra_times, farms=mapped[COL_FARM].tolist())

    npoints = ds.sizes["points"]
    mask_arr, turb_arr, cap_arr = build_static_fields(mapped, npoints=npoints)
    power_da = build_power_field(ds, mapped, power_df)

    mask_da = xr.DataArray(mask_arr, coords={"points": ds["points"]}, dims=("points",), name=VAR_MASK).astype("float32")
    turb_da = xr.DataArray(turb_arr, coords={"points": ds["points"]}, dims=("points",), name=VAR_TURB).astype("float32")
    cap_da  = xr.DataArray(cap_arr,  coords={"points": ds["points"]}, dims=("points",), name=VAR_CAP).astype("float32")

    out_ds = xr.Dataset({VAR_POW: power_da, VAR_MASK: mask_da, VAR_TURB: turb_da, VAR_CAP: cap_da})

    # Chunking (safe default)
    time_chunk = 56   # 7 days @ 3-hourly
    points_chunk = min(4096, npoints)
    encoding = {
        VAR_POW:  {"chunks": (time_chunk, points_chunk)},
        VAR_MASK: {"chunks": (points_chunk,)},
        VAR_TURB: {"chunks": (points_chunk,)},
        VAR_CAP:  {"chunks": (points_chunk,)},
    }

    print(f"[WRITE] Appending {list(out_ds.data_vars)} to {ZARR_PATH}")
    out_ds.to_zarr(ZARR_PATH, mode="a", consolidated=True)#, encoding=encoding)
    print("[DONE]")


if __name__ == "__main__":
    append_to_zarr()
