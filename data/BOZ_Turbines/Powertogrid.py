#!/usr/bin/env python3

"""Project Belgian offshore wind farm power/metadata onto the BOZ CERRA grid.

Adds four variables to ``BOZ.zarr``:
    - power: summed farm power (MW) mapped to the nearest grid cell
    - mask: 1 at cells with at least one farm, NaN elsewhere
    - turbines: number of turbines per grid cell (sums co-located farms)
    - capacity: nameplate capacity per grid cell in MW (sums co-located farms)

This mirrors ``data/NorthSea/Power/powertogrid_northsea.py`` but restricts the
farms to Belgium only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.spatial import cKDTree

# Paths
DATA_ROOT = Path("/mnt/data/weatherloss/WindPower/data")
POWER_CSV = DATA_ROOT / "BOZ_Turbines" / "windpowerdata" / "BE_offshore_per_unit_3H_meanMW_2020-01_to_2025-07.csv"
META_CSV = DATA_ROOT / "NorthSea" / "Power" / "windfarm_metadata.csv"
BOZ_ZARR = DATA_ROOT / "BOZ.zarr"


def _normalize_lon(lon_series: pd.Series, lon_grid: np.ndarray) -> pd.Series:
    """Match longitude convention of the target grid (0..360 vs -180..180)."""
    lon_max = np.nanmax(lon_grid)
    if lon_max > 180:
        return (lon_series + 360) % 360
    return lon_series


def load_metadata() -> pd.DataFrame:
    """Load Belgian farm metadata and coerce numeric fields."""
    df = pd.read_csv(META_CSV)
    required_cols = {"farm", "lat", "lon", "capacity_mw", "turbines", "region"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    df = df[df["region"].str.lower() == "belgium"].copy()
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    df["turbines"] = pd.to_numeric(df["turbines"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df


def load_power_series(target_times: pd.DatetimeIndex, farms: list[str]) -> pd.DataFrame:
    """Read power CSV, keep requested farms, and align to CERRA timestamps."""
    df = pd.read_csv(POWER_CSV)
    if "time" not in df.columns:
        raise ValueError("Power CSV must contain a 'time' column.")

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
    aligned = df.reindex(target_times)
    return aligned


def map_farms_to_grid(ds: xr.Dataset, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Snap each farm to its nearest BOZ grid cell."""
    if "longitude" not in ds or "latitude" not in ds:
        raise ValueError("CERRA dataset must expose 'longitude' and 'latitude' coordinates.")

    lon2d = np.asarray(ds["longitude"])
    lat2d = np.asarray(ds["latitude"])
    ny, nx = lon2d.shape

    meta_df = meta_df.copy()
    meta_df["lon_norm"] = _normalize_lon(meta_df["lon"], lon2d)

    pts_grid = np.c_[lon2d.ravel(), lat2d.ravel()]
    tree = cKDTree(pts_grid)

    dist, idx = tree.query(np.c_[meta_df["lon_norm"], meta_df["lat"]], k=1)
    meta_df["x"] = (idx % nx).astype(int)
    meta_df["y"] = (idx // nx).astype(int)
    meta_df["lon_cell"] = lon2d[meta_df["y"], meta_df["x"]]
    meta_df["lat_cell"] = lat2d[meta_df["y"], meta_df["x"]]
    meta_df["deg_error"] = np.hypot(meta_df["lon_cell"] - meta_df["lon_norm"], meta_df["lat_cell"] - meta_df["lat"])
    meta_df["km_error_approx"] = meta_df["deg_error"] * 111.0
    return meta_df


def build_static_fields(mapped: pd.DataFrame, ny: int, nx: int):
    """Create mask, turbines, and capacity arrays aggregated per grid cell."""
    mask_arr = np.full((ny, nx), np.nan, dtype="float32")
    turb_arr = np.full((ny, nx), np.nan, dtype="float32")
    cap_arr = np.full((ny, nx), np.nan, dtype="float32")

    for (iy, ix), grp in mapped.groupby(["y", "x"]):
        mask_arr[iy, ix] = 1.0
        turb_arr[iy, ix] = grp["turbines"].sum(skipna=True)
        cap_arr[iy, ix] = grp["capacity_mw"].sum(skipna=True)

    return mask_arr, turb_arr, cap_arr


def build_power_field(ds: xr.Dataset, mapped: pd.DataFrame, power_df: pd.DataFrame) -> xr.DataArray:
    """Scatter farm power onto the grid, summing co-located farms."""
    farms = [f for f in power_df.columns if f in set(mapped["farm"])]
    if not farms:
        raise ValueError("No overlapping farms between mapped metadata and power series.")

    mapped_use = mapped[mapped["farm"].isin(farms)]
    ny, nx = ds.sizes["y"], ds.sizes["x"]

    farm_mask = np.zeros((len(farms), ny, nx), dtype="float32")
    lookup = {row["farm"]: (int(row["y"]), int(row["x"])) for _, row in mapped_use.iterrows()}
    for k, farm in enumerate(farms):
        iy, ix = lookup[farm]
        farm_mask[k, iy, ix] = 1.0

    farm_mask_da = xr.DataArray(
        farm_mask,
        coords={"farm": farms, "y": ds["y"], "x": ds["x"]},
        dims=("farm", "y", "x"),
        name="farm_mask_onehot",
    )

    power_da = xr.DataArray(
        power_df[farms].to_numpy(dtype="float32"),
        coords={"time": pd.DatetimeIndex(power_df.index), "farm": farms},
        dims=("time", "farm"),
        name="farm_power_series_MW",
    )

    power_grid = xr.dot(power_da.fillna(0.0), farm_mask_da, dims="farm").astype("float32").rename("power")
    mask2d = (farm_mask_da.sum(dim="farm") > 0)
    power_grid = power_grid.where(mask2d)
    return power_grid.assign_coords(time=ds["time"], y=ds["y"], x=ds["x"])


def append_power_to_zarr(zarr_path: Path):
    """Append power/mask/turbines/capacity variables to the BOZ Zarr store."""
    print(f"[LOAD] Opening {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)

    meta_df = load_metadata()
    mapped = map_farms_to_grid(ds, meta_df)

    cerra_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    power_df = load_power_series(cerra_times, farms=mapped["farm"].tolist())

    mapped = mapped[mapped["farm"].isin(power_df.columns)].reset_index(drop=True)
    if mapped.empty:
        raise ValueError("No farms left after intersecting metadata and power columns.")

    ny, nx = ds.sizes["y"], ds.sizes["x"]
    mask_arr, turb_arr, cap_arr = build_static_fields(mapped, ny=ny, nx=nx)
    power_grid = build_power_field(ds, mapped, power_df)

    mask_da = xr.DataArray(mask_arr, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x"), name="mask").astype("float32")
    turb_da = xr.DataArray(turb_arr, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x"), name="turbines").astype("float32")
    cap_da = xr.DataArray(cap_arr, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x"), name="capacity").astype("float32")

    power_grid.attrs.update({
        "long_name": "Offshore wind power mapped to CERRA grid (sum across co-located farms)",
        "units": "MW",
        "note": "NaN outside turbine cells; time in UTC (3-hourly).",
    })
    mask_da.attrs.update({"long_name": "Wind farm mask", "values": "1 at farm cells, NaN elsewhere"})
    turb_da.attrs.update({"long_name": "Wind turbines per cell", "units": "count"})
    cap_da.attrs.update({"long_name": "Nameplate capacity per cell", "units": "MW"})

    out_ds = xr.Dataset({"power": power_grid, "mask": mask_da, "turbines": turb_da, "capacity": cap_da})
    encoding = {
        "power": {"chunks": (24, ny, nx)},
        "mask": {"chunks": (ny, nx)},
        "turbines": {"chunks": (ny, nx)},
        "capacity": {"chunks": (ny, nx)},
    }

    print(f"[WRITE] Appending variables to {zarr_path}")
    out_ds.to_zarr(zarr_path, mode="a", consolidated=True, encoding=encoding)
    print(f"[DONE] Completed {zarr_path.name}")


def main():
    append_power_to_zarr(BOZ_ZARR)


if __name__ == "__main__":
    main()
