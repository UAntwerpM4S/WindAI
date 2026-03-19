#!/usr/bin/env python3
"""
CERRA → Zarr (grid version, OOM-safe)

Key changes vs. original:
  - _astype_float32() removed: xr.DataArray.astype() triggers full in-memory
    compute on cfgrib-backed lazy arrays → instant OOM.
  - out.astype(cast_map) removed for the same reason.
  - float32 dtype is now specified ONLY in the Zarr encoding dict, so the cast
    happens chunk-by-chunk during the streaming write.
  - Per-month dataset is processed in weekly sub-chunks before writing so the
    full month is never stacked/computed at once.
  - cfgrib index files are written next to each GRIB (avoids re-scanning on
    every open_datasets call).
  - sortby("time") is called on the small (y,x) dataset before chunking,
    not on a large flattened array.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import cfgrib
import zarr
from scipy.spatial import cKDTree

xr.set_options(use_new_combine_kwarg_defaults=True)

# =========================
# CONFIG
# =========================
START = "2015-01-01T00:00:00"
END   = "2025-09-30T21:00:00"

RAW_DIR  = Path("/mnt/weatherloss/WindPower/data/cerra_boz/raw_grib")
OUT_ZARR = Path("/mnt/weatherloss/WindPower/data/EGU26/cerra_EGU_full.zarr")

POWER_CSV = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/BE_UK_offshore_per_unit_3H_meanMW_shifted.csv")
META_CSV  = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

CROP = False
OVERWRITE_TURBINE_VARS_IF_EXIST = True

VAR_MASK = "turbinemask"
VAR_TURB = "turbinecount"
VAR_CAP  = "capacity"
VAR_POW  = "power"

COL_FARM = "farm"
COL_LAT  = "lat"
COL_LON  = "lon"
COL_CAP  = "capacity_mw"
COL_TURB = "turbines"

# Zarr chunk sizes
T_CHUNK = 56    # ~1 week of 3-hourly steps
Y_CHUNK = 128
X_CHUNK = 128
# Weekly sub-chunking frequency to avoid loading a full month at once
WEEK_FREQ = "7D"


# =========================
# GRIB helpers
# =========================

def open_grib_datasets(path: Path):
    """Open GRIB with a persistent on-disk index (avoids rebuilding on every call)."""
    return cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": str(path) + ".idx"})


def drop_coord_safe(da: xr.DataArray, name: str) -> xr.DataArray:
    if name in da.coords and name not in da.dims:
        return da.reset_coords(name, drop=True)
    return da


def clean_common(da: xr.DataArray) -> xr.DataArray:
    """Drop auxiliary coords that cfgrib adds but we don't need.  NO dtype cast here."""
    for c in ["valid_time", "surface", "meanSea", "number",
              "heightAboveGround", "entireAtmosphere", "step"]:
        da = drop_coord_safe(da, c)
    return da


def lon_to_180(lon) -> np.ndarray:
    lon = np.asarray(lon)
    return ((lon + 180.0) % 360.0) - 180.0


def open_month(single_path: Path, pres_path: Path, hgt_path: Path,
               static_ds: xr.Dataset) -> xr.Dataset:
    """
    Open all GRIBs lazily.  Returns a lazy xr.Dataset — nothing is computed yet.
    dtype conversion intentionally omitted here; it happens at Zarr write time.
    """
    single_groups = open_grib_datasets(single_path)
    pres = open_grib_datasets(pres_path)[0]
    hgt  = open_grib_datasets(hgt_path)[0]

    if "isobaricInhPa" in pres.dims or "isobaricInhPa" in pres.coords:
        pres = pres.rename({"isobaricInhPa": "level"})

    t2m    = next(ds["t2m"]    for ds in single_groups if "t2m"    in ds.data_vars)
    mcc    = next(ds["mcc"]    for ds in single_groups if "mcc"    in ds.data_vars)
    w10ds  = next(ds for ds in single_groups if "si10" in ds.data_vars and "wdir10" in ds.data_vars)
    si10   = w10ds["si10"]
    wdir10 = w10ds["wdir10"]
    msl    = next((ds["msl"] for ds in single_groups if "msl" in ds.data_vars), None)

    ws100   = hgt["ws"].sel(heightAboveGround=100.0)
    wdir100 = hgt["wdir"].sel(heightAboveGround=100.0)

    orog = static_ds["orog"]
    lsm  = static_ds["lsm"]

    vars_raw = {
        "t2m":     clean_common(t2m),
        "mcc":     clean_common(mcc),
        "si10":    clean_common(si10),
        "wdir10":  clean_common(wdir10),
        "z":       clean_common(pres["z"]),
        "t":       clean_common(pres["t"]),
        "u":       clean_common(pres["u"]),
        "v":       clean_common(pres["v"]),
        "r":       clean_common(pres["r"]),
        "ws100":   clean_common(ws100),
        "wdir100": clean_common(wdir100),
        "orog":    clean_common(orog),
        "lsm":     clean_common(lsm),
    }
    if msl is not None:
        vars_raw["msl"] = clean_common(msl)

    return xr.Dataset(vars_raw)


# =========================
# Time / Zarr helpers
# =========================

def month_range(start: pd.Timestamp, end: pd.Timestamp):
    cur  = pd.Timestamp(year=start.year, month=start.month, day=1)
    last = pd.Timestamp(year=end.year,   month=end.month,   day=1)
    while cur <= last:
        yield cur.year, cur.month
        cur = cur + pd.offsets.MonthBegin(1)


def fix_time_attrs_and_consolidate(zarr_path: Path):
    g = zarr.open_group(str(zarr_path), mode="a")
    a = dict(g["time"].attrs)
    a.pop("standard_name", None)
    a.pop("long_name", None)
    g["time"].attrs.clear()
    g["time"].attrs.update(a)
    zarr.consolidate_metadata(str(zarr_path))


def make_grid_encoding(ds: xr.Dataset) -> dict:
    """
    Build Zarr encoding with explicit chunks AND dtype=float32.
    Setting dtype here means Zarr casts chunk-by-chunk during the write,
    so we never materialise a full float64 array in RAM.
    """
    enc: dict[str, dict] = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        chunks = []
        for d in dims:
            if d == "time":
                chunks.append(min(T_CHUNK, ds.sizes["time"]))
            elif d == "y":
                chunks.append(min(Y_CHUNK, ds.sizes["y"]))
            elif d == "x":
                chunks.append(min(X_CHUNK, ds.sizes["x"]))
            elif d == "level":
                chunks.append(min(ds.sizes["level"], 64))
            else:
                chunks.append(ds.sizes[d])
        entry: dict = {"chunks": tuple(chunks)}
        # Only request float32 for floating-point vars; leave int/bool alone.
        if np.issubdtype(ds[v].dtype, np.floating):
            entry["dtype"] = "float32"
        enc[v] = entry
    return enc


# =========================
# Main build
# =========================

def build_cerra_grid_zarr():
    start = pd.Timestamp(START)
    end   = pd.Timestamp(END)

    static_ds = open_grib_datasets(RAW_DIR / "cerra_static.grib")[0]

    if OUT_ZARR.exists():
        shutil.rmtree(OUT_ZARR)

    first = True

    for y, m in month_range(start, end):
        print(f"\nProcessing {y}-{m:02d}...")

        ds_full = open_month(
            RAW_DIR / f"cerra_single_{y:04d}_{m:02d}.grib",
            RAW_DIR / f"cerra_pressure_{y:04d}_{m:02d}.grib",
            RAW_DIR / f"cerra_height_{y:04d}_{m:02d}.grib",
            static_ds,
        )

        # Sort by time on the small lazy dataset (cheap), then clip to [start, end]
        ds_full = ds_full.sortby("time").sel(time=slice(start, end))

        # Split month into weekly sub-chunks so we never compute a full month at once
        month_times = pd.DatetimeIndex(ds_full.time.values)
        week_starts = pd.date_range(month_times[0], month_times[-1], freq=WEEK_FREQ)

        for w_idx, ws in enumerate(week_starts):
            we = ws + pd.Timedelta(WEEK_FREQ) - pd.Timedelta("1ns")
            chunk = ds_full.sel(time=slice(ws, we))

            if chunk.sizes.get("time", 0) == 0:
                continue

            print(f"  Week {w_idx + 1}/{len(week_starts)}: "
                  f"{ws.date()} → {min(we, month_times[-1]).date()} "
                  f"({chunk.sizes['time']} steps)")

            if first:
                encoding = make_grid_encoding(chunk)
                chunk.to_zarr(str(OUT_ZARR), mode="w", consolidated=False, encoding=encoding)
                first = False
            else:
                chunk.to_zarr(str(OUT_ZARR), mode="a", append_dim="time", consolidated=False)

            del chunk

        del ds_full

    fix_time_attrs_and_consolidate(OUT_ZARR)
    print("\nZarr grid build complete.")


# =========================
# Turbine metadata + power (grid mode)
# =========================

def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(META_CSV)
    df[COL_CAP]  = pd.to_numeric(df[COL_CAP],  errors="coerce")
    df[COL_TURB] = pd.to_numeric(df[COL_TURB], errors="coerce")
    return df.dropna(subset=[COL_FARM, COL_LAT, COL_LON]).copy()


def load_power_series(target_times: pd.DatetimeIndex, farms: list[str]) -> pd.DataFrame:
    df = pd.read_csv(POWER_CSV)
    time_parsed = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.drop(columns=["time"])
    idx = time_parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    df.index = idx
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[[c for c in df.columns if c in farms]]
    return df.reindex(target_times)


def map_farms_to_yx(ds: xr.Dataset, meta_df: pd.DataFrame) -> pd.DataFrame:
    lat2d = np.asarray(ds["latitude"].values)
    lon2d = lon_to_180(np.asarray(ds["longitude"].values))
    ny, nx = lat2d.shape

    tree = cKDTree(np.c_[lon2d.ravel(), lat2d.ravel()])

    meta = meta_df.copy()
    meta["lon_180"] = lon_to_180(meta[COL_LON].to_numpy())
    dist, idx = tree.query(np.c_[meta["lon_180"].to_numpy(), meta[COL_LAT].to_numpy()], k=1)
    idx = idx.astype(int)
    meta["y"] = (idx // nx).astype(int)
    meta["x"] = (idx % nx).astype(int)
    meta["deg_error"] = dist
    return meta


def build_static_fields_yx(mapped: pd.DataFrame, ny: int, nx: int):
    mask = np.full((ny, nx), np.nan, dtype="float32")
    turb = np.full((ny, nx), np.nan, dtype="float32")
    cap  = np.full((ny, nx), np.nan, dtype="float32")
    for (iy, ix), grp in mapped.groupby(["y", "x"]):
        iy, ix = int(iy), int(ix)
        mask[iy, ix] = 1.0
        turb[iy, ix] = grp[COL_TURB].sum(skipna=True)
        cap[iy, ix]  = grp[COL_CAP].sum(skipna=True)
    return mask, turb, cap


def delete_vars_if_exist(store_path: Path, varnames: list[str]):
    g = zarr.open_group(str(store_path), mode="a")
    for v in varnames:
        if v in g:
            del g[v]


def write_power_field_to_zarr_yx(store_path: Path, ds: xr.Dataset,
                                  mapped: pd.DataFrame, power_df: pd.DataFrame):
    """Write power(time,y,x) cell-by-cell so we never allocate a dense (time,y,x) array."""
    g = zarr.open_group(str(store_path), mode="a")

    ntime = ds.sizes["time"]
    ny    = ds.sizes["y"]
    nx    = ds.sizes["x"]

    if VAR_POW in g:
        del g[VAR_POW]

    z = g.create_dataset(
        VAR_POW,
        shape=(ntime, ny, nx),
        chunks=(min(T_CHUNK, ntime), min(Y_CHUNK, ny), min(X_CHUNK, nx)),
        dtype="float32",
        fill_value=np.nan,
        overwrite=True,
    )
    z.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]

    farms_by_cell = mapped.groupby(["y", "x"])[COL_FARM].apply(list)
    for (iy, ix), farms in farms_by_cell.items():
        iy, ix = int(iy), int(ix)
        cols = [f for f in farms if f in power_df.columns]
        if not cols:
            continue
        summed = power_df[cols].sum(axis=1, skipna=True, min_count=len(cols)).to_numpy(dtype="float32")
        z[:, iy, ix] = summed


def append_turbine_vars_to_cerra_grid_zarr():
    ds = xr.open_zarr(OUT_ZARR, consolidated=True)

    if OVERWRITE_TURBINE_VARS_IF_EXIST:
        delete_vars_if_exist(OUT_ZARR, [VAR_POW, VAR_MASK, VAR_TURB, VAR_CAP])

    meta   = load_metadata()
    mapped = map_farms_to_yx(ds, meta)

    cerra_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    power_df    = load_power_series(cerra_times, farms=mapped[COL_FARM].tolist())

    ny, nx = ds.sizes["y"], ds.sizes["x"]
    mask_arr, turb_arr, cap_arr = build_static_fields_yx(mapped, ny=ny, nx=nx)

    out_static = xr.Dataset({
        VAR_MASK: xr.DataArray(mask_arr, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x")),
        VAR_TURB: xr.DataArray(turb_arr, coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x")),
        VAR_CAP:  xr.DataArray(cap_arr,  coords={"y": ds["y"], "x": ds["x"]}, dims=("y", "x")),
    })

    encoding = {
        VAR_MASK: {"chunks": (min(Y_CHUNK, ny), min(X_CHUNK, nx)), "dtype": "float32"},
        VAR_TURB: {"chunks": (min(Y_CHUNK, ny), min(X_CHUNK, nx)), "dtype": "float32"},
        VAR_CAP:  {"chunks": (min(Y_CHUNK, ny), min(X_CHUNK, nx)), "dtype": "float32"},
    }
    out_static.to_zarr(OUT_ZARR, mode="a", consolidated=False, encoding=encoding)

    write_power_field_to_zarr_yx(OUT_ZARR, ds, mapped, power_df)

    fix_time_attrs_and_consolidate(OUT_ZARR)
    print("Turbine metadata and power written.")


def main():
    if CROP:
        raise RuntimeError("This grid script is intended for CROP=False (full y/x grid).")

    print("--- Phase 1: Building CERRA grid Zarr ---")
    build_cerra_grid_zarr()

    print("\n--- Phase 2: Appending turbine metadata & power ---")
    append_turbine_vars_to_cerra_grid_zarr()

    print("\nDone.")


if __name__ == "__main__":
    main()