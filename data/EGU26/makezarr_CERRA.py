#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import cfgrib
import zarr
from scipy.spatial import cKDTree

# -------------------------
# CONFIG
# -------------------------
START     = "2015-01-01T00:00:00"
END       = "2025-09-30T21:00:00"

RAW_DIR   = Path("/mnt/weatherloss/WindPower/data/cerra_boz/raw_grib")
OUT_ZARR  = Path("/mnt/weatherloss/WindPower/data/EGU26/cerra_EGU_xy.zarr")

POWER_CSV = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/BE_UK_offshore_per_unit_3H_meanMW_shifted.csv")
META_CSV  = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

CROP      = True
LAT_MIN, LAT_MAX = 48.5, 57.5
LON_MIN, LON_MAX = -6.0, 5.0

VAR_MASK, VAR_TURB, VAR_CAP, VAR_POW          = "turbinemask", "turbinecount", "capacity", "power"
COL_FARM, COL_LAT, COL_LON, COL_CAP, COL_TURB = "farm", "lat", "lon", "capacity_mw", "turbines"

# Zarr chunks
T_CHUNK   = 56
Y_CHUNK   = 128
X_CHUNK   = 128
LEV_CHUNK = 1

xr.set_options(use_new_combine_kwarg_defaults=True)

# -------------------------
# CFGRIB SAFE OPEN
# -------------------------
def open_cfgrib_datasets_safe(path: Path, use_sidecar_index: bool = False):
    """
    Robust cfgrib opener.

    - If use_sidecar_index=False (default): do NOT read/write *.idx next to GRIB
      (avoids EOFError from corrupted/truncated index files; safer on shared FS).
    - If True: use default cfgrib behavior, but auto-delete corrupted *.idx on EOFError and retry.
    """
    kwargs = {}
    if not use_sidecar_index:
        # disables sidecar index entirely
        kwargs["indexpath"] = ""

    try:
        return cfgrib.open_datasets(str(path), **kwargs)
    except EOFError:
        # Corrupted/truncated idx -> delete and retry
        for idx in glob.glob(str(path) + ".*.idx"):
            try:
                os.remove(idx)
            except OSError:
                pass
        return cfgrib.open_datasets(str(path), **kwargs)

# -------------------------
# HELPERS
# -------------------------
def clean_da(da: xr.DataArray) -> xr.DataArray:
    drop = [
        "valid_time", "surface", "meanSea", "number",
        "heightAboveGround", "entireAtmosphere", "step",
    ]
    return da.drop_vars([c for c in drop if c in da.coords])

def lon_to_180(lon):
    lon = np.asarray(lon)
    return ((lon + 180.0) % 360.0) - 180.0

def crop_to_latlon_envelope_xy(ds: xr.Dataset,
                               lat_min: float, lat_max: float,
                               lon_min: float, lon_max: float) -> xr.Dataset:
    lat = ds["latitude"]
    lon = ds["longitude"]
    lon180 = xr.apply_ufunc(lon_to_180, lon)

    m = (lat >= lat_min) & (lat <= lat_max) & (lon180 >= lon_min) & (lon180 <= lon_max)
    yy, xx = np.where(m.values)
    if yy.size == 0:
        raise ValueError("Crop box contains no grid points (check bounds / lon wrap).")

    y0, y1 = int(yy.min()), int(yy.max())
    x0, x1 = int(xx.min()), int(xx.max())
    return ds.isel(y=slice(y0, y1 + 1), x=slice(x0, x1 + 1))

def open_month(single_p: Path, pres_p: Path, hgt_p: Path, static_ds: xr.Dataset) -> xr.Dataset:
    data_vars: dict = {}

    # Single levels
    sets_s = open_cfgrib_datasets_safe(single_p)

    data_vars["t2m"]    = clean_da(next(ds["t2m"] for ds in sets_s if "t2m" in ds))
    data_vars["mcc"]    = clean_da(next(ds["mcc"] for ds in sets_s if "mcc" in ds))

    # --- NEW: mean sea level pressure ---
    # Most CERRA single-level GRIBs expose this as "msl"
    data_vars["msl"]    = clean_da(next(ds["msl"] for ds in sets_s if "msl" in ds))

    w10                 = next(ds for ds in sets_s if "si10" in ds)
    data_vars["si10"]   = clean_da(w10["si10"])
    data_vars["wdir10"] = clean_da(w10["wdir10"])

    # Pressure levels
    sets_p = open_cfgrib_datasets_safe(pres_p)
    ds_p   = sets_p[0].rename({"isobaricInhPa": "level"})
    for v in ["z", "t", "u", "v", "r"]:
        data_vars[v] = clean_da(ds_p[v])

    # Height levels (100 m)
    sets_h = open_cfgrib_datasets_safe(hgt_p)
    h      = sets_h[0].sel(heightAboveGround=100.0)
    data_vars["ws100"]   = clean_da(h["ws"])
    data_vars["wdir100"] = clean_da(h["wdir"])

    # Static
    data_vars["orog"] = clean_da(static_ds["orog"])
    data_vars["lsm"]  = clean_da(static_ds["lsm"])

    return xr.Dataset(data_vars)


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(META_CSV)
    df[COL_CAP]  = pd.to_numeric(df[COL_CAP], errors="coerce")
    df[COL_TURB] = pd.to_numeric(df[COL_TURB], errors="coerce")
    df = df.dropna(subset=[COL_FARM, COL_LAT, COL_LON]).copy()

    # enforce [-180,180]
    df[COL_LON] = lon_to_180(df[COL_LON].values)
    return df

def load_power_series(target_times: pd.DatetimeIndex, farms: list[str]) -> pd.DataFrame:
    """
    Match the ERA5 logic:
      - expect a 'time' column if present, otherwise treat first col as time index
      - parse as UTC then drop tz => tz-naive
      - keep only farm columns that exist in power_df
      - reindex to target_times
    """
    df = pd.read_csv(POWER_CSV)

    # Case A: explicit time column
    if "time" in df.columns:
        time_parsed = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True, errors="coerce"))
        df = df.drop(columns=["time"])
        idx = time_parsed.tz_convert("UTC").tz_localize(None)
        df.index = idx
    else:
        # Case B: first column is index already
        df = pd.read_csv(POWER_CSV, index_col=0, parse_dates=True)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

    df = df.apply(pd.to_numeric, errors="coerce")

    keep_cols = [c for c in df.columns if c in farms]
    if not keep_cols:
        raise ValueError("No overlapping farm columns between metadata and power CSV.")
    df = df[keep_cols]

    out = df.reindex(target_times)

    nan_share = float(out.isna().mean().mean())
    print(f"[POWER] NaN share after reindex to ds.time: {nan_share:.3f}")
    print("[POWER] time head (power):", out.index[:3].to_list())
    print("[POWER] time head (ds):   ", target_times[:3].to_list())

    return out

# -------------------------
# PHASE 1: BUILD ZARR (native y/x)
# -------------------------
def build_cerra_xy_zarr():
    start = pd.Timestamp(START)
    end   = pd.Timestamp(END)

    static_ds = open_cfgrib_datasets_safe(RAW_DIR / "cerra_static.grib")[0]

    if OUT_ZARR.exists():
        shutil.rmtree(OUT_ZARR)

    first_write = True

    months = pd.date_range(start, end, freq="MS")
    for t in months:
        y, m = t.year, t.month
        print(f"\nProcessing {y}-{m:02d}...")

        single_p = RAW_DIR / f"cerra_single_{y:04d}_{m:02d}.grib"
        pres_p   = RAW_DIR / f"cerra_pressure_{y:04d}_{m:02d}.grib"
        hgt_p    = RAW_DIR / f"cerra_height_{y:04d}_{m:02d}.grib"

        # Skip missing files cleanly
        missing = [p for p in [single_p, pres_p, hgt_p] if not p.exists()]
        if missing:
            print("[WARN] Missing GRIB(s), skipping month:", ", ".join(str(p.name) for p in missing))
            continue

        ds = open_month(single_p, pres_p, hgt_p, static_ds)

        ds = ds.sortby("time").sel(time=slice(start, end))
        if ds.sizes.get("time", 0) == 0:
            continue

        if CROP:
            ds = crop_to_latlon_envelope_xy(ds, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

        # enforce lon [-180,180] on (y,x)
        ds = ds.assign_coords(longitude=xr.apply_ufunc(lon_to_180, ds.longitude))

        if first_write:
            enc = {}
            for v in ds.data_vars:
                dims = ds[v].dims
                if dims == ("time", "level", "y", "x"):
                    enc[v] = {"chunks": (T_CHUNK, LEV_CHUNK, Y_CHUNK, X_CHUNK), "dtype": "float32"}
                elif dims == ("time", "y", "x"):
                    enc[v] = {"chunks": (T_CHUNK, Y_CHUNK, X_CHUNK), "dtype": "float32"}
                elif dims == ("level", "y", "x"):
                    enc[v] = {"chunks": (LEV_CHUNK, Y_CHUNK, X_CHUNK), "dtype": "float32"}
                elif dims == ("y", "x"):
                    enc[v] = {"chunks": (Y_CHUNK, X_CHUNK), "dtype": "float32"}
                else:
                    enc[v] = {"dtype": "float32"}

            ds.to_zarr(OUT_ZARR, mode="w", consolidated=True, encoding=enc)
            first_write = False
        else:
            ds.to_zarr(OUT_ZARR, mode="a", append_dim="time", consolidated=True)

        del ds

    print("\nZarr build complete.")

# -------------------------
# PHASE 2: APPEND TURBINES + POWER (native y/x)
# -------------------------
def append_turbine_vars_xy():
    ds = xr.open_zarr(OUT_ZARR, consolidated=True)
    meta = load_metadata()

    lon2d = ds.longitude.values
    lat2d = ds.latitude.values
    ny, nx = lon2d.shape

    # KD-tree in (lon,lat) using the GRID convention (already [-180,180])
    tree = cKDTree(np.c_[lon2d.ravel(), lat2d.ravel()])
    _, flat_idx = tree.query(np.c_[meta[COL_LON].values, meta[COL_LAT].values], k=1)
    meta["flat_idx"] = flat_idx.astype(int)

    # Farm name overlap check (prevents silent all-NaN)
    meta_farms = set(meta[COL_FARM].astype(str))
    print("[META] farms:", len(meta_farms))

    # ---- 1) static vars ----
    mask = np.full((ny, nx), np.nan, dtype="float32")
    turb = np.full((ny, nx), np.nan, dtype="float32")
    cap  = np.full((ny, nx), np.nan, dtype="float32")

    grouped = meta.groupby("flat_idx").agg({COL_TURB: "sum", COL_CAP: "sum"})
    for fi, row in grouped.iterrows():
        iy, ix = np.unravel_index(int(fi), (ny, nx))
        mask[iy, ix] = 1.0
        turb[iy, ix] = float(row[COL_TURB])
        cap[iy, ix]  = float(row[COL_CAP])

    xr.Dataset({
        VAR_MASK: (("y", "x"), mask),
        VAR_TURB: (("y", "x"), turb),
        VAR_CAP:  (("y", "x"), cap),
    }).to_zarr(OUT_ZARR, mode="a", consolidated=True)
    print("Static metadata written.")

    # ---- 2) power ----
    target_times = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    power_df = load_power_series(target_times, farms=list(meta_farms))

    power_cols = set(map(str, power_df.columns))
    overlap = meta_farms & power_cols
    print("[POWER] column overlap with meta farms:", len(overlap))

    g = zarr.open_group(str(OUT_ZARR), mode="a")
    z_pow = g.create_dataset(
        VAR_POW,
        shape=(ds.sizes["time"], ny, nx),
        chunks=(T_CHUNK, Y_CHUNK, X_CHUNK),
        dtype="float32",
        fill_value=np.nan,
        overwrite=True,
    )
    z_pow.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]

    # Write: sum farms per cell; NaN if all are NaN at that time
    cell_to_farms = meta.groupby("flat_idx")[COL_FARM].apply(list).to_dict()

    wrote_cells = 0
    for fi, farms in cell_to_farms.items():
        cols = [str(f) for f in farms if str(f) in power_df.columns]
        if not cols:
            continue

        arr = power_df[cols].sum(axis=1, min_count=1).to_numpy(dtype="float32")
        if np.all(np.isnan(arr)):
            continue

        iy, ix = np.unravel_index(int(fi), (ny, nx))
        z_pow[:, iy, ix] = arr
        wrote_cells += 1

    zarr.consolidate_metadata(str(OUT_ZARR))
    print(f"Power written to {wrote_cells} grid cells.")
    print("Power data written and metadata consolidated.")

# -------------------------
def main():
    print("=" * 60)
    print("  CERRA → Zarr (native y/x, lat/lon envelope crop, robust cfgrib)")
    print(f"  Period : {START}  →  {END}")
    print(f"  Output : {OUT_ZARR}")
    if CROP:
        print(f"  Crop   : lat[{LAT_MIN},{LAT_MAX}], lon[{LON_MIN},{LON_MAX}] -> envelope in (y,x)")
    print("=" * 60)

    build_cerra_xy_zarr()
    append_turbine_vars_xy()
    print("\nDone.")

if __name__ == "__main__":
    main()
