#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import cfgrib
import zarr
from scipy.spatial import cKDTree

xr.set_options(use_new_combine_kwarg_defaults=True)


LAT_MIN, LAT_MAX = 42.5, 63.5
LON_MIN, LON_MAX = -12.0, 11.0


RAW_DIR   = Path("/mnt/weatherloss/WindPower/data/NorthSea/ERA5/raw_grib")
OUT_ZARR  = Path("/mnt/weatherloss/WindPower/data/EGU26/era5_EGU.zarr")
POWER_CSV = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/BE_UK_offshore_per_unit_3H_meanMW_shifted.csv")
META_CSV  = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

REBUILD_ERA5_ZARR = True          
OVERWRITE_IF_EXISTS = True        

VAR_MASK = "turbinemask"
VAR_TURB = "turbinecount"
VAR_CAP  = "capacity"
VAR_POW  = "power"
COL_FARM = "farm"
COL_LAT  = "lat"
COL_LON  = "lon"
COL_CAP  = "capacity_mw"
COL_TURB = "turbines"


def lon_to_180(lon):
    lon = np.asarray(lon)
    return ((lon + 180.0) % 360.0) - 180.0

def fix_lon(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign_coords(longitude=("longitude", lon_to_180(ds.longitude.values)))
    return ds.sortby("longitude")

def clean(da: xr.DataArray) -> xr.DataArray:
    for c in ["valid_time", "surface", "number", "step"]:
        if c in da.coords and c not in da.dims:
            da = da.reset_coords(c, drop=True)
    return da

def crop(ds: xr.Dataset) -> xr.Dataset:
    lat_lo, lat_hi = sorted([LAT_MIN, LAT_MAX])
    lon_lo, lon_hi = sorted([LON_MIN, LON_MAX])

    ds = fix_lon(ds)

    if ds.latitude[0] > ds.latitude[-1]:
        ds = ds.sel(latitude=slice(lat_hi, lat_lo))
    else:
        ds = ds.sel(latitude=slice(lat_lo, lat_hi))

    return ds.sel(longitude=slice(lon_lo, lon_hi))

def pressure_coord_to_level(ds: xr.Dataset) -> xr.Dataset:
    if "isobaricInhPa" in ds.dims or "isobaricInhPa" in ds.coords:
        ds = ds.rename({"isobaricInhPa": "level"})
    return ds

def open_month(single_path: Path, pres_path: Path, static_ds: xr.Dataset) -> xr.Dataset:
    single = cfgrib.open_datasets(str(single_path))[0]
    pres = cfgrib.open_datasets(str(pres_path))[0]

    pres = pressure_coord_to_level(pres)

    ds = xr.Dataset(
        {
            "sp":   clean(single["sp"]),
            "msl":  clean(single["msl"]),
            "u10":  clean(single["u10"]),
            "v10":  clean(single["v10"]),
            "t2m":  clean(single["t2m"]),
            "mcc":  clean(single["mcc"]),
            "u100": clean(single["u100"]),
            "v100": clean(single["v100"]),

            "z": clean(pres["z"]),
            "t": clean(pres["t"]),
            "u": clean(pres["u"]),
            "v": clean(pres["v"]),
            "q": clean(pres["q"]),
            "r": clean(pres["r"]),

            "z_sfc": clean(static_ds["z"]),
            "lsm":   clean(static_ds["lsm"]),
        }
    )

    ds = crop(ds).sortby("time")

    if "step" in ds.coords:
        ds = ds.drop_vars("step")

    ds = pressure_coord_to_level(ds)
    return ds

def month_iter():
    for y in range(2015, 2026):
        for m in range(1, 13):
            if (y > 2025) or (y == 2025 and m > 9):
                return
            yield y, m

def build_era5_zarr():
    print(f"[ERA5] Building Zarr at: {OUT_ZARR}")
    static_ds = cfgrib.open_datasets(str(RAW_DIR / "era5_static.grib"))[0]

    y0, m0 = 2015, 1
    ds0 = open_month(
        RAW_DIR / f"era5_single_{y0:04d}_{m0:02d}.grib",
        RAW_DIR / f"era5_pressure_{y0:04d}_{m0:02d}.grib",
        static_ds,
    )

    if OUT_ZARR.exists():
        import shutil
        shutil.rmtree(OUT_ZARR)

    ds0.to_zarr(str(OUT_ZARR), mode="w", consolidated=False)

    for y, m in month_iter():
        if y == 2015 and m == 1:
            continue
        ds = open_month(
            RAW_DIR / f"era5_single_{y:04d}_{m:02d}.grib",
            RAW_DIR / f"era5_pressure_{y:04d}_{m:02d}.grib",
            static_ds,
        )
        ds.to_zarr(str(OUT_ZARR), mode="a", append_dim="time", consolidated=False)

    zarr.consolidate_metadata(str(OUT_ZARR))
    print("[ERA5] Done building + consolidated metadata.")


def _normalize_lon(lon_series: pd.Series, lon_grid: np.ndarray) -> pd.Series:
    """Match longitude convention of the target grid (0..360 vs -180..180)."""
    lon_max = np.nanmax(lon_grid)
    if lon_max > 180:
        return (lon_series + 360.0) % 360.0
    return lon_series

def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(META_CSV)
    required = {COL_FARM, COL_LAT, COL_LON, COL_CAP, COL_TURB}
    missing = required - set(df.columns)


    df[COL_CAP]  = pd.to_numeric(df[COL_CAP], errors="coerce")
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

def map_farms_to_cells(ds: xr.Dataset, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Snap each farm to nearest ERA5 grid cell using KDTree on (lon,lat) mesh.
    Returns meta_df with integer indices lat_i, lon_i and flat cell id.
    """
    lat = np.asarray(ds["latitude"].values)
    lon = np.asarray(ds["longitude"].values)

    meta_df = meta_df.copy()
    meta_df["lon_norm"] = _normalize_lon(meta_df[COL_LON], lon)

    lon2, lat2 = np.meshgrid(lon, lat)  # (nlat, nlon)
    pts = np.c_[lon2.ravel(), lat2.ravel()]
    tree = cKDTree(pts)

    q = np.c_[meta_df["lon_norm"].to_numpy(), meta_df[COL_LAT].to_numpy()]
    dist, flat_idx = tree.query(q, k=1)

    nlon = lon.size
    meta_df["cell"] = flat_idx.astype(int)
    meta_df["lat_i"] = (flat_idx // nlon).astype(int)
    meta_df["lon_i"] = (flat_idx %  nlon).astype(int)
    meta_df["deg_error"] = dist
    meta_df["km_error_approx"] = meta_df["deg_error"] * 111.0
    return meta_df

def build_static_fields(mapped: pd.DataFrame, nlat: int, nlon: int):
    mask = np.full((nlat, nlon), np.nan, dtype="float32")
    turb = np.full((nlat, nlon), np.nan, dtype="float32")
    cap  = np.full((nlat, nlon), np.nan, dtype="float32")

    # sums over farms sharing the same cell
    for cell, grp in mapped.groupby("cell"):
        i = int(grp["lat_i"].iloc[0])
        j = int(grp["lon_i"].iloc[0])
        mask[i, j] = 1.0
        turb[i, j] = grp[COL_TURB].sum(skipna=True)
        cap[i, j]  = grp[COL_CAP].sum(skipna=True)

    return mask, turb, cap

def build_power_field(ds: xr.Dataset, mapped: pd.DataFrame, power_df: pd.DataFrame) -> xr.DataArray:
    """
    power(time, latitude, longitude) with STRICT NaN propagation for co-located farms:
      - sum across farms in the same cell
      - if any farm is NaN at time t, cell becomes NaN at time t
    Implemented via min_count=len(cols).
    """
    ntime = ds.sizes["time"]
    nlat  = ds.sizes["latitude"]
    nlon  = ds.sizes["longitude"]

    out = np.full((ntime, nlat, nlon), np.nan, dtype="float32")

    farms_available = set(power_df.columns)
    mapped = mapped[mapped[COL_FARM].isin(farms_available)].copy()
    if mapped.empty:
        return xr.DataArray(
            out,
            coords={"time": ds["time"], "latitude": ds["latitude"], "longitude": ds["longitude"]},
            dims=("time", "latitude", "longitude"),
            name=VAR_POW,
        )

    farms_by_cell = mapped.groupby("cell")[COL_FARM].apply(list)

    for cell, farms in farms_by_cell.items():
        row = mapped[mapped["cell"] == cell].iloc[0]
        i = int(row["lat_i"])
        j = int(row["lon_i"])

        cols = [f for f in farms if f in power_df.columns]
        if not cols:
            continue

        summed = power_df[cols].sum(axis=1, skipna=True, min_count=len(cols))
        out[:, i, j] = summed.to_numpy(dtype="float32")

    power_da = xr.DataArray(
        out,
        coords={"time": ds["time"], "latitude": ds["latitude"], "longitude": ds["longitude"]},
        dims=("time", "latitude", "longitude"),
        name=VAR_POW,
    )

    # Ensure NaN where no farms mapped
    has_farm = np.zeros((nlat, nlon), dtype=bool)
    for cell in farms_by_cell.index.astype(int):
        ii = cell // nlon
        jj = cell %  nlon
        has_farm[ii, jj] = True
    farm_mask = xr.DataArray(
        has_farm,
        coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
        dims=("latitude", "longitude"),
    )
    return power_da.where(farm_mask)

def delete_vars_if_exist(store_path: Path, varnames: list[str]):
    g = zarr.open_group(str(store_path), mode="a")
    for v in varnames:
        if v in g:
            print(f"[DELETE] Removing existing variable from zarr: {v}")
            del g[v]

def append_turbine_vars_to_era5_zarr():
    print(f"[LOAD] Opening Zarr: {OUT_ZARR}")
    ds = xr.open_zarr(OUT_ZARR, consolidated=True)

    for d in ("time", "latitude", "longitude"):
        if d not in ds.dims:
            raise ValueError(f"Expected dim '{d}' in ERA5 Zarr. Found dims: {dict(ds.dims)}")

    new_vars = [VAR_POW, VAR_MASK, VAR_TURB, VAR_CAP]
    clashes = [v for v in new_vars if v in ds.data_vars]

    if clashes and not OVERWRITE_IF_EXISTS:
        raise ValueError(
            f"Variables already exist in the Zarr: {clashes}. "
            f"Set OVERWRITE_IF_EXISTS=True if you want to overwrite them."
        )
    if clashes and OVERWRITE_IF_EXISTS:
        delete_vars_if_exist(OUT_ZARR, clashes)

    meta = load_metadata()
    mapped = map_farms_to_cells(ds, meta)

    era5_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    power_df = load_power_series(era5_times, farms=mapped[COL_FARM].tolist())

    nlat = ds.sizes["latitude"]
    nlon = ds.sizes["longitude"]

    mask_arr, turb_arr, cap_arr = build_static_fields(mapped, nlat=nlat, nlon=nlon)
    power_da = build_power_field(ds, mapped, power_df)

    mask_da = xr.DataArray(
        mask_arr,
        coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
        dims=("latitude", "longitude"),
        name=VAR_MASK,
    ).astype("float32")

    turb_da = xr.DataArray(
        turb_arr,
        coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
        dims=("latitude", "longitude"),
        name=VAR_TURB,
    ).astype("float32")

    cap_da = xr.DataArray(
        cap_arr,
        coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
        dims=("latitude", "longitude"),
        name=VAR_CAP,
    ).astype("float32")

    out_ds = xr.Dataset({VAR_POW: power_da, VAR_MASK: mask_da, VAR_TURB: turb_da, VAR_CAP: cap_da})

    # Reasonable default chunking for lat/lon grids
    time_chunk = min(56, ds.sizes["time"])    # 7 days @ 3-hourly (adjust if hourly)
    lat_chunk  = min(64, nlat)
    lon_chunk  = min(64, nlon)
    encoding = {
        VAR_POW:  {"chunks": (time_chunk, lat_chunk, lon_chunk)},
        VAR_MASK: {"chunks": (lat_chunk, lon_chunk)},
        VAR_TURB: {"chunks": (lat_chunk, lon_chunk)},
        VAR_CAP:  {"chunks": (lat_chunk, lon_chunk)},
    }

    print(f"[WRITE] Appending {list(out_ds.data_vars)} to {OUT_ZARR}")
    out_ds.to_zarr(OUT_ZARR, mode="a", consolidated=True, encoding=encoding)
    print("[DONE] Added turbine variables.")


def main():
    if REBUILD_ERA5_ZARR:
        build_era5_zarr()
    append_turbine_vars_to_era5_zarr()

if __name__ == "__main__":
    main()

