"""
Crops raw CERRA GRIB files to the North Sea domain 
Input:
    data/cerra_boz/raw_grib/
        cerra_pressure_YYYY_MM*.grib
        cerra_height_YYYY_MM*.grib
        cerra_single_YYYY_MM*.grib
        cerra_static.grib

Output:
    data/NorthSea/Cerra/cerra_crop.zarr  (consolidated Zarr)
"""

from __future__ import annotations

import glob
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable

import cfgrib
import numpy as np
import xarray as xr

# --- Configuration ---
DATA_DIR = Path("/mnt/data/weatherloss/WindPower/data")
RAW_GRIB_DIR = DATA_DIR / "cerra_boz" / "raw_grib"
OUT_DIR = DATA_DIR / "NorthSea" / "Cerra"
OUT_ZARR = OUT_DIR / "cerra_crop.zarr"

# Spatial window (same as crop.ipynb; larger than original BOZ)
LAT_MIN, LAT_MAX = 49.0, 59.0
LON_MIN, LON_MAX = -6.0, 12.0

# Time window to keep
T_START = "2020-01-01"
T_END = "2025-07-31 21:00"


# --- Utility Functions ---


def index_box(ds: xr.Dataset, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX):
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    if np.any(lon > 180):
        lon = ((lon + 180) % 360) - 180
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    if not np.any(mask):
        raise ValueError("No grid points in specified box.")
    yy, xx = np.where(mask)
    return slice(int(yy.min()), int(yy.max()) + 1), slice(int(xx.min()), int(xx.max()) + 1)


def open_all_messages(path: Path):
    return cfgrib.open_datasets(path, backend_kwargs={"decode_timedelta": True, "indexpath": ""})


def crop_and_merge(msgs, ys, xs):
    cropped = []
    for ds in msgs:
        dsc = ds.isel(y=ys, x=xs)
        if "heightAboveGround" in dsc.coords and dsc.coords["heightAboveGround"].size == 1:
            dsc = dsc.reset_coords("heightAboveGround", drop=True).squeeze()
        cropped.append(dsc)
    aligned = xr.align(*cropped, join="inner")
    return xr.merge(aligned, compat="override", combine_attrs="drop")


def drop_common(ds: xr.Dataset) -> xr.Dataset:
    drop_candidates = ["valid_time", "step", "surface", "meanSea", "entireAtmosphere"]
    drops = [v for v in drop_candidates if v in ds.variables or v in ds.coords]
    return ds.drop_vars(drops, errors="ignore")


def dir_to_sin_cos(da_deg: xr.DataArray):
    rad = np.deg2rad((da_deg % 360.0))
    return np.sin(rad), np.cos(rad)


def flatten_height(ds: xr.Dataset) -> xr.Dataset:
    vars_with_hag = [v for v in ds.data_vars if "heightAboveGround" in ds[v].dims]
    new_vars = {}
    for v in vars_with_hag:
        for h in ds["heightAboveGround"].values:
            vname = f"{v}{int(h)}"
            new_da = ds[v].sel(heightAboveGround=h).squeeze(drop=True)
            new_da.name = vname
            new_da.attrs.update({"long_name": f"{v} at {int(h)} m", "height": int(h)})
            new_vars[vname] = new_da
    ds = ds.assign(**new_vars)
    ds = ds.drop_vars(vars_with_hag + ["heightAboveGround"])
    return ds


def month_key(path: Path) -> str | None:
    """Return YYYY-MM key from filename or None if it cannot be parsed."""
    m = re.search(r"_(\d{4})_(\d{2})", path.name)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def map_months(files: Iterable[str | Path]) -> Dict[str, Path]:
    mapped: Dict[str, Path] = {}
    for f in files:
        key = month_key(Path(f))
        if key:
            mapped[key] = Path(f)
    return mapped


def add_time_features(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds  # Skip if no time dimension

    year_len = xr.where(ds.time.dt.is_leap_year, 366, 365)
    doy_phase = 2 * np.pi * (ds.time.dt.dayofyear - 1) / year_len
    ds = ds.assign(
        doy_sin=xr.DataArray(np.sin(doy_phase), dims=["time"], coords={"time": ds.time}),
        doy_cos=xr.DataArray(np.cos(doy_phase), dims=["time"], coords={"time": ds.time}),
    )
    return ds


def add_direction_features(ds: xr.Dataset) -> xr.Dataset:
    if "wdir10" in ds:
        wdir10_sin, wdir10_cos = dir_to_sin_cos(ds["wdir10"])
        ds = ds.assign(
            wdir10_sin=wdir10_sin.assign_attrs(long_name="sine of 10m wind direction", units="1"),
            wdir10_cos=wdir10_cos.assign_attrs(long_name="cosine of 10m wind direction", units="1"),
        )
        ds = ds.drop_vars("wdir10")
    if "wdir" in ds:
        wdir_sin, wdir_cos = dir_to_sin_cos(ds["wdir"])
        ds = ds.assign(
            wdir_sin=wdir_sin.assign_attrs(long_name="sine of wind direction", units="1"),
            wdir_cos=wdir_cos.assign_attrs(long_name="cosine of wind direction", units="1"),
        )
        ds = ds.drop_vars("wdir")
    return ds


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    # Rename, cast, coords
    if "isobaricInhPa" in ds.coords:
        ds = ds.rename({"isobaricInhPa": "level"})
    ds = ds.map(lambda x: x.astype("float32") if np.issubdtype(x.dtype, np.floating) else x)
    # Ensure longitude is -180 to 180 if it exists
    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)

    # Convert orography to surface geopotential
    if "orog" in ds:
        g = 9.80665
        phis = (ds["orog"].astype("float32") * g).assign_attrs(
            long_name="surface geopotential", units="m2 s-2", description="Φs = g * orography"
        )
        ds = ds.drop_vars("orog").assign(surface_geopotential=phis)

    # Rename si10 -> ws10 if present
    if "si10" in ds:
        ds = ds.rename({"si10": "ws10"})
        ds["ws10"].attrs.update({"long_name": "10 m wind speed", "units": "m s-1"})
    return ds


# --- Core Processing Function ---


def build_month_dataset(
    pressure_path: Path, height_path: Path, single_path: Path, y_slice: slice, x_slice: slice
) -> xr.Dataset:
    """Loads, crops, merges, and processes time-dependent data for a single month."""

    def load_group(files):
        datasets = []
        for path in files:
            msgs = open_all_messages(path)
            datasets.append(crop_and_merge(msgs, y_slice, x_slice))
        merged = xr.combine_by_coords(datasets, combine_attrs="drop")
        return drop_common(merged)

    ds_pressure = load_group([pressure_path])
    ds_height = load_group([height_path])
    ds_single = load_group([single_path])

    # Time-slice *before* merging to reduce memory footprint
    ds_pressure = ds_pressure.sel(time=slice(T_START, T_END))
    ds_height = ds_height.sel(time=slice(T_START, T_END))
    ds_single = ds_single.sel(time=slice(T_START, T_END))

    # Merge ONLY the time-dependent datasets
    ds = xr.merge([ds_height, ds_pressure, ds_single], compat="override", combine_attrs="drop")

    if "time" not in ds.coords or ds.time.size == 0:
        # Return empty dataset if no data for the month
        return ds.drop_vars(ds.data_vars)

    ds = add_time_features(ds)
    ds = add_direction_features(ds)

    if "heightAboveGround" in ds.coords:
        ds = flatten_height(ds)

    ds = normalize_dataset(ds)
    ds = ds.sortby("time")

    # NOTE: no manual chunking here; we let xarray write contiguous arrays
    return ds


# --- Main Execution ---


def main():
    xr.set_options(use_new_combine_kwarg_defaults=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pressure_files = sorted(glob.glob(str(RAW_GRIB_DIR / "*pressure_*.grib")))
    height_files = sorted(glob.glob(str(RAW_GRIB_DIR / "*height_*.grib")))
    single_files = sorted(glob.glob(str(RAW_GRIB_DIR / "*single_*.grib")))
    static_path = RAW_GRIB_DIR / "cerra_static.grib"

    if not (pressure_files and height_files and single_files and static_path.exists()):
        raise SystemExit("Missing GRIB files in raw_grib; please check downloads.")

    pressure_by_month = map_months(pressure_files)
    height_by_month = map_months(height_files)
    single_by_month = map_months(single_files)
    months = sorted(set(pressure_by_month) & set(height_by_month) & set(single_by_month))
    if not months:
        raise SystemExit("No overlapping monthly GRIB files found.")

    # 1. Determine crop slices
    y_slice, x_slice = index_box(
        cfgrib.open_dataset(
            pressure_files[0],
            backend_kwargs={"indexpath": "", "decode_timedelta": True},
        )
    )

    # 2. Process and write static data
    ds_static = crop_and_merge(open_all_messages(static_path), y_slice, x_slice)
    ds_static = drop_common(ds_static)
    ds_static = normalize_dataset(ds_static)

    # Filter to only keep static variables (those without a time dimension)
    keep_static = [v for v in ds_static.data_vars if "time" not in ds_static[v].dims]
    if keep_static:
        ds_static = ds_static[keep_static]

    # Drop any lingering time coordinate/variables from the static set to avoid conflicts later
    for coord in list(ds_static.coords):
        if coord == "time" or "time" in ds_static[coord].dims:
            ds_static = ds_static.drop_vars(coord)

    # Clean up previous Zarr store
    if OUT_ZARR.exists():
        shutil.rmtree(OUT_ZARR)

    store_initialized = False
    wrote_time_data = False

    # Write static data first (if any)
    if ds_static.data_vars:
        ds_static.to_zarr(OUT_ZARR, mode="w", consolidated=False)
        store_initialized = True
        print(f"Wrote static data (variables: {list(ds_static.data_vars)}) to {OUT_ZARR}")
    else:
        print("Warning: No static data found to write. Zarr store will be initialized by first time-step data.")

    # 3. Process and append monthly time-dependent data
    for month in months:
        ds = build_month_dataset(
            pressure_by_month[month],
            height_by_month[month],
            single_by_month[month],
            y_slice,
            x_slice,
        )

        if "time" not in ds.coords or ds.time.size == 0:
            print(f"Skipped month {month}: No data found in time window.")
            continue

        # First time-dependent write
        if not wrote_time_data:
            if store_initialized:
                # Static already written, add time-dependent variables (no append_dim yet)
                ds.to_zarr(OUT_ZARR, mode="a", consolidated=False)
                mode_used = "a (first time data, static kept)"
            else:
                # No static: create the store from this dataset
                ds.to_zarr(OUT_ZARR, mode="w", consolidated=False)
                store_initialized = True
                mode_used = "w (first time data, no static)"
        else:
            # Subsequent months: append along time
            ds.to_zarr(
                OUT_ZARR,
                mode="a",
                append_dim="time",
                consolidated=False,
            )
            mode_used = "a (append_dim='time')"

        wrote_time_data = True
        print(f"Wrote month {month} with {ds.time.size} steps [{mode_used}].")

    if not wrote_time_data:
        raise SystemExit("No time-dependent data written; check time window filters or input files.")

    # 4. Consolidate metadata
    try:
        import zarr

        zarr.consolidate_metadata(str(OUT_ZARR))
    except Exception as exc:  # pragma: no cover - consolidation best effort
        print(f"Warning: failed to consolidate metadata: {exc}")
    else:
        print(f"Consolidated metadata at {OUT_ZARR}")


if __name__ == "__main__":
    main()

