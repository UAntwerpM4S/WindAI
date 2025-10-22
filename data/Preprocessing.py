"""
Build a single consolidated CERRA dataset for the Belgian Offshore Zone (BOZ)
from monthly NetCDFs (surface, height, pressure) + static, then add simple
diagnostics and write Zarr + NetCDF.

Inputs (already cropped to BOZ, Lambert Conformal grid, 157x211):
  /mnt/data/weatherloss/WindPower/data/cerra_boz/nc_boz/
    - cerra_single_YYYY_MM_BOZ.nc   (surface)
    - cerra_height_YYYY_MM_BOZ.nc   (heightAboveGround: 50/100/150/200)
    - cerra_pressure_YYYY_MM_BOZ.nc (isobaricInhPa: 1000..500)
    - cerra_static_BOZ.nc           (orog, lsm)

Output:
  - Cerra_boz.zarr (chunked)
  - Cerra_boz.nc   (monolithic NetCDF)
"""

import os
import glob
import numpy as np
import xarray as xr

def dir_to_sin_cos(da_deg: xr.DataArray):
    """Return sin, cos of a wind direction (degrees). Keeps dims/coords."""
    rad = np.deg2rad((da_deg % 360.0))
    return np.sin(rad), np.cos(rad)

def _preprocess_drop_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Minimal preprocess for open_mfdataset to avoid common coordinate clashes.
    Drops non-essential scalar coords if present.
    """
    drop_candidates = ["valid_time", "step", "surface", "meanSea", "entireAtmosphere"]
    drops = [v for v in drop_candidates if (v in ds.coords) or (v in ds.variables)]
    return ds.drop_vars(drops, errors="ignore")

BASE = "/mnt/data/weatherloss/WindPower/data/cerra_boz/nc_boz"
OUT_ZARR = "Cerra_boz.zarr"
OUT_NC = "Cerra_boz.nc"

surface_files = sorted(glob.glob(os.path.join(BASE, "cerra_single_????_??_BOZ.nc")))
height_files = sorted(glob.glob(os.path.join(BASE, "cerra_height_????_??_BOZ.nc")))
pressure_files = sorted(glob.glob(os.path.join(BASE, "cerra_pressure_????_??_BOZ.nc")))
static_file = os.path.join(BASE, "cerra_static_BOZ.nc")

print(f"Found {len(surface_files)} surface, {len(height_files)} height, {len(pressure_files)} pressure files")

# ---------------------------------------------------------------------
# Open & concat per group, then time-window select
# ---------------------------------------------------------------------

t_start = "2023-01-01"
t_end = "2025-07-31 21:00"

print("Opening surface...")
ds_sfc = xr.open_mfdataset(surface_files, combine="by_coords", preprocess=_preprocess_drop_coords)
print("Opening height...")
ds_hgt = xr.open_mfdataset(height_files, combine="by_coords", preprocess=_preprocess_drop_coords)
print("Opening pressure...")
ds_prs = xr.open_mfdataset(pressure_files, combine="by_coords", preprocess=_preprocess_drop_coords)
print("Opening static...")
ds_sta = xr.open_dataset(static_file)

# Time slicing
ds_sfc = ds_sfc.sel(time=slice(t_start, t_end))
ds_hgt = ds_hgt.sel(time=slice(t_start, t_end))
ds_prs = ds_prs.sel(time=slice(t_start, t_end))

# Keep only static fields we need
keep_static = [v for v in ["orog", "lsm"] if v in ds_sta]
ds_sta = ds_sta[keep_static]

# Merge all groups
print("Merging groups...")
cerra = xr.merge([ds_sta, ds_hgt, ds_prs, ds_sfc], compat="override", combine_attrs="drop")

# ---------------------------------------------------------------------
# Add simple forcings: day-of-year sine/cosine
# ---------------------------------------------------------------------

print("Adding DOY sine/cosine...")
year_len = xr.where(cerra.time.dt.is_leap_year, 366, 365)
doy_phase = 2.0 * np.pi * (cerra.time.dt.dayofyear - 1) / year_len

cerra = cerra.assign(
    doy_sin=xr.DataArray(np.sin(doy_phase), dims=["time"], coords={"time": cerra.time},
                         attrs={"long_name": "sine of Julian day", "units": "1"}),
    doy_cos=xr.DataArray(np.cos(doy_phase), dims=["time"], coords={"time": cerra.time},
                         attrs={"long_name": "cosine of Julian day", "units": "1"}),
)

# ---------------------------------------------------------------------
# Wind-direction sine/cosine (surface and height levels)
# ---------------------------------------------------------------------

print("Computing wind-direction sine/cosine...")
wdir10_sin, wdir10_cos = dir_to_sin_cos(cerra["wdir10"])     # (time, y, x)
wdir_sin, wdir_cos = dir_to_sin_cos(cerra["wdir"])           # (time, heightAboveGround, y, x)

cerra = cerra.assign(
    wdir10_sin=wdir10_sin.assign_attrs(long_name="sine of 10m wind direction", units="1"),
    wdir10_cos=wdir10_cos.assign_attrs(long_name="cosine of 10m wind direction", units="1"),
    wdir_sin=wdir_sin.assign_attrs(long_name="sine of wind direction", units="1"),
    wdir_cos=wdir_cos.assign_attrs(long_name="cosine of wind direction", units="1"),
).drop_vars(["wdir10", "wdir"])

# ---------------------------------------------------------------------
# Synthetic wind power from 100 m wind speed (simple curve)
# ---------------------------------------------------------------------

print("Computing synthetic power from ws@100m...")

ws_pts = np.array(
    [0.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
     11.0, 12.0, 13.0, 14.0, 16.0, 20.0, 24.99, 24.999, 25.0, 25.01, 30.0],
    dtype="float32",
)
p_pts = np.array(
    [0.0, 0.0, 0.05, 0.20, 0.50, 1.00, 1.80, 3.00, 4.50, 6.00,
     7.50, 9.00, 9.50, 9.50, 9.50, 9.50, 9.5, 9.5, 0.00, 0.00, 0.00],
    dtype="float32",
)


ws100 = cerra["ws"].sel(heightAboveGround=100.0)

synthetic_power = xr.apply_ufunc(
    np.interp,
    ws100,                              # x
    xr.DataArray(ws_pts, dims="ws_pts"),
    xr.DataArray(p_pts, dims="ws_pts"),
    input_core_dims=[[], ["ws_pts"], ["ws_pts"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[np.float32],
)

cerra = cerra.assign(
    synthetic_windpower=synthetic_power.assign_attrs(
        long_name="Synthetic wind power from 100 m wind speed",
        units="MW per turbine",
        description="Derived from representative 9.5 MW-class power curve",
    )
)

# ---------------------------------------------------------------------
# Flatten heightAboveGround variables into separate vars (ws50, ws100, ...)
# ---------------------------------------------------------------------

print("Flattening heightAboveGround variables...")
ds = cerra.copy()
vars_with_hag = [v for v in ds.data_vars if "heightAboveGround" in ds[v].dims]

new_vars = {}
for v in vars_with_hag:
    for h in ds["heightAboveGround"].values:
        vname = f"{v}{int(h)}"  # e.g., ws50, wdir_cos150, etc.
        new_da = ds[v].sel(heightAboveGround=h).squeeze(drop=True)
        new_da.name = vname
        new_da.attrs.update({"long_name": f"{v} at {int(h)} m", "height": int(h)})
        new_vars[vname] = new_da

ds = ds.assign(**new_vars)
ds = ds.drop_vars(vars_with_hag + ["heightAboveGround"])
ds = ds.rename({"isobaricInhPa": "level"})
cerra = ds

# ---------------------------------------------------------------------
# Final tweaks: dtypes, coords, orog→surface_geopotential, si10→ws10
# ---------------------------------------------------------------------

print("Final tweaks...")

# Cast floats to float32 to reduce size
cerra = cerra.map(lambda x: x.astype("float32") if np.issubdtype(x.dtype, np.floating) else x)

# Wrap longitude to [-180, 180]
cerra = cerra.assign_coords(longitude=((cerra.longitude + 180.0) % 360.0) - 180.0)

# Convert orography (m) → surface geopotential (m^2 s^-2)
if "orog" in cerra:
    g = 9.80665  # m s^-2
    phis = (cerra["orog"] * g).assign_attrs(
        long_name="surface geopotential", units="m2 s-2",
        description="Φs = g * orography (height above mean sea level)"
    )
    cerra = cerra.drop_vars("orog").assign(surface_geopotential=phis)

# Rename 10 m wind speed
if "si10" in cerra:
    cerra = cerra.rename({"si10": "ws10"})
    cerra["ws10"].attrs.update({"long_name": "10 m wind speed", "units": "m s-1"})

# ---------------------------------------------------------------------
# Write outputs (Zarr + NetCDF)
# ---------------------------------------------------------------------

print("Chunking and writing outputs...")
cerra = cerra.chunk({"time": 24, "y": 157, "x": 211})  # adjust if you like

# Zarr (fast, good for training)
cerra.to_zarr(OUT_ZARR, mode="w", consolidated=True)
print(f"Wrote Zarr: {OUT_ZARR}")

# NetCDF (can be slow for a big file; keep for compatibility)
cerra.to_netcdf(OUT_NC, engine="netcdf4", mode="w")
print(f"Wrote NetCDF: {OUT_NC}")

print("Done.")


