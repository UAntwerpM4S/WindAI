"""
Create an inner subset of the CERRA crop without modifying the original store.

Reads:
    data/NorthSea/Cerra/cerra_crop.zarr (Lambert conformal grid)

Writes:
    data/NorthSea/Cerra/cerra_inner.zarr (same vars, limited lat/lon window)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

# Paths
DATA_DIR = Path("/mnt/data/weatherloss/WindPower/data")
SRC_ZARR = DATA_DIR / "NorthSea" / "Cerra" / "cerra_outer.zarr"
DST_ZARR = DATA_DIR / "NorthSea" / "Cerra" / "cerra_inner2.zarr"

# Desired geographic window
LAT_MIN, LAT_MAX = 50.5, 57.0
LON_MIN, LON_MAX = -4.0, 10.0


def _resolve_lon_bounds(lon_coord: np.ndarray) -> Tuple[float, float]:
    """Return lon_min/lon_max in the dataset's convention (0..360 or -180..180)."""
    lon_max_val = np.nanmax(lon_coord)
    if lon_max_val > 180:  # 0..360 convention
        return (LON_MIN + 360) % 360, (LON_MAX + 360) % 360
    return LON_MIN, LON_MAX


def _compute_slices(ds: xr.Dataset) -> Tuple[slice, slice]:
    """Find minimal y/x slices covering the requested lat/lon box on the LCC grid."""
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError("Dataset must contain 'latitude' and 'longitude' coordinates.")

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    lon_min, lon_max = _resolve_lon_bounds(lon)
    lon_min, lon_max = sorted((lon_min, lon_max))
    lat_min, lat_max = sorted((LAT_MIN, LAT_MAX))

    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    if not np.any(mask):
        raise ValueError("No grid points found within the requested lat/lon bounds.")

    yy, xx = np.where(mask)
    y_slice = slice(int(yy.min()), int(yy.max()) + 1)
    x_slice = slice(int(xx.min()), int(xx.max()) + 1)
    return y_slice, x_slice


def build_inner():
    xr.set_options(use_new_combine_kwarg_defaults=True)
    if not SRC_ZARR.exists():
        raise FileNotFoundError(f"Source Zarr not found: {SRC_ZARR}")

    # 1. Open lazily
    ds = xr.open_zarr(SRC_ZARR, consolidated=True)

    # Compute inner slices
    y_slice, x_slice = _compute_slices(ds)

    # 2. Select the inner region (still Dask-backed)
    ds_inner = ds.isel(y=y_slice, x=x_slice)
    
    # 3. Define a new, efficient chunking scheme for the dimensions
    y_len = ds_inner.sizes["y"]
    x_len = ds_inner.sizes["x"]

    chunk_spec = {
        "time": 24,  # ~3 days of 3‑hourly steps
        "y": min(200, y_len),
        "x": min(200, x_len),
        "level": -1,  # keep levels together
    }

    # Apply rechunking with dimension -> size mapping (what xarray expects)
    ds_rechunked = ds_inner.chunk(chunk_spec)
    
    # Drop any existing chunk metadata
    for name in list(ds_rechunked.variables):
        ds_rechunked[name].encoding.pop("chunks", None)

    # Ensure we don't overwrite the source store
    if DST_ZARR.exists():
        shutil.rmtree(DST_ZARR)

    # 6. Write to Zarr.
    ds_rechunked.to_zarr(DST_ZARR, mode="w", consolidated=True)

    try:
        import zarr
        zarr.consolidate_metadata(str(DST_ZARR))
    except Exception as exc:  # best-effort consolidation
        print(f"Warning: failed to consolidate metadata for {DST_ZARR}: {exc}")


if __name__ == "__main__":
    build_inner()
