#!/usr/bin/env python3

"""Crop BOZ.zarr to a lat/lon box and save as BOZ_inner.zarr.

The source grid is CERRA Lambert Conformal; we use the 2D latitude/longitude
auxiliary coordinates to find the smallest y/x window that covers the target
box, then slice the dataset accordingly.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from pathlib import Path

# Bounds (degrees)
LAT_MIN, LAT_MAX = 51.0, 54.0
LON_MIN, LON_MAX = -4.0, 8.0

# Paths
DATA_ROOT = Path("/mnt/data/weatherloss/WindPower/data")
BOZ_FULL = DATA_ROOT / "BOZ.zarr"
BOZ_INNER = DATA_ROOT / "BOZ_inner.zarr"


def _normalize_lon(lon: np.ndarray) -> np.ndarray:
    """Convert 0..360 longitudes to -180..180 if needed."""
    if np.nanmax(lon) > 180:
        return ((lon + 180) % 360) - 180
    return lon


def compute_slice(ds: xr.Dataset):
    """Return y/x slices covering the requested lat/lon box."""
    lat = np.asarray(ds["latitude"])
    lon = _normalize_lon(np.asarray(ds["longitude"]))

    lat_min, lat_max = sorted((LAT_MIN, LAT_MAX))
    lon_min, lon_max = sorted((LON_MIN, LON_MAX))

    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    if not np.any(mask):
        raise ValueError("No grid points inside requested box.")

    yy, xx = np.where(mask)
    return slice(int(yy.min()), int(yy.max()) + 1), slice(int(xx.min()), int(xx.max()) + 1)


def _safe_rechunk(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure chunk sizes are not larger than the cropped dimensions.

    `open_zarr` preserves the original chunk sizes (157x211 for y/x). After
    cropping to the smaller BOZ window this would leave invalid chunk sizes
    (chunks larger than the dimension), which can confuse downstream readers.
    """
    if ds.chunks is None:
        return ds

    chunks = {}
    for dim, sizes in ds.chunks.items():
        if not sizes:
            continue
        # Keep the original chunk size unless it exceeds the new dimension length.
        chunks[dim] = min(int(sizes[0]), ds.sizes[dim])

    return ds.chunk(chunks)


def main():
    print(f"[LOAD] {BOZ_FULL}")
    ds = xr.open_zarr(BOZ_FULL, consolidated=True)

    y_slice, x_slice = compute_slice(ds)
    print(f"[CROP] y={y_slice}, x={x_slice} -> writing {BOZ_INNER}")

    ds_inner = _safe_rechunk(ds.isel(y=y_slice, x=x_slice))
    ds_inner.to_zarr(BOZ_INNER, mode="w", consolidated=True)
    print("[DONE] Saved BOZ_inner.zarr")


if __name__ == "__main__":
    main()
