from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from scipy.spatial import cKDTree  # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature


TARGET_VAR = "ws100"  # "t2m", "ws10", "ws100"

FORECAST_DIRS: List[Path] = [
    Path("/mnt/weatherloss/WindPower/inference/CI/GraphTransformerNew"),
     Path("/mnt/weatherloss/WindPower/inference/CI/TransformerNew"),
     Path("/mnt/weatherloss/WindPower/inference/CI/GNNNew")
]

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/NorthSea/Cerra/cerra_CI_HR.zarr")

LEAD_HOURS: List[int] = list(range(0, 73, 3))
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
#INIT_END   = pd.Timestamp("2024-08-03 21:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")

MAX_DIST_KM = 5.0

BIAS_DIR = Path("CI_bias_plots")
# -------------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init_time(path: Path) -> pd.Timestamp:
    return pd.to_datetime(FORECAST_FILE_RE.search(path.name).group(1), format="%Y%m%d%H%M%S", utc=True)


def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def list_forecast_files(d: Path, start: pd.Timestamp, end: pd.Timestamp) -> List[Path]:
    start, end = to_utc(start), to_utc(end)
    out = []
    for f in sorted(d.glob("forecast_*.nc")):
        it = parse_init_time(f)
        if start <= it <= end:
            out.append(f)
    return out


def common_init_times(dirs: Sequence[Path], start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    sets = []
    for d in dirs:
        sets.append({parse_init_time(f) for f in list_forecast_files(d, start, end)})
    return sorted(set.intersection(*sets))


def init_to_file_map(
    d: Path, inits: Sequence[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp
) -> Dict[pd.Timestamp, Path]:
    files = list_forecast_files(d, start, end)
    m = {parse_init_time(f): f for f in files}
    return {it: m[it] for it in inits}


def lonlat_to_xy_km(lon: np.ndarray, lat: np.ndarray, lat0: float) -> np.ndarray:
    R_km = 6371.0
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    lat0_rad = np.deg2rad(lat0)
    x = R_km * np.deg2rad(lon) * np.cos(lat0_rad)
    y = R_km * np.deg2rad(lat)
    return np.column_stack([x, y])


def build_truth_points(ds_truth: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat2d = ds_truth["latitude"].values
    lon2d = ds_truth["longitude"].values
    lat = lat2d.reshape(-1)
    lon = lon2d.reshape(-1)
    lat0 = float(np.nanmean(lat))
    xy = lonlat_to_xy_km(lon, lat, lat0)
    return lat, lon, xy


def build_forecast_points(sample_file: Path, lat0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(sample_file) as ds:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
    xy = lonlat_to_xy_km(lon, lat, lat0)
    return lat, lon, xy


def nearest_mapping_all_dirs(
    ds_truth: xr.Dataset,
    files_by_dir: Dict[Path, Dict[pd.Timestamp, Path]],
    inits: Sequence[pd.Timestamp],
    max_dist_km: float,
) -> Tuple[np.ndarray, Dict[Path, np.ndarray], np.ndarray]:
    truth_lat, _, truth_xy = build_truth_points(ds_truth)
    lat0 = float(np.nanmean(truth_lat))

    first_init = inits[0]
    dir_fc_idx_full: Dict[Path, np.ndarray] = {}
    dir_dist_full: Dict[Path, np.ndarray] = {}

    for d in files_by_dir:
        sample = files_by_dir[d][first_init]
        _, _, fc_xy = build_forecast_points(sample, lat0)
        tree = cKDTree(fc_xy)
        dist, idx = tree.query(truth_xy, k=1)
        dir_fc_idx_full[d] = idx.astype(np.int64)
        dir_dist_full[d] = dist.astype(np.float64)

    max_dist = np.zeros(truth_xy.shape[0], dtype=np.float64)
    ok = np.ones(truth_xy.shape[0], dtype=bool)
    for d in files_by_dir:
        ok &= dir_dist_full[d] <= max_dist_km
        max_dist = np.maximum(max_dist, dir_dist_full[d])

    truth_keep_idx = np.where(ok)[0].astype(np.int64)
    keep_dist_km = max_dist[truth_keep_idx]
    dir_fc_idx = {d: dir_fc_idx_full[d][truth_keep_idx] for d in files_by_dir}
    return truth_keep_idx, dir_fc_idx, keep_dist_km

def save_bias_per_lead_netcdf(
    bias_3d: np.ndarray,  # Shape: (leads, y, x)
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    leads: np.ndarray,
    out_path: Path,
    var_name: str,
    model_name: str,
) -> None:
    ds_out = xr.Dataset(
        data_vars={
            "bias": (("lead_time", "y", "x"), bias_3d.astype(np.float32)),
        },
        coords={
            "lead_time": leads,
            "y": np.arange(bias_3d.shape[1]),
            "x": np.arange(bias_3d.shape[2]),
            "latitude": (("y", "x"), lat2d.astype(np.float32)),
            "longitude": (("y", "x"), lon2d.astype(np.float32)),
        },
        attrs={
            "title": "Bias map per lead time",
            "model": model_name,
            "bias_definition": "forecast - truth",
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print(f"Saved lead-dependent NetCDF: {out_path}")


def main() -> None:
    start, end = to_utc(INIT_START), to_utc(INIT_END)
    lead_list = sorted(list(set(map(int, LEAD_HOURS))))
    lead_to_idx = {val: i for i, val in enumerate(lead_list)}

    inits = common_init_times(FORECAST_DIRS, start, end)
    filemaps = {d: init_to_file_map(d, inits, start, end) for d in FORECAST_DIRS}

    ds_truth = xr.open_zarr(CERRA_PATH)
    truth_var = ds_truth[TARGET_VAR]
    truth_time_index = pd.DatetimeIndex(pd.to_datetime(truth_var["time"].values))
    
    lat2d, lon2d = ds_truth["latitude"].values, ds_truth["longitude"].values
    y, x = ds_truth.sizes["y"], ds_truth.sizes["x"]
    
    truth_keep_idx, dir_fc_idx, _ = nearest_mapping_all_dirs(ds_truth, filemaps, inits, MAX_DIST_KM)

    for d in FORECAST_DIRS:
        fc_values_idx = dir_fc_idx[d]
        
        sum_bias = np.zeros((len(lead_list), len(truth_keep_idx)), dtype=np.float64)
        count = np.zeros((len(lead_list), len(truth_keep_idx)), dtype=np.int64)

        for it in inits:
            with xr.open_dataset(filemaps[d][it]) as ds_fc:
                valid_utc = pd.to_datetime(ds_fc["time"].values).tz_localize("UTC")
                leads = ((valid_utc - it) / np.timedelta64(1, "h")).astype(int)

                for i, lead_val in enumerate(leads):
                    if lead_val not in lead_to_idx:
                        continue
                    
                    v_time = valid_utc[i].replace(tzinfo=None) # match truth index
                    if v_time not in truth_time_index:
                        continue
                    
                    lead_idx = lead_to_idx[lead_val]
                    
                    # Get forecast and truth for this specific lead and time
                    fc = ds_fc[TARGET_VAR].isel(time=i, values=fc_values_idx).values
                    tr = truth_var.sel(time=v_time).stack(z=("y", "x")).isel(z=truth_keep_idx).values
                    
                    bias = fc - tr
                    finite = np.isfinite(bias)
                    sum_bias[lead_idx] += np.where(finite, bias, 0.0)
                    count[lead_idx] += finite.astype(np.int64)

        # Reshape back to (Leads, Y, X)
        mean_bias_3d = np.full((len(lead_list), y, x), np.nan, dtype=np.float32)
        for l_idx in range(len(lead_list)):
            ok = count[l_idx] > 0
            if not np.any(ok): continue
            
            flat_bias = np.full(y * x, np.nan, dtype=np.float32)
            flat_bias[truth_keep_idx[ok]] = (sum_bias[l_idx][ok] / count[l_idx][ok])
            mean_bias_3d[l_idx] = flat_bias.reshape(y, x)

        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%Z")
        out_nc = BIAS_DIR / f"bias_per_lead_{TARGET_VAR}_{d.name}.nc"
        save_bias_per_lead_netcdf(mean_bias_3d, lat2d, lon2d, np.array(lead_list), out_nc, TARGET_VAR, d.name)

    ds_truth.close()


if __name__ == "__main__":
    main()
