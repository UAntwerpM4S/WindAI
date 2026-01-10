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
    Path("/mnt/weatherloss/WindPower/inference/CI/GraphTransformer"),
     Path("/mnt/weatherloss/WindPower/inference/CI/Transformer"),
     Path("/mnt/weatherloss/WindPower/inference/CI/GNN")
]

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/NorthSea/Cerra/cerra_CI_HR.zarr")

LEAD_HOURS: List[int] = list(range(3, 73, 3))
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

def save_bias_netcdf(
    bias_2d: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    out_path: Path,
    var_name: str,
    model_name: str,
) -> None:
    y, x = bias_2d.shape

    ds_out = xr.Dataset(
        data_vars={
            "bias": (("y", "x"), bias_2d.astype(np.float32)),
        },
        coords={
            "y": np.arange(y, dtype=np.int64),
            "x": np.arange(x, dtype=np.int64),
            "latitude": (("y", "x"), lat2d.astype(np.float32)),
            "longitude": (("y", "x"), lon2d.astype(np.float32)),
        },
        attrs={
            "title": "Mean bias map (forecast - truth)",
            "variable": var_name,
            "model": model_name,
            "bias_definition": "forecast - truth",
        },
    )

    ds_out["bias"].attrs.update(
        long_name=f"Mean bias of {var_name} (forecast - truth)",
        units="m/s" if var_name.startswith("ws") else "",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print(f"Saved bias NetCDF: {out_path}")


def plot_bias_map(
    bias_2d: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    # symmetric color limits around 0 (robust to outliers)
    finite = np.isfinite(bias_2d)
    if np.any(finite):
        vmax = float(np.nanpercentile(np.abs(bias_2d[finite]), 99))
        if vmax == 0:
            vmax = float(np.nanmax(np.abs(bias_2d[finite]))) if np.nanmax(np.abs(bias_2d[finite])) > 0 else 1.0
    else:
        vmax = 1.0
    

    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # pcolormesh with curvilinear lat/lon
    m = ax.pcolormesh(
        lon2d, lat2d, bias_2d,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=-1, vmax=1,
        shading="auto",
    )
    #cbar = plt.colorbar(m, ax=ax, shrink=0.4, pad=0.02)
    #cbar.set_label(f"Bias [m/s]",fontsize=16)
    # --- expand extent a bit so edges don't get clipped ---
    lon_min = float(np.nanmin(lon2d))
    lon_max = float(np.nanmax(lon2d))
    lat_min = float(np.nanmin(lat2d))
    lat_max = float(np.nanmax(lat2d))

    pad_deg = 0.5  # try 0.2–1.0 depending on how much you want
    ax.set_extent(
        [lon_min - pad_deg, lon_max + pad_deg, lat_min - pad_deg, lat_max + pad_deg],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7, zorder=3)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    start, end = to_utc(INIT_START), to_utc(INIT_END)
    lead_keep = set(map(int, LEAD_HOURS))

    # 1) common init times across dirs
    inits = common_init_times(FORECAST_DIRS, start, end)
    print(f"Common init runs across all dirs: {len(inits)}")
    filemaps = {d: init_to_file_map(d, inits, start, end) for d in FORECAST_DIRS}

    # 2) open truth once
    ds_truth = xr.open_zarr(CERRA_PATH)
    truth_var = ds_truth[TARGET_VAR]
    truth_time_index = pd.DatetimeIndex(pd.to_datetime(truth_var["time"].values))

    lat2d = ds_truth["latitude"].values
    lon2d = ds_truth["longitude"].values
    y, x = ds_truth.sizes["y"], ds_truth.sizes["x"]
    n_truth = y * x

    truth_keep_idx, dir_fc_idx, keep_dist_km = nearest_mapping_all_dirs(ds_truth, filemaps, inits, MAX_DIST_KM)
    print(f"Truth grid points total: {n_truth}")
    print(f"Kept truth points (<= {MAX_DIST_KM} km in all dirs): {len(truth_keep_idx)}")
    print(f"Nearest-distance stats [km]: min={keep_dist_km.min():.3f}, median={np.median(keep_dist_km):.3f}, max={keep_dist_km.max():.3f}")

    # 4) compute mean bias map per model
    for d in FORECAST_DIRS:
        fc_values_idx = dir_fc_idx[d]  # (K,)

        sum_bias = np.zeros(len(truth_keep_idx), dtype=np.float64)
        count = np.zeros(len(truth_keep_idx), dtype=np.int64)

        for it in inits:
            fpath = filemaps[d][it]
            ds_fc = xr.open_dataset(fpath)

            valid_times = pd.to_datetime(ds_fc["time"].values)  # naive
            valid_utc = valid_times.tz_localize("UTC")
            leads = ((valid_utc - it) / np.timedelta64(1, "h")).astype(int)

            # keep only requested leads
            tmask = np.isin(leads, list(lead_keep))
            if not np.any(tmask):
                ds_fc.close()
                continue

            times_sel = pd.DatetimeIndex(valid_times[tmask])
            leads_sel = leads[tmask].astype(int)

            common_times = times_sel.intersection(truth_time_index)
            if common_times.empty:
                ds_fc.close()
                continue

            common_utc = common_times.tz_localize("UTC")
            common_leads = ((common_utc - it) / np.timedelta64(1, "h")).astype(int).to_numpy()
            keep2 = np.isin(common_leads, list(lead_keep))
            if not np.any(keep2):
                ds_fc.close()
                continue

            common_times = common_times[keep2]

            # forecast on mapped truth points
            fc = ds_fc[TARGET_VAR].sel(time=common_times).isel(values=fc_values_idx).values  # (T, K)
            ds_fc.close()

            # truth on truth points
            tr = (
                truth_var
                .sel(time=common_times)
                .stack(values=("y", "x"))
                .isel(values=truth_keep_idx)
                .values
            )  # (T, K)

            bias = fc - tr  # (T, K)

            finite = np.isfinite(bias)
            sum_bias += np.nansum(np.where(finite, bias, 0.0), axis=0)
            count += np.sum(finite, axis=0).astype(np.int64)

        mean_bias = np.full(len(truth_keep_idx), np.nan, dtype=np.float32)
        ok = count > 0
        mean_bias[ok] = (sum_bias[ok] / count[ok]).astype(np.float32)

        bias_flat = np.full(n_truth, np.nan, dtype=np.float32)
        bias_flat[truth_keep_idx] = mean_bias
        bias_2d = bias_flat.reshape(y, x)

        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        #out_png = BIAS_DIR / f"bias_{TARGET_VAR}_{d.name}_{stamp}.png"
        out_nc  = BIAS_DIR / f"bias_{TARGET_VAR}_{d.name}_{stamp}.nc"

        title = ""
        #plot_bias_map(bias_2d, lat2d, lon2d, title, out_png)
        save_bias_netcdf(bias_2d, lat2d, lon2d, out_nc, TARGET_VAR, d.name)


    ds_truth.close()


if __name__ == "__main__":
    main()
