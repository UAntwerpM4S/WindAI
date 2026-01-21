from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from scipy.spatial import cKDTree  # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# -------------------- USER SETTINGS --------------------
TARGET_VAR = "ws10" # "ws100"

FORECAST_DIRS: List[Path] = [
    Path("/mnt/weatherloss/WindPower/inference/CI/TFtest1"),
    Path("/mnt/weatherloss/WindPower/inference/CI/TFtest2")
]

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/NorthSea/Cerra/cerra_CI_HR.zarr")
PLOT_DIR = Path("CI_plots")

LEAD_HOURS: List[int] = list(range(3, 73, 3))
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2024-08-31 21:00:00", tz="UTC")

# Nearest-neighbor acceptance threshold (truth point kept only if ALL dirs have a forecast point within this distance)
MAX_DIST_KM = 5.0
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


def init_to_file_map(d: Path, inits: Sequence[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp) -> Dict[pd.Timestamp, Path]:
    files = list_forecast_files(d, start, end)
    m = {parse_init_time(f): f for f in files}
    return {it: m[it] for it in inits}


def lonlat_to_xy_km(lon: np.ndarray, lat: np.ndarray, lat0: float) -> np.ndarray:
    """
    Fast local projection (equirectangular) to kilometers.
    Good for regional domains like North Sea.
    """
    R_km = 6371.0
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    lat0_rad = np.deg2rad(lat0)

    x = R_km * np.deg2rad(lon) * np.cos(lat0_rad)
    y = R_km * np.deg2rad(lat)
    return np.column_stack([x, y])


def build_truth_points(ds_truth: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      truth_lat_flat (N,)
      truth_lon_flat (N,)
      truth_xy_km (N,2)
    """
    lat2d = ds_truth["latitude"].values
    lon2d = ds_truth["longitude"].values
    lat = lat2d.reshape(-1)
    lon = lon2d.reshape(-1)
    lat0 = float(np.nanmean(lat))
    xy = lonlat_to_xy_km(lon, lat, lat0)
    return lat, lon, xy


def build_forecast_points(sample_file: Path, lat0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      fc_lat (M,)
      fc_lon (M,)
      fc_xy_km (M,2)
    """
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
    """
    Build nearest-neighbor mapping from truth points -> forecast 'values' indices for each directory.
    Keeps only truth points where ALL directories have nearest distance <= max_dist_km.

    Returns:
      truth_keep_idx (K,) indices into flattened truth (y,x)
      dir_fc_idx[d]  (K,) indices into forecast 'values' for that directory
      keep_dist_km   (K,) max distance across dirs for each kept truth point
    """
    truth_lat, truth_lon, truth_xy = build_truth_points(ds_truth)
    lat0 = float(np.nanmean(truth_lat))

    # Build a KD-tree per dir using its grid from a representative forecast file (first init)
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



    # Keep truth points that are close enough in *all* dirs
    max_dist = np.zeros(truth_xy.shape[0], dtype=np.float64)
    ok = np.ones(truth_xy.shape[0], dtype=bool)
    for d in files_by_dir:
        ok &= dir_dist_full[d] <= max_dist_km
        max_dist = np.maximum(max_dist, dir_dist_full[d])

    truth_keep_idx = np.where(ok)[0].astype(np.int64)
    keep_dist_km = max_dist[truth_keep_idx]

    dir_fc_idx = {d: dir_fc_idx_full[d][truth_keep_idx] for d in files_by_dir}

    return truth_keep_idx, dir_fc_idx, keep_dist_km


def aggregate_by_lead(dfs: List[pd.DataFrame], lead_hours: Sequence[int]) -> pd.DataFrame:
    all_df = pd.concat(dfs, ignore_index=True)
    out = all_df.groupby("lead_hours", as_index=False).agg(count=("RMSE", "size"), RMSE=("RMSE", "mean"))
    req = pd.DataFrame({"lead_hours": sorted(set(map(int, lead_hours)))})
    return req.merge(out, on="lead_hours", how="left").sort_values("lead_hours")

def save_rmse_csv(results: List[Tuple[str, pd.DataFrame]], out_dir: Path, var: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, df in results:
        out_path = out_dir / f"rmse_{var}_{label}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved RMSE table: {out_path}")

def plot_rmse(results: List[Tuple[str, pd.DataFrame]], out_path: Path) -> None:
    plt.figure(figsize=(7.5, 4.2))
    for label, df in results:
        plt.plot(df["lead_hours"], df["RMSE"], marker="o", lw=1.8, label=label)
    plt.title(f"RMSE vs Lead Time for {TARGET_VAR}", fontsize=12)
    plt.xlabel("Lead time [hours]")
    plt.ylabel("RMSE")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

def parse_pressure_level_var(name: str) -> tuple[str, float] | tuple[None, None]:
    """
    If name looks like 'z_500' or 'u_850', return ('z', 500.0).
    Otherwise return (None, None).
    """
    m = re.match(r"^(?P<base>[A-Za-z]\w*)_(?P<lev>\d{3,4})$", name)
    if not m:
        return (None, None)
    return (m.group("base"), float(m.group("lev")))

def truth_dataarray(ds_truth: xr.Dataset, target_var: str) -> xr.DataArray:
    """
    Returns a DataArray shaped (time, y, x) for truth, regardless of whether
    target_var is surface (e.g. ws100) or pressure-level (e.g. z_500).
    """
    base, lev = parse_pressure_level_var(target_var)

    # Surface-style variable exists directly
    if target_var in ds_truth.data_vars:
        return ds_truth[target_var]

    # Pressure-level style: ds_truth has base var with 'level' dim
    if base is not None and base in ds_truth.data_vars and "level" in ds_truth[base].dims:
        # Prefer exact match; if float formatting differs, nearest is safer
        # (you can remove method="nearest" if you want strict exact matching)
        return ds_truth[base].sel(level=lev, method="nearest")

    raise KeyError(
        f"Truth dataset has no variable '{target_var}'. "
        f"Tried direct '{target_var}', and level mapping '{base}' @ {lev}."
    )


def main() -> None:
    start, end = to_utc(INIT_START), to_utc(INIT_END)

    # 1) Common init times
    inits = common_init_times(FORECAST_DIRS, start, end)
    print(f"Common init runs across all dirs: {len(inits)}")

    filemaps = {d: init_to_file_map(d, inits, start, end) for d in FORECAST_DIRS}

    # 2) Open truth once
    ds_truth = xr.open_zarr(CERRA_PATH)

    # 3) Build nearest mapping truth->forecast per dir, keep only points close in all dirs
    truth_keep_idx, dir_fc_idx, keep_dist_km = nearest_mapping_all_dirs(ds_truth, filemaps, inits, MAX_DIST_KM)
    print(f"Truth grid points total: {ds_truth.dims['y'] * ds_truth.dims['x']}")
    print(f"Kept truth points (<= {MAX_DIST_KM} km in all dirs): {len(truth_keep_idx)}")
    print(f"Nearest-distance stats over kept points [km]: min={keep_dist_km.min():.3f}, "
          f"median={np.median(keep_dist_km):.3f}, max={keep_dist_km.max():.3f}")

    # Precompute truth variable stacked (but only select needed times per run below)
    lead_keep = set(map(int, LEAD_HOURS))
    results: List[Tuple[str, pd.DataFrame]] = []

    for d in FORECAST_DIRS:
        per_run: List[pd.DataFrame] = []

        fc_values_idx = dir_fc_idx[d]  # (K,)

        for it in inits:
            fpath = filemaps[d][it]
            ds_fc = xr.open_dataset(fpath)

            valid_times = pd.to_datetime(ds_fc["time"].values)  # naive
            valid_utc = valid_times.tz_localize("UTC")
            leads = ((valid_utc - it) / np.timedelta64(1, "h")).astype(int)

            tmask = np.isin(leads, list(lead_keep))
            if not np.any(tmask):
                ds_fc.close()
                continue

            times_sel = valid_times[tmask]
            leads_sel = leads[tmask].astype(int)

            # Forecast: (time, values) -> pick our nearest-mapped values
            fc = ds_fc[TARGET_VAR].isel(time=tmask, values=fc_values_idx).values  # (T, K)
            ds_fc.close()

            # Truth: select only needed times, then stack y,x, then pick truth_keep_idx
            tr_da = truth_dataarray(ds_truth, TARGET_VAR)
            tr_da = truth_dataarray(ds_truth, TARGET_VAR)

            tr = (
                tr_da
                .sel(time=times_sel)
                .stack(values=("y", "x"))
                .isel(values=truth_keep_idx)
    .values
)

            tr = (
                tr_da
                .sel(time=times_sel)
                .stack(values=("y", "x"))
                .isel(values=truth_keep_idx)
                .values
            )  # (T, K)

            rmse = np.sqrt(np.nanmean((fc - tr) ** 2, axis=1))
            per_run.append(pd.DataFrame({"lead_hours": leads_sel, "RMSE": rmse.astype(float)}))

        results.append((d.name, aggregate_by_lead(per_run, LEAD_HOURS)))

    ds_truth.close()

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = PLOT_DIR / f"rmse_{TARGET_VAR}_{stamp}.png"
    save_rmse_csv(results, PLOT_DIR, TARGET_VAR)
    plot_rmse(results, out)


if __name__ == "__main__":
    main()
