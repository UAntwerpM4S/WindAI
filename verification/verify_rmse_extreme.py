"""
RMSE vs lead time with two parallel panels: top N% extreme (high) and bottom N% (low).

For standard weather variables (ws100, ws10, etc.):
  - Parallel workers over all CERRA cells
  - Per-cell extreme threshold

When "power" is in TARGET_VARS:
  - Direct power RMSE at Belgian farm cells (if directory has power variable)
  - Power-curve RMSE at Belgian farm cells (always, derived from ws100)
  - Also produces ws100 and ws10 extreme RMSE at farm cells only
  - Extreme threshold is on total Belgian power (scalar per timestep)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

# -------------------- SETTINGS --------------------
TARGET_VARS = ["power"]  # add "power" to trigger power + power-curve plots

FORECAST_DIRS = {
    "NoPowerTF":      Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerTF"),
    "NoPowerGT":      Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerGT"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGTRollout"),
    "BigTF": Path("/mnt/weatherloss/WindPower/inference/EGU/BigTransformer"),
}

CERRA_PATH    = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")
METADATA_PATH = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")
SPECS_PATH    = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/turbine_specs.csv")
COUNTS_PATH   = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/wind_farm_turbine_counts.csv")
OUT_DIR       = Path("EGU_extreme")

INIT_START         = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END           = pd.Timestamp("2025-06-28 21:00:00", tz="UTC")
LEAD_HOURS         = list(range(3, 37, 3))
EXTREME_PERCENTILE = 95   # top N% and bottom (100-N)%
MAX_DIST_KM        = 2.0
N_WORKERS          = 8
# --------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True,
    )


# ── Power-curve helpers (from verify_powercurve.py) ───────────────────────────

@dataclass(frozen=True)
class TurbineSpec:
    cut_in: float
    rated_ws: float
    cut_out: float
    rated_power: float


def power_curve(ws: np.ndarray, spec: TurbineSpec) -> np.ndarray:
    ws  = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws, dtype=np.float32)
    ramp  = (ws >= spec.cut_in) & (ws < spec.rated_ws)
    rated = (ws >= spec.rated_ws) & (ws < spec.cut_out)
    denom = spec.rated_ws ** 3 - spec.cut_in ** 3
    a = 1.0 / denom
    b = spec.cut_in ** 3 / denom
    out[ramp]  = spec.rated_power * (a * ws[ramp] ** 3 - b)
    out[rated] = spec.rated_power
    return out


def load_specs(path: Path) -> Dict[str, TurbineSpec]:
    df = pd.read_csv(path)
    return {
        row["turbine_type (name-capacity-type)"]: TurbineSpec(
            cut_in=float(row["cut_in_ms"]),
            rated_ws=float(row["rated_ws_ms"]),
            cut_out=float(row["cut_out_ms"]),
            rated_power=float(row["rated_power_mw"]),
        )
        for _, row in df.iterrows()
    }


def load_farm_metadata(regions: List[str] = ["Belgium"]) -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH)
    return meta[meta["region"].str.lower().isin([r.lower() for r in regions])].copy()


def get_farm_cerra_indices(meta: pd.DataFrame, cerra_lat: np.ndarray, cerra_lon: np.ndarray) -> np.ndarray:
    be_unique = meta.drop_duplicates(subset=["cerra_grid_lat", "cerra_grid_lon"])
    cerra_keys = {(round(la, 6), round(lo, 6)): i
                  for i, (la, lo) in enumerate(zip(cerra_lat, cerra_lon))}
    indices = []
    for _, row in be_unique.iterrows():
        key = (round(row["cerra_grid_lat"], 6), round(row["cerra_grid_lon"], 6))
        if key not in cerra_keys:
            raise ValueError(f"CERRA cell not found: {key}")
        indices.append(cerra_keys[key])
    return np.array(indices, dtype=int)


def build_counts_matrix(meta: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    counts = pd.read_csv(COUNTS_PATH).set_index("farm")
    type_cols = [c for c in counts.columns if c.lower() != "total"]

    seen: list = []
    for _, row in meta.iterrows():
        cell = (int(row["cerra_y"]), int(row["cerra_x"]))
        if cell not in seen:
            seen.append(cell)

    cell_to_type: Dict = defaultdict(lambda: defaultdict(float))
    for _, row in meta.iterrows():
        farm = row["farm"]
        cell = (int(row["cerra_y"]), int(row["cerra_x"]))
        if farm not in counts.index:
            raise ValueError(f"Farm '{farm}' not in counts file.")
        for tcol in type_cols:
            cell_to_type[cell][tcol] += float(counts.at[farm, tcol])

    mat = np.zeros((len(seen), len(type_cols)), dtype=np.float32)
    for ci, cell in enumerate(seen):
        for tj, tcol in enumerate(type_cols):
            mat[ci, tj] = cell_to_type[cell].get(tcol, 0.0)

    return type_cols, mat


# ── Workers ───────────────────────────────────────────────────────────────────

def _read_one_file_weather(args):
    """
    Worker for standard weather variables.
    Returns (init_iso, high_mse, low_mse) where each is {lh: float}.
    Per-cell extreme masking: high = ob >= thresh_high, low = ob <= thresh_low.
    cell_subset: optional int array to select a subset of forecast cells (e.g. farm cells).
    """
    nc_path, init_iso, lead_hours, var_name, cerra_cache_items, thresh_high, thresh_low, cell_subset = args

    init        = pd.Timestamp(init_iso)
    cerra_cache = {iso: arr for iso, arr in cerra_cache_items}
    result_h, result_l = {}, {}

    try:
        with h5py.File(str(nc_path), "r") as f:
            tv  = f["time"]
            raw = nc4.num2date(
                tv[:],
                tv.attrs["units"].decode(),
                tv.attrs.get("calendar", b"standard").decode(),
            )
            fc_times       = [pd.Timestamp(str(t)).tz_localize("UTC") for t in raw]
            fc_time_to_idx = {t.isoformat(): j for j, t in enumerate(fc_times)}
            var_all        = f[var_name][:, :]  # (n_times, n_cells)

        for lh in lead_hours:
            valid_iso = (init + pd.Timedelta(hours=lh)).isoformat()
            if valid_iso not in fc_time_to_idx or valid_iso not in cerra_cache:
                continue
            fc_vals = var_all[fc_time_to_idx[valid_iso]]
            if cell_subset is not None:
                fc_vals = fc_vals[cell_subset]
            ob_vals = cerra_cache[valid_iso]

            mask_h = ob_vals >= thresh_high
            mask_l = ob_vals <= thresh_low
            if mask_h.sum() > 0:
                result_h[lh] = float(np.nanmean((fc_vals[mask_h] - ob_vals[mask_h]) ** 2))
            if mask_l.sum() > 0:
                result_l[lh] = float(np.nanmean((fc_vals[mask_l] - ob_vals[mask_l]) ** 2))

    except Exception as e:
        print(f"  WORKER ERROR {Path(nc_path).name}: {e}", flush=True)

    return init_iso, result_h, result_l


# ── CERRA / threshold helpers ─────────────────────────────────────────────────

def compute_thresholds(
    ds_cerra: xr.Dataset,
    var_idx: int,
    cerra_dates: pd.DatetimeIndex,
    valid_times: List[pd.Timestamp],
    percentile: int,
    cell_subset: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-cell (or scalar if cell_subset has 1 aggregated value) high and low thresholds.
    Returns (thresh_high, thresh_low).
    """
    cerra_date_to_idx = {d: i for i, d in enumerate(cerra_dates)}
    idxs = [cerra_date_to_idx[t] for t in valid_times if t in cerra_date_to_idx]
    bulk = ds_cerra["data"].isel(time=idxs, variable=var_idx, ensemble=0).values
    if cell_subset is not None:
        bulk = bulk[:, cell_subset]
    thresh_high = np.nanpercentile(bulk, percentile,       axis=0)
    thresh_low  = np.nanpercentile(bulk, 100 - percentile, axis=0)
    return thresh_high, thresh_low


def run_parallel(
    label: str,
    var: str,
    fmap: Dict,
    common_inits: List,
    cerra_cache_items: list,
    thresh_high: np.ndarray,
    thresh_low: np.ndarray,
    cell_subset: np.ndarray | None = None,
) -> Tuple[List[float], List[float]]:
    """Run the multiprocessing pool for one (label, var) pair. Returns (rmse_high, rmse_low)."""
    tasks = [
        (str(fmap[init]), init.isoformat(), LEAD_HOURS, var,
         cerra_cache_items, thresh_high, thresh_low, cell_subset)
        for init in common_inits
    ]

    cache_h: Dict[str, Dict[int, float]] = {}
    cache_l: Dict[str, Dict[int, float]] = {}

    with Pool(processes=N_WORKERS) as pool:
        for n_done, (init_iso, rh, rl) in enumerate(
            pool.imap_unordered(_read_one_file_weather, tasks, chunksize=4)
        ):
            cache_h[init_iso] = rh
            cache_l[init_iso] = rl
            if n_done % 200 == 0:
                print(f"  {label}/{var}: {n_done}/{len(common_inits)} done...", flush=True)

    lead_mse_h = {lh: [] for lh in LEAD_HOURS}
    lead_mse_l = {lh: [] for lh in LEAD_HOURS}
    for init in common_inits:
        iso = init.isoformat()
        for lh in LEAD_HOURS:
            if lh in cache_h.get(iso, {}):
                lead_mse_h[lh].append(cache_h[iso][lh])
            if lh in cache_l.get(iso, {}):
                lead_mse_l[lh].append(cache_l[iso][lh])

    rmse_h = [np.sqrt(np.mean(lead_mse_h[lh])) if lead_mse_h[lh] else np.nan for lh in LEAD_HOURS]
    rmse_l = [np.sqrt(np.mean(lead_mse_l[lh])) if lead_mse_l[lh] else np.nan for lh in LEAD_HOURS]
    return rmse_h, rmse_l


# ── Power sequential processing ───────────────────────────────────────────────

def process_power_sequential(
    label: str,
    fmap: Dict,
    common_inits: List,
    cerra_power_cache: Dict[str, float],
    fc_indices: np.ndarray,
    type_order: List[str],
    specs: Dict[str, TurbineSpec],
    counts_matrix: np.ndarray,
    thresh_high: float,
    thresh_low: float,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    Process one forecast directory for power.
    Returns {'direct': (rmse_h, rmse_l), 'curve': (rmse_h, rmse_l)}.
    'direct' is None if forecast has no power variable.
    """
    # check if power is available
    sample = next(iter(fmap.values()))
    with h5py.File(str(sample), "r") as f:
        has_power = "power" in f

    mse_direct_h: Dict[int, list] = {lh: [] for lh in LEAD_HOURS}
    mse_direct_l: Dict[int, list] = {lh: [] for lh in LEAD_HOURS}
    mse_curve_h:  Dict[int, list] = {lh: [] for lh in LEAD_HOURS}
    mse_curve_l:  Dict[int, list] = {lh: [] for lh in LEAD_HOURS}

    for count, init in enumerate(common_inits):
        if count % 500 == 0:
            print(f"  {label} power: {count}/{len(common_inits)}...", flush=True)
        try:
            with h5py.File(str(fmap[init]), "r") as f:
                tv  = f["time"]
                raw = nc4.num2date(
                    tv[:],
                    tv.attrs["units"].decode(),
                    tv.attrs.get("calendar", b"standard").decode(),
                )
                fc_time_to_idx = {
                    pd.Timestamp(str(t)).tz_localize("UTC").isoformat(): j
                    for j, t in enumerate(raw)
                }
                ws100_all = f["ws100"][:, :]
                power_all = f["power"][:, :] if has_power else None

            for lh in LEAD_HOURS:
                viso = (init + pd.Timedelta(hours=lh)).isoformat()
                if viso not in fc_time_to_idx or viso not in cerra_power_cache:
                    continue
                tidx   = fc_time_to_idx[viso]
                obs_mw = cerra_power_cache[viso]

                # extreme mask on total observed power
                is_high = obs_mw >= thresh_high
                is_low  = obs_mw <= thresh_low

                # power curve
                ws_cells = ws100_all[tidx][fc_indices]  # (n_farm_cells,)
                curve_mw = float(sum(
                    np.sum(power_curve(ws_cells, specs[t]) * counts_matrix[:, j])
                    for j, t in enumerate(type_order)
                ))
                se_curve = (curve_mw - obs_mw) ** 2
                if is_high:
                    mse_curve_h[lh].append(se_curve)
                if is_low:
                    mse_curve_l[lh].append(se_curve)

                # direct power
                if has_power:
                    fc_mw = float(np.sum(power_all[tidx][fc_indices]))
                    se_direct = (fc_mw - obs_mw) ** 2
                    if is_high:
                        mse_direct_h[lh].append(se_direct)
                    if is_low:
                        mse_direct_l[lh].append(se_direct)

        except Exception as e:
            print(f"  Skipped {fmap[init].name}: {e}")

    def to_rmse(d):
        return [np.sqrt(np.mean(d[lh])) if d[lh] else np.nan for lh in LEAD_HOURS]

    results = {"curve": (to_rmse(mse_curve_h), to_rmse(mse_curve_l))}
    if has_power:
        results["direct"] = (to_rmse(mse_direct_h), to_rmse(mse_direct_l))
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_extreme_panels(
    curves: List[Tuple[str, List[float], List[float]]],
    var: str,
    n_inits: int,
    percentile: int,
    out_path: Path,
    ylabel: str = "RMSE",
) -> None:
    """
    Two-panel plot: left = top (100-percentile)% extreme, right = bottom (100-percentile)%.
    curves: list of (label, rmse_high, rmse_low)
    """
    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        f"{var} RMSE — extreme percentiles  (n={n_inits} inits)",
        fontsize=12,
    )
    axes[0].set_title(f"Top {100 - percentile}%  (high {var})", fontsize=11)
    axes[1].set_title(f"Bottom {100 - percentile}%  (low {var})", fontsize=11)

    for i, (label, rmse_h, rmse_l) in enumerate(curves):
        is_curve = "powercurve" in label.lower()
        color    = colors[i % len(colors)]
        ls       = "--" if is_curve else "-"
        mk       = ""   if is_curve else markers[i % len(markers)]
        axes[0].plot(LEAD_HOURS, rmse_h, lw=1.5, ls=ls, marker=mk, color=color, label=label)
        axes[1].plot(LEAD_HOURS, rmse_l, lw=1.5, ls=ls, marker=mk, color=color, label=label)

    for ax in axes:
        ax.set_xlabel("Lead time [hours]")
        ax.set_ylabel(ylabel)
        ax.set_xticks(LEAD_HOURS)
        ax.legend(title="Run", framealpha=0.8, fontsize=8)
        ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mp.set_start_method("spawn", force=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    do_power = "power" in TARGET_VARS
    weather_vars = [v for v in TARGET_VARS if v != "power"]

    ds_cerra    = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars  = list(ds_cerra.attrs["variables"])
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")

    # ── file maps ─────────────────────────────────────────────────────────────
    dir_file_maps = {}
    for label, fc_dir in FORECAST_DIRS.items():
        fmap = {
            parse_init(f): f
            for f in sorted(fc_dir.glob("forecast_*.nc"))
            if INIT_START <= parse_init(f) <= INIT_END
        }
        print(f"{label}: {len(fmap)} files")
        if fmap:
            dir_file_maps[label] = fmap

    if not dir_file_maps:
        raise RuntimeError("No forecast files found.")

    common_inits = sorted(
        set.intersection(*(set(m) for m in dir_file_maps.values()))
    )
    print(f"Common init times: {len(common_inits)}")

    cerra_date_to_idx = {d: i for i, d in enumerate(cerra_dates)}
    all_valid_times = sorted({
        init + pd.Timedelta(hours=lh)
        for init in common_inits
        for lh in LEAD_HOURS
        if (init + pd.Timedelta(hours=lh)) in cerra_date_to_idx
    })

    # ── standard weather variables (all cells, parallel) ──────────────────────
    for var in weather_vars:
        if var not in cerra_vars:
            print(f"WARNING: '{var}' not in CERRA, skipping.")
            continue
        var_idx = cerra_vars.index(var)
        print(f"\n=== {var} (all cells) ===")

        thresh_high, thresh_low = compute_thresholds(
            ds_cerra, var_idx, cerra_dates, all_valid_times, EXTREME_PERCENTILE
        )
        print(f"  High threshold range: {thresh_high.min():.3f} – {thresh_high.max():.3f}")
        print(f"  Low  threshold range: {thresh_low.min():.3f}  – {thresh_low.max():.3f}")

        needed_idxs = [cerra_date_to_idx[t] for t in all_valid_times]
        print(f"  Preloading {len(needed_idxs)} CERRA timesteps...")
        cerra_bulk = ds_cerra["data"].isel(
            time=needed_idxs, variable=var_idx, ensemble=0
        ).values
        cerra_cache_items = [(t.isoformat(), cerra_bulk[i]) for i, t in enumerate(all_valid_times)]
        del cerra_bulk

        curves = []
        for label, fmap in dir_file_maps.items():
            print(f"  Processing {label}...")
            rmse_h, rmse_l = run_parallel(
                label, var, fmap, common_inits, cerra_cache_items,
                thresh_high, thresh_low, cell_subset=None,
            )
            curves.append((label, rmse_h, rmse_l))
            np.save(OUT_DIR / f"extreme_{var}_{label}.npy",
                    np.column_stack([LEAD_HOURS, rmse_h, rmse_l]))

        plot_extreme_panels(
            curves, var, len(common_inits), EXTREME_PERCENTILE,
            OUT_DIR / f"extreme_{var}.png",
        )

    # ── power mode ────────────────────────────────────────────────────────────
    if do_power:
        if "power" not in cerra_vars:
            print("WARNING: 'power' not in CERRA — skipping power plots.")
        else:
            print("\n=== power (farm cells) ===")
            cerra_lat  = ds_cerra["latitudes"].values
            cerra_lon  = ds_cerra["longitudes"].values
            meta       = load_farm_metadata(["Belgium"])
            total_cap  = float(meta["capacity_mw"].sum())
            fc_indices = get_farm_cerra_indices(meta, cerra_lat, cerra_lon)
            specs      = load_specs(SPECS_PATH)
            type_order, counts_matrix = build_counts_matrix(meta)
            print(f"  Farm cells: {len(fc_indices)}  |  Capacity: {total_cap:.0f} MW")

            power_idx = cerra_vars.index("power")

            # preload CERRA power summed over farm cells (scalar per timestep)
            needed_idxs = [cerra_date_to_idx[t] for t in all_valid_times]
            cerra_power_bulk = ds_cerra["data"].isel(
                time=needed_idxs, variable=power_idx, ensemble=0
            ).values[:, fc_indices]  # (n_times, n_farm_cells)
            cerra_power_cache = {
                t.isoformat(): float(np.nansum(cerra_power_bulk[i]))
                for i, t in enumerate(all_valid_times)
            }
            del cerra_power_bulk

            # scalar thresholds on total Belgian power
            total_power_series = np.array(list(cerra_power_cache.values()))
            thresh_pw_high = float(np.nanpercentile(total_power_series, EXTREME_PERCENTILE))
            thresh_pw_low  = float(np.nanpercentile(total_power_series, 100 - EXTREME_PERCENTILE))
            print(f"  Power threshold high: {thresh_pw_high:.1f} MW  "
                  f"low: {thresh_pw_low:.1f} MW  (total cap: {total_cap:.0f} MW)")

            power_curves: List[Tuple[str, List[float], List[float]]] = []
            for label, fmap in dir_file_maps.items():
                print(f"  Processing {label}...")
                res = process_power_sequential(
                    label, fmap, common_inits, cerra_power_cache,
                    fc_indices, type_order, specs, counts_matrix,
                    thresh_pw_high, thresh_pw_low,
                )
                if "direct" in res:
                    rh, rl = res["direct"]
                    power_curves.append((f"{label}-direct", rh, rl))
                    np.save(OUT_DIR / f"extreme_power_direct_{label}.npy",
                            np.column_stack([LEAD_HOURS, rh, rl]))
                rh, rl = res["curve"]
                power_curves.append((f"{label}-powercurve", rh, rl))
                np.save(OUT_DIR / f"extreme_power_curve_{label}.npy",
                        np.column_stack([LEAD_HOURS, rh, rl]))

            plot_extreme_panels(
                power_curves, "power", len(common_inits), EXTREME_PERCENTILE,
                OUT_DIR / "extreme_power.png",
                ylabel="RMSE [MW]",
            )

            # ── also ws100 and ws10 at farm cells (parallel) ──────────────────
            for var in ["ws100", "ws10"]:
                if var not in cerra_vars:
                    continue
                var_idx = cerra_vars.index(var)
                print(f"\n=== {var} (farm cells only) ===")

                thresh_high, thresh_low = compute_thresholds(
                    ds_cerra, var_idx, cerra_dates, all_valid_times,
                    EXTREME_PERCENTILE, cell_subset=fc_indices,
                )

                needed_idxs = [cerra_date_to_idx[t] for t in all_valid_times]
                cerra_bulk = ds_cerra["data"].isel(
                    time=needed_idxs, variable=var_idx, ensemble=0
                ).values[:, fc_indices]
                cerra_cache_items = [
                    (t.isoformat(), cerra_bulk[i]) for i, t in enumerate(all_valid_times)
                ]
                del cerra_bulk

                curves = []
                for label, fmap in dir_file_maps.items():
                    print(f"  Processing {label}...")
                    rmse_h, rmse_l = run_parallel(
                        label, var, fmap, common_inits, cerra_cache_items,
                        thresh_high, thresh_low, cell_subset=fc_indices,
                    )
                    curves.append((label, rmse_h, rmse_l))
                    np.save(OUT_DIR / f"extreme_{var}_farm_{label}.npy",
                            np.column_stack([LEAD_HOURS, rmse_h, rmse_l]))

                plot_extreme_panels(
                    curves, f"{var} (farm cells)", len(common_inits), EXTREME_PERCENTILE,
                    OUT_DIR / f"extreme_{var}_farm.png",
                )

    ds_cerra.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
