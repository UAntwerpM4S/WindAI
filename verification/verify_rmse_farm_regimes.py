"""
ws100 RMSE vs lead time at the DISTRIBUTED farm cells, SPLIT BY WIND REGIME.

Same skeleton as verify_rmse_farm.py, but instead of one RMSE curve it produces four -- one per
wind-speed regime, so you can see where a run wins or loses:

    0 - 4.5 m/s   (below / around cut-in: power ~ 0, forecast errors barely matter)
    4.5 - 8 m/s   (the steep part of the power curve: errors hurt the most here)
    8 - 12 m/s    (approaching rated)
    12+ m/s       (rated / storm: power saturates, but cut-out timing matters)

The regime of each (cell, valid time) is set by the CERRA TRUTH ws100 there, so a curve reads
"when the true wind was in band B, this run's ws100 RMSE was R" -- a conditional skill, not a
mix of regimes. Binning by truth (not forecast) is what makes it an honest conditional.

Cells come from `turbmask` in the distributed power source zarr (the cells the turbines occupy),
same as verify_rmse_farm.py. REGION toggles BE / UK / both. Every run is scored on the init
times common to all runs and on identical cells, against the same CERRA truth.

Usage:
  python verify_rmse_farm_regimes.py                 # uses the REGION constant below
  python verify_rmse_farm_regimes.py --region UK
"""

import argparse
from pathlib import Path
from multiprocessing import Pool
import multiprocessing as mp
import re
import numpy as np
import pandas as pd
import xarray as xr
import h5py
import netCDF4 as nc4
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# -------------------- SETTINGS --------------------
TARGET_VAR = "ws100"          # the wind speed to score (also defines the regimes)

# Regime edges in m/s and their labels. Bins are [edge_i, edge_{i+1}); last is open-ended.
REGIME_EDGES  = [0.0, 4.5, 8.0, 12.0, np.inf]
REGIME_LABELS = ["0-4.5", "4.5-8", "8-12", "12+"]

FORECAST_DIRS = {
    "HighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaPowerGT"),
    "RegularWeather": Path("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
   # "ExtremelyHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/ExtremelyHighCapacity"),
}

CERRA_PATH   = Path("/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/New_Cerra_A_large.zarr")
TURBMASK_SRC = Path("/mnt/weatherloss/WindPower/data/WPDistr/power_cerra_src.zarr")
TURBINES_CSV = Path("/mnt/weatherloss/WindPower/data/WPDistr/turbines.csv")

REGION = "BE"          # <<< TOGGLE: "BE" | "UK" | "both"  (or pass --region)

OUT_DIR    = Path("WindAI_farm_regimes")

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))

N_WORKERS  = 8
# --------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")
NREG = len(REGIME_LABELS)


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True,
    )


def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def get_farm_cerra_indices(region: str) -> np.ndarray:
    """CERRA cells where the distributed power target exists, optionally split by region.

    turbmask is 1 at every cell holding turbines, NaN elsewhere. For a single region, intersect
    it with that region's cells (same cos-lat KD-tree build_power.py used), so the cells match
    the target's cells exactly.
    """
    ds = xr.open_zarr(TURBMASK_SRC, consolidated=True)
    tm = ds["turbmask"].isel(time=0).values
    ds.close()
    idx = np.where(np.isfinite(tm))[0]

    if region.lower() == "both":
        print(f"Region: both | {idx.size} distributed farm cells")
        return np.sort(idx)

    dsc = xr.open_zarr(CERRA_PATH, consolidated=False)
    lat = np.asarray(dsc["latitudes"]).ravel()
    lon = to_180(np.asarray(dsc["longitudes"]).ravel())
    dsc.close()

    t = pd.read_csv(TURBINES_CSV)
    t = t[t["region"].str.upper() == region.upper()]
    if t.empty:
        raise SystemExit(f"no turbines for region {region!r} (expected BE, UK or both)")

    coslat = np.cos(np.radians(float(lat.mean())))
    tree = cKDTree(np.c_[lon * coslat, lat])
    _, cell = tree.query(np.c_[to_180(t["longitude"]) * coslat, t["latitude"].to_numpy()], k=1)
    sel = np.sort(np.intersect1d(idx, np.unique(cell.astype(int))))
    print(f"Region: {region} | {t.farm.nunique()} farms | {sel.size} of {idx.size} "
          f"distributed farm cells")
    return sel


def _read_one_file(args):
    """Worker: one process per file (h5py avoids netCDF4 heap corruption).

    Returns (init_iso, {lead: (ss, se, n)}) where each of ss/se/n is a length-NREG array:
    per wind regime, the summed squared error, summed signed error, and cell count -- the
    regime set by the CERRA truth ws100 at each cell.
    """
    nc_path, init_iso, lead_hours, var_name, cerra_cache_items, farm_cell_idxs = args

    init        = pd.Timestamp(init_iso)
    cerra_cache = {iso: arr for iso, arr in cerra_cache_items}
    result      = {}

    try:
        with h5py.File(str(nc_path), "r") as f:
            tv  = f["time"]
            raw = nc4.num2date(
                tv[:], tv.attrs["units"].decode(),
                tv.attrs.get("calendar", b"standard").decode(),
            )
            fc_time_to_idx = {pd.Timestamp(str(t)).tz_localize("UTC").isoformat(): j
                              for j, t in enumerate(raw)}
            if var_name not in f:
                return init_iso, {}
            var_all = f[var_name][:, :]                       # (n_times, n_cells)

        edges_inner = REGIME_EDGES[1:-1]                      # interior edges for digitize
        for lh in lead_hours:
            valid_iso = (init + pd.Timedelta(hours=lh)).isoformat()
            if valid_iso not in fc_time_to_idx or valid_iso not in cerra_cache:
                continue
            fc = var_all[fc_time_to_idx[valid_iso]][farm_cell_idxs]
            ob = cerra_cache[valid_iso]                       # CERRA truth, farm cells only
            m = np.isfinite(fc) & np.isfinite(ob)
            if not m.any():
                continue
            err = fc[m] - ob[m]
            reg = np.digitize(ob[m], edges_inner)             # 0..NREG-1 by TRUTH ws100

            ss = np.zeros(NREG); se = np.zeros(NREG); n = np.zeros(NREG)
            np.add.at(ss, reg, err ** 2)
            np.add.at(se, reg, err)
            np.add.at(n, reg, 1.0)
            result[lh] = (ss, se, n)

    except Exception as e:                                    # noqa: BLE001
        print(f"  WORKER ERROR {Path(nc_path).name}: {e}", flush=True)

    return init_iso, result


def main():
    ap = argparse.ArgumentParser(description="ws100 RMSE by wind regime at the farm cells")
    ap.add_argument("--region", default=REGION, choices=["BE", "UK", "both"],
                    help=f"which farm cells to score (default: {REGION})")
    args = ap.parse_args()
    region = args.region

    mp.set_start_method("spawn", force=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds_cerra    = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars  = list(ds_cerra.attrs["variables"])
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")
    if TARGET_VAR not in cerra_vars:
        raise SystemExit(f"{TARGET_VAR!r} not in CERRA variables")

    farm_cell_idxs = get_farm_cerra_indices(region)

    # ── file maps, restricted to the common init times ────────────────────────
    dir_file_maps = {}
    for label, fc_dir in FORECAST_DIRS.items():
        fmap = {parse_init(f): f for f in sorted(fc_dir.glob("forecast_*.nc"))
                if INIT_START <= parse_init(f) <= INIT_END}
        print(f"{label}: {len(fmap)} files")
        if fmap:
            dir_file_maps[label] = fmap
    if not dir_file_maps:
        raise RuntimeError("No forecast files found.")
    common_inits = sorted(set.intersection(*(set(m) for m in dir_file_maps.values())))
    print(f"Common init times: {len(common_inits)}")
    if not common_inits:
        raise SystemExit("no init times common to all runs")

    cerra_date_to_idx = {d: i for i, d in enumerate(cerra_dates)}

    # ── preload CERRA truth ws100 at the farm cells ───────────────────────────
    needed_valid_times = sorted({
        init + pd.Timedelta(hours=lh)
        for init in common_inits for lh in LEAD_HOURS
        if (init + pd.Timedelta(hours=lh)) in cerra_date_to_idx
    })
    needed_cerra_idxs = [cerra_date_to_idx[t] for t in needed_valid_times]
    print(f"Preloading {len(needed_cerra_idxs)} CERRA {TARGET_VAR} timesteps "
          f"({len(farm_cell_idxs)} farm cells)...")
    cerra_bulk = ds_cerra["data"].isel(
        time=needed_cerra_idxs, variable=cerra_vars.index(TARGET_VAR), ensemble=0,
    ).values[:, farm_cell_idxs]
    cerra_cache_items = [(t.isoformat(), cerra_bulk[i]) for i, t in enumerate(needed_valid_times)]
    del cerra_bulk

    COLORS  = plt.cm.tab10.colors
    MARKERS = ["o", "s", "^", "D", "v"]
    color_of = {label: COLORS[i % len(COLORS)] for i, label in enumerate(dir_file_maps)}
    mark_of  = {label: MARKERS[i % len(MARKERS)] for i, label in enumerate(dir_file_maps)}

    # per run: acc[lead] = (ss, se, n) arrays over regimes
    all_rmse = {}   # label -> (NREG, nlead) RMSE
    all_bias = {}   # label -> (NREG, nlead) bias
    reg_counts = np.zeros(NREG)
    rows = []

    for label, fmap in dir_file_maps.items():
        acc = {lh: [np.zeros(NREG), np.zeros(NREG), np.zeros(NREG)] for lh in LEAD_HOURS}
        tasks = [(str(fmap[i]), i.isoformat(), LEAD_HOURS, TARGET_VAR,
                  cerra_cache_items, farm_cell_idxs) for i in common_inits]
        print(f"\nProcessing {label} with {N_WORKERS} workers ({len(tasks)} files)...")
        with Pool(processes=N_WORKERS) as pool:
            for done, (init_iso, res) in enumerate(
                    pool.imap_unordered(_read_one_file, tasks, chunksize=4)):
                for lh, (ss, se, n) in res.items():
                    acc[lh][0] += ss; acc[lh][1] += se; acc[lh][2] += n
                if done % 200 == 0:
                    print(f"  {label}: {done}/{len(tasks)}", flush=True)

        rmse = np.full((NREG, len(LEAD_HOURS)), np.nan)
        bias = np.full((NREG, len(LEAD_HOURS)), np.nan)
        for li, lh in enumerate(LEAD_HOURS):
            ss, se, n = acc[lh]
            with np.errstate(invalid="ignore", divide="ignore"):
                rmse[:, li] = np.sqrt(ss / n)
                bias[:, li] = se / n
            for ri in range(NREG):
                rows.append(dict(run=label, regime=REGIME_LABELS[ri], lead_hours=lh,
                                 rmse=rmse[ri, li], bias=bias[ri, li], n=int(n[ri])))
        all_rmse[label], all_bias[label] = rmse, bias
        # sample counts per regime (same for all runs up to missing files; take the max seen)
        reg_counts = np.maximum(reg_counts, sum(acc[lh][2] for lh in LEAD_HOURS))

    # ── print ─────────────────────────────────────────────────────────────────
    for ri, rl in enumerate(REGIME_LABELS):
        print(f"\n=== {TARGET_VAR} RMSE  |  regime {rl} m/s  |  {region}  |  "
              f"~{int(reg_counts[ri])} cell-samples ===")
        tbl = pd.DataFrame({label: all_rmse[label][ri] for label in dir_file_maps},
                           index=LEAD_HOURS)
        tbl.index.name = "lead_h"
        print(tbl.round(4).to_string())

    # ── plot: one panel per regime ────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ri, ax in enumerate(axs.ravel()):
        for label in dir_file_maps:
            ax.plot(LEAD_HOURS, all_rmse[label][ri], marker=mark_of[label],
                    color=color_of[label], lw=1.6, label=label)
        ax.set_title(f"{REGIME_LABELS[ri]} m/s   (~{int(reg_counts[ri])} samples)")
        ax.set_xlabel("Lead time [h]"); ax.set_ylabel(f"RMSE {TARGET_VAR} [m/s]")
        ax.set_xticks(LEAD_HOURS); ax.grid(True, ls="--", alpha=0.5)
    axs[0, 0].legend(title="Run", framealpha=0.85)
    reg_str = {"BE": "Belgium", "UK": "UK", "both": "BE+UK"}[region]
    fig.suptitle(f"{TARGET_VAR} RMSE vs lead by wind regime — {reg_str} farm cells "
                 f"({len(farm_cell_idxs)} cells, {len(common_inits)} inits)\n"
                 f"regime set by CERRA-truth {TARGET_VAR}", fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / f"rmse_regimes_{TARGET_VAR}_{region}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_png}")

    out_csv = OUT_DIR / f"rmse_regimes_{TARGET_VAR}_{region}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    ds_cerra.close()


if __name__ == "__main__":
    main()
