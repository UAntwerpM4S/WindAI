

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
TARGET_VARS = ["ws100"]

FORECAST_DIRS = {
    "HighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaPowerGT"),
    "RegularWeather": Path("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
    #"WindHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/WindHighCapacity"),
}

CERRA_PATH    = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")

# --- the distributed farm cells (replaces windfarm_metadata.csv's nearest-centroid cerra_y) ---
TURBMASK_SRC = Path("/mnt/weatherloss/WindPower/data/WPDistr/power_cerra_src.zarr")
TURBINES_CSV = Path("/mnt/weatherloss/WindPower/data/WPDistr/turbines.csv")

REGION = "BE"          # <<< TOGGLE: "BE" | "UK" | "both"  (or pass --region)

OUT_DIR    = Path("WPDistrFarm")

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))

N_WORKERS = 8
N_WORKERS  = 8
# --------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True,
    )


def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def get_farm_cerra_indices(region: str) -> np.ndarray:
    """CERRA cells where the DISTRIBUTED power target exists, optionally split by region.

    turbmask is 1 at every cell holding turbines and NaN elsewhere (static in time), so it is
    exactly "where the distributed power exists". For a single region, intersect it with that
    region's cells, assigned from turbines.csv with the same cos-lat KD-tree build_power.py
    used -- so the cells match the target's cells exactly.
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
    """
    Worker: one process per file, uses h5py to avoid netCDF4 heap corruption.
    Returns (init_iso, {lead_hour: mean_squared_error}) over farm cells only.
    """
    nc_path, init_iso, lead_hours, var_name, cerra_cache_items, farm_cell_idxs = args

    init        = pd.Timestamp(init_iso)
    cerra_cache = {iso: arr for iso, arr in cerra_cache_items}
    result      = {}

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

            var_all = f[var_name][:, :]  # (n_times, n_cells)

        for lh in lead_hours:
            valid_iso = (init + pd.Timedelta(hours=lh)).isoformat()
            if valid_iso not in fc_time_to_idx or valid_iso not in cerra_cache:
                continue
            tidx    = fc_time_to_idx[valid_iso]
            fc_vals = var_all[tidx][farm_cell_idxs]  # farm cells only
            ob_vals = cerra_cache[valid_iso]          # already farm cells only
            result[lh] = float(np.nanmean((fc_vals - ob_vals) ** 2))

    except Exception as e:
        print(f"  WORKER ERROR {Path(nc_path).name}: {e}", flush=True)

    return init_iso, result


def main():
    ap = argparse.ArgumentParser(description="RMSE at the distributed farm cells")
    ap.add_argument("--region", default=REGION, choices=["BE", "UK", "both"],
                    help=f"which farm cells to score (default: {REGION})")
    args = ap.parse_args()
    region = args.region

    mp.set_start_method("spawn", force=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds_cerra    = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars  = list(ds_cerra.attrs["variables"])
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")

    # ── farm cell indices into the CERRA grid ─────────────────────────────────
    farm_cell_idxs = get_farm_cerra_indices(region)

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

    STYLE_ORDER = [
    "GraphTransformer (No Power)",
    "BigTransformer (Vanilla Power)",
    "BigTransformer (Vanilla Power + Synthetic)",
    ]
    COLORS  = plt.cm.tab10.colors
    MARKERS = ["o", "s", "^", "D", "v"]

    for var in TARGET_VARS:
        var_idx = cerra_vars.index(var)

        # ── preload CERRA for this variable, farm cells only ──────────────────
        needed_valid_times = sorted({
            init + pd.Timedelta(hours=lh)
            for init in common_inits
            for lh in LEAD_HOURS
            if (init + pd.Timedelta(hours=lh)) in cerra_date_to_idx
        })
        needed_cerra_idxs = [cerra_date_to_idx[t] for t in needed_valid_times]
        print(f"Preloading {len(needed_cerra_idxs)} CERRA timesteps for {var} ({len(farm_cell_idxs)} farm cells)...")
        cerra_bulk = ds_cerra["data"].isel(
            time=needed_cerra_idxs,
            variable=var_idx,
            ensemble=0,
        ).values[:, farm_cell_idxs]  # (n_times, n_farm_cells)

        cerra_cache_items = [
            (t.isoformat(), cerra_bulk[i])
            for i, t in enumerate(needed_valid_times)
        ]
        del cerra_bulk
        print("CERRA preload done.")

        fig, ax = plt.subplots(figsize=(9, 5))

        for i, (label, fmap) in enumerate(dir_file_maps.items()):
            print(f"\nProcessing {label} / {var} with {N_WORKERS} workers...")

            tasks = [
                (
                    str(fmap[init]),
                    init.isoformat(),
                    LEAD_HOURS,
                    var,
                    cerra_cache_items,
                    farm_cell_idxs,
                )
                for init in common_inits
            ]

            sq_cache: dict[str, dict[int, float]] = {}

            with Pool(processes=N_WORKERS) as pool:
                for n_done, (init_iso, result) in enumerate(
                    pool.imap_unordered(_read_one_file, tasks, chunksize=4)
                ):
                    sq_cache[init_iso] = result
                    if n_done % 200 == 0:
                        print(f"  {label}/{var}: {n_done}/{len(common_inits)} done...",
                              flush=True)

            # ── aggregate RMSE per lead hour ──────────────────────────────────
            lead_mse = {lh: [] for lh in LEAD_HOURS}
            for init in common_inits:
                init_iso = init.isoformat()
                for lh in LEAD_HOURS:
                    if lh in sq_cache.get(init_iso, {}):
                        lead_mse[lh].append(sq_cache[init_iso][lh])

            leads     = sorted(lead_mse)
            mean_rmse = [
                np.sqrt(np.mean(lead_mse[lh])) if lead_mse[lh] else np.nan
                for lh in leads
            ]

            df = pd.DataFrame({"lead_hours": leads, "RMSE": mean_rmse})
            print(f"\n{label} — {var} (farm cells)\n{df.to_string(index=False)}")

            style_idx = STYLE_ORDER.index(label) if label in STYLE_ORDER else i
            ax.plot(
                df["lead_hours"], df["RMSE"],
                marker=MARKERS[style_idx % len(MARKERS)],
                color=COLORS[style_idx % len(COLORS)],
                lw=1.5, label=label,
            )
            # np.save(OUT_DIR / f"rmse_farm_{var}_{label}.npy",
            #         df[["lead_hours", "RMSE"]].values)

        region_str = {"BE": "Belgium", "UK": "UK", "both": "BE+UK"}[region]
        ax.set_title(
            f"RMSE vs Lead Time — {var} — {region_str} distributed farm cells "
            f"({len(farm_cell_idxs)} cells)\n"
            f"Aug 2024 - July 2025",
            fontsize=12,
        )
        ax.set_xlabel("Lead time [hours]")
        ax.set_ylabel("RMSE")
        ax.set_xticks(LEAD_HOURS)
        ax.legend(title="Run", framealpha=0.8)
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()
        out_png = OUT_DIR / f"rmse_farm_{var}_{region}.png"   # region in the name: BE and UK
        fig.savefig(out_png, dpi=150)                          # runs must not overwrite each other
        plt.close(fig)
        print(f"Saved: {out_png}")

    ds_cerra.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
