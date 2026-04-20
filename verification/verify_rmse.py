"""
RMSE vs lead time against CERRA truth.
"""

from pathlib import Path
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import re
import numpy as np
import pandas as pd
import xarray as xr
import h5py
import netCDF4 as nc4
import matplotlib.pyplot as plt

# -------------------- SETTINGS --------------------
TARGET_VARS = ["ws100"] #, "ws100"] #, "t_850", "q_700", "t2m", "z_500"]

FORECAST_DIRS = {
    "NoPowerTF":      Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerTF"),
    "NoPowerGT":      Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerGT"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGT"),
    "VanillaPowerTF": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerTF"),
    "SyntheticGT": Path("/mnt/weatherloss/WindPower/inference/EGU/SyntheticGT"),
    "VanillaPowerGTRollout": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGTRollout"),
}

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")
OUT_DIR    = Path("EGU_large")

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-02-28 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))

N_WORKERS  = 8
# --------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True,
    )


def _read_one_file(args):
    """
    Worker: one process per file, uses h5py to avoid netCDF4 heap corruption.
    Returns {lead_hour: mean_squared_error}.
    """
    nc_path, init_iso, lead_hours, var_name, cerra_cache_items = args

    init        = pd.Timestamp(init_iso)
    cerra_cache = {iso: arr for iso, arr in cerra_cache_items}
    result      = {}

    try:
        import h5py
        import netCDF4 as nc4

        with h5py.File(str(nc_path), "r") as f:
            tv  = f["time"]
            raw = nc4.num2date(
                tv[:],
                tv.attrs["units"].decode(),
                tv.attrs.get("calendar", b"standard").decode(),
            )
            fc_times       = [pd.Timestamp(str(t)).tz_localize("UTC") for t in raw]
            fc_time_to_idx = {t.isoformat(): j for j, t in enumerate(fc_times)}

            # Plain slice — no fancy indexing on h5py dataset
            var_all = f[var_name][:, :]   # (n_times, n_cells)

        # All file I/O done — pure numpy from here
        for lh in lead_hours:
            valid_iso = (init + pd.Timedelta(hours=lh)).isoformat()
            if valid_iso not in fc_time_to_idx or valid_iso not in cerra_cache:
                continue
            tidx    = fc_time_to_idx[valid_iso]
            fc_vals = var_all[tidx]
            ob_vals = cerra_cache[valid_iso]
            result[lh] = float(np.nanmean((fc_vals - ob_vals) ** 2))

    except Exception as e:
        print(f"  WORKER ERROR {Path(nc_path).name}: {e}", flush=True)

    return init_iso, result


def main():
    mp.set_start_method("spawn", force=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v"]

    for var in TARGET_VARS:
        var_idx = cerra_vars.index(var)

        # ── preload CERRA for this variable ───────────────────────────────────
        needed_valid_times = sorted({
            init + pd.Timedelta(hours=lh)
            for init in common_inits
            for lh in LEAD_HOURS
            if (init + pd.Timedelta(hours=lh)) in cerra_date_to_idx
        })
        needed_cerra_idxs = [cerra_date_to_idx[t] for t in needed_valid_times]
        print(f"Preloading {len(needed_cerra_idxs)} CERRA timesteps for {var}...")
        cerra_bulk = ds_cerra["data"].isel(
            time=needed_cerra_idxs,
            variable=var_idx,
            ensemble=0,
        ).values  # (n_times, n_cells)

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
            print(f"\n{label} — {var}\n{df.to_string(index=False)}")

            ax.plot(
                df["lead_hours"], df["RMSE"],
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                lw=1.5, label=label,
            )

            np.save(OUT_DIR / f"rmse_{var}_{label}.npy",
                    df[["lead_hours", "RMSE"]].values)

        ax.set_title(f"RMSE vs Lead Time — {var} Aug 24 - Feb 25 ({len(common_inits)} )",
        #ax.set_title(f"RMSE vs Lead Time — {var}  (n={len(common_inits)} inits)",
                     fontsize=13)
        ax.set_xlabel("Lead time [hours]")
        ax.set_ylabel("RMSE")
        ax.legend(title="Run", framealpha=0.8)
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"rmse_{var}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {OUT_DIR / f'rmse_{var}.png'}")

    ds_cerra.close()
    print("\nDone.")


if __name__ == "__main__":
    main()