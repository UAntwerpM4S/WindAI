"""
power_verify.py  —  Verify wind power forecasts against CERRA observations.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
import h5py
import netCDF4 as nc4
import matplotlib.pyplot as plt

# -------------------- SETTINGS --------------------

REGIONS = ["Belgium"]

METADATA_CSV = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

FORECAST_DIRS = {
    "BigTransformer (Vanilla + Synthetic power)":      Path("/mnt/weatherloss/WindPower/inference/EGU/SyntheticTF"),
    "BigTransformer (Vanilla power)":          Path("/mnt/weatherloss/WindPower/inference/EGU/BigTransformerRollout"),
  #  "GraphTransformer (Vanilla power)":  Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGTRollout"),
}

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")
OUT_DIR    = Path("EGU_large")

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 18:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 40, 3))

# --------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True,
    )


def load_farms(csv_path: Path, regions=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["region", "farm", "capacity_mw", "lat", "lon",
              "cerra_y", "cerra_grid_lat", "cerra_grid_lon"]
    df = df[needed].dropna(subset=["lat", "lon", "capacity_mw"])
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    df = df.dropna(subset=["capacity_mw"])
    if regions is not None:
        df = df[df["region"].str.lower().isin([r.lower() for r in regions])]
    return df.reset_index(drop=True)


def read_forecast(nc_path: Path, init: pd.Timestamp,
                  lead_hours: list, cell_cerra_idxs: np.ndarray,
                  valid_iso_set: set) -> dict:
    result = {}
    with h5py.File(str(nc_path), "r") as f:
        tv  = f["time"]
        raw = nc4.num2date(
            tv[:],
            tv.attrs["units"].decode(),
            tv.attrs.get("calendar", b"standard").decode(),
        )
        fc_times       = [pd.Timestamp(str(t)).tz_localize("UTC") for t in raw]
        fc_time_to_idx = {t.isoformat(): j for j, t in enumerate(fc_times)}
        power_all = f["power"][:, :]

    power_sel = power_all[:, cell_cerra_idxs]

    for lh in lead_hours:
        viso = (init + pd.Timedelta(hours=lh)).isoformat()
        if viso in fc_time_to_idx and viso in valid_iso_set:
            tidx       = fc_time_to_idx[viso]
            result[lh] = float(np.sum(power_sel[tidx]))

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Metadata ---
    farms_df = load_farms(METADATA_CSV, regions=REGIONS)
    cell_map = {}
    for _, row in farms_df.iterrows():
        idx = int(row["cerra_y"])
        cell_map[idx] = cell_map.get(idx, 0.0) + float(row["capacity_mw"])

    grand_total_cap = sum(cell_map.values())
    cell_cerra_idxs = np.array(list(cell_map.keys()))
    print(f"Region: {REGIONS} | Capacity: {grand_total_cap:.1f} MW")

    # --- 2. Load CERRA ---
    ds_cerra    = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars  = list(ds_cerra.attrs["variables"])
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")
    cerra_date_to_idx = {d: i for i, d in enumerate(cerra_dates)}
    power_idx   = cerra_vars.index("power")

    # --- 3. File maps ---
    dir_file_maps = {}
    for label, fc_dir in FORECAST_DIRS.items():
        fmap = {parse_init(f): f for f in sorted(fc_dir.glob("forecast_*.nc"))
                if INIT_START <= parse_init(f) <= INIT_END}
        if fmap:
            dir_file_maps[label] = fmap
            print(f"{label}: {len(fmap)} files")

    common_inits = sorted(set.intersection(*(set(m) for m in dir_file_maps.values())))
    print(f"Common inits: {len(common_inits)}")
    print(f"Init range: {common_inits[0]} to {common_inits[-1]}")

    # --- 4. Preload CERRA ---
    needed_valid = sorted({
        i + pd.Timedelta(hours=lh)
        for i in common_inits for lh in LEAD_HOURS
        if (i + pd.Timedelta(hours=lh)) in cerra_date_to_idx
    })
    print(f"Preloading CERRA ({len(needed_valid)} steps)...")
    cerra_bulk = ds_cerra["data"].isel(
        time=[cerra_date_to_idx[t] for t in needed_valid],
        variable=power_idx, ensemble=0,
    ).values[:, cell_cerra_idxs]
    cerra_obs_cache = {t.isoformat(): cerra_bulk[i] for i, t in enumerate(needed_valid)}
    ds_cerra.close()
    del cerra_bulk

    valid_iso_set = {
        iso for iso, arr in cerra_obs_cache.items()
        if not np.any(np.isnan(arr))
    }
    common_inits = [
        init for init in common_inits
        if all(
            (init + pd.Timedelta(hours=lh)).isoformat() in valid_iso_set
            for lh in LEAD_HOURS
        )
    ]
    print(f"Inits with full lead hour coverage: {len(common_inits)}")
    print(f"CERRA preload done.")

    # --- 5. Process sequentially ---
    fig, (ax_mae, ax_rmse) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for label, fmap in dir_file_maps.items():
        print(f"\nProcessing {label}...")
        lead_errors = {lh: [] for lh in LEAD_HOURS}

        for count, init in enumerate(common_inits):
            if count % 200 == 0:
                print(f"  {count}/{len(common_inits)}...", flush=True)
            try:
                fc = read_forecast(
                    fmap[init], init, LEAD_HOURS,
                    cell_cerra_idxs, valid_iso_set,
                )
                for lh, fc_mw in fc.items():
                    viso   = (init + pd.Timedelta(hours=lh)).isoformat()
                    obs_mw = float(np.sum(cerra_obs_cache[viso]))
                    lead_errors[lh].append(fc_mw - obs_mw)  # signed for RMSE

            except Exception as e:
                print(f"  Skipping {fmap[init].name}: {e}")

        leads    = sorted(lead_errors)
        errors   = [np.array(lead_errors[lh]) if lead_errors[lh] else np.array([np.nan]) for lh in leads]

        mae_pct  = [np.mean(np.abs(e)) / grand_total_cap * 100 for e in errors]
        rmse_pct = [np.sqrt(np.mean(e ** 2)) / grand_total_cap * 100 for e in errors]

        ax_mae.plot(leads, mae_pct, marker="o", label=label)
        ax_rmse.plot(leads, rmse_pct, marker="o", label=label)

        np.save(OUT_DIR / f"mae_{label}.npy",  np.column_stack([leads, mae_pct]))
        np.save(OUT_DIR / f"rmse_{label}.npy", np.column_stack([leads, rmse_pct]))

    # --- 6. Finalise ---
    region_str = ", ".join(REGIONS) if REGIONS else "All"
    date_range = f"{common_inits[0].date()} – {common_inits[-1].date()}"
    suptitle   = f"Region: {region_str}  ({date_range})"

    for ax, metric in [(ax_mae, "MAE"), (ax_rmse, "RMSE")]:
        ax.set_title(f"Power {metric} Comparison")
        ax.set_xlabel("Lead time [hours]")
        ax.set_ylabel(f"{metric} [% of Total Capacity]")
        ax.set_xticks(LEAD_HOURS)
        ax.legend()
        ax.grid(True, ls="--", alpha=0.5)

    fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mae_rmse_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nDone. Plot saved to {OUT_DIR}/mae_rmse_comparison.png")


if __name__ == "__main__":
    main()