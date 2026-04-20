"""
verify_power_forecasts_sequential.py
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

REGIONS = ["BE"]

METADATA_CSV = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

FORECAST_DIRS = {
    "GT": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGT"),
    "TF": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerTF"),
    "GTROllout": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGTRollout"),
    "Synthetic": Path("/mnt/weatherloss/WindPower/inference/EGU/SyntheticGT"),
}

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")
OUT_DIR    = Path("EGU_large")

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-02-28 18:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))

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
                  cell_caps: np.ndarray, valid_iso_set: set) -> dict:
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
    cell_caps       = np.array(list(cell_map.values()))
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

    # Only keep timesteps with valid (non-NaN) observations
    valid_iso_set = {
        iso for iso, arr in cerra_obs_cache.items()
        if not np.all(np.isnan(arr))
    }
    # Only keep inits where ALL lead hours have valid obs
    common_inits = [
        init for init in common_inits
        if all(
            (init + pd.Timedelta(hours=lh)).isoformat() in valid_iso_set
            for lh in LEAD_HOURS
        )
    ]
    print(f"Inits with full lead hour coverage: {len(common_inits)}")
    print(f"Valid obs timesteps: {len(valid_iso_set)}/{len(cerra_obs_cache)}")

    # Per-lead-hour coverage
    for lh in LEAD_HOURS:
        count = sum(
            1 for init in common_inits
            if (init + pd.Timedelta(hours=lh)).isoformat() in valid_iso_set
        )
        print(f"  Lead {lh:2d}h: {count} valid obs timesteps")

    print("CERRA preload done.")
    # Check a few raw values to detect time shifts
    test_init = common_inits[100]
    print(f"\nTime shift check — init: {test_init}")
    with h5py.File(str(dir_file_maps["GT"][test_init]), "r") as f:
        tv  = f["time"]
        raw = nc4.num2date(tv[:], tv.attrs["units"].decode(),
                           tv.attrs.get("calendar", b"standard").decode())
        fc_times = [pd.Timestamp(str(t)).tz_localize("UTC") for t in raw]
        fc_time_to_idx = {t.isoformat(): j for j, t in enumerate(fc_times)}
        power_all = f["power"][:, :]

    power_sel = power_all[:, cell_cerra_idxs]

    for lh in [3, 15, 30]:
        valid     = test_init + pd.Timedelta(hours=lh)
        viso      = valid.isoformat()
        tidx      = fc_time_to_idx.get(viso)
        fc_mw     = float(np.sum(power_sel[tidx])) if tidx is not None else None
        obs_arr   = cerra_obs_cache.get(viso)
        obs_mw    = float(np.nansum(obs_arr)) if obs_arr is not None else None
        print(f"  +{lh:2d}h  valid={valid}  fc={fc_mw:.0f} MW  obs={obs_mw:.0f} MW  err={abs(fc_mw-obs_mw):.0f} MW")

    # Also check obs at adjacent timesteps to detect offset
    valid3 = test_init + pd.Timedelta(hours=3)
    for offset in [-6, -3, 0, 3, 6]:
        t   = valid3 + pd.Timedelta(hours=offset)
        arr = cerra_obs_cache.get(t.isoformat())
        mw  = float(np.nansum(arr)) if arr is not None else None
        print(f"  obs at valid+{offset:+d}h ({t}): {mw:.0f} MW" if mw is not None else f"  obs at {t}: None")
    # --- 5. Process sequentially ---
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, fmap in dir_file_maps.items():
        print(f"\nProcessing {label}...")
        lead_errors = {lh: [] for lh in LEAD_HOURS}

        for count, init in enumerate(common_inits):
            if count % 200 == 0:
                print(f"  {count}/{len(common_inits)}...", flush=True)
            try:
                fc = read_forecast(
                    fmap[init], init, LEAD_HOURS,
                    cell_cerra_idxs, cell_caps, valid_iso_set,
                )
                for lh, fc_mw in fc.items():
                    viso   = (init + pd.Timedelta(hours=lh)).isoformat()
                    obs_mw = float(np.nansum(cerra_obs_cache[viso]))
                    lead_errors[lh].append(abs(fc_mw - obs_mw))

            except Exception as e:
                print(f"  Skipping {fmap[init].name}: {e}")

        leads   = sorted(lead_errors)
        mae_mw  = [np.mean(lead_errors[lh]) if lead_errors[lh] else np.nan for lh in leads]
        mae_pct = [x / grand_total_cap * 100 if not np.isnan(x) else np.nan for x in mae_mw]

        print(f"{label} sample sizes: { {lh: len(lead_errors[lh]) for lh in leads} }")
        print(f"{label} MAE MW:  {[round(x, 1) for x in mae_mw]}")
        print(f"{label} MAE %:   {[round(x, 2) for x in mae_pct]}")

        ax.plot(leads, mae_pct, marker="o", label=label)
        np.save(OUT_DIR / f"mae_{label}.npy", np.column_stack([leads, mae_pct]))

    # --- 6. Finalise ---
    region_str = ", ".join(REGIONS) if REGIONS else "All"
    ax.set_title(
        f"Power MAE Comparison\nRegion: {region_str}  "
        f"({common_inits[0].date()} – {common_inits[-1].date()})"
    )
    ax.set_xlabel("Lead time [hours]")
    ax.set_ylabel("MAE [% of Total Capacity]")
    ax.set_xticks(LEAD_HOURS)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mae_comparison.png", dpi=150)
    print(f"\nDone. Plot saved to {OUT_DIR}/mae_comparison.png")


if __name__ == "__main__":
    main()