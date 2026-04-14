"""
RMSE vs lead time against CERRA truth.
Grids are identical (24444 cells), so no spatial mapping needed.
One plot per variable; each forecast directory is a separate curve.
Only init times present in ALL directories are used (fair comparison).
"""
 
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
 
# -------------------- SETTINGS --------------------
TARGET_VARS = ["ws10","ws100", "t_850","q_700","t2m"]
 
FORECAST_DIRS = {
    "NoPowerGT": Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerGT"),
    "NoPowerTF":  Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerTF"),
    "VanillaPowerGT":  Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGT"),
    "VanillaPowerTF":  Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerTF"),
}
 
CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")
OUT_DIR    = Path("EGU_plots")
 
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 39, 3))
# --------------------------------------------------
 
FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")
 
 
def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True,
    )
 
 
def main():
    ds_cerra   = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars = list(ds_cerra.attrs["variables"])
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")
 
    OUT_DIR.mkdir(parents=True, exist_ok=True)
 
    # --- collect file maps per directory ---
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
 
    # --- intersect init times across all directories ---
    common_inits = sorted(
        set.intersection(*(set(m) for m in dir_file_maps.values()))
    )
    print(f"Common init times: {len(common_inits)}")
 
    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v"]
 
    for var in TARGET_VARS:
        var_idx = cerra_vars.index(var)
        fig, ax = plt.subplots(figsize=(9, 5))
 
        for i, (label, fmap) in enumerate(dir_file_maps.items()):
            lead_errors = {lh: [] for lh in LEAD_HOURS}
 
            for init in common_inits:
                ds_fc = xr.open_dataset(fmap[init])
                fc_times = pd.to_datetime(ds_fc["time"].values).tz_localize("UTC")
 
                for lh in LEAD_HOURS:
                    valid = init + pd.Timedelta(hours=lh)
                    if valid not in fc_times or valid not in cerra_dates:
                        continue
 
                    fc_vals = ds_fc[var].sel(time=valid.replace(tzinfo=None)).values
                    tr_vals = ds_cerra["data"].isel(
                        time=np.where(cerra_dates == valid)[0][0],
                        variable=var_idx,
                        ensemble=0,
                    ).values
 
                    lead_errors[lh].append(
                        np.sqrt(np.nanmean((fc_vals - tr_vals) ** 2))
                    )
 
                ds_fc.close()
 
            leads     = sorted(lead_errors)
            mean_rmse = [np.mean(lead_errors[lh]) if lead_errors[lh] else np.nan
                         for lh in leads]
 
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
 
        ax.set_title(f"RMSE vs Lead Time — {var}  (n={len(common_inits)} inits)",
                     fontsize=13)
        ax.set_xlabel("Lead time [hours]")
        ax.set_ylabel("RMSE [m/s]")
        ax.legend(title="Run", framealpha=0.8)
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"rmse_{var}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {OUT_DIR / f'rmse_{var}.png'}")
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()