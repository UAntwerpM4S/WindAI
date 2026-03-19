
"""
Verify RMSE vs lead time against CERRA truth.
Nearest-neighbor mapping built once (CERRA -> forecast), then reused.
Supports multiple variables and multiple forecast directories.
One plot per variable; each directory is a separate curve.
Only init times present in ALL directories are used (fair comparison).
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# -------------------- SETTINGS --------------------
TARGET_VARS = ["ws100"] #, "ws100","z_500"]   # <-- add/remove variables here

# Dict of label -> directory.  Each entry becomes one curve per plot.
FORECAST_DIRS = {
    "GraphTransformer":  Path("/mnt/weatherloss/WindPower/inference/CI/GraphTransformerNew"),
    #"Transformer":  Path("/mnt/weatherloss/WindPower/inference/CI/TransformerNew"),
        "GNN150":    Path("/mnt/weatherloss/WindPower/inference/CI/GNN150"),
         "GNN156":    Path("/mnt/weatherloss/WindPower/inference/CI/GNN156"),
}

CERRA_PATH  = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/Cerra_A.zarr")
OUT_DIR     = Path("CI_plots")

INIT_START  = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END    = pd.Timestamp("2024-9-15 21:00:00", tz="UTC")
LEAD_HOURS  = list(range(3, 73, 3))
MAX_DIST_KM = 2.0   # drop CERRA points with no nearby forecast point
# --------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True
    )


def to_xy(lon, lat):
    lat0 = np.deg2rad(np.mean(lat))
    R = 6371.0
    return np.column_stack([
        R * np.deg2rad(lon) * np.cos(lat0),
        R * np.deg2rad(lat)
    ])


def build_index_mapping(fc_lat, fc_lon, cerra_lat, cerra_lon, max_dist_km):
    tree = cKDTree(to_xy(fc_lon, fc_lat))
    dist, fc_idx = tree.query(to_xy(cerra_lon, cerra_lat), k=1)
    ok = dist <= max_dist_km
    print(f"  CERRA cells matched within {max_dist_km} km: {ok.sum()} / {len(ok)}")
    print(f"  Max distance among matched points: {dist[ok].max():.4f} km")
    return np.where(ok)[0], fc_idx[ok]


def collect_rmse(files, var, cerra_keep, fc_indices, cerra_dates, ds_cerra, var_idx):
    """Loop over forecast files and collect per-lead RMSE for one variable."""
    lead_rmse = {lh: [] for lh in LEAD_HOURS}

    for fpath in files:
        init = parse_init(fpath)
        ds_fc = xr.open_dataset(fpath)
        fc_times = pd.to_datetime(ds_fc["time"].values).tz_localize("UTC")

        for lh in LEAD_HOURS:
            valid = init + pd.Timedelta(hours=lh)
            if valid not in fc_times:
                continue
            if valid not in cerra_dates:
                continue

            t_fc = int(np.where(fc_times == valid)[0][0])
            t_tr = int(np.where(cerra_dates == valid)[0][0])

            fc_vals = ds_fc[var].isel(time=t_fc).values[fc_indices]
            tr_vals = ds_cerra["data"].isel(
                time=t_tr, variable=var_idx, ensemble=0
            ).values[cerra_keep]

            rmse = float(np.sqrt(np.nanmean((fc_vals - tr_vals) ** 2)))
            lead_rmse[lh].append(rmse)

        ds_fc.close()

    leads     = sorted(lead_rmse)
    mean_rmse = [np.mean(lead_rmse[lh]) if lead_rmse[lh] else np.nan for lh in leads]
    counts    = [len(lead_rmse[lh]) for lh in leads]
    return pd.DataFrame({"lead_hours": leads, "RMSE": mean_rmse, "n_inits": counts})


def main():
    # --- open CERRA once ---
    ds_cerra    = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars  = list(ds_cerra.attrs["variables"])
    cerra_lat   = ds_cerra["latitudes"].values
    cerra_lon   = ds_cerra["longitudes"].values
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")

    for var in TARGET_VARS:
        if var not in cerra_vars:
            raise ValueError(f"Variable '{var}' not found in CERRA. Available: {cerra_vars}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- build mapping once (shared grid) ---
    print("\n--- Building nearest-neighbour mapping (shared grid) ---")
    first_file = None
    for fc_dir in FORECAST_DIRS.values():
        candidates = sorted(fc_dir.glob("forecast_*.nc"))
        candidates = [f for f in candidates
                      if len(xr.open_dataset(f).data_vars) > 0
                      and INIT_START <= parse_init(f) <= INIT_END]
        if candidates:
            first_file = candidates[0]
            break

    if first_file is None:
        raise RuntimeError("No valid forecast files found in any directory.")

    ds0 = xr.open_dataset(first_file)
    cerra_keep, fc_indices = build_index_mapping(
        ds0["latitude"].values, ds0["longitude"].values,
        cerra_lat, cerra_lon, MAX_DIST_KM
    )
    ds0.close()

    # --- collect file lists per directory (init -> path) ---
    dir_file_maps = {}
    for label, fc_dir in FORECAST_DIRS.items():
        print(f"\n--- Directory: {label} ({fc_dir}) ---")
        files = sorted(fc_dir.glob("forecast_*.nc"))
        files = [f for f in files
                 if len(xr.open_dataset(f).data_vars) > 0
                 and INIT_START <= parse_init(f) <= INIT_END]
        print(f"  Valid forecast files in period: {len(files)}")
        if not files:
            print("  WARNING: no files found, skipping.")
            continue
        dir_file_maps[label] = {parse_init(f): f for f in files}

    if not dir_file_maps:
        raise RuntimeError("No valid forecast files found in any directory.")

    # --- intersect init times across all directories ---
    common_inits = sorted(set.intersection(*(set(m.keys()) for m in dir_file_maps.values())))
    print(f"\n--- Overlapping init times across all directories: {len(common_inits)} ---")
    for label, fmap in dir_file_maps.items():
        n_dropped = len(fmap) - len(common_inits)
        print(f"  {label}: {len(fmap)} total, {n_dropped} dropped (not in all dirs)")

    dir_data = {
        label: [fmap[t] for t in common_inits]
        for label, fmap in dir_file_maps.items()
    }

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for var in TARGET_VARS:
        var_idx = cerra_vars.index(var)
        print(f"\n=== Variable: {var} ===")

        fig, ax = plt.subplots(figsize=(9, 5))
        dfs = {}
        for i, (label, files) in enumerate(dir_data.items()):
            print(f"  Processing directory: {label}")
            df = collect_rmse(files, var, cerra_keep, fc_indices,
                            cerra_dates, ds_cerra, var_idx)
            dfs[label] = df
            print(df.to_string(index=False))

            ax.plot(
                df["lead_hours"], df["RMSE"],
                marker=markers[i % len(markers)],   # different marker
                lw=1,
                color=colors[i % len(colors)],
                label=label
            )

        ax.set_title(f"RMSE vs Lead Time — {var}  (n={len(common_inits)} inits)", fontsize=13)
        ax.set_xlabel("Lead time [hours]")
        ax.set_ylabel("RMSE [m/s]")
        ax.legend(title="Run", framealpha=0.8)
        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()

        out_png = OUT_DIR / f"rmse_test_{var}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_png}")

    for label, df in dfs.items():
            safe_label = label.replace(" ", "_")
            out_npy = OUT_DIR / f"rmse_{var}_{safe_label}.npy"
            np.save(out_npy, df[["lead_hours", "RMSE"]].values)
            print(f"  Saved: {out_npy}")

    print("\nDone.")


if __name__ == "__main__":
    main()