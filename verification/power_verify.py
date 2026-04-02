

from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# -------------------- SETTINGS --------------------
FORECAST_DIRS = {
   # "NoPower2":     Path("/mnt/weatherloss/WindPower/inference/EGU/NoPower2"),
    "VanillaPower": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPower"),
    "Finetune":     Path("/mnt/weatherloss/WindPower/inference/EGU/Finetune"),
    "TinyPower":    Path("/mnt/weatherloss/WindPower/inference/EGU/TinyPower"),
}

CERRA_PATH    = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/Cerra_A.zarr")
METADATA_PATH = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")
OUT_DIR       = Path("power_plots")

INIT_START  = pd.Timestamp("2024-08-03 00:00:00", tz="UTC")
INIT_END    = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS  = list(range(3, 31, 3))
MAX_DIST_KM = 2.0
# --------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True
    )


def to_xy(lon, lat):
    R = 6371.0
    lat0 = np.deg2rad(np.mean(lat))
    return np.column_stack([
        R * np.deg2rad(lon) * np.cos(lat0),
        R * np.deg2rad(lat)
    ])


def get_belgian_metadata() -> tuple[np.ndarray, float]:
    """Return unique CERRA cell indices for Belgian farms and total capacity in MW."""
    meta = pd.read_csv(METADATA_PATH)
    be = meta[meta["region"].str.lower() == "belgium"]
    total_capacity_mw = float(be["capacity_mw"].sum())
    be_unique = be.drop_duplicates(subset=["cerra_grid_lat", "cerra_grid_lon"])
    print(f"  Belgian farms: {len(be)}, unique CERRA cells: {len(be_unique)}")
    print(f"  Total Belgian capacity: {total_capacity_mw:.1f} MW")
    return be_unique, total_capacity_mw


def get_belgian_cerra_indices(cerra_lat: np.ndarray, cerra_lon: np.ndarray) -> np.ndarray:
    be_unique, _ = get_belgian_metadata()
    cerra_keys = {(round(la, 6), round(lo, 6)): i
                  for i, (la, lo) in enumerate(zip(cerra_lat, cerra_lon))}
    indices = []
    for _, row in be_unique.iterrows():
        key = (round(row["cerra_grid_lat"], 6), round(row["cerra_grid_lon"], 6))
        if key not in cerra_keys:
            raise ValueError(f"CERRA cell not found: {key}")
        indices.append(cerra_keys[key])
    return np.array(indices, dtype=int)


def build_fc_indices(fc_lat, fc_lon, cerra_lat, cerra_lon, cerra_keep, max_dist_km):
    tree = cKDTree(to_xy(fc_lon, fc_lat))
    dist, fc_idx = tree.query(to_xy(cerra_lon[cerra_keep], cerra_lat[cerra_keep]), k=1)
    if dist.max() > max_dist_km:
        raise ValueError(
            f"Some CERRA cells have no forecast point within {max_dist_km} km. "
            f"Max distance: {dist.max():.4f} km"
        )
    print(f"  Max NN distance: {dist.max():.4f} km")
    return fc_idx


def collect_mae(files, cerra_keep, fc_indices, cerra_dates, ds_cerra, var_idx):
    """Collect per-lead MAE in MW (summed over Belgian cells)."""
    lead_errors = {lh: [] for lh in LEAD_HOURS}

    for fpath in files:
        init = parse_init(fpath)
        ds_fc = xr.open_dataset(fpath)
        fc_times = pd.to_datetime(ds_fc["time"].values).tz_localize("UTC")

        for lh in LEAD_HOURS:
            valid = init + pd.Timedelta(hours=lh)
            if valid not in fc_times or valid not in cerra_dates:
                continue

            t_fc = int(np.where(fc_times == valid)[0][0])
            t_tr = int(np.where(cerra_dates == valid)[0][0])

            fc_vals = ds_fc["power"].isel(time=t_fc).values[fc_indices]
            tr_vals = ds_cerra["data"].isel(
                time=t_tr, variable=var_idx, ensemble=0
            ).values[cerra_keep]

            # Sum over cells → total Belgian power at this valid time
            fc_total = float(np.nansum(fc_vals))
            tr_total = float(np.nansum(tr_vals))
            lead_errors[lh].append(abs(fc_total - tr_total))

        ds_fc.close()

    leads    = sorted(lead_errors)
    mean_mae = [np.mean(lead_errors[lh]) if lead_errors[lh] else np.nan for lh in leads]
    counts   = [len(lead_errors[lh]) for lh in leads]
    return pd.DataFrame({"lead_hours": leads, "MAE_MW": mean_mae, "n_inits": counts})


def plot_mae(results: dict[str, pd.DataFrame], col: str, ylabel: str, title: str, out_path: Path) -> None:
    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, df) in enumerate(results.items()):
        ax.plot(df["lead_hours"], df[col],
                marker=markers[i % len(markers)], lw=1.5,
                color=colors[i % len(colors)], label=label)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Lead time [hours]")
    ax.set_ylabel(ylabel)
    ax.legend(title="Run", framealpha=0.8)
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ds_cerra    = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars  = list(ds_cerra.attrs["variables"])
    cerra_lat   = ds_cerra["latitudes"].values
    cerra_lon   = ds_cerra["longitudes"].values
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")

    if "power" not in cerra_vars:
        raise ValueError(f"'power' not found in CERRA. Available: {cerra_vars}")
    var_idx = cerra_vars.index("power")

    _, total_capacity_mw = get_belgian_metadata()
    cerra_keep = get_belgian_cerra_indices(cerra_lat, cerra_lon)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- collect and intersect file lists ---
    dir_file_maps = {}
    for label, fc_dir in FORECAST_DIRS.items():
        files = sorted(fc_dir.glob("forecast_*.nc"))
        files = [f for f in files if INIT_START <= parse_init(f) <= INIT_END]
        print(f"{label}: {len(files)} files in period")
        if files:
            dir_file_maps[label] = {parse_init(f): f for f in files}

    if not dir_file_maps:
        raise RuntimeError("No valid forecast files found.")

    common_inits = sorted(set.intersection(*(set(m.keys()) for m in dir_file_maps.values())))
    print(f"\nOverlapping init times: {len(common_inits)}")

    dir_data = {
        label: [fmap[t] for t in common_inits]
        for label, fmap in dir_file_maps.items()
    }

    # --- build NN mapping once ---
    any_file = next(iter(dir_data.values()))[0]
    ds0 = xr.open_dataset(any_file)
    fc_indices = build_fc_indices(
        ds0["latitude"].values, ds0["longitude"].values,
        cerra_lat, cerra_lon, cerra_keep, MAX_DIST_KM
    )
    ds0.close()

    # --- compute MAE per directory ---
    dfs = {}
    for label, files in dir_data.items():
        print(f"  {label} ...", end=" ", flush=True)
        df = collect_mae(files, cerra_keep, fc_indices, cerra_dates, ds_cerra, var_idx)
        df["MAE_pct"] = df["MAE_MW"] / total_capacity_mw * 100.0
        dfs[label] = df
        print("done")

    n = len(common_inits)
    plot_mae(
        dfs, col="MAE_MW",
        ylabel="MAE [MW]",
        title=f"Power MAE vs Lead Time — Belgian offshore  (n={n} inits)",
        out_path=OUT_DIR / "mae_power_belgium_MW.png",
    )
    plot_mae(
        dfs, col="MAE_pct",
        ylabel=f"MAE [% of {total_capacity_mw:.0f} MW]",
        title=f"Power MAE vs Lead Time — Belgian offshore  (n={n} inits)",
        out_path=OUT_DIR / "mae_power_belgium_pct.png",
    )

    for label, df in dfs.items():
        np.save(OUT_DIR / f"mae_power_{label}.npy", df[["lead_hours", "MAE_MW", "MAE_pct"]].values)

    print("\nDone.")


if __name__ == "__main__":
    main()