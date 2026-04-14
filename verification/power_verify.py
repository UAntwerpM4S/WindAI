"""
verify_power_forecasts.py
=========================
MAE vs lead time for wind-power forecasts against CERRA truth.

Cell-to-farm matching replicates append_turbine_vars_xy() exactly:
  - KD-tree is built over the FULL CERRA grid (all y*x points)
  - Each farm is matched to its nearest CERRA cell globally
  - Farms sharing the same cell have their capacities summed
This ensures the verified cells are identical to those used during training.

MAE is expressed as % of grand total capacity.
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

# Region filter: None           → all farms in the CSV
#                ["Belgium"]    → Belgian farms only
#                ["UK"]         → UK only
#                ["Belgium","UK"] → both explicitly
REGIONS = None
# REGIONS = ["Belgium"]

METADATA_CSV = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

FORECAST_DIRS = {
    # "NoPowerGT":      Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerGT"),
    # "NoPowerTF":      Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerTF"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGT"),
    "VanillaPowerTF": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerTF"),
}

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")
OUT_DIR    = Path("EGU_plots")

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 39, 3))   # 3 h … 36 h

# --------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True,
    )


def load_farms(csv_path: Path, regions=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["region", "farm", "capacity_mw", "lat", "lon"]
    df = df[needed].dropna(subset=["lat", "lon", "capacity_mw"])
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    df = df.dropna(subset=["capacity_mw"])
    if regions is not None:
        df = df[df["region"].isin(regions)]
        if df.empty:
            raise ValueError(
                f"No farms found for regions: {regions}. "
                f"Available: {pd.read_csv(csv_path)['region'].unique().tolist()}"
            )
    return df.reset_index(drop=True)


def build_cells(ds_cerra: xr.Dataset, farms_df: pd.DataFrame) -> list[dict]:
    """
    Replicates append_turbine_vars_xy() exactly:
      - KD-tree over the FULL (y, x) CERRA grid in (lon, lat) space
      - Each farm snaps to its globally nearest CERRA cell
      - Cells are grouped by flat index; capacities summed per cell

    Returns a list of dicts, one per unique matched cell:
        {
            'cerra_flat_idx': int,
            'cerra_lat':      float,
            'cerra_lon':      float,
            'fc_value_idx':   int,    # filled later by match_to_forecast()
            'total_cap_mw':   float,
            'farms':          list[str],
        }
    """
    lat2d = ds_cerra["latitudes"].values    # (y, x)
    lon2d = ds_cerra["longitudes"].values  # (y, x)
    ny, nx = lat2d.shape

    # Build tree over full grid — same as original script (lon, lat order)
    tree = cKDTree(np.c_[lon2d.ravel(), lat2d.ravel()])

    # Query with farm (lon, lat) — same order as original
    farm_lonlat = farms_df[["lon", "lat"]].values
    _, flat_indices = tree.query(farm_lonlat, k=1)

    # Group farms by matched flat index
    cell_map: dict[int, dict] = {}
    for i, row in farms_df.iterrows():
        flat_idx = int(flat_indices[i])
        iy, ix   = np.unravel_index(flat_idx, (ny, nx))
        if flat_idx not in cell_map:
            cell_map[flat_idx] = {
                "cerra_flat_idx": flat_idx,
                "cerra_lat":      float(lat2d[iy, ix]),
                "cerra_lon":      float(lon2d[iy, ix]),
                "fc_value_idx":   -1,
                "total_cap_mw":   0.0,
                "farms":          [],
            }
        cell_map[flat_idx]["total_cap_mw"] += float(row["capacity_mw"])
        cell_map[flat_idx]["farms"].append(row["farm"])

    return list(cell_map.values())


def match_to_forecast(cells: list[dict], fc_lat: np.ndarray, fc_lon: np.ndarray) -> None:
    """
    For each CERRA cell, find the nearest point in the forecast grid
    (1-D values dim) and store its index in cell['fc_value_idx'].
    Modifies cells in-place.
    """
    tree = cKDTree(np.c_[fc_lon, fc_lat])
    cell_coords = np.array([[c["cerra_lon"], c["cerra_lat"]] for c in cells])
    _, indices  = tree.query(cell_coords, k=1)
    for cell, idx in zip(cells, indices):
        cell["fc_value_idx"] = int(idx)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── observations ──────────────────────────────────────────────────────────
    ds_cerra    = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars  = list(ds_cerra.attrs["variables"])
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")
    power_idx   = cerra_vars.index("power")

    # ── build cell map (mirrors original zarr-building script) ────────────────
    farms_df = load_farms(METADATA_CSV, regions=REGIONS)
    region_label = (
        "Belgium only" if REGIONS == ["Belgium"]
        else (", ".join(REGIONS) if REGIONS else "All regions")
    )
    print(f"\nRegion filter : {region_label}  ({len(farms_df)} farms)")

    cells = build_cells(ds_cerra, farms_df)
    grand_total_cap = sum(c["total_cap_mw"] for c in cells)

    print(f"Unique CERRA cells matched : {len(cells)}")
    print(f"Grand total capacity       : {grand_total_cap:.1f} MW\n")
    print(f"{'Flat idx':<12}  {'Lat':>8}  {'Lon':>8}  {'Cap (MW)':>10}  Farms")
    for c in cells:
        print(f"  {c['cerra_flat_idx']:<12}  {c['cerra_lat']:>8.3f}  "
              f"{c['cerra_lon']:>8.3f}  {c['total_cap_mw']:>10.1f}  "
              f"{', '.join(c['farms'])}")

    # ── match CERRA cells → forecast grid (done once) ─────────────────────────
    first_dir   = next(iter(FORECAST_DIRS.values()))
    sample_file = sorted(first_dir.glob("forecast_*.nc"))[0]
    ds_sample   = xr.open_dataset(sample_file)
    fc_lat = ds_sample["latitude"].values.astype(float)
    fc_lon = ds_sample["longitude"].values.astype(float)
    ds_sample.close()

    match_to_forecast(cells, fc_lat, fc_lon)

    # ── collect file maps per directory ───────────────────────────────────────
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

    # ── intersect init times across all directories (fair comparison) ─────────
    common_inits = sorted(
        set.intersection(*(set(m) for m in dir_file_maps.values()))
    )
    print(f"\nCommon init times: {len(common_inits)}")

    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v"]

    fname_suffix = (
        "belgium" if REGIONS == ["Belgium"]
        else ("_".join(REGIONS).lower() if REGIONS else "all")
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (label, fmap) in enumerate(dir_file_maps.items()):
        lead_errors = {lh: [] for lh in LEAD_HOURS}

        for init in common_inits:
            ds_fc    = xr.open_dataset(fmap[init])
            fc_times = pd.to_datetime(ds_fc["time"].values).tz_localize("UTC")

            for lh in LEAD_HOURS:
                valid = init + pd.Timedelta(hours=lh)
                if valid not in fc_times or valid not in cerra_dates:
                    continue

                # forecast power [per unit, shape (values,)]
                fc_power = ds_fc["power"].sel(
                    time=valid.replace(tzinfo=None)
                ).values

                # observed power [per unit, shape (values,)]
                obs_power = ds_cerra["data"].isel(
                    time=np.where(cerra_dates == valid)[0][0],
                    variable=power_idx,
                    ensemble=0,
                ).values

                # capacity-weighted aggregate [MW] over all matched cells
                fc_mw  = sum(fc_power[c["fc_value_idx"]]  * c["total_cap_mw"] for c in cells)
                obs_mw = sum(obs_power[c["fc_value_idx"]] * c["total_cap_mw"] for c in cells)

                lead_errors[lh].append(abs(fc_mw - obs_mw))

            ds_fc.close()

        leads    = sorted(lead_errors)
        mean_mae = [
            np.mean(lead_errors[lh]) / grand_total_cap * 100
            if lead_errors[lh] else np.nan
            for lh in leads
        ]

        df = pd.DataFrame({"lead_hours": leads, "MAE_pct": mean_mae})
        print(f"\n{label} — power ({region_label})\n{df.to_string(index=False)}")

        ax.plot(
            df["lead_hours"], df["MAE_pct"],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            lw=1.5, label=label,
        )

        np.save(OUT_DIR / f"mae_power_{label}_{fname_suffix}.npy",
                df[["lead_hours", "MAE_pct"]].values)

    ax.set_title(
        f"Power MAE vs Lead Time  ({region_label})  (n={len(common_inits)} inits)\n"
        f"Total capacity: {grand_total_cap:.0f} MW  |  {len(cells)} grid cells",
        fontsize=12,
    )
    ax.set_xlabel("Lead time [hours]")
    ax.set_ylabel("MAE  [% of total capacity]")
    ax.set_xticks(LEAD_HOURS)
    ax.legend(title="Run", framealpha=0.8)
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"mae_power_{fname_suffix}.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / f'mae_power_{fname_suffix}.png'}")
    print("Done.")


if __name__ == "__main__":
    main()