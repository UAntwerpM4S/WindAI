"""
MAE vs lead time for wind power, verified against CERRA truth.
- Model 'power' variable vs CERRA power (solid line, if available)
- Power-curve derived from ws100 vs CERRA power (dotted line, always)
Only Belgian offshore wind farm cells are used.
MAE reported in MW and as % of total installed capacity.
Only init times present in ALL directories are used.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FORECAST_DIRS: List[Path] = [
  #  Path("/mnt/weatherloss/WindPower/inference/EGU/NoPower"),
    Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPower"),
    Path("/mnt/weatherloss/WindPower/inference/EGU/Finetune"),
    Path("/mnt/weatherloss/WindPower/inference/EGU/TinyPower"),
]

CERRA_PATH    = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/Cerra_A.zarr")
COUNTS_PATH   = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/wind_farm_turbine_counts.csv")
SPECS_PATH    = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/turbine_specs.csv")
METADATA_PATH = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")
PLOT_DIR      = Path("/mnt/weatherloss/WindPower/verification/EGU_plots")

INIT_START  = pd.Timestamp("2024-08-03 00:00:00", tz="UTC")
INIT_END    = pd.Timestamp("2024-10-31 21:00:00", tz="UTC")
LEAD_HOURS  = list(range(3, 73, 3))
MAX_DIST_KM = 2.0

# ---------------------------------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init_time(path: Path) -> pd.Timestamp:
    match = FORECAST_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse init time from: {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S", utc=True)


# ------------------------- File collection ---------------------------------


def get_common_files(dirs: List[Path]) -> Dict[Path, List[Path]]:
    dir_maps: Dict[Path, Dict[pd.Timestamp, Path]] = {}
    for fc_dir in dirs:
        files = sorted(fc_dir.glob("forecast_*.nc"))
        if not files:
            raise FileNotFoundError(f"No forecast_*.nc files in {fc_dir}")
        time_map = {parse_init_time(f): f for f in files}
        time_map = {t: f for t, f in time_map.items() if INIT_START <= t <= INIT_END}
        if not time_map:
            raise ValueError(f"No files in {fc_dir.name} within date window.")
        dir_maps[fc_dir] = time_map

    common_inits = sorted(set.intersection(*(set(m.keys()) for m in dir_maps.values())))
    print(f"\n--- Init time intersection across {len(dirs)} directories ---")
    for fc_dir, fmap in dir_maps.items():
        n_dropped = len(fmap) - len(common_inits)
        print(f"  {fc_dir.name}: {len(fmap)} total, {n_dropped} dropped → {len(common_inits)} used")

    return {fc_dir: [fmap[t] for t in common_inits] for fc_dir, fmap in dir_maps.items()}


# ------------------------- Metadata / capacity -----------------------------


def load_belgian_metadata() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH)
    be = meta[meta["region"].str.lower() == "belgium"]
    print(f"  Belgian farms: {len(be)}, "
          f"unique CERRA cells: {len(be.drop_duplicates(subset=['cerra_grid_lat', 'cerra_grid_lon']))}")
    return be


def get_total_capacity_mw(meta: pd.DataFrame) -> float:
    cap = float(meta["capacity_mw"].sum())
    print(f"  Total Belgian capacity: {cap:.1f} MW")
    return cap


# ------------------------- CERRA setup ------------------------------------


def to_xy(lon, lat):
    R = 6371.0
    lat0 = np.deg2rad(np.mean(lat))
    return np.column_stack([
        R * np.deg2rad(lon) * np.cos(lat0),
        R * np.deg2rad(lat)
    ])


def get_belgian_cerra_indices(meta: pd.DataFrame, cerra_lat: np.ndarray, cerra_lon: np.ndarray) -> np.ndarray:
    """Return flat indices into CERRA array for unique Belgian turbine cells."""
    be_unique = meta.drop_duplicates(subset=["cerra_grid_lat", "cerra_grid_lon"])
    cerra_keys = {(round(la, 6), round(lo, 6)): i
                  for i, (la, lo) in enumerate(zip(cerra_lat, cerra_lon))}
    indices = []
    for _, row in be_unique.iterrows():
        key = (round(row["cerra_grid_lat"], 6), round(row["cerra_grid_lon"], 6))
        if key not in cerra_keys:
            raise ValueError(f"CERRA cell not found for {row['farm']}: {key}")
        indices.append(cerra_keys[key])
    return np.array(indices, dtype=int)


def build_fc_indices(fc_lat, fc_lon, cerra_lat, cerra_lon, cerra_keep) -> np.ndarray:
    """For each Belgian CERRA cell, find the nearest forecast grid point."""
    tree = cKDTree(to_xy(fc_lon, fc_lat))
    dist, fc_idx = tree.query(to_xy(cerra_lon[cerra_keep], cerra_lat[cerra_keep]), k=1)
    if dist.max() > MAX_DIST_KM:
        raise ValueError(f"Max NN distance {dist.max():.4f} km exceeds {MAX_DIST_KM} km.")
    print(f"  Max NN distance: {dist.max():.4f} km")
    return fc_idx


# ------------------------- Power-curve ------------------------------------


@dataclass(frozen=True)
class TurbineSpec:
    cut_in: float
    rated_ws: float
    cut_out: float
    rated_power: float


def power_curve(ws: np.ndarray, spec: TurbineSpec) -> np.ndarray:
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws, dtype=np.float32)
    ramp = (ws >= spec.cut_in) & (ws < spec.rated_ws)
    out[ramp] = spec.rated_power * ((ws[ramp] - spec.cut_in) / (spec.rated_ws - spec.cut_in)) ** 3
    out[(ws >= spec.rated_ws) & (ws < spec.cut_out)] = spec.rated_power
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


def build_counts_matrix(meta: pd.DataFrame) -> Tuple[List[str], np.ndarray, List[Tuple[int, int]]]:
    counts = pd.read_csv(COUNTS_PATH).set_index("farm")
    type_cols = [c for c in counts.columns if c.lower() != "total"]

    # unique cells in same order as cerra_keep
    seen = []
    for _, row in meta.iterrows():
        cell = (int(row["cerra_y"]), int(row["cerra_x"]))
        if cell not in seen:
            seen.append(cell)
    cells = seen

    cell_to_type_counts: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in meta.iterrows():
        farm = row["farm"]
        cell = (int(row["cerra_y"]), int(row["cerra_x"]))
        if farm not in counts.index:
            raise ValueError(f"Farm '{farm}' not in counts file.")
        for tcol in type_cols:
            n = float(counts.at[farm, tcol])
            if n > 0:
                cell_to_type_counts[cell][tcol] += n

    counts_matrix = np.zeros((len(cells), len(type_cols)), dtype=np.float32)
    for ci, cell in enumerate(cells):
        for tj, tcol in enumerate(type_cols):
            counts_matrix[ci, tj] = cell_to_type_counts[cell].get(tcol, 0.0)

    return type_cols, counts_matrix, cells


# ------------------------- MAE vs CERRA -----------------------------------


def collect_mae_vs_cerra(
    files: List[Path],
    fc_var: str,
    fc_indices: np.ndarray,
    cerra_keep: np.ndarray,
    cerra_dates: pd.DatetimeIndex,
    ds_cerra: xr.Dataset,
    cerra_var_idx: int,
    *,
    type_order: List[str] | None = None,
    specs: Dict[str, TurbineSpec] | None = None,
    counts_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Collect per-lead MAE (summed over Belgian cells) vs CERRA power.
    fc_var: 'power' for model power, 'ws100' for power-curve path.
    """
    cerra_date_index = {t: i for i, t in enumerate(cerra_dates)}
    lead_errors = {lh: [] for lh in LEAD_HOURS}

    for fpath in files:
        init = parse_init_time(fpath)
        with xr.open_dataset(fpath) as ds_fc:
            if fc_var == "power" and "power" not in ds_fc:
                continue  # skip files missing power variable
            fc_times = pd.to_datetime(ds_fc["time"].values).tz_localize("UTC")
            if fc_var == "power":
                fc_all = ds_fc["power"].values[:, fc_indices]   # (time, 7)
            else:
                # power-curve: read ws100, apply curve → (time, 7)
                ws_all = ds_fc["ws100"].values[:, fc_indices]   # (time, 7)
                fc_all = np.zeros_like(ws_all, dtype=np.float32)
                for j, tname in enumerate(type_order):
                    fc_all += power_curve(ws_all, specs[tname]) * counts_matrix[:, j]

        for lh in LEAD_HOURS:
            valid = init + pd.Timedelta(hours=lh)
            t_fc = np.where(fc_times == valid)[0]
            t_tr = cerra_date_index.get(valid)
            if len(t_fc) == 0 or t_tr is None:
                continue

            fc_total = float(np.nansum(fc_all[t_fc[0]]))
            tr_vals  = ds_cerra["data"].isel(
                time=t_tr, variable=cerra_var_idx, ensemble=0
            ).values[cerra_keep]
            tr_total = float(np.nansum(tr_vals))

            lead_errors[lh].append(abs(fc_total - tr_total))

    leads    = sorted(lead_errors)
    mean_mae = [np.mean(lead_errors[lh]) if lead_errors[lh] else np.nan for lh in leads]
    counts   = [len(lead_errors[lh]) for lh in leads]
    return pd.DataFrame({"lead_hours": leads, "MAE_MW": mean_mae, "n_inits": counts})


# ------------------------- Plotting ---------------------------------------


def plot_mae(
    results: List[Tuple[str, pd.DataFrame]],
    out_path: Path,
    y_label: str,
    mae_col: str,
    n_inits: int,
) -> None:
    color_palette = ["green", "blue", "black", "red"]
    base_colors: Dict[str, str] = {}
    color_idx = 0

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, df in results:
        base_label = label.split("-")[0]
        if base_label not in base_colors:
            base_colors[base_label] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

        is_powercurve = "powercurve" in label.lower()
        ax.plot(
            df["lead_hours"], df[mae_col],
            lw=1.8,
            ls="--" if is_powercurve else "-",
            marker="" if is_powercurve else "o",
            markersize=4,
            label=label,
            color=base_colors[base_label],
        )

    ax.set_title(f"MAE vs Lead Time — Belgian offshore  (n={n_inits} inits)", fontsize=12)
    ax.set_xlabel("Lead time [hours]")
    ax.set_ylabel(y_label)
    ax.legend(framealpha=0.8)
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ------------------------- Main -------------------------------------------


def main() -> None:
    # --- CERRA setup ---
    ds_cerra      = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars    = list(ds_cerra.attrs["variables"])
    cerra_lat     = ds_cerra["latitudes"].values
    cerra_lon     = ds_cerra["longitudes"].values
    cerra_dates   = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")

    if "power" not in cerra_vars:
        raise ValueError(f"'power' not in CERRA. Available: {cerra_vars}")
    cerra_var_idx = cerra_vars.index("power")

    # --- metadata ---
    meta             = load_belgian_metadata()
    total_cap_mw     = get_total_capacity_mw(meta)
    cerra_keep       = get_belgian_cerra_indices(meta, cerra_lat, cerra_lon)
    specs            = load_specs(SPECS_PATH)
    type_order, counts_matrix, _ = build_counts_matrix(meta)

    # --- files ---
    common_files = get_common_files(FORECAST_DIRS)
    n_inits      = len(next(iter(common_files.values())))

    # --- build NN mapping once (shared grid) ---
    any_file   = common_files[FORECAST_DIRS[0]][0]
    with xr.open_dataset(any_file) as ds0:
        fc_indices = build_fc_indices(
            ds0["latitude"].values, ds0["longitude"].values,
            cerra_lat, cerra_lon, cerra_keep,
        )

    # --- compute MAE per directory ---
    mae_results_mw:  List[Tuple[str, pd.DataFrame]] = []
    mae_results_pct: List[Tuple[str, pd.DataFrame]] = []

    for fc_dir in FORECAST_DIRS:
        label = fc_dir.name
        files = common_files[fc_dir]
        print(f"\nProcessing {label} ...")

        has_power = any("power" in xr.open_dataset(f).data_vars for f in files[:5])

        if has_power:
            print(f"  Computing model power MAE ...")
            df_model = collect_mae_vs_cerra(
                files, "power", fc_indices, cerra_keep,
                cerra_dates, ds_cerra, cerra_var_idx,
            )
            df_model["MAE_pct"] = df_model["MAE_MW"] / total_cap_mw * 100.0
            mae_results_mw.append((f"{label}-power",      df_model))
            mae_results_pct.append((f"{label}-power",     df_model))
        else:
            print(f"  No 'power' variable — skipping model power curve.")

        print(f"  Computing power-curve MAE ...")
        df_pc = collect_mae_vs_cerra(
            files, "ws100", fc_indices, cerra_keep,
            cerra_dates, ds_cerra, cerra_var_idx,
            type_order=type_order, specs=specs, counts_matrix=counts_matrix,
        )
        df_pc["MAE_pct"] = df_pc["MAE_MW"] / total_cap_mw * 100.0
        mae_results_mw.append((f"{label}-powercurve",  df_pc))
        mae_results_pct.append((f"{label}-powercurve", df_pc))

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    plot_mae(mae_results_mw,  PLOT_DIR / f"mae_power_MW_{ts}.png",
             y_label="MAE [MW]", mae_col="MAE_MW", n_inits=n_inits)
    plot_mae(mae_results_pct, PLOT_DIR / f"mae_power_pct_{ts}.png",
             y_label=f"MAE [% of {total_cap_mw:.0f} MW]", mae_col="MAE_pct", n_inits=n_inits)

    print("\nDone.")


if __name__ == "__main__":
    main()