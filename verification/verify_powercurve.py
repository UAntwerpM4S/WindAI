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

import h5py
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FORECAST_DIRS: List[Path] = [
     Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerTFRollout"),
      Path("/mnt/weatherloss/WindPower/inference/EGU/SyntheticTF"),
#    Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGTRollout"),
    #Path("/mnt/weatherloss/WindPower/inference/EGU/BigTransformer"),
    Path("/mnt/weatherloss/WindPower/inference/EGU/BigTransformerRollout"),
    #Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerMAE"),
]

LABELS: Dict[str, str] = {
    "VanillaPower" : "Normal GT",
    "NoPowerGTRollout" :             "BigTransformer (no power)",
    "VanillaPowerGTRollout": "GraphTransformer (vanilla power)",
    "SyntheticTF": "BigTransformer (Vanilla + Synthetic power)",
    "BigTransformer" : "Normal TF",
    "BigTransformerRollout": "BigTransformer (Vanilla power)",
}

CERRA_PATH    = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")
COUNTS_PATH   = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/wind_farm_turbine_counts.csv")
SPECS_PATH    = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/turbine_specs.csv")
METADATA_PATH = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")
PLOT_DIR      = Path("EGU_large")

INIT_START  = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END    = pd.Timestamp("2024-10-31 18:00:00", tz="UTC")
LEAD_HOURS  = list(range(3, 40, 3))
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


# def power_curve(ws: np.ndarray, spec: TurbineSpec) -> np.ndarray:
#     ws = np.asarray(ws, dtype=float)
#     out = np.zeros_like(ws, dtype=np.float32)
#     ramp = (ws >= spec.cut_in) & (ws < spec.rated_ws)
#     out[ramp] = spec.rated_power * ((ws[ramp] - spec.cut_in) / (spec.rated_ws - spec.cut_in)) ** 3
#     out[(ws >= spec.rated_ws) & (ws < spec.cut_out)] = spec.rated_power
#     return out
def power_curve(ws: np.ndarray, spec: TurbineSpec) -> np.ndarray:
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws, dtype=np.float32)
    ramp = (ws >= spec.cut_in) & (ws < spec.rated_ws)
    #out[ramp] = spec.rated_power * ((ws[ramp] - spec.cut_in) / (spec.rated_ws - spec.cut_in)) ** 3
    denom = (spec.rated_ws**3 - spec.cut_in**3)
    a = 1 / denom
    b = spec.cut_in**3 / denom
    out[ramp] = spec.rated_power * (a * ws[ramp]**3 - b) 
    rated = (ws >= spec.rated_ws) & (ws < spec.cut_out)
    out[rated] = spec.rated_power
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


def read_forecast_file(
    nc_path: Path,
    init: pd.Timestamp,
    fc_indices: np.ndarray,
    valid_iso_set: set,
    *,
    fc_var: str,
    type_order: List[str] | None = None,
    specs: Dict[str, TurbineSpec] | None = None,
    counts_matrix: np.ndarray | None = None,
) -> dict:
    """Return {lead_hour: total_mw} for one forecast file using h5py."""
    result = {}
    with h5py.File(str(nc_path), "r") as f:
        if fc_var == "power" and "power" not in f:
            return result
        tv  = f["time"]
        raw = nc4.num2date(
            tv[:],
            tv.attrs["units"].decode(),
            tv.attrs.get("calendar", b"standard").decode(),
        )
        fc_time_to_idx = {
            pd.Timestamp(str(t)).tz_localize("UTC").isoformat(): j
            for j, t in enumerate(raw)
        }
        if fc_var == "power":
            data_all = f["power"][:, :]
        else:
            ws_all   = f["ws100"][:, :]
            data_all = np.zeros((ws_all.shape[0], len(fc_indices)), dtype=np.float32)
            for j, tname in enumerate(type_order):
                data_all += power_curve(ws_all[:, fc_indices], specs[tname]) * counts_matrix[:, j]

    data_sel = data_all[:, fc_indices] if fc_var == "power" else data_all

    for lh in LEAD_HOURS:
        viso = (init + pd.Timedelta(hours=lh)).isoformat()
        if viso in fc_time_to_idx and viso in valid_iso_set:
            result[lh] = float(np.sum(data_sel[fc_time_to_idx[viso]]))

    return result


def collect_mae_vs_cerra(
    files: List[Path],
    fc_var: str,
    fc_indices: np.ndarray,
    cerra_obs_cache: dict,
    valid_iso_set: set,
    *,
    type_order: List[str] | None = None,
    specs: Dict[str, TurbineSpec] | None = None,
    counts_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    lead_errors  = {lh: [] for lh in LEAD_HOURS}
    lead_sq_errs = {lh: [] for lh in LEAD_HOURS}
    lead_biases  = {lh: [] for lh in LEAD_HOURS}

    for count, fpath in enumerate(files):
        if count % 500 == 0:
            print(f"    {count}/{len(files)}", flush=True)
        init = parse_init_time(fpath)
        try:
            fc = read_forecast_file(
                fpath, init, fc_indices, valid_iso_set,
                fc_var=fc_var, type_order=type_order,
                specs=specs, counts_matrix=counts_matrix,
            )
            for lh, fc_mw in fc.items():
                if np.isnan(fc_mw):
                    continue
                viso = (init + pd.Timedelta(hours=lh)).isoformat()
                obs_mw = float(np.sum(cerra_obs_cache[viso]))
                err = fc_mw - obs_mw
                lead_errors[lh].append(abs(err))
                lead_sq_errs[lh].append(err ** 2)
                lead_biases[lh].append(err)
        except Exception as e:
            print(f"    Skipped {fpath.name}: {e}")

    leads    = sorted(lead_errors)
    mean_mae = [np.mean(lead_errors[lh])                   if lead_errors[lh]  else np.nan for lh in leads]
    rmse     = [np.sqrt(np.mean(lead_sq_errs[lh]))         if lead_sq_errs[lh] else np.nan for lh in leads]
    bias     = [np.mean(lead_biases[lh])                   if lead_biases[lh]  else np.nan for lh in leads]
    counts   = [len(lead_errors[lh]) for lh in leads]
    return pd.DataFrame({"lead_hours": leads, "MAE_MW": mean_mae, "RMSE_MW": rmse, "Bias_MW": bias, "n_inits": counts})


# ------------------------- Plotting ---------------------------------------


def plot_metrics(
    results: List[Tuple[str, pd.DataFrame]],
    out_path_side: Path,
    out_path_bias: Path,
    mae_col: str,
    rmse_col: str,
    bias_col: str,
    y_label: str,
    n_inits: int,
) -> None:
    color_palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
    base_colors: Dict[str, str] = {}
    color_idx = 0

    # --- Figure 1: MAE + RMSE side by side ---
    fig1, (ax_mae, ax_rmse) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig1.suptitle(f"Belgian offshore  (n={n_inits} inits)", fontsize=12)

    # --- Figure 2: Bias alone ---
    fig2, ax_bias = plt.subplots(1, 1, figsize=(7, 5))
    fig2.suptitle(f"Belgian offshore  (n={n_inits} inits)", fontsize=12)

    for label, df in results:
        base_label = label.split("-")[0]
        if base_label not in base_colors:
            base_colors[base_label] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

        is_powercurve = "powercurve" in label.lower()
        color  = base_colors[base_label]
        ls     = "--" if is_powercurve else "-"
        marker = ""   if is_powercurve else "o"

        kwargs = dict(lw=1.8, ls=ls, marker=marker, markersize=4, color=color, label=label)
        ax_mae.plot(df["lead_hours"],  df[mae_col],  **kwargs)
        ax_rmse.plot(df["lead_hours"], df[rmse_col], **kwargs)
        ax_bias.plot(df["lead_hours"], df[bias_col], **kwargs)

    for ax, title in zip([ax_mae, ax_rmse], ["MAE", "RMSE"]):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Lead time [hours]")
        ax.set_ylabel(y_label)
        ax.legend(framealpha=0.8, fontsize=8)
        ax.grid(True, ls="--", alpha=0.5)

    ax_bias.set_title("Bias (fc − obs)", fontsize=11)
    ax_bias.set_xlabel("Lead time [hours]")
    ax_bias.set_ylabel(y_label)
    ax_bias.axhline(0, color="black", lw=0.8, ls=":")
    ax_bias.legend(framealpha=0.8, fontsize=8)
    ax_bias.grid(True, ls="--", alpha=0.5)

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig(out_path_side, dpi=200)
    plt.close(fig1)
    print(f"Saved: {out_path_side}")

    fig2.savefig(out_path_bias, dpi=200)
    plt.close(fig2)
    print(f"Saved: {out_path_bias}")


# ------------------------- Main -------------------------------------------


def main() -> None:
    # --- CERRA setup ---
    ds_cerra      = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars    = list(ds_cerra.attrs["variables"])
    cerra_lat     = ds_cerra["latitudes"].values
    cerra_lon     = ds_cerra["longitudes"].values
    cerra_dates   = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")
    cerra_date_to_idx = {d: i for i, d in enumerate(cerra_dates)}

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
    common_inits = [parse_init_time(f) for f in next(iter(common_files.values()))]

    # --- fc_indices: grids are identical so fc_indices == cerra_keep ---
    fc_indices = cerra_keep

    # --- preload CERRA obs for all needed valid times ---
    needed_valid = sorted({
        init + pd.Timedelta(hours=lh)
        for init in common_inits for lh in LEAD_HOURS
        if (init + pd.Timedelta(hours=lh)) in cerra_date_to_idx
    })
    print(f"Preloading CERRA ({len(needed_valid)} timesteps × {len(cerra_keep)} cells)...")
    cerra_bulk = ds_cerra["data"].isel(
        time=[cerra_date_to_idx[t] for t in needed_valid],
        variable=cerra_var_idx, ensemble=0,
    ).values[:, cerra_keep]
    cerra_obs_cache = {t.isoformat(): cerra_bulk[i] for i, t in enumerate(needed_valid)}
    valid_iso_set   = {iso for iso, arr in cerra_obs_cache.items() if not np.any(np.isnan(arr))}
    ds_cerra.close()
    del cerra_bulk
    print("CERRA preload done.")

    # --- compute MAE per directory ---
    mae_results_mw:  List[Tuple[str, pd.DataFrame]] = []
    mae_results_pct: List[Tuple[str, pd.DataFrame]] = []

    for fc_dir in FORECAST_DIRS:
        label = fc_dir.name
        files = common_files[fc_dir]
        print(f"\nProcessing {label} ...")

        with h5py.File(str(files[0]), "r") as f:
            has_power = "power" in f

        label = LABELS.get(fc_dir.name, fc_dir.name)
        if has_power:
            print(f"  power MAE/RMSE...")
            df_model = collect_mae_vs_cerra(
                files, "power", fc_indices, cerra_obs_cache, valid_iso_set,
            )
            df_model["MAE_pct"]  = df_model["MAE_MW"]  / total_cap_mw * 100.0
            df_model["RMSE_pct"] = df_model["RMSE_MW"] / total_cap_mw * 100.0
            df_model["Bias_pct"] = df_model["Bias_MW"]  / total_cap_mw * 100.0
        
            mae_results_mw.append((label, df_model))   # ← ADD THIS
            mae_results_pct.append((label, df_model))  # ← AND THIS
        else:
            print(f"  No 'power' variable — skipping.")

        print(f"  power-curve MAE/RMSE...")
        df_pc = collect_mae_vs_cerra(
            files, "ws100", fc_indices, cerra_obs_cache, valid_iso_set,
            type_order=type_order, specs=specs, counts_matrix=counts_matrix,
        )
        df_pc["MAE_pct"]  = df_pc["MAE_MW"]  / total_cap_mw * 100.0
        df_pc["RMSE_pct"] = df_pc["RMSE_MW"] / total_cap_mw * 100.0
        df_pc["Bias_pct"] = df_pc["Bias_MW"] / total_cap_mw * 100.0
        mae_results_mw.append((f"{label}-powercurve",  df_pc))
        mae_results_pct.append((f"{label}-powercurve", df_pc))

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    plot_metrics(mae_results_mw,  PLOT_DIR / f"metrics_MW_{ts}.png",
                 PLOT_DIR / f"bias_MW_{ts}.png",
                 mae_col="MAE_MW",  rmse_col="RMSE_MW",  bias_col="Bias_MW",
                 y_label="[MW]", n_inits=n_inits)
    plot_metrics(mae_results_pct, PLOT_DIR / f"metrics_pct_{ts}.png",
                 PLOT_DIR / f"bias_pct_{ts}.png",
                 mae_col="MAE_pct", rmse_col="RMSE_pct", bias_col="Bias_pct",
                 y_label=f"[% of {total_cap_mw:.0f} MW]", n_inits=n_inits)

    print("\nDone.")


if __name__ == "__main__":
    main()