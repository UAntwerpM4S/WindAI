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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERIFY_BELGIUM_ONLY = True

FORECAST_DIRS: List[Path] = [
    Path("/mnt/weatherloss/WindPower/inference/EGU/NoPower2"),
    Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPower"),
    Path("/mnt/weatherloss/WindPower/inference/EGU/LowPower"),
    Path("/mnt/weatherloss/WindPower/inference/EGU/TinyPower"),
]



OBS_BE_TOTAL_PATH = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/BE_UK_offshore_per_unit_3H_meanMW_shifted.csv")
OBS_FARM_DIR      = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power")
OBS_FARM_PATH: Path | None = None

COUNTS_PATH   = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/wind_farm_turbine_counts.csv")
SPECS_PATH    = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/turbine_specs.csv")
METADATA_PATH = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

PLOT_DIR = Path("/mnt/weatherloss/WindPower/verification/EGU_plots")

INIT_START = pd.Timestamp("2024-08-03 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2024-10-31 21:00:00", tz="UTC")

LEAD_MIN = 3
LEAD_MAX = 72

DECIMALS = 4
TOL_DEGREES: float | None = None

# ---------------------------------------------------------------------------


def parse_init_time(path: Path) -> pd.Timestamp:
    match = re.search(r"forecast_(\d{14})", path.name)
    if not match:
        raise ValueError(f"Cannot parse init time from filename: {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S", utc=True)


def get_common_files(dirs: List[Path]) -> Dict[Path, List[Path]]:
    dir_maps: Dict[Path, Dict[pd.Timestamp, Path]] = {}
    for fc_dir in dirs:
        files = sorted(fc_dir.glob("forecast_*.nc"))
        if not files:
            raise FileNotFoundError(f"No forecast_*.nc files found in {fc_dir}")
        # ── filter by init date window ──────────────────────────────────────
        time_map = {parse_init_time(f): f for f in files}
        time_map = {
            t: f for t, f in time_map.items()
            if INIT_START <= t <= INIT_END
        }
        if not time_map:
            raise ValueError(
                f"No files in {fc_dir.name} fall within "
                f"[{INIT_START.date()}, {INIT_END.date()}]."
            )
        dir_maps[fc_dir] = time_map

    common_inits = sorted(
        set.intersection(*(set(m.keys()) for m in dir_maps.values()))
    )

    print(f"\n--- Init time intersection across {len(dirs)} directories ---")
    for fc_dir, fmap in dir_maps.items():
        n_dropped = len(fmap) - len(common_inits)
        print(f"  {fc_dir.name}: {len(fmap)} total, {n_dropped} dropped → {len(common_inits)} used")

    return {fc_dir: [fmap[t] for t in common_inits] for fc_dir, fmap in dir_maps.items()}


def _scope_filtered_metadata() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH)
    if VERIFY_BELGIUM_ONLY:
        meta = meta[meta["region"].astype(str).str.lower() == "belgium"]
    return meta


def get_total_capacity_mw() -> float:
    meta = _scope_filtered_metadata()
    if meta.empty:
        scope = "Belgium" if VERIFY_BELGIUM_ONLY else "all"
        raise ValueError(f"No farms found in metadata for scope={scope}.")
    cap = float(meta["capacity_mw"].astype(float).sum())
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError(f"Computed invalid total capacity from metadata: {cap}")
    return cap


# ------------------------- Observations loading -----------------------------


def _find_farm_obs_csv_by_header(directory: Path) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"OBS_FARM_DIR does not exist: {directory}")
    candidates = sorted(directory.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No .csv files found in {directory}")

    must_have_any = {"Belwind Phase 1", "Nobelwind Offshore Windpark", "Northwind", "Rentel Offshore WP"}
    for p in candidates:
        try:
            header = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue
        if len(set(map(str, header)).intersection(must_have_any)) >= 2:
            return p

    raise FileNotFoundError(
        f"Could not auto-detect farm obs CSV in {directory}. "
        f"Set OBS_FARM_PATH explicitly."
    )


def load_obs(time_range: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    tmin = time_range[0].tz_localize(None) if getattr(time_range[0], "tzinfo", None) else time_range[0]
    tmax = time_range[1].tz_localize(None) if getattr(time_range[1], "tzinfo", None) else time_range[1]

    farm_path = OBS_FARM_PATH or _find_farm_obs_csv_by_header(OBS_FARM_DIR)
    obs = pd.read_csv(farm_path)
    time_col = obs.columns[0]
    obs[time_col] = pd.to_datetime(obs[time_col], utc=True).dt.tz_localize(None)
    obs = obs[(obs[time_col] >= tmin) & (obs[time_col] <= tmax)].copy()

    meta = _scope_filtered_metadata()
    farm_names = set(meta["farm"].astype(str).tolist())
    available = [c for c in obs.columns[1:] if c in farm_names]
    if not available:
        raise ValueError(
            f"No matching farm columns found between metadata and obs file: {farm_path}. "
            f"Example farms from metadata: {list(farm_names)[:10]}"
        )

    farm_data = obs[available].apply(pd.to_numeric, errors="coerce")

    # At each timestep, sum only over farms that reported. But if the reporting
    # set varies (some farms have NaN gaps), summing a smaller fleet is not
    # comparable to summing the full fleet at other timesteps.
    # Solution: mark any timestep where at least one farm is NaN as obs_MW=NaN,
    # so compute_mae's dropna skips it. This drops only the affected timesteps,
    # not whole farms — a month-long NaN in farm X only masks that month.
    n_farms_per_row = farm_data.notna().sum(axis=1)
    expected_n = int(n_farms_per_row.max())  # full fleet size

    n_masked = int((n_farms_per_row < expected_n).sum())
    if n_masked > 0:
        print(
            f"  load_obs: {n_masked}/{len(farm_data)} timesteps have at least one "
            f"farm NaN — those timesteps will be excluded from MAE."
        )

    obs_sum = farm_data.sum(axis=1)
    obs_sum[n_farms_per_row < expected_n] = np.nan  # mask incomplete timesteps

    obs["obs_MW"] = obs_sum
    return obs[[time_col, "obs_MW"]].rename(columns={time_col: "time"})


# ------------------------- Metadata helpers -----------------------------


def build_turbine_cells_from_metadata() -> List[Tuple[int, int]]:
    meta = _scope_filtered_metadata()
    cells: List[Tuple[int, int]] = []
    for _, row in meta.iterrows():
        cell = (int(row["cerra_y"]), int(row["cerra_x"]))
        if cell not in cells:
            cells.append(cell)
    if not cells:
        raise ValueError(f"No farms/cells found in metadata.")
    return cells


def build_turbine_points_from_metadata() -> pd.DataFrame:
    meta = _scope_filtered_metadata()
    if meta.empty:
        raise ValueError("No farms found in metadata.")
    meta = meta.copy()
    if "cerra_grid_lat" in meta.columns and "cerra_grid_lon" in meta.columns:
        meta["lat_pt"] = meta["cerra_grid_lat"].astype(float)
        meta["lon_pt"] = meta["cerra_grid_lon"].astype(float)
    elif "lat_cell" in meta.columns and "lon_cell" in meta.columns:
        meta["lat_pt"] = meta["lat_cell"].astype(float)
        meta["lon_pt"] = meta["lon_cell"].astype(float)
    else:
        meta["lat_pt"] = meta["lat"].astype(float)
        meta["lon_pt"] = meta["lon"].astype(float)
    return meta.drop_duplicates(subset=["lat_pt", "lon_pt"])[["lat_pt", "lon_pt"]].reset_index(drop=True)


def build_value_indices_from_latlon(
    sample_forecast_path: Path,
    points: pd.DataFrame,
    *,
    decimals: int = 4,
    tol_degrees: float | None = None,
) -> np.ndarray:
    with xr.open_dataset(sample_forecast_path) as ds:
        if "latitude" not in ds or "longitude" not in ds:
            raise ValueError(f"{sample_forecast_path} missing latitude/longitude variables.")
        latv = ds["latitude"].values.astype(np.float64)
        lonv = ds["longitude"].values.astype(np.float64)
        values_dim = int(ds.sizes["values"])

    key = lambda la, lo: (round(float(la), decimals), round(float(lo), decimals))
    grid_map = {key(la, lo): i for i, (la, lo) in enumerate(zip(latv, lonv))}

    idx: List[int] = []
    bad_points: List[Tuple[float, float]] = []

    for la, lo in points[["lat_pt", "lon_pt"]].itertuples(index=False, name=None):
        vi = grid_map.get(key(la, lo))
        if vi is None:
            bad_points.append((float(la), float(lo)))
        else:
            idx.append(int(vi))

    if bad_points and tol_degrees is not None and tol_degrees > 0:
        remaining = []
        for la, lo in bad_points:
            dlat = latv - la
            dlon = lonv - lo
            dist2 = dlat * dlat + dlon * dlon
            j = int(np.argmin(dist2))
            if float(np.sqrt(dist2[j])) <= tol_degrees:
                idx.append(j)
            else:
                remaining.append((la, lo))
        bad_points = remaining

    if bad_points:
        raise ValueError(
            f"{len(bad_points)} metadata points did not match forecast grid. "
            f"First examples: {bad_points[:10]}. "
            f"Try DECIMALS=3 or set TOL_DEGREES (e.g. 5e-4)."
        )

    idx_arr = np.array(idx, dtype=int)
    if idx_arr.size and int(idx_arr.max()) >= values_dim:
        raise ValueError(f"Mapped turbine indices out of bounds. max_idx={int(idx_arr.max())}, values_dim={values_dim}.")
    if len(np.unique(idx_arr)) < len(idx_arr):
        print("Warning: multiple metadata points mapped to the same forecast 'values' index.")
    return idx_arr


# ------------------------- Forecast reading -----------------------------


def dir_has_power(files: List[Path]) -> bool:
    with xr.open_dataset(files[0]) as ds:
        return "power" in ds


def _check_indices_in_bounds(path: Path, values_dim: int, turbine_value_indices: np.ndarray) -> None:
    if turbine_value_indices.size == 0:
        raise ValueError("turbine_value_indices is empty.")
    max_idx = int(np.max(turbine_value_indices))
    min_idx = int(np.min(turbine_value_indices))
    if min_idx < 0 or max_idx >= values_dim:
        raise ValueError(
            f"turbine_value_indices out of bounds for {path}. "
            f"min={min_idx}, max={max_idx}, values_dim={values_dim}."
        )


def load_forecast_power(path: Path, turbine_value_indices: np.ndarray) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        values_dim = int(ds.sizes["values"])
        _check_indices_in_bounds(path, values_dim, turbine_value_indices)
        if "power" not in ds:
            raise KeyError(f"Dataset missing 'power' variable: {path}")
        power = ds["power"].isel(values=turbine_value_indices).sum(dim="values")
        out = power.to_series().rename("fcst_MW").reset_index()

    init_time = parse_init_time(path)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["init_time"] = init_time
    out["lead_hours"] = ((out["time"] - init_time) / np.timedelta64(1, "h")).astype(int)
    out["time"] = out["time"].dt.tz_localize(None)
    return out[(out["lead_hours"] >= LEAD_MIN) & (out["lead_hours"] <= LEAD_MAX)]


def load_dir_forecasts(files: List[Path], turbine_value_indices: np.ndarray) -> pd.DataFrame:
    frames = []
    for p in files:
        try:
            frames.append(load_forecast_power(p, turbine_value_indices))
        except KeyError as exc:
            print(f"  Skipping {p.name}: {exc}")
    if not frames:
        raise RuntimeError("No usable forecasts found.")
    return pd.concat(frames, ignore_index=True)


# ------------------------- Power-curve path -------------------------------


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


def build_counts_matrix(cells: Sequence[Tuple[int, int]]) -> Tuple[List[str], np.ndarray]:
    counts = pd.read_csv(COUNTS_PATH).set_index("farm")
    type_cols = [c for c in counts.columns if c.lower() not in {"total"}]
    meta = _scope_filtered_metadata()

    cell_to_type_counts: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in meta.iterrows():
        farm = row["farm"]
        cell = (int(row["cerra_y"]), int(row["cerra_x"]))
        if farm not in counts.index:
            raise ValueError(f"Farm '{farm}' not found in counts file.")
        for tcol in type_cols:
            n = float(counts.at[farm, tcol])
            if n > 0:
                cell_to_type_counts[cell][tcol] += n

    counts_matrix = np.zeros((len(cells), len(type_cols)), dtype=np.float32)
    for ci, cell in enumerate(cells):
        for tj, tcol in enumerate(type_cols):
            counts_matrix[ci, tj] = cell_to_type_counts[cell].get(tcol, 0.0)

    return type_cols, counts_matrix


def load_forecast_powercurve(
    path: Path,
    cell_indices: np.ndarray,
    type_order: Sequence[str],
    specs: Dict[str, TurbineSpec],
    counts_matrix: np.ndarray,
) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        values_dim = int(ds.sizes["values"])
        _check_indices_in_bounds(path, values_dim, cell_indices)
        ws = ds["ws100"].isel(values=cell_indices).to_numpy()

        total = np.zeros(ws.shape[0], dtype=np.float32)
        for j, tname in enumerate(type_order):
            if tname not in specs:
                raise ValueError(f"Turbine type '{tname}' missing from specs.")
            total += np.sum(power_curve(ws, specs[tname]) * counts_matrix[:, j], axis=1)

        init_time = parse_init_time(path)
        valid_time_utc = pd.to_datetime(ds["time"].values).tz_localize("UTC")
        lead_hours = ((valid_time_utc - init_time) / np.timedelta64(1, "h")).astype(int)

    return pd.DataFrame({
        "time": valid_time_utc.tz_localize(None),
        "init_time": init_time.tz_localize(None),
        "lead_hours": lead_hours,
        "pc_MW": total,
    })[(lead_hours >= LEAD_MIN) & (lead_hours <= LEAD_MAX)]


def load_dir_powercurve(
    files: List[Path],
    cell_indices: np.ndarray,
    type_order: Sequence[str],
    specs: Dict[str, TurbineSpec],
    counts_matrix: np.ndarray,
) -> pd.DataFrame:
    frames = [load_forecast_powercurve(p, cell_indices, type_order, specs, counts_matrix) for p in files]
    return pd.concat(frames, ignore_index=True)


# ------------------------- Metrics + plotting -----------------------------


def compute_mae(
    fcst: pd.DataFrame,
    obs: pd.DataFrame,
    value_col: str,
    *,
    as_percent: bool = False,
    capacity_mw: float | None = None,
) -> pd.DataFrame:
    if as_percent and (capacity_mw is None or capacity_mw <= 0):
        raise ValueError("capacity_mw must be positive when as_percent=True.")
    merged = fcst.merge(obs, on="time", how="inner").dropna(subset=[value_col, "obs_MW"])
    if merged.empty:
        raise ValueError("No overlapping forecast/obs times after merge.")
    err = merged[value_col] - merged["obs_MW"]
    if as_percent:
        err = err / capacity_mw * 100.0
    merged["err"] = err
    return (
        merged.groupby("lead_hours")
        .agg(count=("err", "size"), MAE=("err", lambda s: s.abs().mean()))
        .reset_index()
        .sort_values("lead_hours")
    )


def plot_mae(results: List[Tuple[str, pd.DataFrame]], out_path: Path, y_label: str) -> None:
    plt.figure(figsize=(7, 4))
    color_palette = ["green", "blue", "black", "red"]
    base_colors: Dict[str, str] = {}
    color_idx = 0

    for label, df in results:
        base_label = label.split("-")[0]
        if base_label not in base_colors:
            base_colors[base_label] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

        is_powercurve = "powercurve" in label.lower()
        plt.plot(
            df["lead_hours"], df["MAE"],
            marker="" if is_powercurve else "o",
            lw=1.8,
            ls="--" if is_powercurve else "-",
            label=label,
            color=base_colors.get(base_label),
        )
        plt.axhline(
            df["MAE"].mean(),
            color=base_colors.get(base_label),
            lw=1.0,
            ls="--" if is_powercurve else "-",
            alpha=0.6,
        )

    plt.title("MAE vs Lead Time", fontsize=12)
    plt.xlabel("Lead time [hours]")
    plt.ylabel(y_label)
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main() -> None:
    scope = "Belgium-only" if VERIFY_BELGIUM_ONLY else "All farms"
    total_capacity_mw = get_total_capacity_mw()
    print(f"Scope: {scope}")
    print(f"Total capacity (from metadata): {total_capacity_mw:.1f} MW")

    if not VERIFY_BELGIUM_ONLY:
        farm_path = OBS_FARM_PATH or _find_farm_obs_csv_by_header(OBS_FARM_DIR)
        print(f"Using farm-level obs file: {farm_path}")

    cells  = build_turbine_cells_from_metadata()
    points = build_turbine_points_from_metadata()
    specs  = load_specs(SPECS_PATH)
    type_order, counts_matrix = build_counts_matrix(cells)

    # --- intersect init times across all directories ---
    common_files = get_common_files(FORECAST_DIRS)

    series_per_dir = []
    global_time_min: pd.Timestamp | None = None
    global_time_max: pd.Timestamp | None = None

    for fc_dir in FORECAST_DIRS:
        label = fc_dir.name
        files = common_files[fc_dir]
        print(f"\nProcessing {label} ({len(files)} init times) ...")

        turbine_value_indices = build_value_indices_from_latlon(
            files[0], points, decimals=DECIMALS, tol_degrees=TOL_DEGREES,
        )

        has_power = dir_has_power(files)
        if not has_power:
            print(f"  '{label}' has no 'power' variable; using power-curve baseline only.")

        fcst    = load_dir_forecasts(files, turbine_value_indices) if has_power else None
        pc_fcst = load_dir_powercurve(files, turbine_value_indices, type_order, specs, counts_matrix)

        for df in (fcst, pc_fcst):
            if df is None:
                continue
            global_time_min = df["time"].min() if global_time_min is None else min(global_time_min, df["time"].min())
            global_time_max = df["time"].max() if global_time_max is None else max(global_time_max, df["time"].max())

        series_per_dir.append((label, has_power, fcst, pc_fcst))

    if global_time_min is None or global_time_max is None:
        raise RuntimeError("No forecasts processed.")

    obs = load_obs((global_time_min, global_time_max))

    mae_results_mw:  List[Tuple[str, pd.DataFrame]] = []
    mae_results_pct: List[Tuple[str, pd.DataFrame]] = []

    for label, has_power, fcst, pc_fcst in series_per_dir:
        if has_power and fcst is not None:
            mae_results_mw.append((f"{label}-model",      compute_mae(fcst, obs, "fcst_MW")))
            mae_results_pct.append((f"{label}-model",     compute_mae(fcst, obs, "fcst_MW", as_percent=True, capacity_mw=total_capacity_mw)))
        mae_results_mw.append((f"{label}-powercurve",     compute_mae(pc_fcst, obs, "pc_MW")))
        mae_results_pct.append((f"{label}-powercurve",    compute_mae(pc_fcst, obs, "pc_MW", as_percent=True, capacity_mw=total_capacity_mw)))

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    plot_mae(mae_results_mw,  PLOT_DIR / f"mae_combined_MW_{ts}.png",  y_label="MAE [MW]")
    plot_mae(mae_results_pct, PLOT_DIR / f"mae_combined_pct_{ts}.png", y_label="MAE [% of total capacity]")
    print(f"Saved plots: mae_combined_MW_{ts}.png and mae_combined_pct_{ts}.png")


if __name__ == "__main__":
    main()


def diagnose_alignment(
    fc_dir: Path,
    turbine_value_indices: np.ndarray,
    obs: pd.DataFrame,
    n_files: int = 2,
) -> None:
    """
    Print side-by-side forecast power vs obs_MW for the first n_files init times.
    Use this to visually check time alignment.
    """
    files = sorted(fc_dir.glob("forecast_*.nc"))[:n_files]
    print(f"\n{'='*70}")
    print(f"ALIGNMENT DIAGNOSTIC — {fc_dir.name}")
    print(f"{'='*70}")
    for p in files:
        init = parse_init_time(p)
        with xr.open_dataset(p) as ds:
            if "power" not in ds:
                print(f"  {p.name}: no 'power' variable, skipping")
                continue
            power = ds["power"].isel(values=turbine_value_indices).sum(dim="values")
            fc_times = pd.to_datetime(ds["time"].values).dt.tz_localize(None)
            fc_vals  = power.values

        df_fc = pd.DataFrame({"time": fc_times, "fcst_MW": fc_vals})
        df_fc["lead_hours"] = ((pd.to_datetime(df_fc["time"]) - init.tz_localize(None))
                               / np.timedelta64(1, "h")).astype(int)
        merged = df_fc.merge(obs, on="time", how="inner")

        print(f"\n  Init: {init}  ({len(merged)} matched timesteps)")
        print(f"  {'lead_h':>7}  {'valid_time':>22}  {'fcst_MW':>10}  {'obs_MW':>10}  {'err':>10}")
        for _, row in merged.head(10).iterrows():
            err = row["fcst_MW"] - row["obs_MW"]
            print(f"  {int(row['lead_hours']):>7}  {str(row['time']):>22}  "
                  f"{row['fcst_MW']:>10.1f}  {row['obs_MW']:>10.1f}  {err:>10.1f}")
    print(f"{'='*70}\n")