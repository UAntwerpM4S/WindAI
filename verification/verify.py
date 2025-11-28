#!/usr/bin/env python
"""
Combined verification: model forecast vs power-curve baseline on one plot.

- Reads forecast netCDFs (forecast_*.nc) from configured directories.
- Aggregates forecast power over Belgian turbine cells and compares to observed
  total power (verification/BE_offshore_3H_totalMW.csv).
- Computes a power-curve baseline from ws100 using turbine specs + counts for
  the same Belgian cells, and compares to the same observations.
- Outputs a single MAE plot with both baselines per forecast directory.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FORECAST_DIRS: List[Path] = [
     Path("/mnt/data/weatherloss/WindPower/inference/TransWithL25"),
     Path("/mnt/data/weatherloss/WindPower/inference/TransNoL2Last")
]

OBS_PATH = Path("/mnt/data/weatherloss/WindPower/verification/BE_offshore_3H_totalMW.csv")
COUNTS_PATH = Path("/mnt/data/weatherloss/WindPower/data/NorthSea/Power/wind_farm_turbine_counts.csv")
SPECS_PATH = Path("/mnt/data/weatherloss/WindPower/data/NorthSea/Power/turbine_specs.csv")
METADATA_PATH = Path("/mnt/data/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

PLOT_DIR = Path("/mnt/data/weatherloss/WindPower/verification/Plots")

FORECAST_NX = 217 #211 #217 
FORECAST_NY = 237 #157 #237 

LEAD_MIN = 3
LEAD_MAX = 24

# ---------------------------------------------------------------------------


def parse_init_time(path: Path) -> pd.Timestamp:
    match = re.search(r"forecast_(\d{14})", path.name)
    if not match:
        raise ValueError(f"Cannot parse init time from filename: {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S", utc=True)


def load_obs(time_range: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    obs = pd.read_csv(OBS_PATH)
    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    total_col = next((c for c in obs.columns if c.lower().startswith("total") and c.lower().endswith("mw")), None)
    if total_col is None:
        raise ValueError("Could not find total MW column in observations.")
    obs["time"] = obs["time"].dt.tz_localize(None)
    tmin = time_range[0].tz_localize(None) if getattr(time_range[0], "tzinfo", None) else time_range[0]
    tmax = time_range[1].tz_localize(None) if getattr(time_range[1], "tzinfo", None) else time_range[1]
    obs = obs[(obs["time"] >= tmin) & (obs["time"] <= tmax)].copy()
    return obs[["time", total_col]].rename(columns={total_col: "obs_MW"})


# ------------------------- Model forecast path -----------------------------


def build_turbine_cells_from_metadata() -> List[Tuple[int, int]]:
    meta = pd.read_csv(METADATA_PATH)
    meta_be = meta[meta["region"].str.lower() == "belgium"]
    cells = []
    for _, row in meta_be.iterrows():
        cell = (int(row["y"]), int(row["x"]))
        if cell not in cells:
            cells.append(cell)
    if not cells:
        raise ValueError("No Belgian farms found in metadata.")
    return cells


def build_forecast_indices(cells: Sequence[Tuple[int, int]]) -> np.ndarray:
    flat_idx_grid = np.arange(FORECAST_NY * FORECAST_NX).reshape(FORECAST_NY, FORECAST_NX)
    return np.array([int(flat_idx_grid[y, x]) for (y, x) in cells], dtype=int)


def load_forecast_power(path: Path, turbine_value_indices: np.ndarray) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    values_dim = ds.sizes.get("values")
    if values_dim != FORECAST_NX * FORECAST_NY:
        raise ValueError(f"Unexpected values dimension in {path}: {values_dim}")

    power = ds["power"].isel(values=turbine_value_indices).sum(dim="values")
    out = power.to_series().rename("fcst_MW").reset_index()

    init_time = parse_init_time(path)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["init_time"] = init_time
    out["lead_hours"] = ((out["time"] - init_time) / np.timedelta64(1, "h")).astype(int)
    out["time"] = out["time"].dt.tz_localize(None)
    return out[(out["lead_hours"] >= LEAD_MIN) & (out["lead_hours"] <= LEAD_MAX)]


def load_dir_forecasts(forecast_dir: Path, turbine_value_indices: np.ndarray) -> pd.DataFrame:
    files = sorted(forecast_dir.glob("forecast_*.nc"))
    if not files:
        raise FileNotFoundError(f"No forecast_*.nc files found in {forecast_dir}")
    frames = [load_forecast_power(p, turbine_value_indices) for p in files]
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


def build_counts_matrix(cells: Sequence[Tuple[int, int]]) -> Tuple[List[str], np.ndarray]:
    counts = pd.read_csv(COUNTS_PATH).set_index("farm")
    type_cols = [c for c in counts.columns if c.lower() not in {"total"}]

    meta = pd.read_csv(METADATA_PATH)
    meta_be = meta[meta["region"].str.lower() == "belgium"]

    cell_to_type_counts: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in meta_be.iterrows():
        farm = row["farm"]
        cell = (int(row["y"]), int(row["x"]))
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
    path: Path, cell_indices: np.ndarray, type_order: Sequence[str], specs: Dict[str, TurbineSpec], counts_matrix: np.ndarray
) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    ws = ds["ws100"].isel(values=cell_indices).to_numpy()  # shape: time, cell

    total = np.zeros(ws.shape[0], dtype=np.float32)
    for j, tname in enumerate(type_order):
        if tname not in specs:
            raise ValueError(f"Turbine type '{tname}' missing from specs.")
        pc = power_curve(ws, specs[tname])
        total += np.sum(pc * counts_matrix[:, j], axis=1)

    init_time = parse_init_time(path)
    valid_time_utc = pd.to_datetime(ds["time"].values).tz_localize("UTC")
    lead_hours = ((valid_time_utc - init_time) / np.timedelta64(1, "h")).astype(int)
    return pd.DataFrame(
        {
            "time": valid_time_utc.tz_localize(None),
            "init_time": init_time.tz_localize(None),
            "lead_hours": lead_hours,
            "pc_MW": total,
        }
    )[(lead_hours >= LEAD_MIN) & (lead_hours <= LEAD_MAX)]


def load_dir_powercurve(
    forecast_dir: Path, cell_indices: np.ndarray, type_order: Sequence[str], specs: Dict[str, TurbineSpec], counts_matrix: np.ndarray
) -> pd.DataFrame:
    files = sorted(forecast_dir.glob("forecast_*.nc"))
    if not files:
        raise FileNotFoundError(f"No forecast_*.nc files found in {forecast_dir}")
    frames = [load_forecast_powercurve(p, cell_indices, type_order, specs, counts_matrix) for p in files]
    return pd.concat(frames, ignore_index=True)


# ------------------------- Metrics + plotting -----------------------------


def compute_mae(fcst: pd.DataFrame, obs: pd.DataFrame, value_col: str) -> pd.DataFrame:
    merged = fcst.merge(obs, on="time", how="inner").dropna(subset=[value_col, "obs_MW"])
    if merged.empty:
        raise ValueError("No overlapping forecast/obs times after merge.")
    merged["err"] = merged[value_col] - merged["obs_MW"]
    metrics = (
        merged.groupby("lead_hours")
        .agg(count=("err", "size"), MAE=("err", lambda s: s.abs().mean()))
        .reset_index()
        .sort_values("lead_hours")
    )
    return metrics


def plot_mae(results: List[Tuple[str, pd.DataFrame]], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    base_colors: Dict[str, str] = {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_idx = 0

    for label, df in results:
        base_label = label.split("-")[0]
        if base_label not in base_colors:
            base_colors[base_label] = color_cycle[color_idx % len(color_cycle)] if color_cycle else None
            color_idx += 1

        is_powercurve = "powercurve" in label.lower()
        linestyle = "--" if is_powercurve else "-"
        marker = "" if is_powercurve else "o"
        plt.plot(
            df["lead_hours"],
            df["MAE"],
            marker=marker,
            lw=1.8,
            ls=linestyle,
            label=label,
            color=base_colors.get(base_label),
        )
    plt.title("MAE vs Lead Time (Belgian cells)", fontsize=12)
    plt.xlabel("Lead time [hours]")
    plt.ylabel("MAE [MW]")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved MAE plot to {out_path}")


def main() -> None:
    cells = build_turbine_cells_from_metadata()
    turbine_value_indices = build_forecast_indices(cells)
    specs = load_specs(SPECS_PATH)
    type_order, counts_matrix = build_counts_matrix(cells)

    all_forecast_series: List[Tuple[str, pd.DataFrame]] = []
    all_pc_series: List[Tuple[str, pd.DataFrame]] = []
    global_time_min: pd.Timestamp | None = None
    global_time_max: pd.Timestamp | None = None

    for fc_dir in FORECAST_DIRS:
        label = fc_dir.name
        print(f"Processing {label} ...")
        fcst = load_dir_forecasts(fc_dir, turbine_value_indices)
        pc_fcst = load_dir_powercurve(fc_dir, turbine_value_indices, type_order, specs, counts_matrix)

        tmin = min(fcst["time"].min(), pc_fcst["time"].min())
        tmax = max(fcst["time"].max(), pc_fcst["time"].max())
        global_time_min = tmin if global_time_min is None else min(global_time_min, tmin)
        global_time_max = tmax if global_time_max is None else max(global_time_max, tmax)

        all_forecast_series.append((label, fcst))
        all_pc_series.append((label, pc_fcst))

    if global_time_min is None or global_time_max is None:
        raise RuntimeError("No forecasts processed.")

    obs = load_obs((global_time_min, global_time_max))

    mae_results: List[Tuple[str, pd.DataFrame]] = []
    for (label, fcst), (_, pc_fcst) in zip(all_forecast_series, all_pc_series):
        mae_model = compute_mae(fcst, obs, "fcst_MW")
        mae_pc = compute_mae(pc_fcst, obs, "pc_MW")
        mae_results.append((f"{label}-model", mae_model))
        mae_results.append((f"{label}-powercurve", mae_pc))

    plot_name = f"mae_combined_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.png"
    plot_mae(mae_results, PLOT_DIR / plot_name)


if __name__ == "__main__":
    main()
