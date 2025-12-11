#!/usr/bin/env python
"""
Farm-level verification: model forecast vs power-curve baseline.

- Reads forecast netCDFs (forecast_*.nc) from configured directories.
- Builds farm-level series for Belgian farms (model power + power-curve baseline).
- Computes MAE vs observed farm production and plots MAE per farm in subplots.
"""

from __future__ import annotations

import math
import re
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
          Path("/mnt/data/weatherloss/WindPower/inference/BestTransf"),
        Path("/mnt/data/weatherloss/WindPower/inference/BestGraph")
]

OBS_PATH = Path("/mnt/data/weatherloss/WindPower/verification/BE_offshore_3H_totalMW.csv")
COUNTS_PATH = Path("/mnt/data/weatherloss/WindPower/data/NorthSea/Power/wind_farm_turbine_counts.csv")
SPECS_PATH = Path("/mnt/data/weatherloss/WindPower/data/NorthSea/Power/turbine_specs.csv")
METADATA_PATH = Path("/mnt/data/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")

PLOT_DIR = Path("/mnt/data/weatherloss/WindPower/verification/Plots")

FORECAST_NX = 211
FORECAST_NY = 157

LEAD_MIN = 3
LEAD_MAX = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_init_time(path: Path) -> pd.Timestamp:
    match = re.search(r"forecast_(\d{14})", path.name)
    if not match:
        raise ValueError(f"Cannot parse init time from filename: {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S", utc=True)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
def load_obs_by_farm(time_range: Tuple[pd.Timestamp, pd.Timestamp], farm_names: Sequence[str]) -> pd.DataFrame:
    obs = pd.read_csv(OBS_PATH)
    obs["time"] = pd.to_datetime(obs["time"], utc=True).dt.tz_localize(None)

    missing = [f for f in farm_names if f not in obs.columns]
    if missing:
        raise ValueError(f"Missing farm columns in observations: {missing}")

    tmin = time_range[0].tz_localize(None) if getattr(time_range[0], "tzinfo", None) else time_range[0]
    tmax = time_range[1].tz_localize(None) if getattr(time_range[1], "tzinfo", None) else time_range[1]
    obs = obs[(obs["time"] >= tmin) & (obs["time"] <= tmax)]

    long = (
        obs[["time", *farm_names]]
        .melt(id_vars="time", var_name="farm", value_name="obs_MW")
        .dropna(subset=["obs_MW"])
    )
    return long


# ---------------------------------------------------------------------------
# Farm metadata + mapping
# ---------------------------------------------------------------------------
def build_belgian_farms() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH)
    meta_be = meta[meta["region"].str.lower() == "belgium"]
    farms = meta_be[["farm", "x", "y"]].drop_duplicates()
    if farms.empty:
        raise ValueError("No Belgian farms found in metadata.")

    farms["x"] = farms["x"].astype(int)
    farms["y"] = farms["y"].astype(int)

    flat_idx_grid = np.arange(FORECAST_NY * FORECAST_NX).reshape(FORECAST_NY, FORECAST_NX)
    farms["value_idx"] = [int(flat_idx_grid[y, x]) for _, (x, y) in farms[["x", "y"]].iterrows()]
    return farms


# ---------------------------------------------------------------------------
# Model forecasts
# ---------------------------------------------------------------------------
def dir_has_power(forecast_dir: Path) -> bool:
    files = sorted(forecast_dir.glob("forecast_*.nc"))
    if not files:
        raise FileNotFoundError(f"No forecast_*.nc files found in {forecast_dir}")
    with xr.open_dataset(files[0]) as ds:
        return "power" in ds


def load_forecast_power_farms(path: Path, farms: pd.DataFrame) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        values_dim = ds.sizes.get("values")
        if values_dim != FORECAST_NX * FORECAST_NY:
            raise ValueError(f"Unexpected values dimension in {path}: {values_dim}")

        idx = xr.DataArray(farms["value_idx"].to_numpy(), dims="farm", coords={"farm": farms["farm"]})
        power = ds["power"].isel(values=idx)
        df = power.to_dataframe(name="fcst_MW").reset_index()

        init_time = parse_init_time(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["init_time"] = init_time
        df["lead_hours"] = ((df["time"] - init_time) / np.timedelta64(1, "h")).astype(int)

    df["time"] = df["time"].dt.tz_localize(None)
    df["init_time"] = df["init_time"].dt.tz_localize(None)
    return df[(df["lead_hours"] >= LEAD_MIN) & (df["lead_hours"] <= LEAD_MAX)]


def load_dir_forecasts(forecast_dir: Path, farms: pd.DataFrame) -> pd.DataFrame:
    files = sorted(forecast_dir.glob("forecast_*.nc"))
    if not files:
        raise FileNotFoundError(f"No forecast_*.nc files found in {forecast_dir}")
    frames = []
    for p in files:
        try:
            frames.append(load_forecast_power_farms(p, farms))
        except KeyError as exc:
            print(f"Skipping {p.name}: {exc}")
            continue
    if not frames:
        raise RuntimeError(f"No usable forecasts found in {forecast_dir}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Power curve baseline
# ---------------------------------------------------------------------------
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


def build_counts_matrix(farms: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    counts = pd.read_csv(COUNTS_PATH).set_index("farm")
    type_cols = [c for c in counts.columns if c.lower() not in {"total"}]

    missing = [f for f in farms["farm"] if f not in counts.index]
    if missing:
        raise ValueError(f"Farm(s) missing in counts file: {missing}")

    counts_matrix = counts.loc[farms["farm"], type_cols].to_numpy(dtype=np.float32)
    return type_cols, counts_matrix


def load_forecast_powercurve_farms(
    path: Path, farms: pd.DataFrame, type_order: Sequence[str], specs: Dict[str, TurbineSpec], counts_matrix: np.ndarray
) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        idx = xr.DataArray(farms["value_idx"].to_numpy(), dims="farm", coords={"farm": farms["farm"]})
        ws = ds["ws100"].isel(values=idx).to_numpy()  # shape: time, farm

        total = np.zeros_like(ws, dtype=np.float32)
        for j, tname in enumerate(type_order):
            if tname not in specs:
                raise ValueError(f"Turbine type '{tname}' missing from specs.")
            pc = power_curve(ws, specs[tname])
            total += pc * counts_matrix[:, j][None, :]

        init_time = parse_init_time(path)
        valid_time_utc = pd.to_datetime(ds["time"].values).tz_localize("UTC")
        lead_hours = ((valid_time_utc - init_time) / np.timedelta64(1, "h")).astype(int)

    records = []
    for fi, farm in enumerate(farms["farm"]):
        df = pd.DataFrame(
            {
                "time": valid_time_utc.tz_localize(None),
                "init_time": init_time.tz_localize(None),
                "lead_hours": lead_hours,
                "farm": farm,
                "pc_MW": total[:, fi],
            }
        )
        df = df[(df["lead_hours"] >= LEAD_MIN) & (df["lead_hours"] <= LEAD_MAX)]
        records.append(df)
    return pd.concat(records, ignore_index=True)


def load_dir_powercurve(
    forecast_dir: Path, farms: pd.DataFrame, type_order: Sequence[str], specs: Dict[str, TurbineSpec], counts_matrix: np.ndarray
) -> pd.DataFrame:
    files = sorted(forecast_dir.glob("forecast_*.nc"))
    if not files:
        raise FileNotFoundError(f"No forecast_*.nc files found in {forecast_dir}")
    frames = [load_forecast_powercurve_farms(p, farms, type_order, specs, counts_matrix) for p in files]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics + plotting
# ---------------------------------------------------------------------------
def compute_mae_by_farm(fcst: pd.DataFrame, obs: pd.DataFrame, value_col: str) -> pd.DataFrame:
    merged = fcst.merge(obs, on=["time", "farm"], how="inner").dropna(subset=[value_col, "obs_MW"])
    if merged.empty:
        raise ValueError("No overlapping forecast/obs times after merge.")
    merged["err"] = merged[value_col] - merged["obs_MW"]
    metrics = (
        merged.groupby(["farm", "lead_hours"])
        .agg(count=("err", "size"), MAE=("err", lambda s: s.abs().mean()))
        .reset_index()
        .sort_values(["farm", "lead_hours"])
    )
    return metrics


def plot_mae_by_farm(results: List[Tuple[str, pd.DataFrame]], farms: Sequence[str], out_path: Path) -> None:
    n_farms = len(farms)
    ncols = 3 if n_farms > 3 else n_farms
    nrows = math.ceil(n_farms / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.3 * nrows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()

    base_colors: Dict[str, str] = {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_idx = 0

    for ax, farm in zip(axes_arr, farms):
        for label, df in results:
            sub = df[df["farm"] == farm]
            if sub.empty:
                continue

            base_label = label.split("-")[0]
            if base_label not in base_colors:
                base_colors[base_label] = color_cycle[color_idx % len(color_cycle)] if color_cycle else None
                color_idx += 1

            is_powercurve = "powercurve" in label.lower()
            linestyle = "--" if is_powercurve else "-"
            marker = "" if is_powercurve else "o"
            ax.plot(
                sub["lead_hours"],
                sub["MAE"],
                marker=marker,
                lw=1.6,
                ls=linestyle,
                label=label,
                color=base_colors.get(base_label),
            )
        ax.set_title(farm, fontsize=10)
        ax.grid(True, ls="--", alpha=0.6)

    # Remove unused axes when farm count doesn't fill the grid.
    for ax in axes_arr[n_farms:]:
        ax.set_visible(False)

    fig.suptitle("MAE vs Lead Time (per farm)", fontsize=13)
    fig.text(0.5, 0.04, "Lead time [hours]", ha="center")
    fig.text(0.04, 0.5, "MAE [MW]", va="center", rotation="vertical")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved MAE plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    farms = build_belgian_farms()
    specs = load_specs(SPECS_PATH)
    type_order, counts_matrix = build_counts_matrix(farms)

    series_per_dir: List[Tuple[str, bool, pd.DataFrame | None, pd.DataFrame]] = []
    global_time_min: pd.Timestamp | None = None
    global_time_max: pd.Timestamp | None = None

    for fc_dir in FORECAST_DIRS:
        label = fc_dir.name
        print(f"Processing {label} ...")
        has_power = dir_has_power(fc_dir)
        if not has_power:
            print(f"  '{label}' has no 'power' variable; using power-curve baseline only.")
        fcst = load_dir_forecasts(fc_dir, farms) if has_power else None
        pc_fcst = load_dir_powercurve(fc_dir, farms, type_order, specs, counts_matrix)

        for df in (fcst, pc_fcst):
            if df is None:
                continue
            tmin = df["time"].min()
            tmax = df["time"].max()
            global_time_min = tmin if global_time_min is None else min(global_time_min, tmin)
            global_time_max = tmax if global_time_max is None else max(global_time_max, tmax)

        series_per_dir.append((label, has_power, fcst, pc_fcst))

    if global_time_min is None or global_time_max is None:
        raise RuntimeError("No forecasts processed.")

    obs = load_obs_by_farm((global_time_min, global_time_max), farms["farm"].tolist())

    mae_results: List[Tuple[str, pd.DataFrame]] = []
    for label, has_power, fcst, pc_fcst in series_per_dir:
        if has_power:
            mae_model = compute_mae_by_farm(fcst, obs, "fcst_MW")
            mae_results.append((f"{label}-model", mae_model))
        mae_pc = compute_mae_by_farm(pc_fcst, obs, "pc_MW")
        mae_results.append((f"{label}-powercurve", mae_pc))

    plot_name = f"mae_by_farm_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.png"
    plot_mae_by_farm(mae_results, farms["farm"].tolist(), PLOT_DIR / plot_name)


if __name__ == "__main__":
    main()
