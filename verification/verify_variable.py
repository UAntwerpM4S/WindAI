#!/usr/bin/env python
"""
Verify a single forecasted variable against the original CERRA BOZ data.

- Configure TARGET_VAR and FORECAST_DIRS below.
- Loads forecasts (forecast_*.nc) and the matching variable from data/BOZ.zarr.
- Flattens both to (time, values) in the same y/x order, then plots MAE vs lead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_VAR = "ws100"

FORECAST_DIRS: List[Path] = [
    Path("/mnt/data/weatherloss/WindPower/inference/GraphBOZepoch2"),
    Path("/mnt/data/weatherloss/WindPower/inference/GraphBOZepoch4"),
    Path("/mnt/data/weatherloss/WindPower/inference/GraphBOZepoch5"),
    Path("/mnt/data/weatherloss/WindPower/inference/GraphBOZepoch7"),
    Path("/mnt/data/weatherloss/WindPower/inference/GraphBOZepoch10"),
    Path("/mnt/data/weatherloss/WindPower/inference/GraphBOZepoch14"),
    Path("/mnt/data/weatherloss/WindPower/inference/GraphBOZepoch17")
]

CERRA_PATH = Path("/mnt/data/weatherloss/WindPower/data/BOZ.zarr")
PLOT_DIR = Path("/mnt/data/weatherloss/WindPower/verification/Plots")

FORECAST_NX = 211
FORECAST_NY = 157

LEAD_MIN = 3
LEAD_MAX = 24

# ---------------------------------------------------------------------------


def parse_init_time(path: Path) -> pd.Timestamp:
    match = re.search(r"forecast_(\d{14})", path.name)
    if not match:
        raise ValueError(f"Cannot parse init time from filename: {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S", utc=True)


def _squeeze_singletons(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if dim == "time":
            continue
        if da.sizes.get(dim, 1) == 1:
            da = da.isel({dim: 0}, drop=True)
    return da


def _as_flat_values(da: xr.DataArray) -> xr.DataArray:
    if "values" in da.dims:
        return da.transpose("time", "values")
    if {"y", "x"}.issubset(da.dims):
        return da.transpose("time", "y", "x").stack(values=("y", "x")).transpose("time", "values")
    raise ValueError(f"Unsupported dimensions for variable: {da.dims}")


def load_cerra_values(var_name: str) -> xr.DataArray:
    ds = xr.open_zarr(CERRA_PATH)
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in {CERRA_PATH}")

    da = _squeeze_singletons(ds[var_name])
    if "time" not in da.coords:
        raise ValueError(f"No time coordinate found for '{var_name}' in {CERRA_PATH}")

    time = pd.to_datetime(da["time"].values)
    if getattr(time, "tz", None) is not None:
        time = time.tz_localize(None)
    da = da.assign_coords(time=time)
    flat = _as_flat_values(da)
    return flat


def load_forecast_values(path: Path, var_name: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in {path}")

    init_time = parse_init_time(path)
    valid_time_utc = pd.to_datetime(ds["time"].values).tz_localize("UTC")
    valid_time_naive = valid_time_utc.tz_localize(None)
    lead_hours = ((valid_time_utc - init_time) / np.timedelta64(1, "h")).astype(int)

    da = _squeeze_singletons(ds[var_name])
    da = da.assign_coords(time=valid_time_naive)
    flat = _as_flat_values(da).assign_coords(lead_hours=("time", lead_hours))
    values_dim = flat.sizes.get("values")
    if values_dim is None or values_dim != FORECAST_NX * FORECAST_NY:
        raise ValueError(f"Unexpected values dimension in {path}: {values_dim}")
    return flat


def compute_mae_against_cerra(fc_da: xr.DataArray, cerra_da: xr.DataArray) -> pd.DataFrame:
    fc_aligned, cerra_aligned = xr.align(fc_da, cerra_da, join="inner")
    if fc_aligned.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping forecast/cerra times after alignment.")

    err = (fc_aligned - cerra_aligned).abs().mean(dim="values")
    leads = fc_aligned.coords["lead_hours"].values
    df = pd.DataFrame({"lead_hours": leads, "MAE": err.values})
    df = df[(df["lead_hours"] >= LEAD_MIN) & (df["lead_hours"] <= LEAD_MAX)]
    return df


def load_dir_metrics(forecast_dir: Path, cerra_da: xr.DataArray) -> pd.DataFrame:
    files = sorted(forecast_dir.glob("forecast_*.nc"))
    if not files:
        raise FileNotFoundError(f"No forecast_*.nc files found in {forecast_dir}")

    frames: List[pd.DataFrame] = []
    for path in files:
        fc_da = load_forecast_values(path, TARGET_VAR)
        frames.append(compute_mae_against_cerra(fc_da, cerra_da))

    all_df = pd.concat(frames, ignore_index=True)
    grouped = (
        all_df.groupby("lead_hours")
        .agg(count=("MAE", "size"), MAE=("MAE", "mean"))
        .reset_index()
        .sort_values("lead_hours")
    )
    return grouped


def plot_mae(results: List[Tuple[str, pd.DataFrame]], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for label, df in results:
        plt.plot(df["lead_hours"], df["MAE"], marker="o", lw=1.8, label=label)
    plt.title(f"MAE vs Lead Time for {TARGET_VAR}", fontsize=12)
    plt.xlabel("Lead time [hours]")
    plt.ylabel(f"MAE [{TARGET_VAR} units]")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved MAE plot to {out_path}")


def main() -> None:
    cerra_da = load_cerra_values(TARGET_VAR)
    if cerra_da.sizes.get("values") != FORECAST_NX * FORECAST_NY:
        raise ValueError(f"CERRA grid size mismatch: {cerra_da.sizes.get('values')} values found.")

    mae_results = []
    for fc_dir in FORECAST_DIRS:
        print(f"Processing {fc_dir.name} ...")
        metrics = load_dir_metrics(fc_dir, cerra_da)
        mae_results.append((fc_dir.name, metrics))

    plot_name = f"mae_{TARGET_VAR}_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.png"
    plot_mae(mae_results, PLOT_DIR / plot_name)


if __name__ == "__main__":
    main()
