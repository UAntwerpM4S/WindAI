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
    Path("/mnt/data/weatherloss/WindPower/inference/BestGraph"),
    Path("/mnt/data/weatherloss/WindPower/inference/BestTransf"),
    Path("/mnt/data/weatherloss/WindPower/inference/Regular")
]

CERRA_PATH = Path("/mnt/data/weatherloss/WindPower/data/BOZ.zarr")
PLOT_DIR = Path("/mnt/data/weatherloss/WindPower/verification/Plots")

FORECAST_NX = 211
FORECAST_NY = 157

LEAD_MIN = 3
LEAD_MAX = 72

COORD_DECIMALS = 5  # precision used to match lat/lon between forecast and CERRA
LEVEL_DIM_CANDIDATES = ("level",)  # order of precedence when looking for level dims

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


def _drop_duplicate_times(da: xr.DataArray, label: str) -> Tuple[xr.DataArray, np.ndarray | None]:
    """Ensure the time index has unique entries to keep xr.align happy."""
    idx = pd.Index(pd.to_datetime(da["time"].values))
    if idx.has_duplicates:
        dup_values = idx[idx.duplicated(keep="first")].unique()
        mask = ~idx.duplicated(keep="first")
        print(
            f"Warning: {label} contains {len(dup_values)} duplicate time value(s); "
            f"dropping duplicates (e.g. {dup_values[:3].tolist()})."
        )
        da = da.isel(time=mask)
        return da, mask
    return da, None


def _latlon_index(lat: np.ndarray, lon: np.ndarray) -> pd.Index:
    lat_r = np.round(np.asarray(lat, dtype=float), COORD_DECIMALS)
    lon_r = np.round(np.asarray(lon, dtype=float), COORD_DECIMALS)
    pairs = np.array(list(zip(lat_r.tolist(), lon_r.tolist())), dtype=object)
    return pd.Index(pairs, name="latlon")


def _intersection_index(a: pd.Index, b: pd.Index, label: str) -> pd.Index:
    common = a.intersection(b)
    if common.empty:
        raise ValueError(f"No overlapping {label} between forecast and CERRA.")
    return common


def _cerra_latlon_index() -> pd.Index:
    ds = xr.open_zarr(CERRA_PATH)
    lat_flat = ds["latitude"].stack(values=("y", "x")).values
    lon_flat = ds["longitude"].stack(values=("y", "x")).values
    return _latlon_index(lat_flat, lon_flat)


def _split_base_and_level(var_name: str) -> Tuple[str, str | None]:
    match = re.match(r"(.+)_([0-9]+)$", var_name)
    if match:
        return match.group(1), match.group(2)
    return var_name, None


def _select_level_slice(da: xr.DataArray, level_token: str, display_name: str) -> xr.DataArray:
    level_dim = next((dim for dim in LEVEL_DIM_CANDIDATES if dim in da.dims), None)
    if level_dim is None:
        raise ValueError(f"Variable '{da.name}' has no level dimension; cannot resolve '{display_name}'.")

    level_coord = da.coords.get(level_dim)
    numeric_level = None
    try:
        numeric_level = float(level_token)
    except ValueError:
        pass

    if level_coord is not None:
        if numeric_level is not None and np.issubdtype(level_coord.dtype, np.number):
            matches = np.where(np.isclose(level_coord.values.astype(float), numeric_level))[0]
            if matches.size:
                return da.isel({level_dim: int(matches[0])})
            try:
                return da.sel({level_dim: numeric_level})
            except Exception:
                pass
        try:
            return da.sel({level_dim: level_token})
        except Exception:
            pass
        if numeric_level is not None and np.issubdtype(level_coord.dtype, np.number):
            idx = int(np.argmin(np.abs(level_coord.values.astype(float) - numeric_level)))
            print(
                f"Level '{level_token}' not found exactly for '{display_name}'; "
                f"using nearest {level_coord.values[idx]} (index {idx})."
            )
            return da.isel({level_dim: idx})

    if numeric_level is not None:
        idx = int(numeric_level)
        if 0 <= idx < da.sizes[level_dim]:
            print(
                f"Level coordinate match for '{display_name}' failed; using index {idx} along '{level_dim}'."
            )
            return da.isel({level_dim: idx})
    raise ValueError(f"Could not resolve level '{level_token}' for variable '{display_name}'.")


def _get_variable_da(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """Return the requested variable; supports level suffixes like 'z_500'."""
    if var_name in ds:
        return ds[var_name]

    base, level = _split_base_and_level(var_name)
    if level is None:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")
    if base not in ds:
        raise ValueError(f"Variable '{base}' (base for '{var_name}') not found in dataset.")

    da = _select_level_slice(ds[base], level, var_name)
    da.name = var_name
    return da


def load_cerra_values(var_name: str, value_index: pd.Index | None = None) -> xr.DataArray:
    ds = xr.open_zarr(CERRA_PATH)
    da = _squeeze_singletons(_get_variable_da(ds, var_name))
    if "time" not in da.coords:
        raise ValueError(f"No time coordinate found for '{var_name}' in {CERRA_PATH}")

    time = pd.to_datetime(da["time"].values)
    if getattr(time, "tz", None) is not None:
        time = time.tz_localize(None)
    da = da.assign_coords(time=time)
    da, _ = _drop_duplicate_times(da, "CERRA")
    flat = _as_flat_values(da)
    if "values" in flat.indexes:
        flat = flat.reset_index("values", drop=True)
    latlon_idx = _latlon_index(ds["latitude"].stack(values=("y", "x")).values, ds["longitude"].stack(values=("y", "x")).values)
    flat = flat.assign_coords(values=("values", latlon_idx))
    if value_index is not None:
        common = value_index.intersection(flat.indexes["values"])
        if common.empty:
            raise ValueError("No overlapping lat/lon between CERRA and forecasts.")
        flat = flat.sel(values=common)
    return flat


def load_forecast_values(path: Path, var_name: str, value_index: pd.Index | None = None) -> xr.DataArray:
    ds = xr.open_dataset(path)

    init_time = parse_init_time(path)
    valid_time_utc = pd.to_datetime(ds["time"].values).tz_localize("UTC")
    valid_time_naive = valid_time_utc.tz_localize(None)
    lead_hours = ((valid_time_utc - init_time) / np.timedelta64(1, "h")).astype(int)

    da = _squeeze_singletons(_get_variable_da(ds, var_name))
    da = da.assign_coords(time=valid_time_naive)
    da, mask = _drop_duplicate_times(da, f"forecast {path.name}")
    if mask is not None:
        lead_hours = lead_hours[mask]
    flat = _as_flat_values(da).assign_coords(lead_hours=("time", lead_hours))
    if "values" in flat.indexes:
        flat = flat.reset_index("values", drop=True)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    flat = flat.assign_coords(values=("values", _latlon_index(lat, lon)))
    if value_index is not None:
        common = value_index.intersection(flat.indexes["values"])
        if common.empty:
            raise ValueError(f"No overlapping lat/lon between {path} and CERRA.")
        flat = flat.sel(values=common)
    return flat


def compute_mae_against_cerra(fc_da: xr.DataArray, cerra_da: xr.DataArray) -> pd.DataFrame:
    time_idx = _intersection_index(
        pd.Index(fc_da.indexes["time"]), pd.Index(cerra_da.indexes["time"]), "times"
    )
    value_idx = _intersection_index(
        pd.Index(fc_da.indexes["values"]), pd.Index(cerra_da.indexes["values"]), "grid points"
    )

    fc_sel = fc_da.sel(time=time_idx, values=value_idx)
    cerra_sel = cerra_da.sel(time=time_idx, values=value_idx)
    if fc_sel.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping forecast/cerra times after intersection.")

    err = np.abs(fc_sel - cerra_sel).mean(dim="values")
    leads = fc_sel.coords["lead_hours"].values
    df = pd.DataFrame({"lead_hours": leads, "MAE": err.values})
    df = df[(df["lead_hours"] >= LEAD_MIN) & (df["lead_hours"] <= LEAD_MAX)]
    return df


def load_dir_metrics(forecast_dir: Path, cerra_da: xr.DataArray) -> pd.DataFrame:
    files = sorted(forecast_dir.glob("forecast_*.nc"))
    if not files:
        raise FileNotFoundError(f"No forecast_*.nc files found in {forecast_dir}")

    frames: List[pd.DataFrame] = []
    for path in files:
        fc_da = load_forecast_values(path, TARGET_VAR, cerra_da.indexes["values"])
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
    plt.ylabel(f"MAE [m/s]")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved MAE plot to {out_path}")


def main() -> None:
    # Build a common lat/lon index across CERRA and the first file of each dir to avoid per-file warnings.
    cerra_latlon = _cerra_latlon_index()
    common_values = cerra_latlon
    for fc_dir in FORECAST_DIRS:
        first = next(iter(sorted(fc_dir.glob("forecast_*.nc"))), None)
        if first is None:
            raise FileNotFoundError(f"No forecast_*.nc files found in {fc_dir}")
        with xr.open_dataset(first) as ds:
            latlon = _latlon_index(ds["latitude"].values, ds["longitude"].values)
        common_values = _intersection_index(common_values, latlon, f"lat/lon for {fc_dir.name}")
    print(f"Using {len(common_values)} common grid points across all directories and CERRA.")

    cerra_da = load_cerra_values(TARGET_VAR, common_values)
    if cerra_da.sizes.get("values") != FORECAST_NX * FORECAST_NY:
        print(f"CERRA subset to {cerra_da.sizes.get('values')} values for common grid.")

    mae_results = []
    for fc_dir in FORECAST_DIRS:
        print(f"Processing {fc_dir.name} ...")
        metrics = load_dir_metrics(fc_dir, cerra_da)
        mae_results.append((fc_dir.name, metrics))

    plot_name = f"mae_{TARGET_VAR}_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.png"
    plot_mae(mae_results, PLOT_DIR / plot_name)


if __name__ == "__main__":
    main()
