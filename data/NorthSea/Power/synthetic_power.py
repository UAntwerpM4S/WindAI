#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================
# Paths / Config
# =============================================================================
POWER_DIR = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power")
CERRA_DIR = POWER_DIR.parent / "Cerra"  # same level as Power

HR_ZARR_IN = CERRA_DIR / "cerra_HR.zarr"
LR_ZARR_IN = CERRA_DIR / "cerra_LR.zarr"

METADATA_CSV = POWER_DIR / "windfarm_metadata.csv"
SPECS_CSV = POWER_DIR / "turbine_specs.csv"
COUNTS_CSV = POWER_DIR / "wind_farm_turbine_counts.csv"

N_HR_VARIANTS = 8
SEED0 = 12345

# Grouping precision for lat_cell/lon_cell identity (defensive against float noise)
LATLON_DECIMALS = 10


# =============================================================================
# Power curve (capacity factor), then MW = CF * installed_capacity
# =============================================================================
@dataclass(frozen=True)
class TurbineSpec:
    cut_in: float
    rated_ws: float
    cut_out: float
    rated_mw: float


def cf_curve(ws: np.ndarray, spec: TurbineSpec) -> np.ndarray:
    """
    Capacity factor in [0,1]:
      0 below cut-in
      cubic ramp to 1 between cut-in and rated_ws
      1 between rated_ws and cut-out
      0 above cut-out
    """
    ws = np.asarray(ws, dtype=np.float32)
    out = np.zeros_like(ws, dtype=np.float32)

    ramp = (ws >= spec.cut_in) & (ws < spec.rated_ws)
    if np.any(ramp):
        out[ramp] = ((ws[ramp] - spec.cut_in) / (spec.rated_ws - spec.cut_in)) ** 3

    rated = (ws >= spec.rated_ws) & (ws < spec.cut_out)
    if np.any(rated):
        out[rated] = 1.0

    return out


# =============================================================================
# Load specs / build sampling pool
# =============================================================================
def load_specs(path: Path) -> Dict[str, TurbineSpec]:
    df = pd.read_csv(path)
    name_col = "turbine_type (name-capacity-type)"
    required = [name_col, "cut_in_ms", "rated_ws_ms", "cut_out_ms", "rated_power_mw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    specs: Dict[str, TurbineSpec] = {}
    for _, r in df.iterrows():
        name = str(r[name_col]).strip()
        specs[name] = TurbineSpec(
            cut_in=float(r["cut_in_ms"]),
            rated_ws=float(r["rated_ws_ms"]),
            cut_out=float(r["cut_out_ms"]),
            rated_mw=float(r["rated_power_mw"]),
        )
    return specs


def build_sampling_pool(
    metadata_csv: Path,
    counts_csv: Path,
    specs: Dict[str, TurbineSpec],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build pool of unique (lat_cell, lon_cell) entries.
    If multiple farms share a cell center, aggregate them:
      - turbines_sum
      - capacity_sum_mw
      - type counts summed over farms at that lat/lon cell
    """
    meta = pd.read_csv(metadata_csv)
    required_meta = {"farm", "lat_cell", "lon_cell", "turbines", "capacity_mw"}
    missing = required_meta - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    counts = pd.read_csv(counts_csv)
    if "farm" not in counts.columns:
        raise ValueError(f"'farm' column missing from {counts_csv}")
    counts = counts.set_index("farm")

    # turbine type columns = everything except Total (case-insensitive)
    type_cols = [c for c in counts.columns if c.lower() != "total"]
    if not type_cols:
        raise ValueError("No turbine type columns found in counts (after excluding Total).")

    # ensure types exist in specs
    missing_specs = [t for t in type_cols if t not in specs]
    if missing_specs:
        raise ValueError(
            "Some turbine types in wind_farm_turbine_counts.csv are missing from turbine_specs.csv. "
            f"Examples: {missing_specs[:10]}"
        )

    # defensive lat/lon keying
    meta["lat_cell_key"] = meta["lat_cell"].astype(float).round(LATLON_DECIMALS)
    meta["lon_cell_key"] = meta["lon_cell"].astype(float).round(LATLON_DECIMALS)

    # Ensure all meta farms exist in counts
    bad_farms = meta.loc[~meta["farm"].isin(counts.index), "farm"].unique().tolist()
    if bad_farms:
        raise ValueError(
            "Some farms in windfarm_metadata.csv are missing from wind_farm_turbine_counts.csv. "
            f"Examples: {bad_farms[:10]}"
        )

    counts_sub = counts[type_cols].copy()
    counts_sub["farm"] = counts_sub.index

    meta2 = meta.merge(counts_sub.reset_index(drop=True), on="farm", how="left")

    agg = {t: "sum" for t in type_cols}
    agg.update({"turbines": "sum", "capacity_mw": "sum"})

    pool = (
        meta2.groupby(["lat_cell_key", "lon_cell_key"], as_index=False)
        .agg(agg)
        .rename(columns={"turbines": "turbines_sum", "capacity_mw": "capacity_sum_mw"})
    )
    if pool.empty:
        raise ValueError("Sampling pool is empty after grouping. Check metadata/counts.")

    return pool, type_cols


# =============================================================================
# Build randomized maps + compute power (MW)
# =============================================================================
def build_random_maps_for_grid(
    ds_hr: xr.Dataset,
    pool: pd.DataFrame,
    type_cols: List[str],
    specs: Dict[str, TurbineSpec],
    *,
    seed: int,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, Dict[str, xr.DataArray]]:
    """
    For every (y,x) with lsm==0:
      - choose a random pool row (lat/lon cell group)
      - assign turbines/capacity for that group
      - build per-type installed capacity maps (MW) for power computation
    For lsm!=0:
      - mask, turbines, capacity will be NaN (per your request)
    Returns:
      mask_map(y,x): 1 over sea cells, NaN otherwise
      turbines_map(y,x): turbines_sum over sea cells, NaN otherwise
      capacity_map(y,x): capacity_sum_mw over sea cells, NaN otherwise
      cap_type_maps: dict[type] -> installed capacity MW at that type per cell (y,x), 0 where land (won't matter)
    """
    if "lsm" not in ds_hr:
        raise KeyError("Dataset missing variable 'lsm'")

    lsm = ds_hr["lsm"]
    sea = (lsm == 0).values  # numpy bool
    ny, nx = sea.shape

    rng = np.random.default_rng(seed)
    n_pool = len(pool)

    sea_idx = np.argwhere(sea)
    picks = rng.integers(0, n_pool, size=sea_idx.shape[0], dtype=np.int32)

    # Prepare arrays
    mask_np = np.full((ny, nx), np.nan, dtype=np.float32)
    turbines_np = np.full((ny, nx), np.nan, dtype=np.float32)
    capacity_np = np.full((ny, nx), np.nan, dtype=np.float32)

    turbines_sum = pool["turbines_sum"].to_numpy(dtype=np.float32)
    capacity_sum = pool["capacity_sum_mw"].to_numpy(dtype=np.float32)

    mask_np[sea] = 1.0
    turbines_np[sea] = turbines_sum[picks]
    capacity_np[sea] = capacity_sum[picks]

    coords = {"y": ds_hr["y"], "x": ds_hr["x"]}
    mask_map = xr.DataArray(mask_np, dims=("y", "x"), coords=coords, name="mask")
    turbines_map = xr.DataArray(turbines_np, dims=("y", "x"), coords=coords, name="turbines")
    capacity_map = xr.DataArray(capacity_np, dims=("y", "x"), coords=coords, name="capacity")

    # per-type installed capacity maps in MW (used for computing power)
    # We'll store 0 on land; power will be NaN on land anyway after we apply mask.
    cap_type_maps: Dict[str, xr.DataArray] = {}
    for t in type_cols:
        rated = float(specs[t].rated_mw)
        ntype = pool[t].to_numpy(dtype=np.float32)

        cap_np = np.zeros((ny, nx), dtype=np.float32)
        cap_np[sea] = ntype[picks] * rated

        cap_type_maps[t] = xr.DataArray(cap_np, dims=("y", "x"), coords=coords)

    return mask_map, turbines_map, capacity_map, cap_type_maps


def compute_power_mw(
    ws100: xr.DataArray,
    cap_type_maps: Dict[str, xr.DataArray],
    specs: Dict[str, TurbineSpec],
    mask_map: xr.DataArray,
) -> xr.DataArray:
    """
    power(time,y,x) = sum_t [ CF_t(ws100) * Cap_t(y,x) ]
    Then apply mask: NaN where mask is NaN (i.e. land)
    """
    syn = None
    for t, cap_map in cap_type_maps.items():
        sp = specs[t]
        cf = xr.apply_ufunc(
            cf_curve,
            ws100,
            kwargs={"spec": sp},
            dask="parallelized",          # IMPORTANT: enables chunked/dask arrays without rechunking
            output_dtypes=[np.float32],
        )
        term = cf * cap_map.astype(np.float32)
        syn = term if syn is None else (syn + term)

    syn = syn.astype(np.float32)
    # apply mask: where mask is NaN -> NaN power
    syn = syn.where(~xr.ufuncs.isnan(mask_map))
    syn.name = "power"
    syn.attrs.update(
        {
            "long_name": "Power (MW) from ws100 + randomized farm turbine mix",
            "units": "MW",
            "note": "Randomized turbines/capacity assigned to lsm==0; power computed from ws100 and turbine specs.",
        }
    )
    return syn


# =============================================================================
# Writers
# =============================================================================
def write_hr_variant(
    ds_hr_in: xr.Dataset,
    pool: pd.DataFrame,
    type_cols: List[str],
    specs: Dict[str, TurbineSpec],
    out_path: Path,
    *,
    seed: int,
) -> None:
    if "ws100" not in ds_hr_in:
        raise KeyError("HR dataset missing 'ws100'")

    mask_map, turbines_map, capacity_map, cap_type_maps = build_random_maps_for_grid(
        ds_hr_in, pool, type_cols, specs, seed=seed
    )
    power_new = compute_power_mw(ds_hr_in["ws100"], cap_type_maps, specs, mask_map)

    # Replace mask/turbines/capacity/power in the original dataset
    drop_vars = [v for v in ["mask", "turbines", "capacity", "power"] if v in ds_hr_in.variables]
    ds_out = ds_hr_in.drop_vars(drop_vars)

    ds_out = ds_out.assign(
        mask=mask_map.astype(np.float32),
        turbines=turbines_map.astype(np.float32),
        capacity=capacity_map.astype(np.float32),
        power=power_new,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_zarr(out_path, mode="w", consolidated=True)
    print(f"Wrote HR variant: {out_path}")


def write_lr_all_nan(ds_lr_in: xr.Dataset, out_path: Path) -> None:
    """
    LR: per your instruction, set mask/turbines/capacity/power to NaN everywhere,
    so the 'outer' dataset won't provide these signals.
    """
    # Drop and replace
    drop_vars = [v for v in ["mask", "turbines", "capacity", "power"] if v in ds_lr_in.variables]
    ds_out = ds_lr_in.drop_vars(drop_vars)

    # mask/turbines/capacity are (y,x)
    if "lsm" not in ds_lr_in:
        raise KeyError("LR dataset missing 'lsm'")
    nan_yx = xr.full_like(ds_lr_in["lsm"], np.nan, dtype=np.float32)

    # power is (time,y,x) -> use ws100 for shape
    if "ws100" not in ds_lr_in:
        raise KeyError("LR dataset missing 'ws100'")
    nan_tyx = xr.full_like(ds_lr_in["ws100"], np.nan, dtype=np.float32)

    ds_out = ds_out.assign(
        mask=nan_yx.rename("mask"),
        turbines=nan_yx.rename("turbines"),
        capacity=nan_yx.rename("capacity"),
        power=nan_tyx.rename("power"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_zarr(out_path, mode="w", consolidated=True)
    print(f"Wrote LR dataset (all NaN for mask/turbines/capacity/power): {out_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("Loading turbine specs + building sampling pool...")
    specs = load_specs(SPECS_CSV)
    pool, type_cols = build_sampling_pool(METADATA_CSV, COUNTS_CSV, specs)

    print(f"Pool size (unique lat_cell/lon_cell groups): {len(pool)}")
    print(f"Number of turbine types: {len(type_cols)}")

    print("Opening original Zarr datasets...")
    ds_hr = xr.open_zarr(HR_ZARR_IN, consolidated=True)
    ds_lr = xr.open_zarr(LR_ZARR_IN, consolidated=True)

    # LR: one output (NaNs for these four variables)
    write_lr_all_nan(ds_lr, POWER_DIR / "cerra_LR_rand.zarr")

    # HR: 8 randomized variants
    for i in range(1, N_HR_VARIANTS + 1):
        seed = SEED0 + i
        out = POWER_DIR / f"cerra_HR_{i}.zarr"
        print(f"Creating HR variant {i}/{N_HR_VARIANTS} (seed={seed}) ...")
        write_hr_variant(ds_hr, pool, type_cols, specs, out, seed=seed)


if __name__ == "__main__":
    main()
