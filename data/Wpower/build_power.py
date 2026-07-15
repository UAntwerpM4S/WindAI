#!/usr/bin/env python3
"""Build the distributed power target on the CERRA grid.

Spread each farm's observed power across the cells its turbines actually occupy, weighted by
CAPACITY, and SUM where farms share a cell (power is extensive). Write anemoi-ready source zarrs:

  power_cerra_src.zarr : power, capacityfactor, capacity, turbinecount, turbmask   (New_Cerra grid)
  power_era5_src.zarr   : the same variables projected onto the ERA5 grid (cutout companion)

Distribution (per farm f, cell c, time t):

    share(f, c)   = capacity of f's turbines in c  /  f's total capacity          (sum_c share = 1)
    power(c, t)   = SUM_f  P_obs(f, t) * share(f, c)                                (extensive: SUM)
    capacity(c)   = SUM_f  capacity of f's turbines in c                           (static)
    capacityfactor(c, t) = power(c, t) / capacity(c)                               (in [0, 1])

NaN rule: power is extensive, so a cell is NaN at t if ANY farm contributing to it is NaN at t
(you cannot sum a known and an unknown). Contrast wpx, an intensive wind speed, where a missing
farm simply drops out of a weighted mean. With Belgium's cells heavily shared this NaNs more
cells than you might expect; the printed valid fraction shows how much.

Inputs are the Wpower metadata (this directory): turbines.csv (farm, lon, lat, capacity_mw) and
power_obs.csv (per-farm MW, UTC, one column per farm). No power-curve inversion, no turbine-count
weighting -- both are wrong here (see README / farm_metadata.py).

Masking is DEFERRED for v1: calm (P=0) and storm shutdowns are kept (both forecastable from the
wind field); curtailment/outage detection is future work. The only NaNs are missing observations.

Usage:
  python build_power.py [--no-era5] [--cerra ZARR] [--era5 ZARR]
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from anemoi.datasets import open_dataset
from scipy.spatial import cKDTree


@dataclass
class Config:
    cerra_zarr: Path = Path("/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/New_Cerra_A_large.zarr")
    era5_zarr:  Path = Path("/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/era5_A_large.zarr")
    wpower_dir: Path = Path(__file__).resolve().parent
    out_cerra:  Path = Path("/mnt/weatherloss/WindPower/data/Wpower/power_cerra_src.zarr")
    out_era5:   Path = Path("/mnt/weatherloss/WindPower/data/Wpower/power_era5_src.zarr")
    time_chunk: int = 512

    @property
    def turbines_csv(self) -> Path: return self.wpower_dir / "turbines.csv"
    @property
    def power_csv(self) -> Path: return self.wpower_dir / "power_obs.csv"


def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def load_cerra_grid(path: Path):
    ds = open_dataset(str(path))
    return (np.asarray(ds.latitudes, dtype=float),
            np.asarray(ds.longitudes, dtype=float),
            pd.DatetimeIndex(ds.dates))


# =============================================================================
# turbines -> cells -> capacity-share weights
# =============================================================================
def assign_turbine_cells(turbines: pd.DataFrame, cerra_lat, cerra_lon180) -> np.ndarray:
    """Nearest CERRA cell per turbine (cos-lat scaled)."""
    coslat = np.cos(np.radians(float(cerra_lat.mean())))
    tree = cKDTree(np.c_[cerra_lon180 * coslat, cerra_lat])
    _, cell = tree.query(np.c_[to_180(turbines["longitude"]) * coslat,
                               turbines["latitude"].to_numpy()], k=1)
    return cell.astype(int)


def build_weights(turbines: pd.DataFrame, farms: list[str]):
    """Return (farm_cell_idx, W, cap_cell, count_cell).

    W[c, f] = capacity share of farm f in farm-cell c (each farm's column sums to 1).
    cap_cell[c] = total turbine capacity in c;  count_cell[c] = turbines in c.
    """
    farm_cell_idx = np.sort(turbines["cell"].unique())
    cpos = {int(c): i for i, c in enumerate(farm_cell_idx)}
    fpos = {f: j for j, f in enumerate(farms)}

    W = np.zeros((farm_cell_idx.size, len(farms)), dtype=np.float64)
    fc_cap = turbines.groupby(["farm", "cell"])["capacity_mw"].sum()
    farm_cap = turbines.groupby("farm")["capacity_mw"].sum()
    for (farm, cell), cap in fc_cap.items():
        W[cpos[int(cell)], fpos[farm]] = cap / farm_cap[farm]

    cap_cell = (turbines.groupby("cell")["capacity_mw"].sum()
                .reindex(farm_cell_idx).to_numpy(dtype=np.float64))
    count_cell = (turbines.groupby("cell").size()
                  .reindex(farm_cell_idx).to_numpy(dtype=np.float64))
    return farm_cell_idx, W, cap_cell, count_cell


# =============================================================================
# paint power onto cells (extensive SUM, strict NaN)
# =============================================================================
def paint_power(P: np.ndarray, W: np.ndarray, drop_nan: bool = False) -> np.ndarray:
    """P (T, F) farm power with NaNs, W (C, F) capacity shares -> power (T, C).

    Extensive SUM over farms. By default (drop_nan=False) a cell is NaN at t if ANY farm
    contributing to it (W>0) is NaN at t -- the correct rule for a summed quantity. With
    drop_nan=True, NaN contributors are simply omitted (used only for the ERA5 companion, which
    must stay finite; see write_era5_companion).
    """
    finite = np.isfinite(P)
    power = np.where(finite, P, 0.0) @ W.T                      # (T, C), NaN farms count as 0
    if drop_nan:
        # a cell is valid if at least one contributing farm is finite at t
        reachable = (finite.astype(float) @ (W.T > 0)) > 0
        power = np.where(reachable, power, np.nan).astype(np.float32)
        return power
    contrib = (W.T > 0)                                        # (F, C) which farms feed each cell
    nan_hits = (~finite).astype(float) @ contrib               # (T, C) count of NaN contributors
    return np.where(nan_hits > 0, np.nan, power).astype(np.float32)


# =============================================================================
# streamed zarr writer (matches build_wpx.py)
# =============================================================================
def _write_streamed(out: Path, data_vars_fn, dates, lat, lon, n_cells, chunk):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    lat32, lon32 = np.asarray(lat, "float32"), np.asarray(lon, "float32")
    for k, s in enumerate(range(0, dates.size, chunk)):
        e = min(s + chunk, dates.size)
        dvars = {name: (("time", "values"), arr) for name, arr in data_vars_fn(s, e).items()}
        dsb = xr.Dataset(dvars, coords={"time": ("time", dates.values[s:e]),
                                        "latitude": ("values", lat32),
                                        "longitude": ("values", lon32)})
        if k == 0:
            enc = {n: {"chunks": (chunk, n_cells)} for n in dvars}
            dsb.to_zarr(out, mode="w", consolidated=False, encoding=enc)
        else:
            dsb.to_zarr(out, mode="a", append_dim="time", consolidated=False)
    zarr.consolidate_metadata(str(out))


def _scatter(block_len, n_cells, farm_cell_idx, **fields):
    """Build full-grid (block_len, n_cells) arrays, NaN off-farm, from farm-cell values."""
    out = {}
    for name, vals in fields.items():
        a = np.full((block_len, n_cells), np.nan, "float32")
        a[:, farm_cell_idx] = vals
        out[name] = a
    return out


def write_cerra_src(cfg, power_fc, cf_fc, cap_cell, count_cell, farm_cell_idx,
                    lat, lon, dates, n_cells):
    T = dates.size
    cap_t = np.broadcast_to(cap_cell.astype("float32"), (cfg.time_chunk + 1, cap_cell.size))
    cnt_t = np.broadcast_to(count_cell.astype("float32"), (cfg.time_chunk + 1, count_cell.size))

    def blocks(s, e):
        n = e - s
        return _scatter(n, n_cells, farm_cell_idx,
                        power=power_fc[s:e], capacityfactor=cf_fc[s:e],
                        capacity=cap_t[:n], turbinecount=cnt_t[:n],
                        turbmask=np.ones((n, farm_cell_idx.size), "float32"))
    _write_streamed(cfg.out_cerra, blocks, dates, lat, lon, n_cells, cfg.time_chunk)


def write_era5_companion(cfg, P, farms, farm_cell_idx, cerra_lat, cerra_lon180, dates,
                         W, cap_cell, count_cell):
    """Project the farms onto the coarser ERA5 grid so the cutout variable-set matches.

    Each CERRA farm cell maps to its nearest ERA5 cell; power / capacity / turbinecount SUM into
    that ERA5 cell (extensive), capacityfactor = power / capacity. To guarantee the companion is
    not all-NaN (anemoi-datasets' statistics pass needs finite values), power here DROPS NaN
    contributors instead of the strict rule used for the inner grid.
    """
    e = open_dataset(str(cfg.era5_zarr))
    elat = np.asarray(e.latitudes, dtype=float)
    elon180 = to_180(np.asarray(e.longitudes, dtype=float))
    en = elat.size

    coslat = np.cos(np.radians(float(elat.mean())))
    tree = cKDTree(np.c_[elon180 * coslat, elat])
    _, ecell = tree.query(np.c_[cerra_lon180[farm_cell_idx] * coslat,
                                cerra_lat[farm_cell_idx]], k=1)
    ecell = ecell.astype(int)
    e_idx = np.unique(ecell)
    epos = {int(c): i for i, c in enumerate(e_idx)}

    # aggregation matrix: ERA5 cell <- its CERRA farm cells
    A = np.zeros((e_idx.size, farm_cell_idx.size))
    for j, ec in enumerate(ecell):
        A[epos[int(ec)], j] = 1.0

    # power on the CERRA farm cells (drop-NaN so the boundary stays finite), then sum into ERA5
    power_fc = paint_power(P, W, drop_nan=True)                 # (T, C_cerra)
    power_e = np.where(np.isfinite(power_fc), power_fc, 0.0) @ A.T
    reach = (np.isfinite(power_fc).astype(float) @ A.T) > 0
    power_e = np.where(reach, power_e, np.nan).astype(np.float32)

    cap_e = (cap_cell @ A.T).astype("float32")                 # static
    cnt_e = (count_cell @ A.T).astype("float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        cf_e = (power_e / cap_e[None, :]).astype("float32")

    cap_t = np.broadcast_to(cap_e, (cfg.time_chunk + 1, cap_e.size))
    cnt_t = np.broadcast_to(cnt_e, (cfg.time_chunk + 1, cnt_e.size))

    def blocks(s, ee):
        n = ee - s
        return _scatter(n, en, e_idx,
                        power=power_e[s:ee], capacityfactor=cf_e[s:ee],
                        capacity=cap_t[:n], turbinecount=cnt_t[:n],
                        turbmask=np.ones((n, e_idx.size), "float32"))
    _write_streamed(cfg.out_era5, blocks, dates, elat, elon180, en, cfg.time_chunk)
    return e_idx.size, en


def verify_written(out: Path, lat, lon, expect_mask_cells: int) -> None:
    ds = xr.open_zarr(out, consolidated=True)
    mid = ds.sizes["time"] // 2
    n_pow = int(np.isfinite(ds["power"].isel(time=mid).values).sum())
    n_tm = int(np.nansum(ds["turbmask"].isel(time=mid).values))
    cf = ds["capacityfactor"].isel(time=mid).values
    cf_max = float(np.nanmax(cf)) if np.isfinite(cf).any() else np.nan
    aligned = (np.allclose(ds["latitude"].values, np.asarray(lat, "float32"))
               and np.allclose(ds["longitude"].values, np.asarray(lon, "float32")))
    ds.close()
    if n_tm != expect_mask_cells:
        raise ValueError(f"{out.name}: turbmask sum {n_tm} != {expect_mask_cells} expected cells")
    if not aligned:
        raise ValueError(f"{out.name}: written lat/lon do not match the source grid")
    if not (np.isnan(cf_max) or cf_max <= 1.05):
        raise ValueError(f"{out.name}: capacityfactor max {cf_max:.3f} > 1.05")
    print(f"      verified: power non-NaN={n_pow}, turbmask={n_tm}, CF_max={cf_max:.3f}, grid aligned")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-era5", action="store_true", help="skip the ERA5 companion zarr")
    ap.add_argument("--cerra", type=Path, default=None, help="override CERRA zarr path")
    ap.add_argument("--era5", type=Path, default=None, help="override ERA5 zarr path")
    args = ap.parse_args()
    cfg = Config()
    if args.cerra: cfg.cerra_zarr = args.cerra
    if args.era5:  cfg.era5_zarr = args.era5

    print("[1/6] CERRA grid")
    cerra_lat, cerra_lon, cerra_dates = load_cerra_grid(cfg.cerra_zarr)
    cerra_lon180 = to_180(cerra_lon)
    n_cells = cerra_lat.size
    print(f"      {n_cells} cells | {cerra_dates.size} dates "
          f"({cerra_dates[0]:%Y-%m-%d} .. {cerra_dates[-1]:%Y-%m-%d})")

    print("[2/6] turbines -> cells -> capacity-share weights")
    turbines = pd.read_csv(cfg.turbines_csv)
    turbines["cell"] = assign_turbine_cells(turbines, cerra_lat, cerra_lon180)
    power = pd.read_csv(cfg.power_csv, index_col=0, parse_dates=True)
    farms = list(power.columns)
    assert set(turbines["farm"]) == set(farms), "turbines.csv and power_obs.csv farm sets differ"
    farm_cell_idx, W, cap_cell, count_cell = build_weights(turbines, farms)
    shared = int(((W > 0).sum(1) > 1).sum())
    if not np.allclose(W.sum(0), 1.0):
        raise ValueError("capacity shares do not sum to 1 per farm")
    print(f"      {len(turbines)} turbines | {len(farms)} farms | {farm_cell_idx.size} cells "
          f"({shared} shared) | {cap_cell.sum():.1f} MW placed")

    print("[3/6] align observations to the CERRA clock")
    idx = power.index.tz_convert(None) if power.index.tz else power.index
    P = power.set_axis(idx).reindex(cerra_dates)[farms].to_numpy(dtype=float)   # (T, F)
    n_match = int(np.isfinite(P).any(1).sum())
    if n_match == 0:
        raise ValueError("no observation timestamp matches the CERRA clock")
    print(f"      {n_match} of {cerra_dates.size} CERRA times have >=1 reporting farm")

    print("[4/6] paint power (capacity-weighted SUM, strict NaN)")
    power_fc = paint_power(P, W)                                # (T, C)
    with np.errstate(invalid="ignore", divide="ignore"):
        cf_fc = (power_fc / cap_cell[None, :].astype("float32")).astype("float32")
    cf_max = float(np.nanmax(cf_fc))
    valid = np.isfinite(power_fc).mean()
    print(f"      painted valid fraction {valid:.4f} | CF max {cf_max:.3f}")
    if cf_max > 1.05:
        raise ValueError(f"capacity factor exceeds 1.05 ({cf_max:.3f}) -- capacity/obs mismatch")

    print(f"[5/6] write {cfg.out_cerra.name}")
    write_cerra_src(cfg, power_fc, cf_fc, cap_cell, count_cell, farm_cell_idx,
                    cerra_lat, cerra_lon, cerra_dates, n_cells)
    verify_written(cfg.out_cerra, cerra_lat, cerra_lon, farm_cell_idx.size)

    if not args.no_era5:
        print(f"[6/6] write {cfg.out_era5.name}")
        n_ef, en = write_era5_companion(cfg, P, farms, farm_cell_idx,
                                        cerra_lat, cerra_lon180, cerra_dates,
                                        W, cap_cell, count_cell)
        e = open_dataset(str(cfg.era5_zarr))
        verify_written(cfg.out_era5, np.asarray(e.latitudes, float),
                       to_180(np.asarray(e.longitudes, float)), n_ef)
        print(f"      ERA5 companion: {n_ef} farm cells of {en}")
    else:
        print("[6/6] ERA5 companion skipped")

    print("done.")


if __name__ == "__main__":
    main()
