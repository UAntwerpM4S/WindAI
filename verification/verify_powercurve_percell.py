"""
Per-cell MAE/RMSE vs lead time, one figure per unique Belgian CERRA cell.
Produces 7 plots (4 single-farm cells, 3 shared-farm cells).

Same methodology as verify_powercurve.py (model 'power' + power-curve vs CERRA
truth), but the error is kept PER CELL instead of summed over cells. Each cell
is normalised by its OWN installed capacity.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import netCDF4 as nc4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# Reuse everything from the main script (config + helpers)
from verify_powercurve import (
    FORECAST_DIRS, COLOR_MAP, CERRA_PATH, SPECS_PATH, LEAD_HOURS,
    parse_init_time, load_belgian_metadata, get_belgian_cerra_indices,
    power_curve, load_specs, build_counts_matrix, get_common_files, _safe,
)

OUT_DIR = Path("WindAI_power_percell")


def read_forecast_percell(nc_path, init, fc_indices, valid_iso_set, fc_var,
                          type_order=None, specs=None, counts_matrix=None):
    """Return {lead_hour: per-cell MW vector} for one forecast file."""
    result = {}
    with h5py.File(str(nc_path), "r") as f:
        if fc_var == "power" and "power" not in f:
            return result
        tv  = f["time"]
        raw = nc4.num2date(
            tv[:], tv.attrs["units"].decode(),
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
            result[lh] = data_sel[fc_time_to_idx[viso]]      # (n_cells,)
    return result


def collect_percell(files, fc_var, fc_indices, cerra_obs_cache, valid_iso_set, n_cells, **kw):
    """Accumulate per-(lead, cell) errors -> return MAE/RMSE/Bias arrays (n_leads, n_cells)."""
    lead_to_idx = {lh: i for i, lh in enumerate(LEAD_HOURS)}
    n_leads = len(LEAD_HOURS)
    abs_s  = np.zeros((n_leads, n_cells))
    sq_s   = np.zeros((n_leads, n_cells))
    bias_s = np.zeros((n_leads, n_cells))
    cnt    = np.zeros((n_leads, n_cells), dtype=np.int64)

    for k, fpath in enumerate(files):
        if k % 500 == 0:
            print(f"    {k}/{len(files)}", flush=True)
        init = parse_init_time(fpath)
        try:
            fc = read_forecast_percell(fpath, init, fc_indices, valid_iso_set, fc_var, **kw)
        except Exception as e:
            print(f"    Skipped {fpath.name}: {e}")
            continue
        for lh, fc_vec in fc.items():
            obs_vec = cerra_obs_cache[(init + pd.Timedelta(hours=lh)).isoformat()]
            err = fc_vec - obs_vec
            fin = np.isfinite(err)
            l = lead_to_idx[lh]
            abs_s[l]  += np.where(fin, np.abs(err), 0.0)
            sq_s[l]   += np.where(fin, err ** 2, 0.0)
            bias_s[l] += np.where(fin, err, 0.0)
            cnt[l]    += fin

    with np.errstate(invalid="ignore", divide="ignore"):
        mae  = np.where(cnt > 0, abs_s / cnt, np.nan)
        rmse = np.where(cnt > 0, np.sqrt(sq_s / cnt), np.nan)
        bias = np.where(cnt > 0, bias_s / cnt, np.nan)
    return mae, rmse, bias


def main() -> None:
    ds_cerra    = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars  = list(ds_cerra.attrs["variables"])
    cerra_lat   = ds_cerra["latitudes"].values
    cerra_lon   = ds_cerra["longitudes"].values
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")
    cerra_date_to_idx = {d: i for i, d in enumerate(cerra_dates)}
    cerra_var_idx = cerra_vars.index("power")

    # ALL Belgian farms (every cell, single + shared)
    meta = load_belgian_metadata()
    cerra_keep = get_belgian_cerra_indices(meta, cerra_lat, cerra_lon)
    specs = load_specs(SPECS_PATH)
    type_order, counts_matrix, _ = build_counts_matrix(meta)
    n_cells = len(cerra_keep)

    # per-cell capacity + labels, in the SAME order as cerra_keep (be_unique order)
    be_unique    = meta.drop_duplicates(subset=["cerra_grid_lat", "cerra_grid_lon"])
    cap_by_cell  = meta.groupby(["cerra_grid_lat", "cerra_grid_lon"])["capacity_mw"].sum()
    farm_by_cell = meta.groupby(["cerra_grid_lat", "cerra_grid_lon"])["farm"].apply(list)
    cell_caps, cell_labels, cell_nfarms = [], [], []
    for r in be_unique.itertuples():
        key = (r.cerra_grid_lat, r.cerra_grid_lon)
        farms = list(farm_by_cell[key])
        cell_caps.append(float(cap_by_cell[key]))
        cell_labels.append(" + ".join(map(str, farms)))
        cell_nfarms.append(len(farms))
    cell_caps = np.array(cell_caps)
    print(f"Cells: {n_cells}  "
          f"({sum(n == 1 for n in cell_nfarms)} single-farm, "
          f"{sum(n > 1 for n in cell_nfarms)} shared)")

    common_files = get_common_files(FORECAST_DIRS)
    common_inits = [parse_init_time(f) for f in next(iter(common_files.values()))]
    fc_indices   = cerra_keep

    needed_valid = sorted({
        init + pd.Timedelta(hours=lh)
        for init in common_inits for lh in LEAD_HOURS
        if (init + pd.Timedelta(hours=lh)) in cerra_date_to_idx
    })
    print(f"Preloading CERRA ({len(needed_valid)} timesteps × {n_cells} cells)...")
    cerra_bulk = ds_cerra["data"].isel(
        time=[cerra_date_to_idx[t] for t in needed_valid],
        variable=cerra_var_idx, ensemble=0,
    ).values[:, cerra_keep]
    cerra_obs_cache = {t.isoformat(): cerra_bulk[i] for i, t in enumerate(needed_valid)}
    valid_iso_set   = {iso for iso, arr in cerra_obs_cache.items() if not np.any(np.isnan(arr))}
    ds_cerra.close()
    del cerra_bulk

    # accumulate per model (power + power-curve), each (n_leads, n_cells)
    results = {}
    for label, files in common_files.items():
        print(f"\nProcessing {label} ...")
        with h5py.File(str(files[0]), "r") as f:
            has_power = "power" in f
        if has_power:
            print("  power ...")
            mae, rmse, bias = collect_percell(files, "power", fc_indices, cerra_obs_cache, valid_iso_set, n_cells)
            results[label] = {"mae": mae, "rmse": rmse, "bias": bias}
        print("  power-curve ...")
        mae, rmse, bias = collect_percell(files, "ws100", fc_indices, cerra_obs_cache, valid_iso_set, n_cells,
                                          type_order=type_order, specs=specs, counts_matrix=counts_matrix)
        results[f"{label}-powercurve"] = {"mae": mae, "rmse": rmse, "bias": bias}

    # one figure per cell
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leads = np.array(LEAD_HOURS)
    for ci in range(n_cells):
        cap = cell_caps[ci]
        fig, (ax_mae, ax_rmse, ax_bias) = plt.subplots(1, 3, figsize=(20, 5))
        tag = "single-farm" if cell_nfarms[ci] == 1 else f"{cell_nfarms[ci]} farms (shared)"
        fig.suptitle(f"{cell_labels[ci]}   [{tag}]   capacity = {cap:.0f} MW", fontsize=12)

        for label, d in results.items():
            base  = label.replace("-powercurve", "")
            is_pc = "powercurve" in label.lower()
            color = COLOR_MAP.get(base, "gray")
            kw = dict(lw=1.6, ls="--" if is_pc else "-",
                      marker="" if is_pc else "o", ms=3, color=color, label=label)
            ax_mae.plot(leads,  d["mae"][:, ci]  / cap * 100.0, **kw)
            ax_rmse.plot(leads, d["rmse"][:, ci] / cap * 100.0, **kw)
            ax_bias.plot(leads, d["bias"][:, ci] / cap * 100.0, **kw)   # signed (fc - obs)

        for ax, t in zip((ax_mae, ax_rmse, ax_bias), ("MAE", "RMSE", "Bias (fc − obs)")):
            ax.set_title(t, fontsize=11)
            ax.set_xlabel("Lead time [hours]")
            ax.set_ylabel("[% of cell capacity]")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=7, framealpha=0.8)
        ax_bias.axhline(0, color="black", lw=0.8, ls=":")

        fig.tight_layout()
        out = OUT_DIR / f"cell_{ci:02d}_{_safe(cell_labels[ci])}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
