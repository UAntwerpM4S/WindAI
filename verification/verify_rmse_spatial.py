"""
Spatial RMSE per lead time against CERRA truth.
Output: one NetCDF per (variable, model) with shape (lead_time, cell).
RMSE at each cell is computed across all common init times.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

# -------------------- SETTINGS --------------------
TARGET_VARS = ["ws10", "ws100"]

FORECAST_DIRS = {
    "NoPowerTF":      Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerTF"),
    "NoPowerGT":      Path("/mnt/weatherloss/WindPower/inference/EGU/NoPowerGT"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerGT"),
    "VanillaPowerTF": Path("/mnt/weatherloss/WindPower/inference/EGU/VanillaPowerTF"),
}

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(0, 39, 3))
OUT_DIR    = Path("EGU_spatial_rmse")
# --------------------------------------------------

FORECAST_FILE_RE = re.compile(r"forecast_(\d{14})")


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(
        FORECAST_FILE_RE.search(path.name).group(1),
        format="%Y%m%d%H%M%S", utc=True,
    )


def list_files(d: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, Path]:
    return {
        parse_init(f): f
        for f in sorted(d.glob("forecast_*.nc"))
        if start <= parse_init(f) <= end
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds_cerra    = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars  = list(ds_cerra.attrs["variables"])
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")
    n_cells     = ds_cerra.sizes["cell"]

    # Fast lookup: valid_time ISO → cerra time index
    cerra_date_to_idx = {d.isoformat(): i for i, d in enumerate(cerra_dates)}

    lead_hours_sorted = sorted(set(LEAD_HOURS))
    lead_to_idx       = {lh: i for i, lh in enumerate(lead_hours_sorted)}
    n_leads           = len(lead_to_idx)

    file_maps = {label: list_files(path, INIT_START, INIT_END)
                 for label, path in FORECAST_DIRS.items()}

    common_inits = sorted(
        set.intersection(*(set(m) for m in file_maps.values()))
    )
    print(f"Common init times: {len(common_inits)}")

    for var in TARGET_VARS:
        if var not in cerra_vars:
            print(f"WARNING: '{var}' not in CERRA, skipping.")
            continue
        var_idx = cerra_vars.index(var)
        print(f"\n=== Variable: {var} ===")

        # Preload all needed CERRA timesteps for this variable
        needed_valid_times = sorted({
            init + pd.Timedelta(hours=lh)
            for init in common_inits
            for lh in LEAD_HOURS
            if (init + pd.Timedelta(hours=lh)).isoformat() in cerra_date_to_idx
        })
        needed_cerra_idxs = [cerra_date_to_idx[t.isoformat()] for t in needed_valid_times]
        print(f"  Preloading {len(needed_cerra_idxs)} CERRA timesteps...")
        cerra_bulk = ds_cerra["data"].isel(
            time=needed_cerra_idxs,
            variable=var_idx,
            ensemble=0,
        ).values  # (n_needed, n_cells)
        cerra_cache = {
            t.isoformat(): cerra_bulk[i]
            for i, t in enumerate(needed_valid_times)
        }
        del cerra_bulk
        print("  CERRA preload done.")

        for label in FORECAST_DIRS:
            print(f"  Processing: {label}")

            sum_sq = np.zeros((n_leads, n_cells), dtype=np.float64)
            count  = np.zeros((n_leads, n_cells), dtype=np.int64)

            for n_init, init in enumerate(common_inits):
                if n_init % 500 == 0:
                    print(f"    {n_init}/{len(common_inits)}...", flush=True)

                try:
                    with h5py.File(str(file_maps[label][init]), "r") as f:
                        tv  = f["time"]
                        raw = nc4.num2date(
                            tv[:],
                            tv.attrs["units"].decode(),
                            tv.attrs.get("calendar", b"standard").decode(),
                        )
                        fc_times = [pd.Timestamp(str(t)).tz_localize("UTC") for t in raw]
                        var_all  = f[var][:, :]  # (n_times, n_cells) — plain slice

                except Exception as e:
                    print(f"    Skipping {file_maps[label][init].name}: {e}")
                    continue

                # Pure numpy from here — no file handles open
                for t_i, fc_time in enumerate(fc_times):
                    lh = int((fc_time - init).total_seconds() / 3600)
                    if lh not in lead_to_idx:
                        continue
                    viso = fc_time.isoformat()
                    if viso not in cerra_cache:
                        continue

                    fc     = var_all[t_i]
                    tr     = cerra_cache[viso]
                    sq_err = (fc - tr) ** 2
                    finite = np.isfinite(sq_err)
                    l_idx  = lead_to_idx[lh]
                    sum_sq[l_idx] += np.where(finite, sq_err, 0.0)
                    count[l_idx]  += finite.astype(np.int64)

            with np.errstate(invalid="ignore"):
                spatial_rmse = np.where(count > 0, np.sqrt(sum_sq / count), np.nan)
            spatial_rmse = spatial_rmse.astype(np.float32)

            leads_arr = np.array(lead_hours_sorted)
            ds_out = xr.Dataset(
                {"rmse": (("lead_time", "cell"), spatial_rmse)},
                coords={
                    "lead_time": leads_arr,
                    "latitude":  ("cell", ds_cerra["latitudes"].values.astype(np.float32)),
                    "longitude": ("cell", ds_cerra["longitudes"].values.astype(np.float32)),
                },
                attrs={
                    "model":    label,
                    "variable": var,
                    "n_inits":  len(common_inits),
                },
            )
            out_path = OUT_DIR / f"spatial_rmse_{var}_{label}.nc"
            ds_out.to_netcdf(out_path)
            print(f"  Saved: {out_path}")

    ds_cerra.close()
    print("\nDone.")


if __name__ == "__main__":
    main()