"""
Spatial RMSE per lead time against CERRA truth.
Output: one NetCDF per (variable, model) with shape (lead_time, cell).
RMSE at each cell is computed across all common init times.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# -------------------- SETTINGS --------------------
TARGET_VARS = ["ws10", "ws100"]

FORECAST_DIRS = {
    "GNNCI": Path("/mnt/weatherloss/WindPower/inference/CI/GNNCI"),
    "GTCI":  Path("/mnt/weatherloss/WindPower/inference/CI/GTCI"),
    "TFCI":  Path("/mnt/weatherloss/WindPower/inference/CI/TFCI"),
}

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/Cerra_A.zarr")
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
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values)
    n_cells     = ds_cerra.sizes["cell"]

    lead_to_idx = {lh: i for i, lh in enumerate(sorted(set(LEAD_HOURS)))}
    n_leads     = len(lead_to_idx)

    file_maps = {label: list_files(path, INIT_START, INIT_END)
                 for label, path in FORECAST_DIRS.items()}

    common_inits = sorted(
        set.intersection(*(set(m) for m in file_maps.values()))
    )
    print(f"Common init times: {len(common_inits)}")

    cerra_time_set = set(cerra_dates)

    for var in TARGET_VARS:
        if var not in cerra_vars:
            print(f"WARNING: '{var}' not in CERRA, skipping.")
            continue
        var_idx = cerra_vars.index(var)
        print(f"\n=== Variable: {var} ===")

        for label in FORECAST_DIRS:
            print(f"  Processing: {label}")

            # accumulate squared errors and counts per (lead, cell)
            sum_sq = np.zeros((n_leads, n_cells), dtype=np.float64)
            count  = np.zeros((n_leads, n_cells), dtype=np.int64)

            for init in common_inits:
                with xr.open_dataset(file_maps[label][init]) as ds_fc:
                    fc_times = pd.to_datetime(ds_fc["time"].values)
                    leads_h  = (
                        (fc_times - init.replace(tzinfo=None)) / np.timedelta64(1, "h")
                    ).astype(int)

                    for i, lh in enumerate(leads_h):
                        if lh not in lead_to_idx:
                            continue
                        if fc_times[i] not in cerra_time_set:
                            continue

                        t_idx = int(np.where(cerra_dates == fc_times[i])[0][0])

                        fc = ds_fc[var].isel(time=i).values        # (n_cells,)
                        tr = ds_cerra["data"].isel(
                            time=t_idx, variable=var_idx, ensemble=0
                        ).values                                    # (n_cells,)

                        sq_err = (fc - tr) ** 2
                        finite = np.isfinite(sq_err)
                        l_idx  = lead_to_idx[lh]
                        sum_sq[l_idx] += np.where(finite, sq_err, 0.0)
                        count[l_idx]  += finite.astype(np.int64)

            with np.errstate(invalid="ignore"):
                spatial_rmse = np.where(count > 0, np.sqrt(sum_sq / count), np.nan)
            spatial_rmse = spatial_rmse.astype(np.float32)  # (n_leads, n_cells)

            leads_arr = np.array(sorted(lead_to_idx))
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