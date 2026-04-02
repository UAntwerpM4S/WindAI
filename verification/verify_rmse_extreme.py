"""
Spatial RMSE per lead time, computed only over the top 5% most extreme
observed (CERRA truth) values of the variable at each valid time.
Output: one NetCDF per (variable, model) with shape (lead_time, cell).
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
LEAD_HOURS = list(range(0, 73, 3))
OUT_DIR    = Path("CI_spatial_rmse_extreme")
EXTREME_PERCENTILE = 95  # top 5% most extreme
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

        # --- compute per-cell threshold over all valid times ---
        # collect all valid times that appear in common inits
        valid_times = sorted({
            init.replace(tzinfo=None) + pd.Timedelta(hours=lh)
            for init in common_inits
            for lh in LEAD_HOURS
        } & cerra_time_set)

        print(f"  Computing {EXTREME_PERCENTILE}th percentile threshold "
              f"over {len(valid_times)} valid times...")

        # shape: (n_valid_times, n_cells)
        truth_all = np.stack([
            ds_cerra["data"].isel(
                time=int(np.where(cerra_dates == vt)[0][0]),
                variable=var_idx,
                ensemble=0,
            ).values
            for vt in valid_times
        ])  # (n_valid_times, n_cells)

        # per-cell threshold: value above which we consider "extreme"
        threshold = np.nanpercentile(truth_all, EXTREME_PERCENTILE, axis=0)  # (n_cells,)
        print(f"  Threshold range: {threshold.min():.3f} – {threshold.max():.3f}")
        print(f"  Threshold mean across cells: {threshold.mean():.3f} m/s")
        del truth_all  # free memory

        # count is model-independent (only depends on truth vs threshold)
        # so compute it once using any one forecast for the timing/lead structure
        first_label = next(iter(FORECAST_DIRS))
        count = np.zeros((n_leads, n_cells), dtype=np.int64)

        for init in common_inits:
            with xr.open_dataset(file_maps[first_label][init]) as ds_fc:
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
                    tr = ds_cerra["data"].isel(
                        time=t_idx, variable=var_idx, ensemble=0
                    ).values                              # (n_cells,)

                    extreme_mask = tr >= threshold        # (n_cells,)
                    l_idx = lead_to_idx[lh]
                    count[l_idx] += extreme_mask.astype(np.int64)

        avg_count_per_cell = count.mean(axis=0)   # mean over leads -> (n_cells,)
        print(f"  Avg count per cell (mean over leads): {avg_count_per_cell.mean():.1f}")
        print(f"  Avg count per cell (min  over leads): {avg_count_per_cell.min():.1f}")
        print(f"  Count at lead=3h  — mean/min over cells: "
              f"{count[lead_to_idx[3]].mean():.1f} / {count[lead_to_idx[3]].min()}")
        print(f"  Count at lead=72h — mean/min over cells: "
              f"{count[lead_to_idx[72]].mean():.1f} / {count[lead_to_idx[72]].min()}")

    ds_cerra.close()
    print("\nDone.")


if __name__ == "__main__":
    main()