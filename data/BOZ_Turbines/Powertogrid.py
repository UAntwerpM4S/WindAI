#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# =========================
# CONFIG (edit paths here)
# =========================
BASE = "/mnt/data/weatherloss/WindPower/data/BOZ_Turbines"
COORD_DIR = os.path.join(BASE, "coordinates")
ENTSOE_CSV = os.path.join(BASE, "windpowerdata", "BE_offshore_per_unit_3H_meanMW_2020-01_to_2025-07.csv")
CERRA_ZARR = "/mnt/data/weatherloss/WindPower/data/Cerra.zarr"


CSV_TO_FARM = {
    "Belwind Phase 1":                 "Belwind",
    "Mermaid Offshore WP":             "Mermaid",
    "Nobelwind Offshore Windpark":     "Nobelwind",
    "Norther Offshore WP":             "Norther",
    "Northwester 2":                   "Northwester2",
    "Northwind":                       "Northwind",
    "Rentel Offshore WP":              "Rentel",
    "Seastar Offshore WP":             "Seastar",
    "Thorntonbank - C-Power - Area NE":"CPower_NE",
    "Thorntonbank - C-Power - Area SW":"CPower_SW",
}

# =========================
# 1) Load CERRA grid (lon/lat + time)
# =========================
ds = xr.open_zarr(CERRA_ZARR, consolidated=True)
lon2d = np.asarray(ds["longitude"])
lat2d = np.asarray(ds["latitude"])
ny, nx = ds.sizes["y"], ds.sizes["x"]
cerra_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
print(f"[CERRA] grid: ny={ny}, nx={nx}; times: {cerra_times.min()} → {cerra_times.max()}")

# KDTree for nearest grid cell queries
pts = np.c_[lon2d.ravel(), lat2d.ravel()]
tree = cKDTree(pts)

# =========================
# 2) Load turbine coordinates and split CPower
# =========================
_HUB_REGEX = re.compile(
    r"(?:^|\b)(?:OSS|OHVS|SUB-?STATION|OFFSHORE\s+HIGH\s+VOLTAGE|HUB|PLATFORM)(?:\b|$)",
    re.I,
)
_LETTER_NUM = re.compile(r"^\s*([A-J])\s*-?\s*(\d+)\s*$", re.I)

def cpower_subfarm(name: str) -> str:
    m = _LETTER_NUM.match(str(name).strip())
    if not m:
        return "CPower_unknown"
    letter = m.group(1).upper()
    n = int(m.group(2))
    if letter in {"A","B","C"}:
        return "CPower_SW"
    if letter == "D":
        return "CPower_SW" if 0 <= n <= 8 else "CPower_NE"
    if letter in {"E","F","G","H","I","J"}:
        return "CPower_NE"
    return "CPower_unknown"

def load_all_turbines(coord_dir):
    rows = []
    for csv in glob.glob(os.path.join(coord_dir, "*_turbines_coords.csv")):
        farm = os.path.basename(csv).replace("_turbines_coords.csv", "")
        df = pd.read_csv(csv)

        cols = {c.lower().strip(): c for c in df.columns}
        name_col = cols.get("name", next((c for c in df.columns if c.lower().startswith("name")), None))
        lon_col  = cols.get("longitude", next((c for c in df.columns if "lon" in c.lower()), None))
        lat_col  = cols.get("latitude", next((c for c in df.columns if "lat" in c.lower()), None))
        if not (name_col and lon_col and lat_col):
            raise ValueError(f"Missing columns in {csv}. Found: {list(df.columns)}")

        sub = df[[name_col, lon_col, lat_col]].copy()
        sub.columns = ["name","lon","lat"]
        sub = sub[~sub["name"].astype(str).str.contains(_HUB_REGEX)]
        sub["farm"] = farm
        rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["name","lon","lat","farm"])

turbines = load_all_turbines(COORD_DIR)
turbines["farm_split"] = turbines["farm"]
mask_cp = turbines["farm"].str.fullmatch("CPower", case=False, na=False)
turbines.loc[mask_cp, "farm_split"] = turbines.loc[mask_cp, "name"].map(cpower_subfarm)

# Drop rare unknown labels
turbines_use = turbines[turbines["farm_split"] != "CPower_unknown"].copy()

# =========================
# 3) Map each farm to nearest CERRA cell (centroid of turbines)
# =========================
farm_key = "farm_split"
centroids = (turbines_use.groupby(farm_key)[["lon","lat"]]
             .mean()
             .rename(columns={"lon":"farm_lon","lat":"farm_lat"}))

# nearest grid cell
dist, idx = tree.query(centroids[["farm_lon","farm_lat"]].values, k=1)
j = (idx % nx).astype(int)
i = (idx // nx).astype(int)

farm_cells = centroids.copy()
farm_cells["y"] = i
farm_cells["x"] = j
farm_cells["lon_cell"] = lon2d[i, j]
farm_cells["lat_cell"] = lat2d[i, j]
farm_cells["deg_error"] = np.hypot(farm_cells["lon_cell"]-farm_cells["farm_lon"],
                                   farm_cells["lat_cell"]-farm_cells["farm_lat"])
farm_cells = farm_cells.reset_index().rename(columns={farm_key: "farm"})

print("[FARMS] mapped:")
print(farm_cells[["farm","y","x","deg_error"]].sort_values("farm").to_string(index=False))

# one-hot mask (farm,y,x)
farms = farm_cells["farm"].tolist()
mask_arr = np.zeros((len(farms), ny, nx), dtype="float32")
lookup = {row["farm"]: (int(row["y"]), int(row["x"])) for _, row in farm_cells.iterrows()}
for k, f in enumerate(farms):
    ii, jj = lookup[f]
    mask_arr[k, ii, jj] = 1.0

farm_mask = xr.DataArray(
    mask_arr, coords={"farm": farms, "y": ds["y"], "x": ds["x"]},
    dims=("farm","y","x"), name="farm_mask_onehot"
)

# =========================
# 4) Load ENTSO-E 3H power & align to CERRA time
# =========================
dfp_raw = pd.read_csv(ENTSOE_CSV, index_col=0, parse_dates=True)
# ensure tz-aware UTC
# ensure UTC, then drop timezone to make times naive (matches CERRA)
if dfp_raw.index.tz is not None:
    dfp_raw.index = dfp_raw.index.tz_convert("UTC").tz_localize(None)
else:
    dfp_raw.index = dfp_raw.index.tz_localize(None)  # assume already UTC
# keep only mapped columns, rename to canonical farm names
keep_cols = [c for c in dfp_raw.columns if c in CSV_TO_FARM]
dfp = dfp_raw[keep_cols].rename(columns=CSV_TO_FARM)
# coerce numeric
dfp = dfp.apply(pd.to_numeric, errors="coerce")
# align to CERRA 3H ticks
dfp_aligned = dfp.reindex(cerra_times)
# keep only farms we have cells for
farms_in_csv = sorted(set(farms) & set(dfp_aligned.columns))
dfp_aligned = dfp_aligned[farms_in_csv].astype("float32")

print(f"[ENTSO-E] aligned shape: {dfp_aligned.shape}, farms: {farms_in_csv[:5]}{'...' if len(farms_in_csv)>5 else ''}")
print(f"[ENTSO-E] time span: {dfp_aligned.index.min()} → {dfp_aligned.index.max()}")

# =========================
# 5) Scatter farm series to grid (sum co-located farms), NaN elsewhere
# =========================
P_farm = xr.DataArray(
    dfp_aligned.to_numpy(dtype="float32"),
    coords={"time": dfp_aligned.index, "farm": farms_in_csv},
    dims=("time","farm"),
    name="farm_power_series_MW",
)
farm_mask_sub = farm_mask.sel(farm=P_farm.farm)

# dot: (time,farm) · (farm,y,x) -> (time,y,x)
power_cell = xr.dot(P_farm.fillna(0.0), farm_mask_sub, dims="farm").astype("float32").rename("power")

# 2D mask of turbine locations; NaN elsewhere
mask2d_bool = (farm_mask_sub.sum(dim="farm") > 0)
mask_da = xr.where(mask2d_bool, 1.0, np.nan).astype("float32").rename("mask")

# Apply NaN outside turbine cells
power_on_cells = power_cell.where(mask2d_bool).astype("float32")

# Attach CERRA coords (already aligned)
power_on_cells = power_on_cells.assign_coords(time=P_farm.time, y=ds["y"], x=ds["x"])
mask_da = mask_da.assign_coords(y=ds["y"], x=ds["x"])

# attrs
power_on_cells.attrs.update({
    "long_name": "Offshore wind power mapped to CERRA grid (sum across co-located farms)",
    "units": "MW",
    "note": "NaN outside turbine cells; time in UTC (3-hourly).",
})
mask_da.attrs.update({
    "long_name": "Turbine-location mask (2D)",
    "values": "1 at farm cells, NaN elsewhere",
})

# =========================
# 6) Sanity checks
# =========================
# a) overlap with CERRA
overlap_rows = int(dfp_aligned.notna().any(axis=1).sum())
print(f"[CHECK] rows with any farm data: {overlap_rows}")

# b) totals compare at a few times
def check_time(idx):
    s = float(np.nansum(P_farm.isel(time=idx).values))
    g = float(np.nansum(power_on_cells.isel(time=idx).values))
    print(f"[CHECK] t[{idx}] {pd.Timestamp(P_farm.time.values[idx])}  series_sum={s:.3f} MW | grid_sum={g:.3f} MW")

for ti in [0, min(100, len(P_farm.time)-1), min(500, len(P_farm.time)-1), len(P_farm.time)-1]:
    check_time(ti)

# c) nonzero cells (should equal number of farm cells that have any data at some time)
nz_cells = int((power_on_cells.notnull().any(dim="time")).sum().item())
print(f"[CHECK] cells with any non-NaN power over time: {nz_cells}")

# =========================
# 7) Write into existing CERRA Zarr (append variables)
# =========================
power_ds = xr.Dataset({"power": power_on_cells, "mask": mask_da})
# chunking
encoding = {
    "power": {"chunks": (24, ny, nx)},
    "mask":  {"chunks": (ny, nx)},
}
power_ds.to_zarr(CERRA_ZARR, mode="a", consolidated=True, encoding=encoding)
print(f"[WRITE] Appended variables ['power','mask'] to {CERRA_ZARR}")
