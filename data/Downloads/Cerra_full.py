#!/usr/bin/env python3
import os, time, queue, threading, calendar
import cdsapi
import numpy as np
import xarray as xr
import cfgrib

# ===================== Config =====================
OUT_ROOT = "/mnt/data/weatherloss/WindPower/data/cerra_boz"
RAW_DIR  = os.path.join(OUT_ROOT, "raw_grib")
NC_DIR   = os.path.join(OUT_ROOT, "nc_boz")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(NC_DIR,  exist_ok=True)

# Years & cadence
YEARS  = ["2022"]   # 2020..2025 inclusive
MONTHS = [f"{m:02d}" for m in range(1, 13)]
TIMES  = ["00:00","03:00","06:00","09:00","12:00","15:00","18:00","21:00"]

# Training domain (deg)
LAT_MIN, LAT_MAX = 49.0, 56.0
LON_MIN, LON_MAX = -6.0, 10.0

# Workers (crop/convert)
N_PROCESSORS = 2

# Silence xarray combine warning and opt into new defaults
xr.set_options(use_new_combine_kwarg_defaults=True)

# ===================== Helpers =====================
def days_in_month(year_str, month_str):
    y = int(year_str); m = int(month_str)
    last = calendar.monthrange(y, m)[1]  # correct 28/29/30/31
    return [f"{d:02d}" for d in range(1, last + 1)]

def index_box(ds, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX):
    """Return y/x slices for the given lat/lon box using dataset 2D coordinates."""
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    # Wrap 0..360 → -180..180 if needed (CERRA often 0..360)
    if np.any(lon > 180):
        lon = ((lon + 180) % 360) - 180
    # Order bounds
    if lat_min > lat_max: lat_min, lat_max = lat_max, lat_min
    if lon_min > lon_max: lon_min, lon_max = lon_max, lon_min
    # Mask & slice
    m = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    if not np.any(m):
        raise ValueError("No grid points in specified box.")
    yy, xx = np.where(m)
    return slice(int(yy.min()), int(yy.max()) + 1), slice(int(xx.min()), int(xx.max()) + 1)

# ===================== Crop & Convert =====================
def crop_grib_to_boz_nc(grib_path, nc_out_path):
    """Open a GRIB (may contain multiple messages), crop all to the same y/x window, merge, save NetCDF."""
    if os.path.exists(nc_out_path) and os.path.getsize(nc_out_path) > 0:
        print(f"✔ exists: {nc_out_path}")
        return
    print(f"… opening GRIB: {grib_path}")

    # Open all messages with persistent cfgrib index (fast on re-runs)
    ds_list = cfgrib.open_datasets(grib_path, backend_kwargs={"decode_timedelta": True})

    # Compute y/x slice once from the first message
    ys, xs = index_box(ds_list[0])

    cropped = []
    for ds in ds_list:
        dsc = ds.isel(y=ys, x=xs)
        # Drop singleton heightAboveGround (cfgrib quirk for single-level height vars)
        if "heightAboveGround" in dsc.coords and dsc.coords["heightAboveGround"].size == 1:
            dsc = dsc.reset_coords("heightAboveGround", drop=True).squeeze()
        cropped.append(dsc)

    # Align along common coords (e.g., time) then merge variables
    aligned = xr.align(*cropped, join="inner")
    ds_out = xr.merge(aligned, compat="override", combine_attrs="drop")

    # Compressed NetCDF (float32)
    comp = dict(zlib=True, complevel=4, shuffle=True, dtype="float32")
    enc = {v: comp for v in ds_out.data_vars}

    tmp = nc_out_path + ".tmp"
    ds_out.to_netcdf(tmp, engine="netcdf4", encoding=enc)
    os.replace(tmp, nc_out_path)
    print(f"✔ saved: {nc_out_path}")

# ===================== CDS download (single thread) =====================
client = cdsapi.Client()

def cds_retrieve_with_retry(dataset, request, target, max_tries=5):
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"✔ raw exists: {target}")
        return
    tries = 0
    while True:
        tries += 1
        try:
            print(f"→ retrieving {dataset} -> {target}")
            client.retrieve(dataset, request, target)
            print(f"✔ downloaded: {target}")
            return
        except Exception as e:
            if tries >= max_tries:
                raise
            wait = min(60 * tries, 300)
            print(f"CDS error ({e}). Retrying in {wait}s [{tries}/{max_tries}]…")
            time.sleep(wait)

# ===================== Queues & Workers =====================
dl_queue = queue.Queue()
proc_queue = queue.Queue()

def downloader():
    while True:
        item = dl_queue.get()
        if item is None:
            break
        dataset, req, grib_path = item
        try:
            cds_retrieve_with_retry(dataset, req, grib_path)
            proc_queue.put((grib_path,))
        finally:
            dl_queue.task_done()

def processor():
    while True:
        item = proc_queue.get()
        if item is None:
            break
        (grib_path,) = item
        base = os.path.basename(grib_path).replace(".grib", "_BOZ.nc")
        nc_out = os.path.join(NC_DIR, base)
        try:
            crop_grib_to_boz_nc(grib_path, nc_out)
        finally:
            proc_queue.task_done()

# ===================== Request builders (NO 'area' for CERRA) =====================
def submit_static_once():
    dataset = "reanalysis-cerra-single-levels"
    req = {
        "variable": ["land_sea_mask", "orography"],  # remove "orography" if not needed
        "level_type": "surface_or_atmosphere",
        "data_type": "reanalysis",
        "product_type": "analysis",
        "format": "grib",
        "year": "2020", "month": "01", "day": "01", "time": "00:00",
    }
    grib_path = os.path.join(RAW_DIR, "cerra_static.grib")
    dl_queue.put((dataset, req, grib_path))

def submit_height_month(y, m):
    dataset = "reanalysis-cerra-height-levels"
    req = {
        "variable": ["wind_speed", "wind_direction"],
        "height_level": ["50_m","100_m","150_m","200_m"],
        "data_type": "reanalysis",
        "product_type": "analysis",
        "format": "grib",
        "year": y, "month": m,
        "day": days_in_month(y, m),
        "time": TIMES,
    }
    grib_path = os.path.join(RAW_DIR, f"cerra_height_{y}_{m}.grib")
    dl_queue.put((dataset, req, grib_path))

def submit_pressure_month(y, m):
    dataset = "reanalysis-cerra-pressure-levels"
    req = {
        "variable": [
            "geopotential","relative_humidity","temperature",
            "u_component_of_wind","v_component_of_wind"
        ],
        "pressure_level": ["500","600","700","750","800","850","900","950","1000"],
        "data_type": "reanalysis",
        "product_type": "analysis",
        "format": "grib",
        "year": y, "month": m,
        "day": days_in_month(y, m),
        "time": TIMES,
    }
    grib_path = os.path.join(RAW_DIR, f"cerra_pressure_{y}_{m}.grib")
    dl_queue.put((dataset, req, grib_path))

def submit_single_month(y, m):
    dataset = "reanalysis-cerra-single-levels"
    req = {
        "variable": [
            "10m_wind_speed","10m_wind_direction","2m_temperature",
            "mean_sea_level_pressure","medium_cloud_cover"
        ],
        "level_type": "surface_or_atmosphere",
        "data_type": "reanalysis",
        "product_type": "analysis",
        "format": "grib",
        "year": y, "month": m,
        "day": days_in_month(y, m),
        "time": TIMES,
    }
    grib_path = os.path.join(RAW_DIR, f"cerra_single_{y}_{m}.grib")
    dl_queue.put((dataset, req, grib_path))

# ===================== Main =====================
def main():
    # Start workers
    t_dl = threading.Thread(target=downloader, daemon=True)
    procs = [threading.Thread(target=processor, daemon=True) for _ in range(N_PROCESSORS)]
    t_dl.start()
    for t in procs: t.start()

    # Submit jobs
    submit_static_once()
    for y in YEARS:
        for m in MONTHS:
            submit_pressure_month(y, m)  # heaviest first
    for y in YEARS:
        for m in MONTHS:
            submit_single_month(y, m)
    for y in YEARS:
        for m in MONTHS:
            submit_height_month(y, m)

    # Wait for downloads to finish
    dl_queue.join()
    # Stop processors
    for _ in procs:
        proc_queue.put(None)
    # Wait for processing to finish
    proc_queue.join()

    print("\nAll done.")
    print(f"Raw GRIB: {RAW_DIR}")
    print(f"BOZ NetCDF: {NC_DIR}")

if __name__ == "__main__":
    main()
