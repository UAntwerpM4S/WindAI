#!/usr/bin/env python3
"""
Download ERA5 (CDS) monthly GRIBs for:
- Pressure levels: z, r, t, u, v on selected pressure levels
- Single levels: u10, v10, t2m, msl, sp
- Static (once): land_sea_mask, orography

Cadence:
- 3-hourly: 00,03,06,09,12,15,18,21 UTC
Period:
- 2018-01 .. 2025-12, but STOP at first month that CDS reports as unavailable.

Output:
- ./raw_grib/era5_static.grib
- ./raw_grib/era5_pressure_YYYY_MM.grib
- ./raw_grib/era5_single_YYYY_MM.grib
"""

import os
import time
import calendar
import cdsapi

# ===================== Config =====================
# Put raw_grib next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw_grib")
os.makedirs(RAW_DIR, exist_ok=True)

# ERA5 datasets
DS_PRESSURE = "reanalysis-era5-pressure-levels"
DS_SINGLE   = "reanalysis-era5-single-levels"

# Years/months/times
START_YEAR = 2015
END_YEAR   = 2025
MONTHS = [f"{m:02d}" for m in range(1, 13)]
TIMES  = ["00:00","03:00","06:00","09:00","12:00","15:00","18:00","21:00"]

# Bounding box in CDS order: [North, West, South, East]
AREA = [65, -15, 35, 25]

# Pressure levels (hPa) as strings
PRESSURE_LEVELS = ["500","600","700","750","800","850","900","950","1000"]

# Variables (download only, no derived calculations)
PRESSURE_VARS = [
    "geopotential",
    "relative_humidity",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
SINGLE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "medium_cloud_cover",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    
]
STATIC_VARS = [
    "land_sea_mask",
    "geopotential",
]

# GRIB output knobs (match CDS “Show API request” style)
DATA_FORMAT = "grib"
DOWNLOAD_FORMAT = "unarchived"

# Retry behavior
MAX_TRIES = 5

# ===================== Helpers =====================
def days_in_month(year: int, month: int):
    last = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, last + 1)]

def is_no_data_error(exc: Exception) -> bool:
    """
    CDS error texts vary a bit. We treat these as "month not available" and stop.
    """
    msg = (str(exc) or "").lower()
    needles = [
        "no data",
        "no matches",
        "no result",
        "no results",
        "not available",
        "cannot find",
        "nothing to download",
        "empty result",
        "request returned no data",
        "there is no data",
    ]
    return any(n in msg for n in needles)

def retrieve_with_retry(client: cdsapi.Client, dataset: str, request: dict, target: str):
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"✔ exists: {target}")
        return

    tries = 0
    while True:
        tries += 1
        try:
            print(f"→ retrieving {dataset} -> {target}")
            client.retrieve(dataset, request, target)
            if not (os.path.exists(target) and os.path.getsize(target) > 0):
                raise RuntimeError(f"Download did not create a non-empty file: {target}")
            print(f"✔ downloaded: {target}")
            return
        except Exception as e:
            if is_no_data_error(e):
                raise  # handled by caller (stop-at-first-missing-month)
            if tries >= MAX_TRIES:
                raise
            wait = min(60 * tries, 300)
            print(f"CDS error ({e}). Retrying in {wait}s [{tries}/{MAX_TRIES}]…")
            time.sleep(wait)

# ===================== Request builders =====================
def build_static_request():
    # One timestamp is enough (static fields)
    return {
        "product_type": ["reanalysis"],
        "variable": STATIC_VARS,
        "year": ["2018"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": DATA_FORMAT,
        "download_format": DOWNLOAD_FORMAT,
        "area": AREA,
    }

def build_pressure_request(y: int, m_str: str):
    m = int(m_str)
    return {
        "product_type": ["reanalysis"],
        "variable": PRESSURE_VARS,
        "pressure_level": PRESSURE_LEVELS,
        "year": [str(y)],
        "month": [m_str],
        "day": days_in_month(y, m),
        "time": TIMES,
        "data_format": DATA_FORMAT,
        "download_format": DOWNLOAD_FORMAT,
        "area": AREA,
    }

def build_single_request(y: int, m_str: str):
    m = int(m_str)
    return {
        "product_type": ["reanalysis"],
        "variable": SINGLE_VARS,
        "year": [str(y)],
        "month": [m_str],
        "day": days_in_month(y, m),
        "time": TIMES,
        "data_format": DATA_FORMAT,
        "download_format": DOWNLOAD_FORMAT,
        "area": AREA,
    }

# ===================== Main =====================
def main():
    client = cdsapi.Client()

    # ---- Static once ----
    static_path = os.path.join(RAW_DIR, "era5_static.grib")
    try:
        retrieve_with_retry(client, DS_SINGLE, build_static_request(), static_path)
    except Exception as e:
        print(f"✖ Failed to download static fields: {e}")
        raise

    # ---- Monthly loop: pressure + single ----
    for year in range(START_YEAR, END_YEAR + 1):
        for m_str in MONTHS:
            pressure_path = os.path.join(RAW_DIR, f"era5_pressure_{year}_{m_str}.grib")
            single_path   = os.path.join(RAW_DIR, f"era5_single_{year}_{m_str}.grib")

            # Pressure first (heavier)
            try:
                retrieve_with_retry(client, DS_PRESSURE, build_pressure_request(year, m_str), pressure_path)
            except Exception as e:
                if is_no_data_error(e):
                    print(f"\n🛑 No data available for {year}-{m_str} (pressure). Stopping as requested.")
                    return
                raise

            # Single levels
            try:
                retrieve_with_retry(client, DS_SINGLE, build_single_request(year, m_str), single_path)
            except Exception as e:
                if is_no_data_error(e):
                    print(f"\n🛑 No data available for {year}-{m_str} (single). Stopping as requested.")
                    return
                raise

            print(f"✅ Done {year}-{m_str}")

    print("\nAll done.")
    print(f"Raw GRIB directory: {RAW_DIR}")

if __name__ == "__main__":
    main()
