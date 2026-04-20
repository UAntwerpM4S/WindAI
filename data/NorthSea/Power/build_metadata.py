"""
Build wind farm metadata CSV covering Belgium (from farm_to_cell_lookup.csv) and UK (manually curated table),
then immediately appends correct cerra_y, cerra_x, cerra_grid_lat, cerra_grid_lon, cerra_distance_km
by matching each farm to the nearest cell in the current CERRA zarr.

Matching logic mirrors append_turbine_vars_xy() exactly:
  - KD-tree over the full (y, x) CERRA grid in (lon, lat) space
  - Each farm snaps to its globally nearest CERRA cell

Outputs: windfarm_metadata.csv with columns:
    region, farm, capacity_mw, lat, lon, turbines, turbine_types,
    cerra_y, cerra_x, cerra_grid_lat, cerra_grid_lon, cerra_distance_km
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# -------------------------
# PATHS
# -------------------------
POWER_DIR  = Path(__file__).resolve().parent
ROOT_DIR   = POWER_DIR.parent.parent.parent
BE_LOOKUP  = ROOT_DIR / "data" / "BOZ_Turbines" / "coordinates" / "farm_to_cell_lookup.csv"
OUTPUT     = POWER_DIR / "windfarm_metadata.csv"
CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")

# -------------------------
# BELGIUM
# -------------------------
BE_TURBINE_INFO: Dict[str, Tuple[str, float]] = {
    "turbines_alstom_haliade_150_6mw":              ("Alstom Haliade 150",              6.0),
    "turbines_mhi_vestas_v164_84mw":                ("MHI Vestas V164-8.4",             8.4),
    "turbines_mhi_vestas_v164_95mw":                ("MHI Vestas V164-9.5",             9.5),
    "turbines_repower_5m126_5mw":                   ("REpower 5M126",                   5.0),
    "turbines_repower_62m126_615mw":                ("REpower 6.15M126",                6.15),
    "turbines_siemens_gamesa_sg_80_167_dd_84mw":    ("Siemens Gamesa SG 8.0-167 DD",    8.4),
    "turbines_siemens_swt_70_154_735mw":            ("Siemens SWT-7.0-154",             7.35),
    "turbines_vestas_v112_33mw":                    ("Vestas V112-3.3",                 3.3),
    "turbines_vestas_v112_3mw":                     ("Vestas V112-3.0",                 3.0),
    "turbines_vestas_v90_3mw":                      ("Vestas V90-3.0",                  3.0),
}

BE_NAME_MAP: Dict[str, str] = {
    "Belwind":      "Belwind Phase 1",
    "Nobelwind":    "Nobelwind Offshore Windpark",
    "Norther":      "Norther Offshore WP",
    "Northwester":  "Northwester 2",
    "Northwester2": "Northwester 2",
    "Northwind":    "Northwind",
    "Rentel":       "Rentel Offshore WP",
    "CPower_NE":    "Thorntonbank - C-Power - Area NE",
    "CPower_SW":    "Thorntonbank - C-Power - Area SW",
    "Mermaid":      "Mermaid Offshore WP",
    "Seastar":      "Seastar Offshore WP",
}


def load_belgium() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with BE_LOOKUP.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            farm_raw = r["farm"]
            farm = BE_NAME_MAP.get(farm_raw, farm_raw)
            lat = float(r["farm_lat"])
            lon = float(r["farm_lon"])
            turbine_entries: List[str] = []
            capacity = 0.0
            turbines = 0
            for col, (model, mw) in BE_TURBINE_INFO.items():
                cnt = int(float(r.get(col, 0) or 0))
                if cnt:
                    turbines += cnt
                    capacity += cnt * mw
                    turbine_entries.append(f"{cnt}x {model}")
            rows.append({
                "region":        "Belgium",
                "farm":          farm,
                "capacity_mw":   f"{capacity:.2f}" if capacity else "NaN",
                "lat":           f"{lat:.6f}",
                "lon":           f"{lon:.6f}",
                "turbines":      str(turbines) if turbines else "NaN",
                "turbine_types": "; ".join(turbine_entries) if turbine_entries else "NaN",
            })
    return rows


# -------------------------
# UK
# -------------------------
def dms_to_decimal(dms: str) -> Tuple[float, float]:
    """Convert strings like '53°59′00″N 3°17′00″W' to signed decimal degrees."""
    dms = dms.strip()
    parts = dms.split()
    if len(parts) != 2:
        raise ValueError(f"Unrecognized DMS format: {dms}")
    lat_str, lon_str = parts

    def parse_one(piece: str) -> float:
        piece = (piece.replace("'", "′").replace("''", "″")
                      .replace("′′", "″").replace("'′", "″"))
        deg_part, rest = piece.split("°", 1)
        hemi = rest[-1]
        rest_core = rest[:-1]
        if "′" in rest_core:
            minutes_part, sec_part = rest_core.split("′", 1)
            seconds_part = sec_part.replace("″", "")
            val = (float(deg_part)
                   + float(minutes_part or 0) / 60
                   + float(seconds_part or 0) / 3600)
        else:
            val = float(deg_part + rest_core)
        if hemi.upper() in {"S", "W"}:
            val = -val
        return val

    return parse_one(lat_str), parse_one(lon_str)


UK_FARMS: Dict[str, Dict[str, str]] = {
    "Barrow":            {"capacity_mw": "90",   "turbines": "30",  "turbine_types": "Vestas V90-3.0MW",                     "coord": "53°59′00″N 3°17′00″W"},
    "BurboBank":         {"capacity_mw": "90",   "turbines": "25",  "turbine_types": "Siemens SWT-3.6-107",                  "coord": "53°29′00″N 3°11′00″W"},
    "BurboBankExtension":{"capacity_mw": "258",  "turbines": "32",  "turbine_types": "Vestas V164-8.0MW",                    "coord": "53°29′00″N 3°11′00″W"},
    "Dudgeon":           {"capacity_mw": "402",  "turbines": "67",  "turbine_types": "Siemens SWT-6.0-154",                  "coord": "53°15′00″N 1°23′00″E"},
    "EastAngliaOne":     {"capacity_mw": "714",  "turbines": "102", "turbine_types": "Siemens SWT-7.0-154",                  "coord": "52°14′04″N 02°29′18″E"},
    "Galloper":          {"capacity_mw": "353",  "turbines": "56",  "turbine_types": "Siemens SWT-6.0-154",                  "coord": "51°43′0″N 01°12′50″E"},
    "GreaterGabbard":    {"capacity_mw": "504",  "turbines": "140", "turbine_types": "Siemens SWT-3.6-107",                  "coord": "51°56′0″N 1°53′0″E"},
    "GunfleetSands":     {"capacity_mw": "173",  "turbines": "50",  "turbine_types": "Siemens SWT-3.6-107 / SWT-6.0-120",   "coord": "51°43′0″N 01°12′50″E"},
    "GwyntyMor":         {"capacity_mw": "576",  "turbines": "160", "turbine_types": "Siemens SWT-3.6-107",                  "coord": "53°27′00″N 03°35′00″W"},
    "HornseaOne":        {"capacity_mw": "1218", "turbines": "174", "turbine_types": "Siemens SWT-7.0-154",                  "coord": "53°53′06″N 01°47′28″E"},
    "HornseaTwo":        {"capacity_mw": "1386", "turbines": "165", "turbine_types": "Siemens SG 8.0-167 DD",               "coord": "53°53′06″N 01°47′28″E"},
    "HumberGateway":     {"capacity_mw": "219",  "turbines": "73",  "turbine_types": "Vestas V112-3.0MW",                    "coord": "53°38′38″N 0°17′35″E"},
    "Lincs":             {"capacity_mw": "270",  "turbines": "75",  "turbine_types": "Siemens SWT-3.6-120",                  "coord": "53°11′0″N 0°29′0″E"},
    "LondonArray":       {"capacity_mw": "630",  "turbines": "175", "turbine_types": "Siemens SWT-3.6-107",                  "coord": "51°38′38″N 1°33′13″E"},
    "Ormonde":           {"capacity_mw": "150",  "turbines": "30",  "turbine_types": "REpower 5MW",                          "coord": "54°06′00″N 3°24′00″W"},
    "RaceBank":          {"capacity_mw": "580",  "turbines": "91",  "turbine_types": "Siemens SWT-6.0-154",                  "coord": "53°16′30″N 0°50′30″E"},
    "Rampion":           {"capacity_mw": "400",  "turbines": "116", "turbine_types": "Vestas V112-3.45MW",                   "coord": "50°40′00″N 0°16′00″W"},
    "RobinRigg":         {"capacity_mw": "180",  "turbines": "60",  "turbine_types": "Vestas V90-3.0MW",                     "coord": "54°45′00″N 3°43′00″W"},
    "TritonKnoll":       {"capacity_mw": "857",  "turbines": "90",  "turbine_types": "Vestas V164-9.5MW",                    "coord": "53°30′00″N 0°48′0″E"},
    "Walney1":           {"capacity_mw": "184",  "turbines": "51",  "turbine_types": "Siemens SWT-3.6-107",                  "coord": "54°03′00″N 3°31′00″W"},
    "Walney2":           {"capacity_mw": "184",  "turbines": "51",  "turbine_types": "Siemens SWT-3.6-107",                  "coord": "54°03′00″N 3°31′00″W"},
    "WalneyExtension":   {"capacity_mw": "659",  "turbines": "87",  "turbine_types": "Siemens SWT-7.0-154 / MHI V164-8.25", "coord": "54°5′17″N 3°44′17″W"},
    "WestOfDuddonSands": {"capacity_mw": "389",  "turbines": "108", "turbine_types": "Siemens SWT-3.6-120",                  "coord": "53°59′00″N 3°28′00″W"},
    "WestermostRough":   {"capacity_mw": "210",  "turbines": "35",  "turbine_types": "Siemens SWT-6.0-154",                  "coord": "53°48′18″N 0°8′56″E"},
}


def load_uk() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for farm, info in UK_FARMS.items():
        coord = info.get("coord")
        if coord and " " in coord:
            lat, lon = dms_to_decimal(coord)
        else:
            lat = lon = float("nan")
        rows.append({
            "region":        "UK",
            "farm":          farm,
            "capacity_mw":   info.get("capacity_mw", "NaN"),
            "lat":           f"{lat:.6f}" if not np.isnan(lat) else "NaN",
            "lon":           f"{lon:.6f}" if not np.isnan(lon) else "NaN",
            "turbines":      info.get("turbines", "NaN"),
            "turbine_types": info.get("turbine_types", "NaN"),
        })
    return rows


# -------------------------
# CERRA CELL MATCHING
# -------------------------
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def append_cerra_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match each farm to its nearest CERRA cell using the same KD-tree logic
    as append_turbine_vars_xy(): tree built over full grid in (lon, lat) order.
    Adds cerra_y, cerra_x, cerra_grid_lat, cerra_grid_lon, cerra_distance_km.
    """
    print(f"Opening CERRA zarr: {CERRA_PATH}")
    ds = xr.open_zarr(CERRA_PATH, consolidated=False)

    # Anemoi datasets store lat/lon as 1-D arrays over the "values" dimension
    lat1d = ds["latitudes"].values   # (n_points,)
    lon1d = ds["longitudes"].values  # (n_points,)
    print(f"  Grid size: {lat1d.size} points")

    # KD-tree in (lon, lat) over full grid — mirrors append_turbine_vars_xy
    tree = cKDTree(np.c_[lon1d, lat1d])

    farm_lon = df["lon"].astype(float).values
    farm_lat = df["lat"].astype(float).values

    _, flat_indices = tree.query(np.c_[farm_lon, farm_lat], k=1)

    grid_lat = lat1d[flat_indices]
    grid_lon = lon1d[flat_indices]
    dist_km  = haversine_km(farm_lat, farm_lon, grid_lat, grid_lon)

    df = df.copy()
    df["cerra_y"]           = flat_indices   # flat index into 1-D values dim
    df["cerra_x"]           = 0              # no x-axis in Anemoi 1-D grid
    df["cerra_grid_lat"]    = grid_lat
    df["cerra_grid_lon"]    = grid_lon
    df["cerra_distance_km"] = dist_km

    return df


# -------------------------
# MAIN
# -------------------------
def main() -> None:
    rows = load_belgium() + load_uk()

    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Drop farms with missing coordinates before spatial matching
    missing = df[df["lat"].isna() | df["lon"].isna()]
    if not missing.empty:
        print(f"[WARN] Dropping {len(missing)} farms with missing lat/lon: "
              f"{missing['farm'].tolist()}")
        df = df.dropna(subset=["lat", "lon"])

    df = append_cerra_coords(df)

    # Print summary
    print(f"\n{'Farm':<35}  {'y':>5}  {'x':>5}  {'Grid lat':>10}  {'Grid lon':>10}  {'Dist (km)':>10}")
    for _, row in df.iterrows():
        print(f"  {row['farm']:<33}  {int(row['cerra_y']):>5}  {int(row['cerra_x']):>5}  "
              f"{row['cerra_grid_lat']:>10.5f}  {row['cerra_grid_lon']:>10.5f}  "
              f"{row['cerra_distance_km']:>10.4f}")
    print(f"\nMax distance : {df['cerra_distance_km'].max():.3f} km  "
          f"({df.loc[df['cerra_distance_km'].idxmax(), 'farm']})")
    print(f"Mean distance: {df['cerra_distance_km'].mean():.3f} km")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nWrote metadata to {OUTPUT}")


if __name__ == "__main__":
    main()