"""
Build wind farm metadata CSV covering Belgium (from farm_to_cell_lookup.csv) and UK (manually curated table).

Outputs: data/NorthSea/Power/windfarm_metadata.csv with columns:
region,farm,capacity_mw,lat,lon,turbines,turbine_types
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

POWER_DIR = Path(__file__).resolve().parent
ROOT_DIR = POWER_DIR.parent.parent.parent
BE_LOOKUP = ROOT_DIR / "data" / "BOZ_Turbines" / "coordinates" / "farm_to_cell_lookup.csv"
OUTPUT = POWER_DIR / "windfarm_metadata.csv"

# Map turbine columns to (readable model, MW per unit)
BE_TURBINE_INFO: Dict[str, Tuple[str, float]] = {
    "turbines_alstom_haliade_150_6mw": ("Alstom Haliade 150", 6.0),
    "turbines_mhi_vestas_v164_84mw": ("MHI Vestas V164-8.4", 8.4),
    "turbines_mhi_vestas_v164_95mw": ("MHI Vestas V164-9.5", 9.5),
    "turbines_repower_5m126_5mw": ("REpower 5M126", 5.0),
    "turbines_repower_62m126_615mw": ("REpower 6.15M126", 6.15),
    "turbines_siemens_gamesa_sg_80_167_dd_84mw": ("Siemens Gamesa SG 8.0-167 DD", 8.4),
    "turbines_siemens_swt_70_154_735mw": ("Siemens SWT-7.0-154", 7.35),
    "turbines_vestas_v112_33mw": ("Vestas V112-3.3", 3.3),
    "turbines_vestas_v112_3mw": ("Vestas V112-3.0", 3.0),
    "turbines_vestas_v90_3mw": ("Vestas V90-3.0", 3.0),
}

# Map farm_to_cell_lookup naming to Belgium series names
BE_NAME_MAP: Dict[str, str] = {
    "Belwind": "Belwind Phase 1",
    "Nobelwind": "Nobelwind Offshore Windpark",
    "Norther": "Norther Offshore WP",
    "Northwester": "Northwester 2",
    "Northwester2": "Northwester 2",
    "Northwind": "Northwind",
    "Rentel": "Rentel Offshore WP",
    "CPower_NE": "Thorntonbank - C-Power - Area NE",
    "CPower_SW": "Thorntonbank - C-Power - Area SW",
    "Mermaid": "Mermaid Offshore WP",
    "Seastar": "Seastar Offshore WP",
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
            rows.append(
                {
                    "region": "Belgium",
                    "farm": farm,
                    "capacity_mw": f"{capacity:.2f}" if capacity else "NaN",
                    "lat": f"{lat:.6f}",
                    "lon": f"{lon:.6f}",
                    "turbines": str(turbines) if turbines else "NaN",
                    "turbine_types": "; ".join(turbine_entries) if turbine_entries else "NaN",
                }
            )
    return rows


def dms_to_decimal(dms: str) -> float:
    """
    Convert strings like 53°59′00″N 3°17′00″W to signed decimal degrees.
    """
    dms = dms.strip()
    parts = dms.split()
    if len(parts) != 2:
        raise ValueError(f"Unrecognized DMS format: {dms}")
    lat_str, lon_str = parts

    def parse_one(piece: str) -> float:
        piece = piece.replace("’", "′").replace("''", "″").replace("′′", "″").replace("’′", "″")
        deg_part, rest = piece.split("°", 1)
        hemi = rest[-1]
        rest_core = rest[:-1]
        if "′" in rest_core:
            minutes_part, sec_part = rest_core.split("′", 1)
            if "″" in sec_part:
                seconds_part = sec_part.replace("″", "")
            else:
                seconds_part = sec_part
            val = float(deg_part) + float(minutes_part or 0) / 60 + float(seconds_part or 0) / 3600
        else:
            # format like 58.1 or 2.8 without minutes/seconds
            val = float(deg_part + rest_core)
        if hemi.upper() in {"S", "W"}:
            val = -val
        return val

    return parse_one(lat_str), parse_one(lon_str)


UK_FARMS: Dict[str, Dict[str, str]] = {
    "Barrow": {
        "capacity_mw": "90",
        "turbines": "30",
        "turbine_types": "Vestas V90-3.0MW",
        "coord": "53°59′00″N 3°17′00″W",
    },
    "BurboBank": {
        "capacity_mw": "90",
        "turbines": "25",
        "turbine_types": "Siemens SWT-3.6-107",
        "coord": "53°29′00″N 3°11′00″W",
    },
    "BurboBankExtension": {
        "capacity_mw": "258",
        "turbines": "32",
        "turbine_types": "Vestas V164-8.0MW",
        "coord": "53°29′00″N 3°11′00″W",
    },
    "Dudgeon": {
        "capacity_mw": "402",
        "turbines": "67",
        "turbine_types": "Siemens SWT-6.0-154",
        "coord": "53°15′00″N 1°23′00″E",
    },
    "EastAngliaOne": {
        "capacity_mw": "714",
        "turbines": "102",
        "turbine_types": "Siemens SWT-7.0-154",
        "coord": "52°14′04″N 02°29′18″E",
    },
    "Galloper": {
        "capacity_mw": "353",
        "turbines": "56",
        "turbine_types": "Siemens SWT-6.0-154",
        "coord": "51°43′0″N 01°12′50″E",
    },
    "GreaterGabbard": {
        "capacity_mw": "504",
        "turbines": "140",
        "turbine_types": "Siemens SWT-3.6-107",
        "coord": "51°56′0″N 1°53′0″E",
    },
    "GunfleetSands": {
        "capacity_mw": "173",
        "turbines": "50",
        "turbine_types": "Siemens SWT-3.6-107 / SWT-6.0-120",
        "coord": "51°43′0″N 01°12′50″E",
    },
    "GwyntyMor": {
        "capacity_mw": "576",
        "turbines": "160",
        "turbine_types": "Siemens SWT-3.6-107",
        "coord": "53°27′00″N 03°35′00″W",
    },
    "HornseaOne": {
        "capacity_mw": "1218",
        "turbines": "174",
        "turbine_types": "Siemens SWT-7.0-154",
        "coord": "53°53′06″N 01°47′28″E",
    },
    "HornseaTwo": {
        "capacity_mw": "1386",
        "turbines": "165",
        "turbine_types": "Siemens SG 8.0-167 DD",
        "coord": "53°53′06″N 01°47′28″E",
    },
    "HumberGateway": {
        "capacity_mw": "219",
        "turbines": "73",
        "turbine_types": "Vestas V112-3.0MW",
        "coord": "53°38′38″N 0°17′35″E",
    }, 
    "Lincs": {
        "capacity_mw": "270",
        "turbines": "75",
        "turbine_types": "Siemens SWT-3.6-120",
        "coord": "53°11′0″N 0°29′0″E",
    },
    "LondonArray": {
        "capacity_mw": "630",
        "turbines": "175",
        "turbine_types": "Siemens SWT-3.6-107",
        "coord": "51°38′38″N 1°33′13″E",
    },
    "Ormonde": {
        "capacity_mw": "150",
        "turbines": "30",
        "turbine_types": "REpower 5MW",
        "coord": "54°06′00″N 3°24′00″W",
    },
    "RaceBank": {
        "capacity_mw": "580",
        "turbines": "91",
        "turbine_types": "Siemens SWT-6.0-154",
        "coord": "53°16′30″N 0°50′30″E",
    },
    "Rampion": {
        "capacity_mw": "400",
        "turbines": "116",
        "turbine_types": "Vestas V112-3.45MW",
        "coord": "50°40′00″N 0°16′00″W",
    },
    "RobinRigg": {
        "capacity_mw": "180",
        "turbines": "60",
        "turbine_types": "Vestas V90-3.0MW",
        "coord": "54°45′00″N 3°43′00″W",
    },
    "TritonKnoll": {
        "capacity_mw": "857",
        "turbines": "90",
        "turbine_types": "Vestas V164-9.5MW",
        "coord": "53°30′00″N 0°48′0″E",
    },
    "Walney1": {
        "capacity_mw": "184",
        "turbines": "51",  # approximate split from total Walney 1+2
        "turbine_types": "Siemens SWT-3.6-107",
        "coord": "54°03′00″N 3°31′00″W",
    },
    "Walney2": {
        "capacity_mw": "184",
        "turbines": "51",  # approximate split from total Walney 1+2
        "turbine_types": "Siemens SWT-3.6-107",
        "coord": "54°03′00″N 3°31′00″W",
    },
    "WalneyExtension": {
        "capacity_mw": "659",
        "turbines": "87",
        "turbine_types": "Siemens SWT-7.0-154 / MHI V164-8.25",
        "coord": "54°5′17″N 3°44′17″W",
    },
    "WestOfDuddonSands": {
        "capacity_mw": "389",
        "turbines": "108",
        "turbine_types": "Siemens SWT-3.6-120",
        "coord": "53°59′00″N 3°28′00″W",
    },
    "WestermostRough": {
        "capacity_mw": "210",
        "turbines": "35",
        "turbine_types": "Siemens SWT-6.0-154",
        "coord": "53°48′18″N 0°8′56″E",
    },
}


def load_uk() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for farm, info in UK_FARMS.items():
        coord = info.get("coord")
        if coord and " " in coord:
            lat, lon = dms_to_decimal(coord)
        else:
            lat = lon = float("nan")
        rows.append(
            {
                "region": "UK",
                "farm": farm,
                "capacity_mw": info.get("capacity_mw", "NaN"),
                "lat": f"{lat:.6f}" if lat == lat else "NaN",
                "lon": f"{lon:.6f}" if lon == lon else "NaN",
                "turbines": info.get("turbines", "NaN"),
                "turbine_types": info.get("turbine_types", "NaN"),
            }
        )
    return rows


def main() -> None:
    rows = load_belgium() + load_uk()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["region", "farm", "capacity_mw", "lat", "lon", "turbines", "turbine_types"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote metadata to {OUTPUT}")


if __name__ == "__main__":
    main()
