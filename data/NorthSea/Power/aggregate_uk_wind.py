"""
Aggregate all UK wind farm CSVs into a single 3-hour mean MW file aligned like the Belgian dataset.

Steps:
- Read every *.csv in data/NorthSea/Power/UK (half-hourly MWh values, timestamp marks the end of the half hour).
- Accumulate energy over 3-hour windows ending at 00:00, 03:00, 06:00, ... (sum six half-hour entries).
- Convert to mean MW by dividing the 3-hour energy by 3.
- Shift timestamps back by 3 hours so the value at, e.g., 18:00 is reported at 15:00.
- Write combined CSV with header `time,<farm1>,<farm2>,...` formatted like BE_offshore_per_unit_3H_meanMW_*.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


THREE_HOURS = timedelta(hours=3)
POWER_DIR = Path(__file__).resolve().parent
UK_DIR = POWER_DIR / "UK"
OUTPUT_CSV = POWER_DIR / "UK_offshore_per_unit_3H_meanMW_shifted.csv"
BELGIUM_CSV = POWER_DIR / "BELGIUM" / "BE_offshore_per_unit_3H_meanMW_2020-01_to_2025-07.csv"
COMBINED_OUTPUT_CSV = POWER_DIR / "BE_UK_offshore_per_unit_3H_meanMW_shifted.csv"

# Start dates for select Belgium farms; earlier timestamps will be forced to NaN.
BELGIUM_STARTS: Dict[str, datetime] = {
    "Northwester 2": datetime(2020, 1, 12, 0, 0),
    "Seastar Offshore WP": datetime(2020, 9, 8, 12, 0),
    "Mermaid Offshore WP": datetime(2020, 7, 1, 6, 0),
}

# Allowed periods (inclusive years). Outside these windows values are set to NaN.
ALLOWED_YEARS: Dict[str, List[int]] = {
    "HornseaTwo": [2023, 2024],
    "HornseaOne": [2021, 2022, 2023],
    "Seagreen": [2023, 2024],
    "MorayEast": [2022, 2023, 2024],
    "TritonKnoll": [2023, 2024],
    "EastAngliaOne": [2021, 2023],
    "WalneyExtension": [2020, 2021, 2022, 2023],
    "LondonArray": [2020, 2021, 2022, 2023],
    "Beatrice": [2020, 2021, 2022, 2023],
    "GwyntyMor": [2020, 2022, 2023],
    "RaceBank": [2020, 2021, 2022, 2023],
    "GreaterGabbard": [2020, 2021, 2022, 2023],
    "Dudgeon": [2020, 2021, 2022, 2023],
    "Rampion": [2021, 2022, 2023],
    "WestOfDuddonSands": [2020, 2021, 2022, 2023],
    "Galloper": [2021, 2022, 2023],
    "SheringhamShoals": [2022, 2023],
    "Lincs": [2020, 2021, 2022, 2023],
    "BurboBankExtension": [2020, 2021, 2022, 2023],
    "HumberGateway": [2020, 2021, 2022, 2023],
    "WestermostRough": [2020, 2021, 2022, 2023],
    "Walney1": [2020, 2021, 2022, 2023],
    "Walney2": [2020, 2021, 2022, 2023],
    "RobinRigg": [2020, 2021, 2022, 2023],
    "GunfleetSands": [2020, 2021, 2022, 2023],
    "Ormonde": [2020],
    "BurboBank": [2020, 2021, 2022, 2023],
    "Barrow": [2021, 2022],
    "Kincardine": [2020, 2021, 2022, 2023],
    "HywindScotland": [2020, 2021, 2022, 2023],
}


@dataclass
class Bucket:
    energy_mwh: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.energy_mwh += value
        self.count += 1


def parse_rows(csv_path: Path) -> Iterator[Tuple[datetime, float]]:
    """Yield datetime (UTC, naive) and energy value for each row."""
    with csv_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                ts_str, val_str = raw.split(",", 1)
            except ValueError as exc:
                raise ValueError(f"{csv_path}: line {line_no} is not 'timestamp,value'") from exc
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError as exc:
                raise ValueError(f"{csv_path}: line {line_no} has invalid timestamp '{ts_str}'") from exc
            try:
                val = float(val_str)
            except ValueError as exc:
                raise ValueError(f"{csv_path}: line {line_no} has invalid value '{val_str}'") from exc
            yield ts, max(0.0, val)  # clamp negative values to 0


def bucket_end(ts: datetime) -> datetime:
    """
    Return the 3-hour bucket end time for a half-hourly timestamp.
    Example: 15:30, 16:00, ..., 18:00 -> bucket end 18:00.
    """
    minutes_since_midnight = ts.hour * 60 + ts.minute
    bucket_minutes = ((minutes_since_midnight + 179) // 180) * 180  # ceil to 3-hour boundary
    base = datetime.combine(ts.date(), datetime.min.time())
    if bucket_minutes == 1440:
        base += timedelta(days=1)
        bucket_minutes = 0
    return base + timedelta(minutes=bucket_minutes)


def format_ts(ts: datetime) -> str:
    """Format timestamp like the Belgium file (UTC)."""
    return ts.strftime("%Y-%m-%d %H:%M:%S+00:00")


def aggregate_farm(csv_path: Path) -> Dict[datetime, float]:
    """Aggregate one farm into shifted 3-hour mean MW keyed by shifted timestamp."""
    buckets: Dict[datetime, Bucket] = defaultdict(Bucket)
    for ts, val in parse_rows(csv_path):
        end = bucket_end(ts)
        buckets[end].add(val)

    aggregated: Dict[datetime, float] = {}
    for end_ts, bucket in buckets.items():
        # Per instructions: sum 3 hours of half-hourly MWh, then divide by 3 to get mean MW.
        mean_mw = bucket.energy_mwh / 3.0
        shifted = end_ts - THREE_HOURS
        aggregated[shifted] = mean_mw
    return aggregated


def collect_farms(csv_paths: Iterable[Path]) -> Tuple[List[str], Dict[str, Dict[datetime, float]]]:
    """Return farm names and their aggregated data maps."""
    farm_data: Dict[str, Dict[datetime, float]] = {}
    names: List[str] = []
    for path in sorted(csv_paths):
        name = path.stem
        names.append(name)
        farm_data[name] = aggregate_farm(path)
    return names, farm_data


def read_belgium(csv_path: Path) -> Tuple[List[str], Dict[datetime, List[str]]]:
    """Read the Belgium 3-hour file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Belgium file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or header[0] != "time":
            raise ValueError(f"Unexpected header in {csv_path}")
        farm_names = header[1:]
        data: Dict[datetime, List[str]] = {}
        for row in reader:
            if not row:
                continue
            ts_str = row[0]
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo:
                ts = ts.replace(tzinfo=None)
            values = row[1:]
            if len(values) < len(farm_names):
                values.extend([""] * (len(farm_names) - len(values)))
            normalized: List[str] = []
            for name, v in zip(farm_names, values):
                val = v if v != "" else "NaN"
                start = BELGIUM_STARTS.get(name)
                if start and ts < start:
                    val = "NaN"
                normalized.append(val)
            data[ts] = normalized
    return farm_names, data


def build_uk_row(ts: datetime, farm_names: List[str], farm_data: Dict[str, Dict[datetime, float]]) -> List[str]:
    row: List[str] = []
    for name in farm_names:
        years = ALLOWED_YEARS.get(name)
        value = farm_data[name].get(ts)
        if value is None or (years is not None and ts.year not in years):
            row.append("NaN")
        else:
            row.append(f"{value:.6f}")
    return row


def main() -> None:
    csv_paths = [p for p in UK_DIR.glob("*.csv") if p.is_file()]
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {UK_DIR}")

    farm_names, farm_data = collect_farms(csv_paths)
    all_times = set()
    for data in farm_data.values():
        all_times.update(data.keys())
    sorted_times = sorted(all_times)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", *farm_names])
        for ts in sorted_times:
            row = [format_ts(ts)]
            row.extend(build_uk_row(ts, farm_names, farm_data))
            writer.writerow(row)

    # Merge with Belgium file
    be_names, be_data = read_belgium(BELGIUM_CSV)
    combined_times = sorted(set(sorted_times) | set(be_data.keys()))
    with COMBINED_OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", *be_names, *farm_names])
        for ts in combined_times:
            row = [format_ts(ts)]
            be_row = be_data.get(ts)
            if be_row is None:
                row.extend(["NaN"] * len(be_names))
            else:
                row.extend(be_row)
            row.extend(build_uk_row(ts, farm_names, farm_data))
            writer.writerow(row)

    print(f"Wrote aggregated file to {OUTPUT_CSV.relative_to(POWER_DIR)}")
    print(f"Wrote combined BE+UK file to {COMBINED_OUTPUT_CSV.relative_to(POWER_DIR)}")


if __name__ == "__main__":
    main()
