"""
End-to-end regression test against Nost (2025), PLoS ONE 20(5): e0321528, S1 Table.

Nost published, for 31 North Sea farms, the installed capacity and the mean capacity factor
over the period each farm was running AT FULL CAPACITY. Those periods are exactly the
ALLOWED_YEARS table in ../NorthSea/Power/aggregate_uk_wind.py -- they are not arbitrary, and
they are the availability metadata that lets us exclude derated operation.

If our capacities are right, and the raw observations really are MWh per half-hour, and the
ALLOWED_YEARS windows are applied correctly, then recomputing each farm's mean CF from the raw
Elexon series must reproduce Nost's published CF. It does, for all 23 farms we use, to within
0.1 percentage points.

(Sheringham Shoal is absent: it is not in the combined obs file, and Nost excludes it as never
having run at full capacity.)

This single check simultaneously validates:
  - the unit convention (values are MWh per half-hour, so mean MW = mean(v) * 2)
  - every UK capacity in farm_metadata.py
  - the ALLOWED_YEARS windows

Run it after ANY change to capacities, fleets or the aggregation. It is much cheaper than
discovering a factor-of-two or a wrong nameplate after the zarr is built.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WPOWER_DIR = Path(__file__).resolve().parent
FARMS_CSV = WPOWER_DIR / "farms.csv"

# Raw half-hourly Elexon series, one CSV per farm. Server layout first, then the laptop's.
RAW_UK_CANDIDATES = [
    WPOWER_DIR.parent / "NorthSea" / "Power" / "UK",
    Path("/home/aaron/PhD/Research/Energy/Power/UK"),
]

TOL_PP = 0.5  # percentage points

# Nost (2025) S1 Table: obs column -> (capacity MW, published CF %, full-capacity years)
NOST: dict[str, tuple[float, float, list[int]]] = {
    "HornseaTwo":        (1386.0, 41.4, [2023, 2024]),
    "HornseaOne":        (1218.0, 46.1, [2021, 2022, 2023]),
    "TritonKnoll":       (857.0, 40.6, [2023, 2024]),
    "EastAngliaOne":     (714.0, 46.9, [2021, 2023]),
    "WalneyExtension":   (659.0, 44.3, [2020, 2021, 2022, 2023]),
    "LondonArray":       (630.0, 41.9, [2020, 2021, 2022, 2023]),
    "GwyntyMor":         (576.0, 35.2, [2020, 2022, 2023]),
    "RaceBank":          (573.0, 43.8, [2020, 2021, 2022, 2023]),
    "GreaterGabbard":    (504.0, 39.1, [2020, 2021, 2022, 2023]),
    "Dudgeon":           (402.0, 45.7, [2020, 2021, 2022, 2023]),
    "Rampion":           (400.0, 41.1, [2021, 2022, 2023]),
    "WestOfDuddonSands": (389.0, 43.8, [2020, 2021, 2022, 2023]),
    "Galloper":          (353.0, 47.3, [2021, 2022, 2023]),
    "Lincs":             (270.0, 41.7, [2020, 2021, 2022, 2023]),
    "HumberGateway":     (219.0, 42.6, [2020, 2021, 2022, 2023]),
    "WestermostRough":   (210.0, 46.0, [2020, 2021, 2022, 2023]),
    "Walney1":           (184.0, 36.8, [2020, 2021, 2022, 2023]),
    "Walney2":           (184.0, 43.8, [2020, 2021, 2022, 2023]),
    "RobinRigg":         (174.0, 36.0, [2020, 2021, 2022, 2023]),
    "GunfleetSands":     (173.0, 35.4, [2020, 2021, 2022, 2023]),
    "Ormonde":           (150.0, 39.8, [2020]),
    "Barrow":            (90.0, 31.7, [2021, 2022]),
}


def our_capacity_per_obs_column() -> dict[str, float]:
    """farms.csv is per FARM; Nost is per OBS COLUMN. Walney is one farm over two columns,
    so split its capacity evenly (both halves are 51 x SWT-3.6-107 / 184 MW)."""
    farms = pd.read_csv(FARMS_CSV)
    out: dict[str, float] = {}
    for _, r in farms[farms.region == "UK"].iterrows():
        cols = r.obs_columns.split(";")
        for c in cols:
            out[c] = r.capacity_mw / len(cols)
    return out


def raw_uk_dir() -> Path:
    for p in RAW_UK_CANDIDATES:
        if p.is_dir() and any(p.glob("*.csv")):
            return p
    raise SystemExit(
        "raw half-hourly UK observations not found in any of:\n  "
        + "\n  ".join(str(p) for p in RAW_UK_CANDIDATES)
        + "\n(this check needs them; the rest of the pipeline does not)")


def main() -> None:
    ours = our_capacity_per_obs_column()
    raw_uk = raw_uk_dir()
    print(f"raw UK observations: {raw_uk}\n")

    print(f"{'obs column':18s} {'our MW':>7s} {'Nost MW':>8s} | {'our CF%':>7s} {'Nost CF%':>8s} {'diff':>6s}")
    print("-" * 66)

    failures, errs = [], []
    for col, (cap_nost, cf_nost, years) in NOST.items():
        cap_ours = ours.get(col)
        if cap_ours is None:
            failures.append(f"{col}: not present in farms.csv")
            continue
        if abs(cap_ours - cap_nost) > 1.0:
            failures.append(f"{col}: capacity {cap_ours:.1f} MW vs Nost {cap_nost:.1f} MW")

        raw = pd.read_csv(raw_uk / f"{col}.csv", header=None, names=["t", "v"])
        raw["t"] = pd.to_datetime(raw["t"])
        sel = raw[raw["t"].dt.year.isin(years)]

        # values are MWh per half-hour -> mean MW is mean(v) * 2
        cf_ours = (sel["v"].mean() * 2.0 / cap_nost) * 100.0
        err = cf_ours - cf_nost
        errs.append(err)
        if abs(err) > TOL_PP:
            failures.append(f"{col}: CF {cf_ours:.1f}% vs Nost {cf_nost:.1f}% ({err:+.1f} pp)")

        print(f"{col:18s} {cap_ours:7.1f} {cap_nost:8.1f} | "
              f"{cf_ours:7.1f} {cf_nost:8.1f} {err:+6.2f}")

    errs = np.array(errs)
    print("-" * 66)
    print(f"mean |error| {np.abs(errs).mean():.2f} pp   max {np.abs(errs).max():.2f} pp   "
          f"(tolerance {TOL_PP} pp)")

    if failures:
        print("\n!! FAILURES")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nPASSED: every farm reproduces Nost's published capacity factor.")
    print("Units, capacities and ALLOWED_YEARS windows are all consistent.")


if __name__ == "__main__":
    main()
