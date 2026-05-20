from pathlib import Path
import pandas as pd
import re
from itertools import combinations

DIRS = {
    "GNN":      Path("/mnt/weatherloss/WindPower/inference/CI/GNNLAM"),
    "GT": Path("/mnt/weatherloss/WindPower/inference/CI/GTLAM"),
    "TF": Path("/mnt/weatherloss/WindPower/inference/CI/TFLAM"),
}

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")

FORECAST_RE = re.compile(r"forecast_(\d{14})")

expected = pd.date_range(INIT_START, INIT_END, freq="3h", tz="UTC")


def extract_inits(directory: Path) -> pd.DatetimeIndex:
    times = []
    for f in directory.glob("forecast_*.nc"):
        m = FORECAST_RE.search(f.name)
        if not m:
            continue
        t = pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S", utc=True)
        if INIT_START <= t <= INIT_END:
            times.append(t)
    return pd.DatetimeIndex(sorted(set(times)))


def main():
    print(f"Expected (3-hourly {INIT_START.date()} – {INIT_END.date()}): {len(expected)}\n")

    inits = {label: extract_inits(d) for label, d in DIRS.items()}

    # Per-directory summary
    print(f"{'Directory':<20}  {'Files':>6}  {'Missing':>7}  {'Coverage':>9}")
    print("-" * 50)
    for label, idx in inits.items():
        missing = len(expected.difference(idx))
        pct = len(idx) / len(expected) * 100
        print(f"{label:<20}  {len(idx):>6}  {missing:>7}  {pct:>8.1f}%")

    # Pairwise intersections
    print("\nPairwise intersections:")
    print(f"  {'Pair':<45}  {'Common':>6}")
    print("  " + "-" * 53)
    for (a, ia), (b, ib) in combinations(inits.items(), 2):
        print(f"  {a} ∩ {b:<25}  {len(ia.intersection(ib)):>6}")

    # Full intersection
    common_all = inits[list(inits)[0]]
    for idx in list(inits.values())[1:]:
        common_all = common_all.intersection(idx)

    print(f"\nAll four in common: {len(common_all)} / {len(expected)} ({len(common_all)/len(expected)*100:.1f}%)")

    # Show first 10 missing with which dirs have them
    missing_all = expected.difference(common_all)
    if len(missing_all):
        print(f"\nFirst 10 timesteps missing from the common set:")
        for t in missing_all[:10]:
            have = [l for l, idx in inits.items() if t in idx]
            print(f"  {t}  present in: {have if have else 'none'}")


if __name__ == "__main__":
    main()
