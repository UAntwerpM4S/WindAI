"""
Step 3 of the Wpower build. Turn the raw combined observations into the canonical
per-FARM power table that the distributed-power target consumes.

    BE_UK_offshore_per_unit_3H_meanMW_shifted.csv   (35 obs columns)
        -> power_obs.csv                            (31 farm columns, matching farms.csv)

Why a separate file rather than using the raw one directly
    The raw file is keyed by ENTSO-E/Elexon OBS COLUMN, not by farm:
      - Walney1 and Walney2 are separate columns, but they are ONE farm here (a single OSM
        polygon covers both, and both are 51 x SWT-3.6-107 / 184 MW). They must be SUMMED.
      - Aberdeen, BurboBank and BurboBankExtension are present but excluded from the farm
        set (see EXCLUDED in farm_metadata.py).
      - SheringhamShoals is NOT in the raw file at all -- Nost excludes it as never having
        run at full capacity, so it is correctly absent.
    power_obs.csv is 1:1 with farms.csv, so build_power.py can join on `farm` with no
    per-column special cases.

NaN rule when summing columns into one farm
    If EITHER Walney1 or Walney2 is NaN, the merged farm is NaN. Power is EXTENSIVE: you
    cannot sum a known and an unknown. (Contrast wpx, an intensive quantity, where a missing
    contributor can simply drop out of a weighted mean.) The same rule applies downstream
    when several farms share a grid cell.

Physical clamp
    A capacity factor above CF_MAX is impossible and indicates corrupt data, so those
    samples are set to NaN rather than trained on. In practice this fires on exactly ONE
    record in the whole dataset (GwyntyMor 2022-11-15 14:30, CF 1.97 in an otherwise healthy
    year whose 99th percentile is 0.97). Cheap insurance against a CF ~2.0 entering the target.

Units: mean MW over the 3 hours beginning at the timestamp. Verified aligned between BE and
UK -- see README ("Time convention").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WPOWER_DIR = Path(__file__).resolve().parent
FARMS_CSV = WPOWER_DIR / "farms.csv"
OUT_CSV = WPOWER_DIR / "power_obs.csv"

# Look in the repo first, then the server path build_wpx.py uses.
RAW_CANDIDATES = [
    WPOWER_DIR / "BE_UK_offshore_per_unit_3H_meanMW_shifted.csv",
    Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/BE_UK_offshore_per_unit_3H_meanMW_shifted.csv"),
]

CF_MAX = 1.05


def raw_path() -> Path:
    for p in RAW_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"no raw observations at any of {RAW_CANDIDATES}")


def main() -> None:
    src = raw_path()
    raw = pd.read_csv(src)
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    raw = raw.set_index("time").sort_index()

    farms = pd.read_csv(FARMS_CSV)

    missing = [c for _, r in farms.iterrows() for c in r.obs_columns.split(";")
               if c not in raw.columns]
    if missing:
        raise SystemExit(f"obs columns missing from {src.name}: {missing}")

    out = pd.DataFrame(index=raw.index)
    clamped = 0

    print(f"source: {src}")
    print(f"{'farm':18s} {'reg':>3s} {'cap MW':>7s} {'valid%':>7s} {'maxCF':>6s} {'meanCF':>7s}  cols")
    print("-" * 78)

    for _, r in farms.iterrows():
        cols = r.obs_columns.split(";")
        block = raw[cols].astype(float)

        # extensive: any contributor NaN -> the farm total is unknown
        s = block.sum(axis=1, min_count=len(cols))

        cf = s / r.capacity_mw
        bad = cf > CF_MAX
        clamped += int(bad.sum())
        s = s.mask(bad)

        out[r.farm] = s
        v = s.dropna()
        print(f"{r.farm:18s} {r.region:>3s} {r.capacity_mw:7.1f} "
              f"{100 * len(v) / len(s):6.1f}% {v.max() / r.capacity_mw:6.2f} "
              f"{v.mean() / r.capacity_mw:7.2f}  {'+'.join(cols)}")

    print("-" * 78)
    print(f"rows {len(out)}   {out.index.min()} -> {out.index.max()}")
    print(f"clamped as CF > {CF_MAX}: {clamped} sample(s)")

    # fleet totals only make sense where every farm reports, which is never; report per region
    for reg in ("BE", "UK"):
        f = farms[farms.region == reg]
        sub = out[f.farm.tolist()]
        any_valid = sub.notna().any(axis=1)
        all_valid = sub.notna().all(axis=1)
        print(f"{reg}: {len(f):2d} farms  timesteps with ANY farm reporting "
              f"{100 * any_valid.mean():5.1f}%   with ALL reporting {100 * all_valid.mean():5.1f}%")

    out.round(3).to_csv(OUT_CSV)
    print(f"\nwrote {OUT_CSV.name}  ({len(out)} rows x {len(out.columns)} farms, mean MW)")

    bad_cf = [c for c in out.columns
              if (out[c] / farms.set_index("farm").loc[c, "capacity_mw"]).max() > CF_MAX]
    if bad_cf:
        raise SystemExit(f"!! farms still exceeding CF {CF_MAX}: {bad_cf}")
    print("every farm's observed power stays within its nameplate capacity.")


if __name__ == "__main__":
    main()
