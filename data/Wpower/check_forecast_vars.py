#!/usr/bin/env python3
"""Which forecast files are missing a variable, and when were they written?

Scans one or more inference directories and reports, per directory:
  - how many forecast_*.nc have / lack the variable
  - the INIT-DATE range of each group (are the missing ones an early block?)
  - the FILE MTIME range of each group (are they leftovers from an older inference pass?)
  - the first and last few offending filenames

The mtime split is the useful bit: if the files lacking the variable were all written days before
the ones that have it, the directory is a mix of two inference passes and the re-run did not
overwrite everything (it probably covered a different set of init times).

Usage:
  python check_forecast_vars.py DIR [DIR ...] [--var capacityfactor] [--list 10]
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

FORECAST_RE = re.compile(r"forecast_(\d{14})")


def var_names(path: Path) -> set[str]:
    """Variable names from the file header only (fast: no data read)."""
    try:
        import netCDF4
        with netCDF4.Dataset(str(path), "r") as ds:
            return set(ds.variables.keys())
    except Exception:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            return set(ds.variables.keys())


def init_of(path: Path) -> str:
    m = FORECAST_RE.search(path.name)
    return m.group(1) if m else "?"


def fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--var", default="capacityfactor")
    ap.add_argument("--list", type=int, default=10, help="how many offenders to print")
    args = ap.parse_args()

    for d in args.dirs:
        files = sorted(d.glob("forecast_*.nc"))
        print(f"\n{'=' * 78}\n{d}\n{'=' * 78}")
        if not files:
            print("  no forecast_*.nc here")
            continue

        have, lack = [], []
        for i, f in enumerate(files):
            (have if args.var in var_names(f) else lack).append(f)
            if i and i % 500 == 0:
                print(f"  scanned {i}/{len(files)}...", flush=True)

        print(f"  {len(files)} files | WITH '{args.var}': {len(have)} | WITHOUT: {len(lack)}")

        for name, group in (("WITH   ", have), ("WITHOUT", lack)):
            if not group:
                continue
            inits = [init_of(f) for f in group]
            mt = [f.stat().st_mtime for f in group]
            print(f"    {name}: init {min(inits)} .. {max(inits)}   "
                  f"written {fmt(min(mt))} .. {fmt(max(mt))}")

        if lack:
            print(f"\n  first {min(args.list, len(lack))} without '{args.var}':")
            for f in lack[:args.list]:
                print(f"    {f.name}   written {fmt(f.stat().st_mtime)}")
            if len(lack) > args.list:
                print(f"  last {min(args.list, len(lack))}:")
                for f in lack[-args.list:]:
                    print(f"    {f.name}   written {fmt(f.stat().st_mtime)}")

            # Is it a clean time split (two passes) or interleaved?
            hv = sorted(f.stat().st_mtime for f in have) if have else []
            lv = sorted(f.stat().st_mtime for f in lack)
            if hv and max(lv) < min(hv):
                print("\n  => ALL the missing files were written BEFORE every good one:"
                      "\n     a leftover older inference pass. The re-run did not cover these"
                      "\n     init times, so the old files were never overwritten.")
            elif hv and min(lv) > max(hv):
                print("\n  => the missing files are the NEWEST: the latest pass wrote files"
                      "\n     without the variable (wrong checkpoint/config?).")
            elif hv:
                print("\n  => mtimes are interleaved: not a clean two-pass split.")


if __name__ == "__main__":
    main()
