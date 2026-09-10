#!/usr/bin/env python3
"""Find the forecast_*.nc file that kills the reader.

A malformed netCDF does not raise -- the C library walks off the end of a buffer and the whole
process dies ("free(): invalid size", "free(): invalid pointer", segfault). Nothing in Python can
catch that, so the only way to name the file is to open each one in its OWN process and see which
child fails to come back.

Phase 1 flags size outliers, which is what a truncated write looks like, and costs one stat call
per file. Phase 2 opens every file in a subprocess and reads exactly what verify_power.py reads
-- the variables, at the same indices -- because a file can have an intact header and a corrupt
data chunk, and only touching the data finds that.

Prints the bad files at the end. Delete or re-run inference for those, nothing else.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# ============================== SETTINGS ==============================
FORECAST_DIRS = [
    Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacity_Finetune"),
]
ONLY_MONTHS = [11]           # month numbers from the filename, [] for all. The crash was traced
                             # to November of the fine-tune run, so scan that and nothing else.
WS_VAR = "ws100"
CF_VAR = "capacityfactor"
NC_ENGINE = "netcdf4"        # scan with the SAME engine verify_power.py uses, or the scan can
                             # pass files that still crash the real run
SIZE_TOL = 0.02              # phase 1: flag files more than this fraction off the median size
WORKERS = 8                  # subprocesses in flight; each one is short-lived
TIMEOUT = 120                # seconds per file before it counts as a hang
# ======================================================================

CHILD = """
import sys, xarray as xr
p, eng, ws, cf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with xr.open_dataset(p, engine=eng) as ds:
    ds["latitude"].values; ds["longitude"].values; ds["time"].values
    ds[ws].values                       # touch the data, not just the header
    if cf in ds:
        ds[cf].values
"""


def check(path: Path) -> tuple[Path, str]:
    """(file, verdict). Anything other than 'ok' means this file breaks the reader."""
    r = subprocess.run(
        [sys.executable, "-c", CHILD, str(path), NC_ENGINE, WS_VAR, CF_VAR],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    if r.returncode == 0:
        return path, "ok"
    if r.returncode < 0:
        return path, f"KILLED by signal {-r.returncode} (heap corruption / segfault)"
    tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
    return path, f"exit {r.returncode}: {tail[:120]}"


def main() -> None:
    for d in FORECAST_DIRS:
        files = sorted(d.glob("forecast_*.nc"))
        if ONLY_MONTHS:
            # forecast_YYYYMMDDHHMMSS.nc -> month is the 5th and 6th digit
            files = [f for f in files if int(f.name[13:15]) in ONLY_MONTHS]
        if not files:
            raise SystemExit(f"no forecast_*.nc in {d} for months {ONLY_MONTHS}")
        print(f"\n{'='*78}\n{d}\n{len(files)} files"
              + (f"  (months {ONLY_MONTHS})" if ONLY_MONTHS else "") + f"\n{'='*78}")

        # ---- phase 1: size outliers (a truncated write is visible without opening anything)
        sizes = np.array([f.stat().st_size for f in files], dtype=float)
        med = float(np.median(sizes))
        off = np.abs(sizes - med) / med
        odd = np.where(off > SIZE_TOL)[0]
        print(f"median size {med/1e6:.2f} MB   min {sizes.min()/1e6:.2f}   max {sizes.max()/1e6:.2f}")
        if odd.size:
            print(f"  {odd.size} size outlier(s) >{100*SIZE_TOL:.0f}% off the median:")
            for i in odd:
                print(f"    {files[i].name}  {sizes[i]/1e6:8.2f} MB  ({100*off[i]:+.1f}%)")
        else:
            print("  no size outliers -- if a file is bad it is corrupt inside, not truncated")

        # ---- phase 2: open every file in its own process
        print(f"\nopening each file in a subprocess ({WORKERS} at a time) ...")
        bad = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for n, (path, verdict) in enumerate(ex.map(check, files), 1):
                if verdict != "ok":
                    bad.append((path, verdict))
                    print(f"  BAD  {path.name}  {verdict}", flush=True)
                if n % 200 == 0:
                    print(f"  {n}/{len(files)}", flush=True)

        print(f"\n{len(bad)} bad file(s) in {d.name}")
        for path, verdict in bad:
            print(f"  {path}\n      {verdict}")


if __name__ == "__main__":
    main()
