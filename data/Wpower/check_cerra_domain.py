"""
Do all 31 farms fall inside the CERRA cutout, and how many grid cells does the target
actually land on?

This is the one question that cannot be answered off the server, and it decides the
experiment: the whole point of adding the UK farms is to raise the number of cells carrying
a target from Belgium's ~16 to something the graph can learn a spatial mapping from. Until
this runs, that number is a lat/lon-binning ESTIMATE (~184), not the real grid.

It also prints the neutral loss weight -- the factor that puts the power channel on equal
footing with a weight-1.0 full-field variable. anemoi's loss SUMS over grid cells and lets
`node_weights: unit-sum` normalise it, while `general_variable` weights are applied raw. So a
target defined at only N of M cells gets N/M of a full-field variable's grid weight. With
N ~ 184 and M ~ 72668 that is ~0.25%, i.e. a neutral weight of ~400. Get this wrong and the
power channel never trains, which is very likely what happened in VanillaPowerGT
(power: 0.0001 -> effectively ~1e-8).

Run:  python check_cerra_domain.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree

WPOWER_DIR = Path(__file__).resolve().parent
TURBINES_CSV = WPOWER_DIR / "turbines.csv"

CERRA_CANDIDATES = [
    Path("/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/New_Cerra_A_large.zarr"),
    Path("/mnt/weatherloss/WindPowerProxy/data/Anemoidatasets/proxy_cerra_A.zarr"),
]

EARTH_R_KM = 6371.0
# A turbine further than this from any grid cell centre is outside the domain. CERRA is
# ~5.5 km, so half a diagonal is ~3.9 km; 10 km is generous and only flags real misses.
OUTSIDE_KM = 10.0


def find_zarr() -> Path:
    for p in CERRA_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("no CERRA zarr found in:\n  " + "\n  ".join(str(p) for p in CERRA_CANDIDATES)
                     + "\nEdit CERRA_CANDIDATES to point at the inner dataset.")


def load_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    g = zarr.open(str(path), mode="r")
    lat = np.asarray(g["latitudes"]).ravel()
    lon = np.asarray(g["longitudes"]).ravel()
    lon = np.where(lon > 180.0, lon - 360.0, lon)   # anemoi stores 0..360
    return lat, lon


def to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Unit sphere -> chordal KD-tree distances behave correctly near the meridian."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def main() -> None:
    zpath = find_zarr()
    lat, lon = load_grid(zpath)
    print(f"CERRA grid: {zpath}")
    print(f"  {len(lat)} cells   lat {lat.min():.2f}..{lat.max():.2f}   "
          f"lon {lon.min():.2f}..{lon.max():.2f}\n")

    t = pd.read_csv(TURBINES_CSV)
    tree = cKDTree(to_xyz(lat, lon))
    chord, idx = tree.query(to_xyz(t.latitude.to_numpy(), t.longitude.to_numpy()), k=1)
    # chord length on the unit sphere -> great-circle km
    t["dist_km"] = 2.0 * EARTH_R_KM * np.arcsin(np.clip(chord / 2.0, 0, 1))
    t["cell"] = idx

    print(f"{'farm':18s} {'reg':>3s} {'turb':>4s} {'cells':>5s} {'maxdist km':>10s}  status")
    print("-" * 62)
    outside = []
    for farm, g in t.groupby("farm", sort=False):
        far = g.dist_km.max()
        ok = far <= OUTSIDE_KM
        if not ok:
            outside.append(farm)
        print(f"{farm:18s} {g.region.iloc[0]:>3s} {len(g):4d} {g.cell.nunique():5d} "
              f"{far:10.1f}  {'in' if ok else 'OUTSIDE DOMAIN'}")

    inside = t[~t.farm.isin(outside)]
    cells = inside.cell.nunique()
    n_grid = len(lat)

    print("-" * 62)
    print(f"farms inside: {inside.farm.nunique()} of {t.farm.nunique()}"
          + (f"   OUTSIDE: {outside}" if outside else ""))
    print(f"turbines inside: {len(inside)} of {len(t)}")
    print()
    print(f"TARGET CELLS: {cells} of {n_grid} grid cells ({100 * cells / n_grid:.3f}%)")

    per = inside.groupby("cell").farm.nunique()
    print(f"  cells hosting >1 farm (contributions SUM): {(per > 1).sum()}")
    print(f"  capacity placed: {inside.capacity_mw.sum():.1f} MW")

    be = inside[inside.region == "BE"]
    print(f"  Belgium alone (the evaluation target): {be.cell.nunique()} cells")

    print()
    print("LOSS WEIGHTING (see module docstring)")
    print(f"  neutral weight for the power channel ~ {n_grid / cells:.0f}")
    print(f"  (Belgium-only would have needed ~{n_grid / max(be.cell.nunique(), 1):.0f})")
    print("  Sweep around the neutral value, e.g. 0.3x / 1x / 3x. Watch ws100 validation")
    print("  as the guardrail. Node weights are area-based, so treat this as the right")
    print("  order of magnitude rather than an exact figure.")


if __name__ == "__main__":
    main()
