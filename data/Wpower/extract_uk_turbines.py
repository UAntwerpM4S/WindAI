"""
Step 1 of the Wpower metadata build. Extract per-farm UK offshore turbine coordinates
from the two OSM GeoJSON exports (in ../NorthSea/Power/), writing CSVs into
Wpower/coordinates/ in the same NAME,LONGITUDE,LATITUDE format as the Belgian
BOZ_Turbines/coordinates/, so both can be globbed by the same loader.

Run this BEFORE farm_metadata.py, which consumes the CSVs written here.

Inputs
    UKWindfarms.geojson          17112 unnamed turbine Points (all UK, onshore included)
    UKPolygoncoordinates.geojson named wind-plant polygons

Method
    Point-in-polygon. Each turbine point is assigned to the named plant polygon that
    contains it. Validated against the curated turbine counts in build_metadata.py:
    20 of 21 polygon-matched farms reproduce the curated count exactly.

Naming traps (why FARM_TO_POLY is an explicit dict and not fuzzy matching)
    - OSM writes 'Hornsea 1', the obs write 'HornseaOne'
    - OSM writes 'Sheringham Shoal' (singular), the obs write 'SheringhamShoals'
    - 'Walney Wind Farm' is an UMBRELLA over 'Walney 1 & 2' + 'Walney 3'; matching it
      naively double-counts. The 'Walney 3' polygon is also broken (20 points, should be
      87), so Walney Extension is recovered as umbrella MINUS 'Walney 1 & 2' == 87 exactly.
    - 'Walney 1 & 2' is one polygon over two obs columns (51 + 51). Safe to merge: both
      are 51 x SWT-3.6-107 / 184 MW, so capacity-share == count-share and the only
      assumption is equal capacity factor between two identical adjacent arrays.

Excluded
    - Burbo Bank (+ Extension): one OSM polygon over two farms with DIFFERENT machines
      (25 x 3.6 MW and 32 x 8.0 MW), spatially interleaved with no recoverable boundary
      (k-means -> 36/21, spacing -> 56/1, principal-axis cut at the known counts -> an
      86 m gap against a 165 m median). Merging would spread 348 MW evenly over 57
      turbines: +70% into the original's cells, -24% into the Extension's. Dropped.
    - The six Scottish farms (Aberdeen, Beatrice, Hywind, Kincardine, Moray East,
      Seagreen): the turbine-point export is bbox-clipped at 56.5 N, so no coordinates
      exist for them at all. Would need a fresh Overpass export.

Derived footprints (no usable OSM polygon, but the turbine points are present)
    - Rampion: no polygon in the export at all.
    - Gunfleet Sands: OSM has only the 12 MW / 2-turbine 'Gunfleet Sands 3' demo, which is
      a SEPARATE ENTSO-E unit -- its output is not in the GunfleetSands obs column, so those
      2 turbines are subtracted, leaving the 48 x SWT-3.6-107 of Gunfleet Sands 1&2.
    Both footprints are recovered by DBSCAN around the curated centroid and validated
    against the curated turbine count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from sklearn.cluster import DBSCAN

WPOWER_DIR = Path(__file__).resolve().parent          # WindAI/data/Wpower
GEOJSON_DIR = WPOWER_DIR.parent / "NorthSea" / "Power"  # where the OSM exports live
TURBINE_GEOJSON = GEOJSON_DIR / "UKWindfarms.geojson"
POLYGON_GEOJSON = GEOJSON_DIR / "UKPolygoncoordinates.geojson"
OUT_DIR = WPOWER_DIR / "coordinates"

EARTH_KM_PER_DEG = 111.0


# -----------------------------------------------------------------------------
# Farm table. capacity_mw / turbines / fleet are the curated values (build_metadata.py);
# `turbines` doubles as the validation target for the geometry join.
# Walney1+Walney2 are merged (one polygon, identical machines).
# -----------------------------------------------------------------------------
FARMS: dict[str, dict] = {
    "Barrow":            dict(poly="Barrow Wind Farm",                       cap=90.0,   n=30,  fleet={"VestasV90-3.0": 30}),
    "Dudgeon":           dict(poly="Dudgeon Offshore Wind Farm",             cap=402.0,  n=67,  fleet={"SiemensSWT-6.0-154": 67}),
    "EastAngliaOne":     dict(poly="East Anglia ONE",                        cap=714.0,  n=102, fleet={"SiemensSWT-7.0-154": 102}),
    "Galloper":          dict(poly="Galloper Wind Farm",                     cap=353.0,  n=56,  fleet={"SiemensSWT-6.0-154": 56}),
    "GreaterGabbard":    dict(poly="Greater Gabbard Wind Farm",              cap=504.0,  n=140, fleet={"SiemensSWT-3.6-107": 140}),
    "GwyntyMor":         dict(poly="Gwynt y Môr Offshore Wind Farm",         cap=576.0,  n=160, fleet={"SiemensSWT-3.6-107": 160}, tol=2),
    "HornseaOne":        dict(poly="Hornsea 1 Offshore Wind Farm",           cap=1218.0, n=174, fleet={"SiemensSWT-7.0-154": 174}),
    "HornseaTwo":        dict(poly="Hornsea 2 Offshore Wind Farm",           cap=1386.0, n=165, fleet={"SiemensSG-8.0-167DD": 165}),
    "HumberGateway":     dict(poly="Humber Gateway Wind Farm",               cap=219.0,  n=73,  fleet={"VestasV112-3.0": 73}),
    "Lincs":             dict(poly="Lincs Wind Farm",                        cap=270.0,  n=75,  fleet={"SiemensSWT-3.6-120": 75}),
    "LondonArray":       dict(poly="London Array Wind Farm",                 cap=630.0,  n=175, fleet={"SiemensSWT-3.6-107": 175}),
    "Ormonde":           dict(poly="Ormonde Wind Farm",                      cap=150.0,  n=30,  fleet={"Repower-5": 30}),
    "RaceBank":          dict(poly="Race Bank Wind Farm",                    cap=580.0,  n=91,  fleet={"SiemensSWT-6.0-154": 91}),
    "RobinRigg":         dict(poly="Robin Rigg Offshore Wind Farm",          cap=180.0,  n=60,  fleet={"VestasV90-3.0": 60}, tol=2),
    "TritonKnoll":       dict(poly="Triton Knoll Wind Farm",                 cap=857.0,  n=90,  fleet={"MHIVestas-9.5-V164": 90}),
    "Walney":            dict(poly="Walney 1 & 2",                           cap=368.0,  n=102, fleet={"SiemensSWT-3.6-107": 102},
                              merged_obs=("Walney1", "Walney2")),
    "WalneyExtension":   dict(poly="Walney Wind Farm", minus="Walney 1 & 2", cap=659.0,  n=87,
                              fleet={"SiemensSWT-7.0-154": 40, "MHIVestas-8.25-V164": 47}),
    "WestOfDuddonSands": dict(poly="West of Duddon Sands Wind Farm",         cap=389.0,  n=108, fleet={"SiemensSWT-3.6-120": 108}),
    "WestermostRough":   dict(poly="Westermost Rough Wind Farm",             cap=210.0,  n=35,  fleet={"SiemensSWT-6.0-154": 35}),
    # --- derived footprints: turbine points exist, no usable polygon ---
    "Rampion":           dict(cluster=(50.6667, -0.2667),                    cap=400.0,  n=116, fleet={"VestasV112-3.45": 116}, tol=4),
    # Gunfleet Sands 1&2 only. The 2 x SWT-6.0-120 demo turbines inside the 'Gunfleet
    # Sands 3' polygon are a SEPARATE 12 MW ENTSO-E unit whose output is not in the
    # GunfleetSands obs column, so they must not be handed any of its power.
    "GunfleetSands":     dict(cluster=(51.7167, 1.2139), minus="Gunfleet Sands 3 Offshore Wind Farm",
                              cap=172.8,  n=48, fleet={"SiemensSWT-3.6-107": 48}, tol=4),
}


def load_points() -> np.ndarray:
    feats = json.loads(TURBINE_GEOJSON.read_text())["features"]
    return np.array([f["geometry"]["coordinates"][:2] for f in feats
                     if f["geometry"]["type"] == "Point"])


def load_rings() -> dict[str, list[np.ndarray]]:
    rings: dict[str, list[np.ndarray]] = {}
    for f in json.loads(POLYGON_GEOJSON.read_text())["features"]:
        name = f["properties"].get("name")
        geom = f["geometry"]
        if not name:
            continue
        if geom["type"] == "MultiPolygon":
            parts = [np.asarray(p[0]) for p in geom["coordinates"]]
        elif geom["type"] == "Polygon":
            parts = [np.asarray(geom["coordinates"][0])]
        else:
            continue
        rings.setdefault(name, []).extend(parts)
    return rings


def inside(pts: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    mask = np.zeros(len(pts), dtype=bool)
    for ring in rings:
        mask |= MplPath(ring).contains_points(pts)
    return mask


def cluster_near(pts: np.ndarray, lat: float, lon: float, expect: int) -> np.ndarray:
    """Recover a farm footprint that OSM has no polygon for: DBSCAN around the curated
    centroid, then keep the cluster whose size is closest to the curated turbine count."""
    near = (np.abs(pts[:, 1] - lat) < 0.35) & (np.abs(pts[:, 0] - lon) < 0.55)
    idx = np.flatnonzero(near)
    if idx.size == 0:
        return np.zeros(len(pts), dtype=bool)

    local = np.c_[pts[idx, 0] * EARTH_KM_PER_DEG * np.cos(np.radians(lat)),
                  pts[idx, 1] * EARTH_KM_PER_DEG]
    labels = DBSCAN(eps=2.0, min_samples=3).fit_predict(local)

    best, best_err = None, None
    for lab in set(labels) - {-1}:
        sel = labels == lab
        err = abs(int(sel.sum()) - expect)
        if best_err is None or err < best_err:
            best, best_err = sel, err

    mask = np.zeros(len(pts), dtype=bool)
    if best is not None:
        mask[idx[best]] = True
    return mask


def main() -> None:
    pts = load_points()
    rings = load_rings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"turbine points in export: {len(pts)}")
    print(f"writing to: {OUT_DIR}\n")
    print(f"{'farm':20s} {'expect':>6s} {'found':>6s} {'MW':>7s} {'MW/turb':>8s}  status")
    print("-" * 74)

    claimed = np.zeros(len(pts), dtype=bool)
    rows, failures = [], []

    for farm, spec in FARMS.items():
        if "cluster" in spec:
            mask = cluster_near(pts, *spec["cluster"], spec["n"])
        else:
            if spec["poly"] not in rings:
                failures.append(f"{farm}: polygon {spec['poly']!r} not in export")
                continue
            mask = inside(pts, rings[spec["poly"]])
        if "minus" in spec:      # Walney Ext = umbrella - Walney 1&2; Gunfleet - the GS3 demo
            mask &= ~inside(pts, rings[spec["minus"]])

        overlap = int((mask & claimed).sum())
        claimed |= mask
        n_found = int(mask.sum())
        tol = spec.get("tol", 0)
        ok = abs(n_found - spec["n"]) <= tol and overlap == 0

        status = "OK" if ok else f"MISMATCH ({n_found - spec['n']:+d})"
        if overlap:
            status += f" !! {overlap} pts double-claimed"
        if not ok:
            failures.append(f"{farm}: expected {spec['n']}, found {n_found}, overlap {overlap}")

        mw_per = spec["cap"] / n_found if n_found else float("nan")
        print(f"{farm:20s} {spec['n']:>6d} {n_found:>6d} {spec['cap']:>7.1f} {mw_per:>8.2f}  {status}")

        farm_pts = pts[mask]
        # sort N->S then W->E so NAME is stable across reruns
        order = np.lexsort((farm_pts[:, 0], -farm_pts[:, 1]))
        farm_pts = farm_pts[order]
        df = pd.DataFrame({
            "NAME": [f"{farm}_{i + 1:03d}" for i in range(len(farm_pts))],
            "LONGITUDE": farm_pts[:, 0],
            "LATITUDE": farm_pts[:, 1],
        })
        df.to_csv(OUT_DIR / f"{farm}_turbines_coords.csv", index=False)
        rows.append(dict(farm=farm, turbines=n_found, capacity_mw=spec["cap"],
                         mw_per_turbine=mw_per,
                         fleet="; ".join(f"{v}x {k}" for k, v in spec["fleet"].items()),
                         obs_columns=";".join(spec.get("merged_obs", (farm,)))))

    print("-" * 74)
    summary = pd.DataFrame(rows)
    print(f"{'TOTAL':20s} {summary.turbines.sum():>13d} {summary.capacity_mw.sum():>7.1f} MW")
    summary.to_csv(OUT_DIR / "uk_farm_summary.csv", index=False)
    print(f"\nwrote {len(rows)} farm CSVs + uk_farm_summary.csv")

    if failures:
        print("\n!! VALIDATION FAILURES")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nall farms reproduce their curated turbine count.")


if __name__ == "__main__":
    main()
