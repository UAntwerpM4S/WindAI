"""
Authoritative farm metadata for the distributed-power target: Belgium + UK.

This is the single source of truth for (farm -> obs column, coordinates, fleet, capacity).
Everything downstream (capacity-share distribution, per-cell capacity forcing, cut-out
masking) reads from here. Run this module directly to print a full audit and write
farm_metadata_full.csv.

WHY CAPACITY, NOT TURBINE COUNT
    The target distributes each farm's observed MW across the cells its turbines occupy:

        power(cell,t) = P_obs(farm,t) * capacity(farm,cell) / capacity_total(farm)

    Weighting by turbine COUNT is only correct when a farm's turbines are identical.
    Belwind, C-Power and Walney Extension have mixed fleets, where count-weighting
    misallocates power *within* a single farm. Capacity-weighting is identical for
    uniform farms and correct for mixed ones. Where the individual machines of a mixed
    fleet cannot be told apart on the map, per-turbine capacity falls back to the farm
    mean (capacity_mw / n_turbines), which is no worse than count-weighting.
    See MIXED_FLEET_UNRESOLVED below for exactly where that happens.

CAPACITY IS EXTENSIVE
    A cell hosting turbines from two farms SUMS their contributions. Contrast the wpx
    (wind speed) target, which is intensive and averages.

RECONCILIATION (all three numbers now agree, which no previous table managed)
    399 turbines, 2261.2 MW for Belgium -- matching the independently verified totals.
    The fix was finding a misfiled coordinate; see BELWIND_HALIADE below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WPOWER_DIR = Path(__file__).resolve().parent     # WindAI/data/Wpower
DATA_DIR = WPOWER_DIR.parent                     # WindAI/data
COORDS = WPOWER_DIR / "coordinates"              # canonical: one CSV per farm, all 31
SPECS_CSV = WPOWER_DIR / "turbine_specs.csv"

# Belgian RAW coordinates. Left in place as source data (other scripts read them), and they
# need cleaning before use: hub rows removed, C-Power split into its two ENTSO-E units, and
# the Haliade reassigned out of Nobelwind's file. This module does that cleaning and writes
# the cleaned per-farm CSVs into COORDS, so `coordinates/` ends up 1:1 with farms.csv.
BE_RAW_COORDS = DATA_DIR / "BOZ_Turbines" / "coordinates"

FARMS_CSV = WPOWER_DIR / "farms.csv"        # one row per farm
TURBINES_CSV = WPOWER_DIR / "turbines.csv"  # one row per turbine, WITH per-turbine capacity

# Substation / offshore-hub records that must never be treated as turbines. Needs a
# SUBSTRING match, not a word-boundary one: BW_OHVS, NWOHVS and OTS all escape \b.
HUB_PATTERN = r"OHVS|OSS|OTS"

# -----------------------------------------------------------------------------
# BELWIND_HALIADE -- a misfiled coordinate, found by geometry
#
# Belwind is 55 x Vestas V90-3.0 PLUS one 6 MW Alstom Haliade 150 demonstrator
# (171 MW / 56 turbines). But Belwind_turbines_coords.csv holds only 55 turbines
# + BW_OHVS, and Nobelwind_turbines_coords.csv holds 51 records for what is really a
# 50-turbine / 165 MW farm. The extra Nobelwind record (NAME == 1) sits 350 m from a
# Belwind turbine but 1036 m from the nearest Nobelwind turbine, against a median
# Nobelwind spacing of 519 m -- i.e. it is inside Belwind's array, not Nobelwind's.
# That is the Haliade. Reassigning it makes turbines (399) and capacity (2261.2 MW)
# both reconcile exactly; leaving it in place breaks both.
# -----------------------------------------------------------------------------
HALIADE_IN_NOBELWIND_CSV = "1"

# -----------------------------------------------------------------------------
# Farms. `fleet` maps a turbine_specs key -> count. `capacity_mw` is the as-built
# nameplate; it is CHECKED against sum(count * rated_power_mw), not derived from it,
# so a disagreement surfaces as a validation failure rather than passing silently.
#
# `obs` lists the column(s) in BE_UK_offshore_per_unit_3H_meanMW_shifted.csv. More than
# one means the columns are summed (the coordinates cannot separate them).
# -----------------------------------------------------------------------------
FARMS: dict[str, dict] = {
    # ---------------- Belgium (10 farms, 399 turbines, 2261.2 MW) ----------------
    "Belwind": dict(
        region="BE", obs=["Belwind Phase 1"], coords="Belwind",
        fleet={"Vestas-3-V90": 55, "AlstomHaliade-6-150": 1}, capacity_mw=171.0,
        extra_coords=("Nobelwind", HALIADE_IN_NOBELWIND_CSV),
        notes="Haliade coordinate misfiled in Nobelwind CSV; see BELWIND_HALIADE.",
    ),
    "Nobelwind": dict(
        region="BE", obs=["Nobelwind Offshore Windpark"], coords="Nobelwind",
        fleet={"Vestas-3.3-V112": 50}, capacity_mw=165.0,
        drop_coords=(HALIADE_IN_NOBELWIND_CSV,),
        notes="51 records in CSV; record 1 is Belwind's Haliade, dropped here.",
    ),
    "CPower_SW": dict(
        region="BE", obs=["Thorntonbank - C-Power - Area SW"], coords="CPower",
        coord_filter="SW",
        fleet={"Repower-5": 6, "Repower-6.15-M126": 24}, capacity_mw=177.6,
        notes="Phase-1 6x REpower 5M assumed to sit in the SW area (see MIXED_FLEET_UNRESOLVED).",
    ),
    "CPower_NE": dict(
        region="BE", obs=["Thorntonbank - C-Power - Area NE"], coords="CPower",
        coord_filter="NE",
        fleet={"Repower-6.15-M126": 24}, capacity_mw=147.6,
        notes="Split by turbine name: A,B,C,D -> SW (30); E-J -> NE (24).",
    ),
    "Mermaid": dict(
        region="BE", obs=["Mermaid Offshore WP"], coords="Mermaid",
        fleet={"SiemensGamesaSG-8.4-167DD": 28}, capacity_mw=235.2,
        notes="Uprated 8.4 MW, not the nominal SG-8.0 the old counts table pointed at.",
    ),
    "Seastar": dict(
        region="BE", obs=["Seastar Offshore WP"], coords="Seastar",
        fleet={"SiemensGamesaSG-8.4-167DD": 30}, capacity_mw=252.0,
        notes="windfarm_metadata.csv said 20 turbines / 168 MW -- wrong; CSV has 30.",
    ),
    "Norther": dict(
        region="BE", obs=["Norther Offshore WP"], coords="Norther",
        fleet={"MHIVestas-8.4-V164": 44}, capacity_mw=369.6,
    ),
    "Northwester2": dict(
        region="BE", obs=["Northwester 2"], coords="Northwester2",
        fleet={"MHIVestas-9.5-V164": 23}, capacity_mw=218.5,
    ),
    "Northwind": dict(
        region="BE", obs=["Northwind"], coords="Northwind",
        fleet={"Vestas-3.0-V112": 72}, capacity_mw=216.0,
    ),
    "Rentel": dict(
        region="BE", obs=["Rentel Offshore WP"], coords="Rentel",
        fleet={"SiemensSWT-7.35-154": 42}, capacity_mw=308.7,
        notes="Uprated 7.35 MW, not the nominal SWT-7.0.",
    ),

    # ---------------- UK (22 farms, 2071 turbines) ----------------
    # Coordinates from extract_uk_turbines.py (OSM point-in-polygon; every farm
    # reproduces its curated turbine count). Several farms run uprated machines, so
    # rated_power_mw is taken from the spec row that matches the as-built MW/turbine.
    "Barrow":            dict(region="UK", obs=["Barrow"], coords="Barrow",
                              fleet={"Vestas-3-V90": 30}, capacity_mw=90.0),
    "Dudgeon":           dict(region="UK", obs=["Dudgeon"], coords="Dudgeon",
                              fleet={"SiemensSWT-6.0-154": 67}, capacity_mw=402.0),
    "EastAngliaOne":     dict(region="UK", obs=["EastAngliaOne"], coords="EastAngliaOne",
                              fleet={"SiemensSWT-7.0-154": 102}, capacity_mw=714.0),
    "Galloper":          dict(region="UK", obs=["Galloper"], coords="Galloper",
                              fleet={"SiemensSWT-6.0-154": 56}, capacity_mw=353.0,
                              cap_tol=0.06,
                              notes="As-built 6.3 MW/turbine (uprated SWT-6.0); no 6.3 spec row."),
    "GreaterGabbard":    dict(region="UK", obs=["GreaterGabbard"], coords="GreaterGabbard",
                              fleet={"SiemensSWT-3.6-107": 140}, capacity_mw=504.0),
    "GunfleetSands":     dict(region="UK", obs=["GunfleetSands"], coords="GunfleetSands",
                              fleet={"SiemensSWT-3.6-107": 48}, capacity_mw=173.0, cap_tol=0.005,
                              notes="2 x SWT-6.0-120 demo (Gunfleet Sands 3) excluded; separate 12 MW unit."),
    "GwyntyMor":         dict(region="UK", obs=["GwyntyMor"], coords="GwyntyMor",
                              fleet={"SiemensSWT-3.6-107": 158}, capacity_mw=576.0, cap_tol=0.015,
                              notes="OSM has 158 of 160 turbines. Nameplate kept at Nost's 576 MW so "
                                    "capacity factor stays true; the 2 unplaced turbines' capacity is "
                                    "spread over the 158 we can place."),
    "HornseaOne":        dict(region="UK", obs=["HornseaOne"], coords="HornseaOne",
                              fleet={"SiemensSWT-7.0-154": 174}, capacity_mw=1218.0),
    "HornseaTwo":        dict(region="UK", obs=["HornseaTwo"], coords="HornseaTwo",
                              fleet={"SiemensGamesaSG-8.4-167DD": 165}, capacity_mw=1386.0,
                              notes="Uprated 8.4 MW; UK_FARMS labelled it SG-8.0 but 1386/165 = 8.4."),
    "HumberGateway":     dict(region="UK", obs=["HumberGateway"], coords="HumberGateway",
                              fleet={"Vestas-3.0-V112": 73}, capacity_mw=219.0),
    "Lincs":             dict(region="UK", obs=["Lincs"], coords="Lincs",
                              fleet={"SiemensSWT-3.6-120": 75}, capacity_mw=270.0),
    "LondonArray":       dict(region="UK", obs=["LondonArray"], coords="LondonArray",
                              fleet={"SiemensSWT-3.6-107": 175}, capacity_mw=630.0),
    "Ormonde":           dict(region="UK", obs=["Ormonde"], coords="Ormonde",
                              fleet={"Repower-5": 30}, capacity_mw=150.0),
    "RaceBank":          dict(region="UK", obs=["RaceBank"], coords="RaceBank",
                              fleet={"SiemensSWT-6.0-154": 91}, capacity_mw=573.0,
                              cap_tol=0.05,
                              notes="573 MW per Nost S1 (and the OSM polygon); the old UK_FARMS table's "
                                    "580 was wrong. As-built 6.30 MW/turbine (uprated SWT-6.0)."),
    "Rampion":           dict(region="UK", obs=["Rampion"], coords="Rampion",
                              fleet={"Vestas-3.4-V112": 116}, capacity_mw=400.0,
                              cap_tol=0.01,
                              notes="Footprint derived by DBSCAN (no OSM polygon); count validates at 116."),
    "RobinRigg":         dict(region="UK", obs=["RobinRigg"], coords="RobinRigg",
                              fleet={"Vestas-3-V90": 59}, capacity_mw=174.0, cap_tol=0.02,
                              notes="174 MW per Nost S1 (and the OSM polygon), not the 180 that 60 x V90 "
                                    "would imply. OSM has 59 of 60 turbines."),
    "TritonKnoll":       dict(region="UK", obs=["TritonKnoll"], coords="TritonKnoll",
                              fleet={"MHIVestas-9.5-V164": 90}, capacity_mw=857.0, cap_tol=0.005),
    "Walney":            dict(region="UK", obs=["Walney1", "Walney2"], coords="Walney",
                              fleet={"SiemensSWT-3.6-107": 102}, capacity_mw=368.0, cap_tol=0.005,
                              notes="Walney 1 + 2 summed: one OSM polygon, and both are 51 x SWT-3.6-107 "
                                    "/ 184 MW, so capacity-share == count-share. Only assumption is equal CF."),
    "WalneyExtension":   dict(region="UK", obs=["WalneyExtension"], coords="WalneyExtension",
                              fleet={"SiemensSWT-7.0-154": 40, "MHIVestas-8.25-V164": 47},
                              capacity_mw=659.0, cap_tol=0.02,
                              notes="Footprint = 'Walney Wind Farm' umbrella MINUS 'Walney 1 & 2' (the "
                                    "'Walney 3' polygon is broken). Mixed fleet, unresolvable on the map."),
    "WestOfDuddonSands": dict(region="UK", obs=["WestOfDuddonSands"], coords="WestOfDuddonSands",
                              fleet={"SiemensSWT-3.6-120": 108}, capacity_mw=389.0, cap_tol=0.005),
    "WestermostRough":   dict(region="UK", obs=["WestermostRough"], coords="WestermostRough",
                              fleet={"SiemensSWT-6.0-154": 35}, capacity_mw=210.0),
}

# -----------------------------------------------------------------------------
# Farms whose fleet is mixed but whose individual machines CANNOT be told apart from
# the coordinates alone. For these, per-turbine capacity degrades to the farm mean.
# The error is bounded and internal to the farm's own footprint.
# -----------------------------------------------------------------------------
# (Belwind is mixed but NOT listed: its lone Haliade is individually identified, so its
#  per-turbine capacities are exact.)
MIXED_FLEET_UNRESOLVED = {
    "CPower_SW":       "6 x Repower-5 (5.0 MW) among 24 x Repower-6.15 (6.15 MW). Which 6 is unknown. "
                       "Bounded: 30 turbines over a handful of cells, 23% capacity spread.",
    "WalneyExtension": "40 x SWT-7.0 (7.0 MW) and 47 x V164-8.25 (8.25 MW) interleaved; nothing in the "
                       "coordinates distinguishes them. Bounded: 18% capacity spread within one farm.",
}

# -----------------------------------------------------------------------------
# Deliberately EXCLUDED, and why. Recorded here so the omissions are auditable.
# -----------------------------------------------------------------------------
EXCLUDED = {
    "BurboBank + BurboBankExtension":
        "One OSM polygon over two farms with different machines (25 x 3.6 MW, 32 x 8.0 MW), "
        "spatially interleaved with no recoverable boundary (k-means 36/21; spacing 56/1; "
        "principal-axis cut at the known counts gives an 86 m gap against a 165 m median). "
        "Merging would spread 348 MW evenly over 57 turbines: +70% into the original's cells, "
        "-24% into the Extension's. Dropped rather than bake in a known misallocation.",
    "SheringhamShoals":
        "Not in the combined observations file (BE_UK_offshore_per_unit_3H_meanMW_shifted.csv has 25 UK "
        "columns and Sheringham is not one of them). And Nost (2025) EXCLUDES it from his analysis -- it "
        "is one of four farms where 'it was not possible to find a period where they were running at full "
        "capacity', i.e. it is derated across its whole record. Its OSM polygon and 88 turbines resolve "
        "cleanly, but the power signal is exactly the unforecastable underproduction we want to keep out.",
    "Aberdeen, Beatrice, HywindScotland, Kincardine, MorayEast, Seagreen":
        "The OSM turbine-point export is bbox-clipped at 56.5 N. These farms are north of it, so "
        "no turbine coordinates exist for them at all. Needs a fresh Overpass export, not better "
        "name matching.",
}


def load_specs() -> pd.DataFrame:
    df = pd.read_csv(SPECS_CSV)
    df = df.rename(columns={df.columns[0]: "turbine_type"})
    return df.set_index("turbine_type")


def _coords_path(name: str) -> Path:
    """UK coords come from COORDS (written by extract_uk_turbines.py); Belgian coords come
    from the raw BOZ_Turbines export, which this module cleans."""
    for d in (BE_RAW_COORDS, COORDS):
        p = d / f"{name}_turbines_coords.csv"
        if p.exists():
            return p
    raise FileNotFoundError(f"no coordinates CSV for {name!r}")


def _cpower_area(name: str) -> str:
    """C-Power's two ENTSO-E units are split by turbine name: A,B,C,D -> SW; E-J -> NE."""
    return "SW" if str(name)[:1] in "ABCD" else "NE"


def load_farm_coords(farm: str) -> pd.DataFrame:
    """Turbine coordinates for one farm, hub rows removed and the Haliade reassigned."""
    spec = FARMS[farm]
    df = pd.read_csv(_coords_path(spec["coords"]))
    df["NAME"] = df["NAME"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df[~df["NAME"].str.contains(HUB_PATTERN, case=False, regex=True)]

    if spec.get("coord_filter"):
        df = df[df["NAME"].map(_cpower_area) == spec["coord_filter"]]
    if spec.get("drop_coords"):
        df = df[~df["NAME"].isin(spec["drop_coords"])]
    if spec.get("extra_coords"):
        src, name = spec["extra_coords"]
        extra = pd.read_csv(_coords_path(src))
        extra["NAME"] = extra["NAME"].astype(str).str.replace(r"\.0$", "", regex=True)
        extra = extra[extra["NAME"] == name].copy()
        extra["NAME"] = f"{farm}_HALIADE"
        df = pd.concat([df, extra], ignore_index=True)

    return df[["NAME", "LONGITUDE", "LATITUDE"]].reset_index(drop=True)


def turbine_capacities(farm: str, coords: pd.DataFrame, specs: pd.DataFrame) -> np.ndarray:
    """Rated MW for each individual turbine of a farm -- the weights the distributed target
    uses: power(cell) = P_obs * sum(capacity of that farm's turbines in the cell) / capacity_total.

    Exact for a uniform fleet, and for Belwind (whose lone Haliade is individually identified).
    For the fleets in MIXED_FLEET_UNRESOLVED the individual machines cannot be told apart, so
    every turbine gets the farm mean -- which is exactly what count-weighting would do, i.e. no
    worse than the rule this scheme replaces.

    Note the farm's NAMEPLATE is always what gets distributed, never sum(count x rated): the
    nameplate is the as-built number the observations are actually generated by (several farms
    run uprated machines), so the per-turbine capacities are rescaled to sum to it exactly.
    """
    fleet = FARMS[farm]["fleet"]
    cap = FARMS[farm]["capacity_mw"]
    n = len(coords)

    if farm == "Belwind":                       # the Haliade is identified by name
        is_haliade = coords["NAME"].str.endswith("_HALIADE").to_numpy()
        rated = np.where(is_haliade,
                         specs.loc["AlstomHaliade-6-150", "rated_power_mw"],
                         specs.loc["Vestas-3-V90", "rated_power_mw"]).astype(float)
    elif len(fleet) == 1 or farm in MIXED_FLEET_UNRESOLVED:
        rated = np.full(n, cap / n, dtype=float)
    else:
        raise AssertionError(f"{farm}: mixed fleet that is neither resolved nor declared unresolvable")

    return rated * (cap / rated.sum())          # renormalise so the farm sums to its nameplate


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    specs = load_specs()
    rows, turbines, problems = [], [], []

    for farm, spec in FARMS.items():
        fleet = spec["fleet"]
        n_fleet = sum(fleet.values())

        unknown = [k for k in fleet if k not in specs.index]
        if unknown:
            problems.append(f"{farm}: turbine types not in turbine_specs: {unknown}")
            continue

        coords = load_farm_coords(farm)
        if len(coords) != n_fleet:
            problems.append(f"{farm}: {len(coords)} coordinates but fleet totals {n_fleet}")

        rated_sum = sum(n * specs.loc[k, "rated_power_mw"] for k, n in fleet.items())
        cap = spec["capacity_mw"]
        tol = spec.get("cap_tol", 0.001)
        rel = abs(rated_sum - cap) / cap
        if rel > tol:
            problems.append(
                f"{farm}: nameplate {cap:.1f} MW vs sum(count x rated) {rated_sum:.1f} MW "
                f"({rel:.1%} > tol {tol:.1%})")

        # cut-out is what the curtailment mask needs; take the most conservative (lowest)
        # in a mixed fleet, since above it ANY machine may already have shut down.
        cut_out = min(specs.loc[k, "cut_out_ms"] for k in fleet)
        cut_in = min(specs.loc[k, "cut_in_ms"] for k in fleet)

        rows.append(dict(
            farm=farm, region=spec["region"],
            obs_columns=";".join(spec["obs"]),
            n_turbines=len(coords),
            capacity_mw=cap,
            mw_per_turbine=cap / len(coords) if len(coords) else np.nan,
            fleet="; ".join(f"{n}x {k}" for k, n in fleet.items()),
            fleet_uniform=len(fleet) == 1,
            capacity_resolvable_per_turbine=farm not in MIXED_FLEET_UNRESOLVED or len(fleet) == 1,
            cut_in_ms=cut_in, cut_out_ms=cut_out,
            sum_rated_mw=round(rated_sum, 2),
            notes=spec.get("notes", ""),
        ))

        # Belgian coords are cleaned here (hub rows dropped, C-Power split, Haliade
        # reassigned), so write the cleaned per-farm CSV out. The UK ones already live in
        # COORDS, written by extract_uk_turbines.py. Result: coordinates/ is 1:1 with farms.csv.
        if spec["region"] == "BE":
            coords.to_csv(COORDS / f"{farm}_turbines_coords.csv", index=False)

        caps = turbine_capacities(farm, coords, specs)
        if not np.isclose(caps.sum(), cap):
            problems.append(f"{farm}: per-turbine capacities sum to {caps.sum():.2f}, not {cap:.2f}")
        turbines.append(pd.DataFrame({
            "farm": farm, "region": spec["region"],
            "turbine": coords["NAME"].to_numpy(),
            "longitude": coords["LONGITUDE"].to_numpy(),
            "latitude": coords["LATITUDE"].to_numpy(),
            "capacity_mw": caps,
        }))

    turb_df = pd.concat(turbines, ignore_index=True) if turbines else pd.DataFrame()
    return pd.DataFrame(rows), turb_df, problems


def main() -> None:
    df, turb, problems = build()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 44)

    print("=" * 118)
    print("FARM METADATA AUDIT")
    print("=" * 118)
    cols = ["farm", "region", "n_turbines", "capacity_mw", "mw_per_turbine",
            "sum_rated_mw", "cut_in_ms", "cut_out_ms", "fleet"]
    print(df[cols].to_string(index=False))

    print()
    for reg in ("BE", "UK"):
        s = df[df.region == reg]
        print(f"{reg}: {len(s):2d} farms   {s.n_turbines.sum():4d} turbines   {s.capacity_mw.sum():8.1f} MW")
    print(f"{'ALL':<3}: {len(df):2d} farms   {df.n_turbines.sum():4d} turbines   {df.capacity_mw.sum():8.1f} MW")

    print()
    print("MIXED FLEETS WHERE PER-TURBINE CAPACITY FALLS BACK TO THE FARM MEAN:")
    fallback = df[~df.capacity_resolvable_per_turbine]
    if fallback.empty:
        print("  (none)")
    for _, r in fallback.iterrows():
        print(f"  {r.farm:18s} {r.fleet}")
        print(f"  {'':18s} -> {MIXED_FLEET_UNRESOLVED[r.farm]}")

    print()
    print("EXCLUDED FARMS:")
    for k, v in EXCLUDED.items():
        print(f"  {k}")
        for line in (v[i:i + 96] for i in range(0, len(v), 96)):
            print(f"      {line}")

    df.to_csv(FARMS_CSV, index=False)
    turb.to_csv(TURBINES_CSV, index=False)
    print()
    print(f"wrote {FARMS_CSV.name}     ({len(df)} farms)")
    print(f"wrote {TURBINES_CSV.name}  ({len(turb)} turbines, {turb.capacity_mw.sum():.1f} MW"
          f" -- per-turbine capacity is the weight the distributed target uses)")

    if problems:
        print()
        print("!! VALIDATION PROBLEMS")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print()
    print("VALIDATION PASSED: every farm's coordinate count matches its fleet, and every")
    print("nameplate capacity matches sum(count x rated_power_mw) within tolerance.")


if __name__ == "__main__":
    main()
