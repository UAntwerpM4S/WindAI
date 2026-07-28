# Wpower — metadata for the spatially-distributed power target

All metadata needed to place observed wind-farm power onto the CERRA grid, for Belgium + UK.

**31 farms · 2,380 turbines · 12,576.2 MW** — Belgium 10 / 399 / 2,261.2 · UK 21 / 1,981 / 10,315.0

> **Belgium is the evaluation target. The UK farms are auxiliary training signal only** — they exist to
> give the loss enough non-zero cells to learn a spatial mapping from. UK observations end in 2023;
> Belgian ones run to 2025, so validation/test naturally lands where only Belgium reports. That is fine
> and intended.

## The target this metadata exists to build

Each farm's observed MW is spread across the cells its turbines actually occupy, weighted by
**capacity**, and cells hosting more than one farm **sum** (power is extensive):

```
power(cell, t) = Σ_farms  P_obs(farm, t) × capacity(farm's turbines in cell) / capacity_total(farm)
```

Capacity-weighting — not turbine-count weighting — because a farm with mixed machines (Belwind,
C-Power, Walney Extension) would otherwise hand a 3 MW turbine the same share as an 8 MW one.
For a uniform fleet the two rules are identical, so this is a strict generalization.

## Files

| file | what it is |
|---|---|
| `turbine_specs.csv` | Per turbine **model**: cut-in, rated wind speed, cut-out, rated MW, rotor Ø. Hand-curated; sources in `../NorthSea/Power/metadatawfarms.txt`. |
| `farm_metadata.py` | **Source of truth.** Declares every farm's obs column(s), coordinates, fleet and nameplate. Run it: it audits, and exits non-zero if anything stops reconciling. |
| `extract_uk_turbines.py` | Builds the UK coordinates from the OSM GeoJSONs in `../NorthSea/Power/`. Run **before** `farm_metadata.py`. |
| `coordinates/` | **All 31 farms**, one CSV each (`NAME,LONGITUDE,LATITUDE`), 1:1 with `farms.csv`. UK written by `extract_uk_turbines.py`; Belgian written by `farm_metadata.py`, *cleaned* from `../BOZ_Turbines/coordinates/` (see below). |
| `build_power_obs.py` | Turns the raw observations into the canonical per-farm table. |
| `BE_UK_offshore_per_unit_3H_meanMW_shifted.csv` | *Raw source.* 35 ENTSO-E/Elexon obs columns, 3-hourly mean MW. Built by `../NorthSea/Power/aggregate_uk_wind.py`. |
| `validate_against_nost.py` | **End-to-end regression test.** Reproduces the published capacity factors of Nøst (2025). Run after any change to capacities, fleets or aggregation. |
| `farms.csv` | *Generated.* One row per farm: obs columns, turbine count, capacity, fleet, cut-in/cut-out, notes. |
| `turbines.csv` | *Generated.* One row per turbine: farm, lon, lat, **capacity_mw**. The per-turbine capacities are the distribution weights. |
| `power_obs.csv` | *Generated.* **The power timeseries.** 16,312 rows × 31 farms, 3-hourly mean MW, 2020-01-01 → 2025-07-31. One column per farm, 1:1 with `farms.csv`. |

`turbines.csv` (where the power goes) and `power_obs.csv` (how much power) are the two files the target
build consumes. Regenerate and check the whole chain with:

```bash
python extract_uk_turbines.py    # UK coordinates  (validates against known turbine counts)
python farm_metadata.py          # farms.csv + turbines.csv  (audits everything)
python build_power_obs.py        # power_obs.csv  (checks every farm stays within nameplate)
python validate_against_nost.py  # end-to-end check against the published CFs
```

All four fail loudly rather than silently emitting a wrong table.

## power_obs.csv — timestamps

**UTC, year-round, no daylight saving.** 3-hourly, 2020-01-01 00:00 → 2025-07-31 21:00 — step-for-step
aligned with CERRA (also UTC, `frequency: 3h`).

**Window convention — a forward mean.** The value at `t` is the **mean power over `[t, t+3h)`**, labelled
by the window's start (that is what the "shifted" in the source filename does). CERRA at `t` is the
*instantaneous* analysis at `t`. So when anemoi joins them, the power target is the average of the coming
3 hours — its centroid sits at `t + 1.5h`, i.e. **the target leads the instantaneous CERRA field by ~1.5 h**.
This is uniform across all 31 farms (BE and UK share the identical convention — verified), and 1.5 h is
inside the data's own 3-hour step, so it does not misalign anything. It simply means the target is *"mean
power over the coming 3 h"*, not *"instantaneous power at the CERRA timestamp"* — a coherent, slightly
forward-looking target. Re-centering on `t` would need the raw half-hourly, which exists for the UK
(`../NorthSea/Power/UK/`) but **not** for Belgium, so doing it one-sided would reintroduce a BE/UK
mismatch. Left as-is.

**Comparing to the ENTSO-E Transparency Platform — mind the DST.** `power_obs` is UTC; the ENTSO-E **web
UI** shows the bidding zone's **local time with daylight saving** (Belgium: **CEST = UTC+2 Apr–Oct**,
CET = UTC+1 Nov–Mar). So `power_obs` 12:00 UTC in April is **14:00 on the website**, and — with the forward
window — equals the mean ENTSO-E per-unit generation over **14:00–17:00 CEST = 12:00–15:00 UTC**. To avoid
the conversion, use the ENTSO-E **API** (returns UTC) and average its 15-min/hourly values over the 3-hour
UTC window.

## power_obs.csv — what's in it

Keyed by **farm**, not by obs column, which is the whole point of it existing:

- **Walney1 + Walney2 are summed** into one `Walney` farm (one OSM polygon covers both; both are
  51 × SWT-3.6-107 / 184 MW). If *either* column is NaN the merged farm is NaN — power is **extensive**,
  so you cannot sum a known and an unknown. The same rule applies downstream when farms share a cell.
- **Aberdeen, BurboBank, BurboBankExtension are dropped** (present in the raw file, excluded from the
  farm set).
- Every farm's observed power stays within its nameplate: max capacity factor **0.95–1.01** across all 31.

Coverage is uneven by design, because `ALLOWED_YEARS` masks each UK farm outside its full-capacity period:

| | valid timesteps |
|---|---|
| Belgium | 87.7–100% per farm; **all 10** reporting simultaneously 87.6% of the time |
| UK | 17.9% (Ormonde, one year) to 71.6% (four years); **never** all 21 at once |

That the UK farms never all report together is expected and harmless — the loss masks per cell, so each
farm contributes gradient only during its own valid window.

A `CF > 1.05 → NaN` clamp guards against corrupt samples. It currently fires **zero** times: the one bad
half-hourly record (GwyntyMor 2022-11-15, CF 1.97) is diluted below threshold by the 3-hour averaging. It
stays in as protection.

## The end-to-end validation

Nøst (2025), *PLoS ONE* 20(5): e0321528, S1 Table publishes the installed capacity and the mean capacity
factor of 31 North Sea farms, measured over the period each was **running at full capacity**.

Recomputing each farm's mean CF from the raw Elexon series over those same windows reproduces his
published CF for **all 23 farms we use, to within 0.09 percentage points** (mean abs error 0.03 pp).

That one check validates, simultaneously:

- the **unit convention** — the raw values are MWh per half-hour, so mean MW = `mean(v) × 2`
- **every UK capacity** in `farm_metadata.py`
- the **`ALLOWED_YEARS`** windows in `../NorthSea/Power/aggregate_uk_wind.py`

It is by far the cheapest way to catch a factor-of-two or a wrong nameplate — much cheaper than finding
it after the zarr is built.

## ⚠️ Do not "improve" ALLOWED_YEARS

`ALLOWED_YEARS` is **Nøst's S1 "Period" column, verbatim** — including the odd-looking entries
(`EastAngliaOne: [2021, 2023]`, `GwyntyMor: [2020, 2022, 2023]`, `Ormonde: [2020]`). They are the periods
each farm ran at full capacity, found by inspecting the production series against installed capacity "to
reveal periods where technical issues are reducing the capacity factors."

**This is the availability metadata needed to exclude derated operation** — the thing the project brief
listed as an unsolved limitation. A naive quality rule (complete records + CF ≤ 1.05) is strictly weaker:
it catches missing data and impossible spikes but sails straight past *sustained derating*. Concrete trap:
Ormonde 2021–23 has complete records and CF ≈ 0.28, which looks perfectly healthy — until you notice
Ormonde's true CF is 39.8%, so those years are a farm running ~30% below par. Exactly what the exclusion
exists to remove.

## Provenance

- **Power observations** — 3-hourly mean MW, 2020-01-01 → 2025-07-31 (16,312 steps), built by
  `../NorthSea/Power/aggregate_uk_wind.py` into `BE_UK_offshore_per_unit_3H_meanMW_shifted.csv`
  (10 Belgian + 25 UK columns). UK raw half-hourly source (Elexon/BMRS, MWh per half-hour) is in
  `Power/UK/`. Belgian obs peak at 0.97–0.99 of every nameplate in `farms.csv` — an independent
  confirmation of the Belgian capacities.
- **Time convention — verified, do not re-litigate.** `aggregate_uk_wind.py` shifts the UK back 3 h to
  match the Belgian convention, and it does so correctly. Regressing peak-correlation lag on longitude
  separation over 2022–23 farm pairs (which removes advection) gives clock offsets of +0.013 (BE–BE),
  −0.071 (UK–UK) and **−0.088 (BE–UK)** in 3-hour steps. A one-step misalignment would put BE–UK near
  ±1.0; the actual figure is ~16 minutes, indistinguishable from the same-region baselines.
- **Belgian coordinates** — cleaned from `../BOZ_Turbines/coordinates/` (left in place as source data;
  other scripts read it). The raw files cannot be used directly: they contain substation rows
  (`BW_OHVS`, `NWOHVS`, `OTS`, …), C-Power is one 54-turbine file rather than its two ENTSO-E units, and
  Belwind's Haliade coordinate is misfiled under Nobelwind. `farm_metadata.py` applies all three fixes and
  writes the cleaned per-farm CSVs into `coordinates/`.
- **UK coordinates** — derived here by point-in-polygon from two OSM exports: `UKWindfarms.geojson`
  (17,112 *unnamed* turbine points, onshore included) and `UKPolygoncoordinates.geojson` (named plant
  polygons). **Every UK farm reproduces its independently-curated turbine count**, which is the
  validation that the join is right.

## Corrections applied (the old tables were wrong — don't use them)

`EGU26/windfarm_metadata.csv` and `NorthSea/Power/build_metadata.py` give **389 turbines / 2,177 MW**
for Belgium, against a verified 399 / 2,261.4. Every discrepancy is now resolved:

- **A misfiled coordinate.** Belwind is 55 × V90-3.0 **plus a 6 MW Alstom Haliade demonstrator**
  (56 turbines / 171 MW), but its CSV holds only 55 turbines + the substation — while Nobelwind's CSV
  holds **51** records for a farm that is really 50 turbines / 165 MW. Nobelwind's record `1` sits
  **350 m from a Belwind turbine but 1,036 m from the nearest Nobelwind turbine** (median Nobelwind
  spacing 519 m): it is inside Belwind's array. It is the Haliade. Reassigning it makes turbines (399)
  *and* capacity (2,261.2 MW) both reconcile exactly. `farm_metadata.py` does this automatically.
- **Seastar is 30 turbines / 252 MW**, not the 20 / 168 MW the old table claims.
- **C-Power's 27/27 area split is a fiction.** The real split is by turbine name — A–D → SW (30),
  E–J → NE (24) — giving 177.6 MW and 147.6 MW, not 162.6 each.
- **Uprated machines were under-labelled:** Hornsea Two is SG-**8.4** (1386/165 = 8.4, labelled 8.0);
  Galloper 6.3 and Race Bank 6.30 (uprated SWT-6.0). Same class of error as Mermaid/Seastar/Rentel.
  The **nameplate is authoritative** — per-turbine capacities are rescaled to sum to it.
- **Three UK capacities corrected against Nøst S1:** Race Bank **573 MW** (the old table said 580),
  Robin Rigg **174 MW** (not the 180 that 60 × V90-3.0 implies), Gwynt y Môr **576 MW** with only 158 of
  its 160 turbines placeable in OSM. All three would fail the capacity-factor check otherwise.
- **Gunfleet Sands' 2 demo turbines** ("Gunfleet Sands 3") are a *separate* 12 MW ENTSO-E unit whose
  output is not in the `GunfleetSands` column, so they get no share of its power. Excluded → 48
  turbines / 172.8 MW.
- `SiemensSWT-3.6-107` rated wind speed corrected **16.5 → 13.5 m/s**. It covers 711 turbines, so this
  matters for any power-curve baseline (not for capacity weighting).

## Known limitations — carry these into the paper

- **Two mixed fleets cannot be resolved from coordinates**, so their per-turbine capacity falls back to
  the farm mean. This is exactly what count-weighting would do, i.e. no worse than the rule being
  replaced, and the error is bounded inside one farm's own footprint:
  - `CPower_SW` — which 6 of its 30 turbines are the 5.0 MW Repower-5 is unknown (23% capacity spread).
  - `WalneyExtension` — 40 × SWT-7.0 and 47 × V164-8.25 are interleaved (18% spread).
- **Burbo Bank + Extension are excluded.** One OSM polygon covers two farms with different machines
  (25 × 3.6 MW, 32 × 8.0 MW), spatially interleaved with no recoverable boundary — k-means splits
  36/21, spacing splits 56/1, and a principal-axis cut at the known counts finds an 86 m gap against a
  165 m median. Merging would spread 348 MW evenly over 57 turbines: **+70%** into the original's
  cells, **−24%** into the Extension's. Dropped rather than bake in a known misallocation.
- **Six Scottish farms have no coordinates at all** (Aberdeen, Beatrice, Hywind, Kincardine, Moray
  East, Seagreen). The OSM turbine-point export is bbox-clipped at **56.5°N** and they lie north of it.
  Recoverable only with a fresh Overpass export — it is not a name-matching problem.
- **Sheringham Shoal is excluded**, despite resolving cleanly (clean polygon, exactly 88 turbines). It is
  not a column in the combined observations file, and Nøst excludes it as one of four farms where "it was
  not possible to find a period where they were running at full capacity" — it is derated across its whole
  record. It looks like free signal; it is actually the unforecastable underproduction we filter out.
- **Belgium has no availability curation.** `ALLOWED_YEARS` covers the UK only; Belgium gets just
  `BELGIUM_STARTS` (commissioning dates). So derating and curtailment go unfiltered on the evaluation
  target. **Accepted and disclosed** — flag it in the paper.
- **Unverified:** whether every farm falls inside the CERRA LAM cutout. The set spans 50.6–54.8°N,
  −3.9–3.1°E, including an Irish Sea cluster (Walney, Barrow, Ormonde, Robin Rigg, West of Duddon).
  The domain lives in `New_Cerra_A_large.zarr` on `/mnt/weatherloss/`. **Check this before building the
  zarr** — if the cutout misses the Irish Sea, the farm list needs revisiting.
