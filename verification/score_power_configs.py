#!/usr/bin/env python3
"""Score wind-farm power forecasts against the observations, comparing runs and methods.

Self-contained: reconstruction, power curves, scoring and plotting all live here.

For every run, up to TWO power forecasts are produced, summed to the regional total, and scored
against the ENTSO-E/Elexon observations in power_obs.csv:

  DIRECT      the model's own `capacityfactor` prediction, mapped back to farms:
                  P_direct(farm,t) = SUM_cell capacity(farm's turbines in cell) * CF_pred(cell,t)
              the adjoint of build_power.py's capacity-weighted distribution (exact for a farm's
              un-shared cells: a perfect CF field reproduces P_obs).

  POWERCURVE  the classical baseline: the model's forecast ws100 at the farm, pushed through that
              farm's own aggregate power curve, built from turbine_specs.csv (cut-in, rated wind,
              cut-out, rated MW) and the farm's fleet in farms.csv:
                  P_curve(farm,t) = A_farm( ws100_forecast(farm,t) )
              A_farm is rescaled so its rated plateau equals the farm's nameplate capacity.
              This is an INSTANTANEOUS estimate: the power AT t.

  CURVE3H     the same power curve, but averaged over the observation's own window (default; use
              --no-curve-window to switch off):
                  P_curve3h(farm,t) = 1/2 [ A_farm(ws100(t)) + A_farm(ws100(t+dt)) ]

              WHY THIS EXISTS. power_obs at t is the MEAN power over [t, t+3h) (see the data
              README: the value is labelled by the window's start, so its centroid leads the
              instantaneous CERRA field by ~1.5 h). The DIRECT forecast is trained on exactly that
              quantity, so it predicts a window mean. The instantaneous POWERCURVE does not: it
              predicts the power AT t and is then graded against an average. Even a perfect
              snapshot forecast is charged a non-zero error by that mismatch, so part of any
              direct-vs-curve gap is a bookkeeping convention rather than forecast skill.

              CURVE3H removes it by making the baseline predict the same thing the observation is.
              The POWERS are averaged, not the winds -- the curve is non-linear (cubic on the ramp),
              so A(mean ws) != mean A(ws). Compare `curve` and `curve3h` on the SAME sample: if
              they barely differ the convention was harmless and the direct-vs-curve result stands;
              if curve3h improves a lot, that much of the gap was never forecast skill.

              Forming the window needs the forecast at t+dt, so with --curve-window on, any
              (init, lead) lacking it is dropped from EVERY series -- all methods stay on one
              identical sample -- and leads that can never form a window are trimmed up front.

A run with only ws100 and no `capacityfactor` (e.g. RegularWeather) still gets a POWERCURVE
forecast and is scored on equal terms -- it simply has no DIRECT line. That comparison is the
point: does predicting power directly beat predicting wind and applying the power curve?

Metrics per farm and for the regional total, by lead time: MAE (MW and % of capacity), RMSE
and BIAS. Bias matters here -- the idealised specs power curve ignores wake
losses, so it should over-predict at high wind; whether the direct forecast removes that is a
concrete, testable claim MAE alone cannot show.

WIND REGIMES.  Every method is ALSO scored split by wind regime, so you can see where each one
wins. The regime is set by the OBSERVED power, converted to an equivalent wind speed through that
farm's own power curve -- so the bins are truth-conditioned (never forecast-conditioned) and are
labelled in m/s, matching verify_rmse_farm_regimes.py. `--regime-by fc-ws` switches to binning on
the forecast wind instead, which answers a different, operational question ("given that my
forecast says 8-12 m/s, how wrong am I?") and is NOT a conditional on truth.

  Two things to keep in mind when reading the regime split:
  * Above rated the power curve is flat, so ALL winds from rated to cut-out map to the same
    observed power. The top bin is therefore "at or above rated" and cannot be subdivided. That
    is a property of power, not of the binning.
  * Truth-conditioned bins reward an under-dispersive forecast in the middle bins and punish it
    in the tails, even with no difference in skill. Here that is a feature rather than a trap:
    the methods differ mainly in dispersion (direct sigma_p/sigma_o ~ 0.67, specs curve ~ 1.11),
    and the regime split is precisely what makes that visible.

Figures:
  1. mae_total_<region>.png     MAE of the TOTAL summed power vs lead, % of total capacity
  2. mae_per_farm_<region>.png  MAE vs lead per farm, % of that farm's capacity
  3. mae_regimes_<region>.png   one panel per wind regime: TOTAL MAE vs lead, per method
Solid = DIRECT, dashed = POWERCURVE (instantaneous), dotted = CURVE3H (window mean), colour =
run. Every run is scored on the init times common to all runs, so the comparison uses an
identical sample.

Usage:
  python score_power_configs.py                     # uses the SETTINGS below
  python score_power_configs.py --region UK
  python score_power_configs.py --runs A=/path/one B=/path/two
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# -------------------- SETTINGS --------------------
# Where farms.csv / turbines.csv / power_obs.csv / turbine_specs.csv live (NOT where this script
# lives, so it can sit in verification/ while the metadata stays with the data).
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")

# One entry per run; the key is the legend label. A run WITHOUT the power variable but WITH ws100
# (e.g. RegularWeather) is fine -- it gets the POWERCURVE forecast only.
FORECAST_DIRS = {
    "HighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaPowerGT"),
    "RegularWeather": Path("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
  #  "SemiHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/SemiHighCapacity"),
}
VAR        = "capacityfactor"        # absent in a run -> POWERCURVE only
REGION     = "BE"                    # BE | UK | all
LEAD_HOURS = list(range(3, 37, 3))
OUT_DIR    = Path("WPDistr_scores")  # relative to cwd

# Evaluation period: restrict the scored inits by INIT time. START/END are inclusive date strings
# (e.g. "2025-01-01"); None = open-ended. SEASON restricts by init month ("all" = no restriction).
START   = None
END     = None
SEASON  = "all"
# Average the power curve over the observation window so the baseline predicts the same quantity
# the observation is (see CURVE3H in the docstring). --no-curve-window restores the old behaviour.
CURVE_WINDOW = True
SEASONS = {"all": None, "DJF": {12, 1, 2}, "MAM": {3, 4, 5}, "JJA": {6, 7, 8}, "SON": {9, 10, 11}}

# Wind-regime edges in m/s, matching verify_rmse_farm_regimes.py. Each edge is converted to a
# capacity-factor threshold through the relevant power curve, and the OBSERVED power is binned
# against those thresholds -- so the split is on truth, expressed in m/s.
REGIME_WS_EDGES = [4.5, 8.0, 12.0]
REGIME_LABELS = ["0-4.5", "4.5-8", "8-12", "12+"]
REGIME_BY = "obs-cf"                 # obs-cf (truth) | fc-ws (forecast wind)
# --------------------------------------------------

FORECAST_RE = re.compile(r"forecast_(\d{14})")
FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")

NREG = len(REGIME_LABELS)

METHOD_LABEL = {"direct": "direct",
                "curve": "power curve (inst.)",
                "curve3h": "power curve (window mean)",
                "curve3hb": "power curve (window mean, BACKWARD control)"}


# =============================================================================
# reconstruction: forecast cells -> per-farm power
# =============================================================================
def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(FORECAST_RE.search(path.name).group(1),
                          format="%Y%m%d%H%M%S", utc=True)


def build_reconstruction(fc_lat, fc_lon, turbines, farms):
    """Assign turbines to forecast cells and build the reconstruction operators.

    Returns:
      cell_idx : forecast-cell indices that hold turbines
      G        : (n_farms, n_cells) with G[f,j] = capacity of farm f in cell_idx[j]
      cap_cell : (n_cells,) total capacity per cell (for a raw-power forecast -> CF)
    """
    coslat = np.cos(np.radians(float(fc_lat.mean())))
    tree = cKDTree(np.c_[to_180(fc_lon) * coslat, fc_lat])
    _, cell = tree.query(np.c_[to_180(turbines["longitude"]) * coslat,
                               turbines["latitude"].to_numpy()], k=1)
    t = turbines.assign(cell=cell.astype(int))

    cell_idx = np.sort(t["cell"].unique())
    cpos = {int(c): j for j, c in enumerate(cell_idx)}
    fpos = {f: i for i, f in enumerate(farms)}

    G = np.zeros((len(farms), cell_idx.size), dtype=np.float64)
    for (farm, c), cap in t.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[fpos[farm], cpos[int(c)]] = cap
    cap_cell = t.groupby("cell")["capacity_mw"].sum().reindex(cell_idx).to_numpy()
    return cell_idx, G, cap_cell


def farm_wind(ws_cells: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Capacity-weighted mean wind per farm. ws_cells (T, C), G (F, C) -> (T, F)."""
    w = G / G.sum(1, keepdims=True)
    return ws_cells @ w.T


# =============================================================================
# per-farm aggregate power curve  (specs -> P(v))
# =============================================================================
def turbine_power(ws, cut_in, rated_ws, cut_out, rated_mw) -> np.ndarray:
    """One turbine: 0 below cut-in, cubic ramp to rated, flat at rated, 0 above cut-out."""
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws)
    ramp = (ws >= cut_in) & (ws < rated_ws)
    out[ramp] = rated_mw * (ws[ramp] ** 3 - cut_in ** 3) / (rated_ws ** 3 - cut_in ** 3)
    out[(ws >= rated_ws) & (ws < cut_out)] = rated_mw
    return out


def build_farm_curves(farms_df, specs, farms) -> dict:
    """farm -> callable(ws) -> MW, the fleet's summed power curve scaled to the nameplate."""
    curves = {}
    meta = farms_df.set_index("farm")
    for farm in farms:
        fleet, cap = meta.loc[farm, "fleet"], float(meta.loc[farm, "capacity_mw"])
        parts = []
        for chunk in str(fleet).split(";"):
            m = FLEET_RE.match(chunk)
            if not m:
                raise SystemExit(f"{farm}: cannot parse fleet entry {chunk!r}")
            count, ttype = int(m.group(1)), m.group(2)
            if ttype not in specs.index:
                raise SystemExit(f"{farm}: turbine type {ttype!r} not in turbine_specs.csv")
            parts.append((count, specs.loc[ttype]))

        scale = cap / sum(c * float(s["rated_power_mw"]) for c, s in parts)   # nameplate wins

        def curve(ws, parts=parts, scale=scale):
            tot = np.zeros_like(np.asarray(ws, dtype=float))
            for count, s in parts:
                tot += count * turbine_power(ws, float(s["cut_in_ms"]), float(s["rated_ws_ms"]),
                                             float(s["cut_out_ms"]), float(s["rated_power_mw"]))
            return tot * scale
        curves[farm] = curve
    return curves


def cf_edges_from_ws(curve, capacity_mw, ws_edges):
    """Capacity-factor thresholds equivalent to the given wind-speed edges, for ONE power curve.

    The curve is monotone non-decreasing, so the thresholds come out increasing and can be fed
    straight to np.digitize on an observed capacity factor. Above rated the curve is flat, which
    is why the top bin is open-ended: every wind from rated to cut-out gives the same power.
    """
    cf = np.asarray([float(curve(np.asarray([e], dtype=float))[0]) / capacity_mw
                     for e in ws_edges], dtype=float)
    # guard against a non-strict curve producing tied thresholds (would silently empty a bin)
    for i in range(1, len(cf)):
        if cf[i] <= cf[i - 1]:
            cf[i] = cf[i - 1] + 1e-9
    return cf


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None, metavar="LABEL=DIR",
                    help="override FORECAST_DIRS: one or more LABEL=/path/to/inference/dir")
    ap.add_argument("--region", default=REGION, choices=["BE", "UK", "all"])
    ap.add_argument("--var", default=VAR, help="the power variable (absent -> POWERCURVE only)")
    ap.add_argument("--leads", type=int, nargs="+", default=LEAD_HOURS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--start", default=START, help="evaluation period start (init date, inclusive)")
    ap.add_argument("--end", default=END, help="evaluation period end (init date, inclusive)")
    ap.add_argument("--season", default=SEASON, choices=list(SEASONS),
                    help="restrict scored inits to a season by init month (default: all)")
    ap.add_argument("--curve-window", dest="curve_window", action="store_true",
                    default=CURVE_WINDOW,
                    help="also score the power curve averaged over the observation window "
                         "(default: on)")
    ap.add_argument("--no-curve-window", dest="curve_window", action="store_false",
                    help="instantaneous power curve only -- the pre-2026 behaviour")
    ap.add_argument("--regime-by", default=REGIME_BY, choices=["obs-cf", "fc-ws"],
                    help="what defines the wind regime: 'obs-cf' bins the OBSERVED power through "
                         "the power curve (truth-conditioned, the default) or 'fc-ws' bins the "
                         "forecast wind at the farm (operational, not a conditional on truth)")
    ap.add_argument("--regime-ws-edges", type=float, nargs="+", default=REGIME_WS_EDGES,
                    help="interior wind-speed edges in m/s (default: 4.5 8 12)")
    ap.add_argument("--curve-window-control", action="store_true",
                    help="also score the BACKWARD window mean 1/2[P(t-dt)+P(t)]. Same smoothing "
                         "as the forward mean but centred at t-dt/2 instead of t+dt/2, which is "
                         "the only way to tell the alignment gain apart from the plain "
                         "two-forecast averaging gain. Costs the first lead.")
    args = ap.parse_args()

    if args.runs:
        runs: dict[str, Path] = {}
        for r in args.runs:
            if "=" not in r:
                raise SystemExit(f"--runs takes LABEL=DIR, got {r!r}")
            label, d = r.split("=", 1)
            runs[label] = Path(d)
    else:
        runs = dict(FORECAST_DIRS)
    args.out.mkdir(parents=True, exist_ok=True)

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    specs = pd.read_csv(WPOWER_DIR / "turbine_specs.csv")
    specs = specs.rename(columns={specs.columns[0]: "turbine_type"}).set_index("turbine_type")
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")

    # The averaging window is the observation step: power_obs at t is the MEAN over [t, t+step).
    # Inferred rather than hard-coded so this stays right if the obs frequency ever changes.
    obs_step_h = float(pd.Series(obs.index).diff().dt.total_seconds().median() / 3600.0)
    WINDOW = pd.Timedelta(hours=obs_step_h)

    farms = (farms_df.farm.tolist() if args.region == "all"
             else farms_df[farms_df.region == args.region].farm.tolist())
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    turbines = turbines[turbines.farm.isin(farms)]
    L = len(args.leads)
    lead_pos = {lh: k for k, lh in enumerate(args.leads)}
    print(f"{args.region}: {len(farms)} farms, {cap.sum():.0f} MW\n")

    # ---- file maps, restricted to init times present in EVERY run ----
    fmaps: dict[str, dict] = {}
    has_power: dict[str, bool] = {}
    for label, d in runs.items():
        files = sorted(d.glob("forecast_*.nc"))
        if not files:
            print(f"{label:16s} NO forecast_*.nc in {d} -- skipping")
            continue
        with xr.open_dataset(files[0]) as ds0:
            hp, hw = args.var in ds0, "ws100" in ds0
        if not hw:
            print(f"{label:16s} no ws100 -- skipping (needed for the power curve)")
            continue
        has_power[label] = hp
        print(f"{label:16s} {len(files):4d} files   "
              f"{'DIRECT + POWERCURVE' if hp else 'POWERCURVE only (no %s)' % args.var}")
        fmaps[label] = {parse_init(f): f for f in files}

    if not fmaps:
        raise SystemExit("no usable runs -- check the dirs")
    common = sorted(set.intersection(*(set(m) for m in fmaps.values())))
    print(f"\ncommon init times across {len(fmaps)} run(s): {len(common)}")
    if not common:
        raise SystemExit("no init times common to all runs")
    for label, fm in fmaps.items():
        if len(fm) > len(common):
            print(f"  note: {label} has {len(fm)} inits, {len(fm) - len(common)} dropped")

    # An inference directory can be HETEROGENEOUS: an earlier pass may have written files without
    # the power variable and a later pass added it. Deciding from files[0] alone then blows up
    # mid-run. Verify, and drop inits where a direct-capable run lacks it, so every series is
    # scored on one identical sample.
    direct_runs = [r for r in fmaps if has_power[r]]
    if direct_runs:
        print(f"\nverifying '{args.var}' is present in all {len(common)} candidate inits "
              f"({len(direct_runs)} run(s) with a direct forecast)...")
        good, missing = [], {r: 0 for r in direct_runs}
        for i, init in enumerate(common):
            ok_all = True
            for r in direct_runs:
                with xr.open_dataset(fmaps[r][init]) as ds:
                    if args.var not in ds:
                        missing[r] += 1
                        ok_all = False
                        break
            if ok_all:
                good.append(init)
            if i and i % 500 == 0:
                print(f"  checked {i}/{len(common)}", flush=True)
        if len(good) < len(common):
            print(f"  DROPPED {len(common) - len(good)} init(s) missing '{args.var}':")
            for r, c in missing.items():
                if c:
                    print(f"    {r}: {c} file(s) without it  <-- that inference run is incomplete")
        common = good
        print(f"  usable init times: {len(common)}")
        if not common:
            raise SystemExit(f"no init has '{args.var}' in every direct run")

    # ---- restrict to the chosen evaluation period (by INIT time) ----
    common_idx = pd.DatetimeIndex(common)
    keep = np.ones(len(common_idx), dtype=bool)
    if args.start:
        keep &= common_idx >= pd.Timestamp(args.start, tz="UTC")
    if args.end:
        keep &= common_idx <= pd.Timestamp(args.end, tz="UTC")
    if args.season != "all":
        keep &= common_idx.month.isin(list(SEASONS[args.season]))
    sfx = ""
    if not keep.all() or args.season != "all":
        common = [t for t, k in zip(common, keep) if k]
        span = (f"{args.start or 'open'} .. {args.end or 'open'}"
                + ("" if args.season == "all" else f" | season {args.season}"))
        print(f"\nevaluation period [{span}]: {len(common)} init times kept")
        if not common:
            raise SystemExit("no init times in the chosen evaluation period")
        tags = ([args.season] if args.season != "all" else []) + \
               ([str(args.start)] if args.start else []) + ([str(args.end)] if args.end else [])
        sfx = "_" + "_".join(tags)                         # keeps period outputs from overwriting

    # ---- how far do the forecasts reach?  (a window needs the forecast at t+step) ----
    if args.curve_window:
        probe = common[0]
        max_lead_h = min_lead_h = None
        for label, fm in fmaps.items():
            with xr.open_dataset(fm[probe]) as ds_p:
                ft = pd.DatetimeIndex(ds_p["time"].values).tz_localize("UTC")
            reach = float((ft.max() - probe).total_seconds() / 3600.0)
            first = float((ft.min() - probe).total_seconds() / 3600.0)
            max_lead_h = reach if max_lead_h is None else min(max_lead_h, reach)
            min_lead_h = first if min_lead_h is None else max(min_lead_h, first)
        print(f"\nobservation window = {obs_step_h:.0f} h (power_obs at t is the mean over "
              f"[t, t+{obs_step_h:.0f}h)); forecasts reach +{max_lead_h:.0f} h")
        keep_leads = [lh for lh in args.leads if lh + obs_step_h <= max_lead_h
                      and (not args.curve_window_control or lh - obs_step_h >= min_lead_h)]
        if len(keep_leads) < len(args.leads):
            gone = [lh for lh in args.leads if lh not in keep_leads]
            why = f"need the forecast at +{max(gone) + obs_step_h:.0f} h"
            if args.curve_window_control:
                why += f" and at +{min(gone) - obs_step_h:.0f} h for the backward control"
            print(f"  lead(s) {gone} h cannot form the window ({why}) and are DROPPED FROM "
                  f"EVERY SERIES, so all methods stay on one identical sample.")
            print("  (--no-curve-window scores them with the instantaneous curve instead)")
            args.leads = keep_leads
            if not args.leads:
                raise SystemExit("no lead time can form the observation window -- "
                                 "run with --no-curve-window")
        L = len(args.leads)
        lead_pos = {lh: k for k, lh in enumerate(args.leads)}

    # ---- drop farms with NO observations in the scored window ----
    # UK observations end in 2023 (ALLOWED_YEARS = Nost's full-capacity periods), so a 2024-25
    # inference window has nothing to score most UK farms against, and the regional TOTAL (which
    # needs EVERY farm reporting) would come out silently all-NaN.
    valid_times = pd.DatetimeIndex(sorted({i + pd.Timedelta(hours=lh)
                                           for i in common for lh in args.leads}))
    valid_times = valid_times.intersection(obs.index)
    if len(valid_times) == 0:
        raise SystemExit("no forecast valid time overlaps power_obs.csv -- check the periods")
    counts = obs.loc[valid_times, farms].notna().sum()
    dead = [f for f in farms if counts[f] == 0]
    if dead:
        print(f"\n{len(dead)} of {len(farms)} {args.region} farm(s) have NO observations in the "
              f"scored window ({valid_times.min():%Y-%m-%d} .. {valid_times.max():%Y-%m-%d}):")
        print("   " + ", ".join(dead))
        farms = [f for f in farms if counts[f] > 0]
        if not farms:
            raise SystemExit(
                f"no {args.region} farm has observations in the forecast window -- nothing to "
                f"score. UK observations end in 2023, so a 2024-25 window can only score BE.")
        cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
        turbines = turbines[turbines.farm.isin(farms)]
        print(f"   -> scoring the remaining {len(farms)} farm(s), total capacity {cap.sum():.0f} MW")
        print("      (the TOTAL below is that reduced fleet, NOT the full region)")

    cap_np = cap.to_numpy()
    total_cap = float(cap.sum())
    curves = build_farm_curves(farms_df, specs, farms)
    F = len(farms)

    # ---- wind-regime machinery ----
    ws_edges = list(args.regime_ws_edges)
    reg_labels = ([f"0-{ws_edges[0]:g}"]
                  + [f"{ws_edges[i]:g}-{ws_edges[i+1]:g}" for i in range(len(ws_edges) - 1)]
                  + [f"{ws_edges[-1]:g}+"])
    R = len(reg_labels)
    # per-farm CF thresholds, and the fleet-aggregate ones for the regional total
    cf_thr = np.stack([cf_edges_from_ws(curves[f], float(cap_np[i]), ws_edges)
                       for i, f in enumerate(farms)])                       # (F, R-1)
    def fleet_curve(ws):
        return sum(curves[f](ws) for f in farms)
    cf_thr_tot = cf_edges_from_ws(fleet_curve, total_cap, ws_edges)          # (R-1,)
    print(f"wind regimes ({args.regime_by}): " + " | ".join(reg_labels) + "  m/s")
    if args.regime_by == "obs-cf":
        print("  binned on OBSERVED power through each farm's own curve (truth-conditioned)")
        print("  fleet-total capacity-factor thresholds: "
              + ", ".join(f"{e:g} m/s -> CF {c:.3f}" for e, c in zip(ws_edges, cf_thr_tot)))
        print("  NOTE: above rated the curve is flat, so the top bin cannot be subdivided.")
    else:
        print("  binned on the FORECAST wind at the farm -- operational, NOT a truth conditional")

    # ---- accumulators, keyed (run, method) ----
    series = ([(r, "direct") for r in fmaps if has_power[r]]
              + [(r, "curve") for r in fmaps]
              + ([(r, "curve3h") for r in fmaps] if args.curve_window else [])
              + ([(r, "curve3hb") for r in fmaps]
                 if args.curve_window and args.curve_window_control else []))
    keys = list(series)
    z2 = lambda: np.zeros((F, L))                                          # noqa: E731
    sae = {s: z2() for s in keys}      # sum |err|      -> MAE
    sse = {s: z2() for s in keys}      # sum err^2      -> RMSE
    sbe = {s: z2() for s in keys}      # sum err        -> bias
    n = {s: z2() for s in keys}
    sae_t = {s: np.zeros(L) for s in keys}   # regional total (all farms reporting)
    sse_t = {s: np.zeros(L) for s in keys}
    sbe_t = {s: np.zeros(L) for s in keys}
    n_t = {s: np.zeros(L) for s in keys}
    # cross-moments of the regional total, for the MSE decomposition and the variance ratio
    sp_t = {s: np.zeros(L) for s in keys}    # sum pred
    so_t = {s: np.zeros(L) for s in keys}    # sum obs
    spp_t = {s: np.zeros(L) for s in keys}   # sum pred^2
    soo_t = {s: np.zeros(L) for s in keys}   # sum obs^2
    spo_t = {s: np.zeros(L) for s in keys}   # sum pred*obs
    # regime split: per-farm pooled over leads (F,R), and regional total per lead (L,R)
    sae_fr = {s: np.zeros((F, R)) for s in keys}
    n_fr = {s: np.zeros((F, R)) for s in keys}
    sae_tr = {s: np.zeros((L, R)) for s in keys}
    sse_tr = {s: np.zeros((L, R)) for s in keys}
    sbe_tr = {s: np.zeros((L, R)) for s in keys}
    n_tr = {s: np.zeros((L, R)) for s in keys}

    def accumulate(s, ppred, ptrue, k, rf=None, rt=None):
        """rf: (F,) regime index per farm; rt: regime index of the regional total."""
        ok = np.isfinite(ptrue) & np.isfinite(ppred)
        if ok.any():
            e = ppred[ok] - ptrue[ok]
            sae[s][ok, k] += np.abs(e); sse[s][ok, k] += e * e
            sbe[s][ok, k] += e;         n[s][ok, k] += 1
            if rf is not None:
                fi = np.where(ok)[0]
                np.add.at(sae_fr[s], (fi, rf[fi]), np.abs(e))
                np.add.at(n_fr[s], (fi, rf[fi]), 1.0)
        if ok.all():                      # total only when every farm reports
            pt, ot = ppred.sum(), ptrue.sum()
            et = pt - ot
            sae_t[s][k] += abs(et); sse_t[s][k] += et * et
            sbe_t[s][k] += et;      n_t[s][k] += 1
            sp_t[s][k] += pt;   so_t[s][k] += ot
            spp_t[s][k] += pt * pt; soo_t[s][k] += ot * ot; spo_t[s][k] += pt * ot
            if rt is not None:
                sae_tr[s][k, rt] += abs(et); sse_tr[s][k, rt] += et * et
                sbe_tr[s][k, rt] += et;      n_tr[s][k, rt] += 1

    def regimes_obs(ptrue):
        """(per-farm regime, total regime) from the OBSERVED power. NaN farms -> bin 0, unused."""
        with np.errstate(invalid="ignore"):
            cf_f = np.divide(ptrue, cap_np, out=np.zeros_like(ptrue), where=cap_np > 0)
        rf = np.array([int(np.digitize(cf_f[i], cf_thr[i])) for i in range(F)])
        rt = int(np.digitize(ptrue.sum() / total_cap, cf_thr_tot)) if np.isfinite(ptrue).all() \
            else None
        return rf, rt

    def regimes_fcws(ws_row):
        """(per-farm regime, total regime) from the FORECAST wind at each farm."""
        rf = np.digitize(ws_row, ws_edges).astype(int)
        wt = float(np.sum(ws_row * cap_np) / total_cap)      # capacity-weighted fleet wind
        return rf, int(np.digitize(wt, ws_edges))

    recon_cache: dict = {}

    def get_recon(lat, lon):
        key = (lat.size, round(float(lat[0]), 4), round(float(lon[-1]), 4))
        if key not in recon_cache:
            recon_cache[key] = build_reconstruction(lat, lon, turbines, farms)
        return recon_cache[key]

    # ---- the runs ----
    no_window = 0                     # (init, lead) samples with no forecast at t+step
    for label, fmap in fmaps.items():
        hp = has_power[label]
        print(f"\nscoring {label} ({len(common)} inits)...")
        for init in common:
            with xr.open_dataset(fmap[init]) as ds:
                cell_idx, G, cap_cell = get_recon(ds["latitude"].values, ds["longitude"].values)
                fc_times = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
                ws_farm = farm_wind(ds["ws100"].values[:, cell_idx], G)      # (T, F)
                have_var = hp and args.var in ds        # defensive: never KeyError mid-run
                if have_var:
                    field = ds[args.var].values[:, cell_idx]
                    # a raw-power forecast is MW/cell -> to CF first, so the reconstruction is
                    # identical in both cases
                    cf = field if args.var == "capacityfactor" else np.divide(
                        field, cap_cell[None, :], out=np.full_like(field, np.nan),
                        where=cap_cell[None, :] > 0)
                    p_direct_all = cf @ G.T

            p_curve_all = np.column_stack([curves[f](ws_farm[:, i]) for i, f in enumerate(farms)])

            t2i = {t: j for j, t in enumerate(fc_times)}
            for lh in args.leads:
                vt = init + pd.Timedelta(hours=lh)
                if vt not in t2i or vt not in obs.index:
                    continue
                j, k = t2i[vt], lead_pos[lh]
                # far edge of the observation's averaging window, [vt, vt+step)
                j3 = t2i.get(vt + WINDOW) if args.curve_window else None
                jb = t2i.get(vt - WINDOW) if (args.curve_window
                                              and args.curve_window_control) else None
                if args.curve_window and (j3 is None
                                          or (args.curve_window_control and jb is None)):
                    no_window += 1
                    continue          # drop from EVERY series -- keep one identical sample
                ptrue = obs.loc[vt, farms].to_numpy(float)
                # one regime per (init, lead), shared by every method so the split is identical
                rf, rt = (regimes_obs(ptrue) if args.regime_by == "obs-cf"
                          else regimes_fcws(ws_farm[j]))
                accumulate((label, "curve"), p_curve_all[j], ptrue, k, rf, rt)
                if args.curve_window:
                    # average the POWERS, not the winds: the curve is non-linear, so
                    # A(mean ws) != mean A(ws)
                    accumulate((label, "curve3h"),
                               0.5 * (p_curve_all[j] + p_curve_all[j3]), ptrue, k, rf, rt)
                    if args.curve_window_control:
                        # same 2-forecast smoothing, centred at t-dt/2 instead of t+dt/2:
                        # the difference from curve3h is misalignment ALONE
                        accumulate((label, "curve3hb"),
                                   0.5 * (p_curve_all[jb] + p_curve_all[j]), ptrue, k, rf, rt)
                if have_var:
                    accumulate((label, "direct"), p_direct_all[j], ptrue, k, rf, rt)

    if args.curve_window and no_window:
        print(f"\nnote: {no_window} (init, lead) sample(s) had no forecast at t+"
              f"{obs_step_h:.0f}h and were dropped from every series")

    # =========================================================================
    # metrics
    # =========================================================================
    with np.errstate(invalid="ignore", divide="ignore"):
        mae   = {s: sae[s] / n[s] for s in keys}
        rmse  = {s: np.sqrt(sse[s] / n[s]) for s in keys}
        bias  = {s: sbe[s] / n[s] for s in keys}
        nmae  = {s: 100.0 * mae[s] / cap_np[:, None] for s in keys}
        mae_t   = {s: sae_t[s] / n_t[s] for s in keys}
        rmse_t  = {s: np.sqrt(sse_t[s] / n_t[s]) for s in keys}
        bias_t  = {s: sbe_t[s] / n_t[s] for s in keys}
        nmae_t  = {s: 100.0 * mae_t[s] / total_cap for s in keys}

    lbl = lambda s: f"{s[0]} · {METHOD_LABEL[s[1]]}"                                   # noqa: E731

    def decompose(s, k=None):
        """MSE decomposition of the regional total.

            MSE = (mean_p - mean_o)^2  +  (sd_p - sd_o)^2  +  2*sd_p*sd_o*(1 - r)
                   bias^2 (systematic)     amplitude          phase / timing

        The first two are calibratable (a constant offset and a gain would remove them); the
        third is genuine timing error and is the only part a better forecast must earn.
        Pass k for one lead, or None to pool over all leads.
        """
        sl = slice(None) if k is None else slice(k, k + 1)
        N = float(np.nansum(n_t[s][sl]))
        if N == 0:
            return dict(mse=np.nan, bias2=np.nan, amp=np.nan, phase=np.nan,
                        sd_p=np.nan, sd_o=np.nan, r=np.nan, var_ratio=np.nan)
        mp = np.nansum(sp_t[s][sl]) / N
        mo = np.nansum(so_t[s][sl]) / N
        vp = max(np.nansum(spp_t[s][sl]) / N - mp * mp, 0.0)
        vo = max(np.nansum(soo_t[s][sl]) / N - mo * mo, 0.0)
        cov = np.nansum(spo_t[s][sl]) / N - mp * mo
        sd_p, sd_o = np.sqrt(vp), np.sqrt(vo)
        r = cov / (sd_p * sd_o) if sd_p > 0 and sd_o > 0 else np.nan
        return dict(mse=np.nansum(sse_t[s][sl]) / N,
                    bias2=(mp - mo) ** 2,
                    amp=(sd_p - sd_o) ** 2,
                    phase=2 * sd_p * sd_o * (1 - r) if np.isfinite(r) else np.nan,
                    sd_p=sd_p, sd_o=sd_o, r=r,
                    var_ratio=sd_p / sd_o if sd_o > 0 else np.nan)

    # ---- 1. total MAE by lead ----
    # up to 4 runs x 4 methods = 16 columns; full labels ran off the line, so number the
    # columns and print the legend above.
    print(f"\nTOTAL {args.region} power — MAE as % of {total_cap:.0f} MW")
    cols = list(series)
    for i, s in enumerate(cols, 1):
        print(f"  [{i:2d}] {lbl(s)}")
    hdr = "lead  " + "".join(f"{f'[{i}]':>9s}" for i in range(1, len(cols) + 1))
    print(hdr); print("-" * len(hdr))
    for lh in args.leads:
        k = lead_pos[lh]
        print(f"+{lh:2d}h  " + "".join(f"{nmae_t[s][k]:8.2f}%" for s in cols))

    # ---- 2. summary pooled over leads ----
    print(f"\nSUMMARY — {args.region} total, pooled over all leads")
    print(f"{'series':34s} {'MAE%':>7s} {'RMSE MW':>9s} {'bias MW':>9s}")
    print("-" * 62)
    summary_rows = []
    for s in series:
        N = np.nansum(n_t[s])
        if N == 0:
            continue
        m = np.nansum(sae_t[s]) / N
        r = np.sqrt(np.nansum(sse_t[s]) / N)
        b = np.nansum(sbe_t[s]) / N
        print(f"{lbl(s):34s} {100*m/total_cap:6.2f}% {r:9.1f} {b:+9.1f}")
        d = decompose(s)
        summary_rows.append(dict(series=lbl(s), run=s[0], method=s[1], mae_mw=m,
                                 nmae_pct=100 * m / total_cap, rmse_mw=r, bias_mw=b,
                                 mse=d["mse"], bias2=d["bias2"], amplitude=d["amp"],
                                 phase=d["phase"], sd_pred=d["sd_p"], sd_obs=d["sd_o"],
                                 corr=d["r"], var_ratio=d["var_ratio"], n=int(N)))

    # =========================================================================
    # plots
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {r: c for r, c in zip(fmaps, plt.cm.tab10.colors)}
    style = {"direct": "-", "curve": "--", "curve3h": ":", "curve3hb": "-."}

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for s in series:
        ax.plot(args.leads, nmae_t[s], marker="o", ms=4, lw=1.8,
                color=colors[s[0]], ls=style[s[1]], label=lbl(s))
    ax.set(xlabel="lead time [h]", ylabel="MAE [% of total capacity]",
           title=f"Total {args.region} power — MAE vs observations ({F} farms, {total_cap:.0f} MW)\n"
                 f"solid = direct CF, dashed = power curve (inst.)"
                 + (", dotted = power curve (window mean)" if args.curve_window else ""))
    ax.set_xticks(args.leads); ax.grid(ls="--", alpha=0.5)
    ax.legend(fontsize=7, ncol=2 if len(series) > 6 else 1)
    fig.tight_layout()
    p = args.out / f"mae_total_{args.region}{sfx}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"\nsaved {p}")

    ncol = 5 if F > 6 else 3
    nrow = int(np.ceil(F / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                            sharex=True, sharey=True, squeeze=False)
    for i, farm in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        for s in series:
            ax.plot(args.leads, nmae[s][i], marker="o", ms=3, lw=1.2,
                    color=colors[s[0]], ls=style[s[1]], label=lbl(s))
        ax.set_title(f"{farm}\n{cap[farm]:.0f} MW", fontsize=8)
        ax.grid(ls="--", alpha=0.4)
    for j in range(F, nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    axs[0][0].legend(fontsize=6)
    fig.supxlabel("lead time [h]"); fig.supylabel("MAE [% of farm capacity]")
    fig.suptitle(f"Per-farm power MAE vs observations — {args.region}  "
                 f"(solid = direct, dashed = curve inst."
                 + (", dotted = curve window mean)" if args.curve_window else ")"))
    fig.tight_layout()
    p = args.out / f"mae_per_farm_{args.region}{sfx}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    # ---- 3. per-farm BIAS ----
    # Sign matters: the idealised specs power curve ignores wake losses, so it should sit ABOVE
    # zero (over-predicting). Whether the direct forecast removes that -- or overshoots into
    # under-prediction -- is invisible in MAE.
    with np.errstate(invalid="ignore", divide="ignore"):
        nbias = {s: 100.0 * bias[s] / cap_np[:, None] for s in series}
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                            sharex=True, sharey=True, squeeze=False)
    for i, farm in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        for s in series:
            ax.plot(args.leads, nbias[s][i], marker="o", ms=3, lw=1.2,
                    color=colors[s[0]], ls=style[s[1]], label=lbl(s))
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_title(f"{farm}\n{cap[farm]:.0f} MW", fontsize=8)
        ax.grid(ls="--", alpha=0.4)
    for j in range(F, nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    axs[0][0].legend(fontsize=6)
    fig.supxlabel("lead time [h]"); fig.supylabel("bias [% of farm capacity]")
    fig.suptitle(f"Per-farm power BIAS (forecast − observed) — {args.region}  "
                 f"(solid = direct, dashed = curve inst."
                 + (", dotted = curve window mean" if args.curve_window else "")
                 + "; above 0 = over-predicting)")
    fig.tight_layout()
    p = args.out / f"bias_per_farm_{args.region}{sfx}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    # ---- 4. MSE decomposition + variance ratio ----
    dec_all = {s: decompose(s) for s in series}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.8))

    labs = [lbl(s) for s in series]
    b2 = np.array([dec_all[s]["bias2"] for s in series])
    am = np.array([dec_all[s]["amp"] for s in series])
    ph = np.array([dec_all[s]["phase"] for s in series])
    x = np.arange(len(labs))
    axL.bar(x, b2, label="bias²  (systematic offset)", color="tab:red")
    axL.bar(x, am, bottom=b2, label="amplitude  (σ mismatch)", color="tab:orange")
    axL.bar(x, ph, bottom=b2 + am, label="phase  (timing, irreducible)", color="tab:blue")
    axL.set_xticks(x); axL.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    axL.set_ylabel("MSE of the regional total [MW²]")
    axL.set_title("MSE decomposition, pooled over leads\n"
                  "red+orange are calibratable; blue must be earned")
    axL.grid(ls="--", alpha=0.4, axis="y"); axL.legend(fontsize=8)

    for s in series:
        vr = [decompose(s, k)["var_ratio"] for k in range(L)]
        axR.plot(args.leads, vr, marker="o", ms=4, lw=1.8,
                 color=colors[s[0]], ls=style[s[1]], label=lbl(s))
    axR.axhline(1.0, color="grey", ls="--", lw=1.2)
    axR.annotate("perfectly dispersed", (args.leads[-1], 1.0), ha="right", va="bottom",
                 fontsize=8, color="grey")
    axR.set(xlabel="lead time [h]", ylabel=r"$\sigma_{pred}\,/\,\sigma_{obs}$",
            title="Variance ratio of the regional total\nbelow 1 = under-dispersive (too smooth)")
    axR.set_xticks(args.leads); axR.grid(ls="--", alpha=0.5); axR.legend(fontsize=7)
    fig.tight_layout()
    p = args.out / f"decomposition_{args.region}{sfx}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    # printed decomposition table
    print(f"\nMSE DECOMPOSITION — {args.region} total, pooled over leads  "
          f"(MSE = bias² + amplitude + phase)")
    print(f"{'series':34s} {'RMSE':>8s} {'bias²':>9s} {'ampl':>9s} {'phase':>9s} "
          f"{'σp/σo':>7s} {'r':>6s}")
    print("-" * 88)
    for s in series:
        d = dec_all[s]
        if not np.isfinite(d["mse"]):
            continue
        tot = d["bias2"] + d["amp"] + d["phase"]
        print(f"{lbl(s):34s} {np.sqrt(d['mse']):8.1f} "
              f"{d['bias2']:9.0f} {d['amp']:9.0f} {d['phase']:9.0f} "
              f"{d['var_ratio']:7.3f} {d['r']:6.3f}"
              f"{'   (check: sum/MSE = %.3f)' % (tot / d['mse']) if d['mse'] > 0 else ''}")

    # ---- 5. the alignment test: instantaneous vs window-mean power curve ----
    # The ONLY difference between the two lines is whether the baseline predicts the same
    # quantity power_obs holds (a mean over [t, t+step)) or an instantaneous value at t. Any gap
    # is the size of the bookkeeping penalty the instantaneous baseline was paying, and it has
    # to be subtracted from the direct-vs-curve margin before that margin means anything.
    if args.curve_window:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
        for r in fmaps:
            axA.plot(args.leads, nmae_t[(r, "curve")], marker="o", ms=4, lw=1.8, ls="--",
                     color=colors[r], label=f"{r} · inst.")
            axA.plot(args.leads, nmae_t[(r, "curve3h")], marker="s", ms=4, lw=1.8, ls=":",
                     color=colors[r], label=f"{r} · window mean")
            if has_power[r]:
                axA.plot(args.leads, nmae_t[(r, "direct")], marker="^", ms=4, lw=1.8, ls="-",
                         color=colors[r], label=f"{r} · direct")
            axB.plot(args.leads,
                     nmae_t[(r, "curve")] - nmae_t[(r, "curve3h")],
                     marker="o", ms=4, lw=1.8, color=colors[r], label=r)
        axA.set(xlabel="lead time [h]", ylabel="MAE [% of total capacity]",
                title="Total power MAE — does aligning the baseline to the\n"
                      "observation window change the ranking?")
        axA.set_xticks(args.leads); axA.grid(ls="--", alpha=0.5)
        axA.legend(fontsize=7, ncol=2)
        axB.axhline(0.0, color="k", lw=1.0)
        axB.set(xlabel="lead time [h]", ylabel="MAE(inst.) − MAE(window mean)  [% of capacity]",
                title="Penalty the instantaneous power curve was paying\n"
                      "for predicting a snapshot against a 3 h mean")
        axB.set_xticks(args.leads); axB.grid(ls="--", alpha=0.5); axB.legend(fontsize=8)
        fig.tight_layout()
        p = args.out / f"curve_window_test_{args.region}{sfx}.png"
        fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

        # ---- what the instantaneous curve would score if it were merely ALIGNED ----
        # Worked in MSE, where the effect is additive: a misalignment tau adds ~2*var*(1-rho(tau))
        # on top of the aligned error. fwd (offset 0), inst (offset dt/2), bwd (offset dt) share
        # everything else, and fwd/bwd share the 2-forecast smoothing as well. So
        #     D = MSE_bwd - MSE_fwd            is the added MSE of a full dt misalignment
        #     MSE_inst - D*(1/2)^p             is the instantaneous curve with alignment removed
        # p = 1 for a rough process, p = 2 for a smooth one; both are reported as a range, because
        # 3-hourly observations cannot pin down the sub-3h behaviour that decides p. Crucially
        # the smoothing bonus never enters: we only ever subtract from inst, never compare to it.
        def pooled_mse(r, meth):
            N = np.nansum(n_t[(r, meth)])
            return np.nansum(sse_t[(r, meth)]) / N if N else np.nan

        print(f"\nALIGNMENT TEST — {args.region} total RMSE [MW], pooled over leads")
        if not args.curve_window_control:
            print("  (re-run with --curve-window-control for the number that actually matters:")
            print("   without it, inst.-vs-fwd confounds the alignment fix with the fact that")
            print("   averaging two forecasts reduces error on its own.)")
        hdr = (f"{'run':20s} {'inst.':>8s} {'fwd':>8s} "
               + (f"{'bwd':>8s} {'aligned inst.':>16s} " if args.curve_window_control else "")
               + f"{'direct':>8s} {'margin was':>11s} {'margin aligned':>15s}")
        print(hdr); print("-" * len(hdr))
        for r in fmaps:
            mi, mf = pooled_mse(r, "curve"), pooled_mse(r, "curve3h")
            di = np.sqrt(pooled_mse(r, "direct")) if has_power[r] else np.nan
            row = f"{r:20s} {np.sqrt(mi):7.1f} {np.sqrt(mf):7.1f} "
            adj = ""
            if args.curve_window_control:
                mb = pooled_mse(r, "curve3hb")
                D = mb - mf
                if not np.isfinite(D) or D <= 0:
                    row += f"{np.sqrt(mb):7.1f} {'none detected':>16s} "
                    adj = "n/a"
                else:
                    lo = np.sqrt(max(mi - D / 2.0, 0.0))     # p = 1  (rough)
                    hi = np.sqrt(max(mi - D / 4.0, 0.0))     # p = 2  (smooth)
                    row += f"{np.sqrt(mb):7.1f} {f'{lo:.0f}-{hi:.0f}':>16s} "
                    adj = f"{lo - di:+.0f} .. {hi - di:+.0f}" if np.isfinite(di) else "n/a"
            print(row + f"{di:7.1f} {np.sqrt(mi) - di:+10.1f} {adj:>15s}")

        print(f"\n  inst. = curve on the forecast at t          (centroid t, offset "
              f"{obs_step_h/2:.1f}h from the obs window)")
        print(f"  fwd   = 1/2[P(t)+P(t+{obs_step_h:.0f}h)]                (centroid "
              f"t+{obs_step_h/2:.1f}h, ALIGNED)")
        if args.curve_window_control:
            print(f"  bwd   = 1/2[P(t-{obs_step_h:.0f}h)+P(t)]               (centroid "
                  f"t-{obs_step_h/2:.1f}h, offset {obs_step_h:.0f}h -- same smoothing as fwd)")
            print("\n  SIGN TEST first: if the convention matters at all, bwd must be clearly WORSE")
            print("  than fwd. It has identical smoothing and is only misaligned, so bwd-fwd is a")
            print("  clean measure of misalignment. If bwd <= fwd there is no penalty to correct")
            print("  and the original instantaneous comparison stands as published.")
            print("\n  'aligned inst.' is the instantaneous curve with only the alignment penalty")
            print("  removed -- still ONE forecast, so it is directly comparable to direct. The")
            print("  range spans error growing linearly (rough) to quadratically (smooth) in the")
            print("  offset; 3-hourly obs cannot decide which, so quote the range, not a point.")
            print("  If 'margin aligned' stays positive across the whole range, the direct-vs-curve")
            print("  result survives. If it crosses zero, it does not.")
        print("\n  Do NOT quote 'fwd' against 'direct' as the headline: fwd uses the forecast at")
        print("  two times and direct at one, so that comparison favours the curve.")

    # ---- 6. the wind-regime split ----
    with np.errstate(invalid="ignore", divide="ignore"):
        nmae_tr = {s: 100.0 * (sae_tr[s] / n_tr[s]) / total_cap for s in keys}     # (L,R)
        nmae_fr = {s: 100.0 * (sae_fr[s] / n_fr[s]) / cap_np[:, None] for s in keys}
        bias_tr = {s: sbe_tr[s] / n_tr[s] for s in keys}

    cols = list(series)
    counts = n_tr[cols[0]].sum(axis=0)
    print(f"\nTOTAL {args.region} POWER BY WIND REGIME — MAE as % of {total_cap:.0f} MW, "
          f"pooled over leads   (regime by {args.regime_by})")
    for i, s in enumerate(cols, 1):
        print(f"  [{i:2d}] {lbl(s)}")
    hdr = (f"{'regime':>10s} {'n':>7s} " + "".join(f"{f'[{i}]':>9s}"
                                                   for i in range(1, len(cols) + 1)))
    print(hdr); print("-" * len(hdr))
    for ri, rl in enumerate(reg_labels):
        row = f"{rl:>10s} {int(counts[ri]):7d} "
        for s in cols:
            N = n_tr[s][:, ri].sum()
            row += f"{(100.0 * sae_tr[s][:, ri].sum() / N / total_cap) if N else np.nan:8.2f}%"
        print(row)
    print("  m/s bands; the last one is 'at or above rated' and cannot be subdivided.")

    print(f"\nBIAS BY WIND REGIME [MW] — {args.region} total, pooled over leads "
          f"(+ = over-predict)")
    print(hdr); print("-" * len(hdr))
    for ri, rl in enumerate(reg_labels):
        row = f"{rl:>10s} {int(counts[ri]):7d} "
        for s in cols:
            N = n_tr[s][:, ri].sum()
            row += f"{(sbe_tr[s][:, ri].sum() / N) if N else np.nan:8.0f} "
        print(row)

    nrw = 2 if R > 2 else 1
    ncw = int(np.ceil(R / nrw))
    fig, axs = plt.subplots(nrw, ncw, figsize=(5.2 * ncw, 4.2 * nrw), squeeze=False, sharex=True)
    for ri in range(R):
        ax = axs[ri // ncw][ri % ncw]
        for s in series:
            ax.plot(args.leads, nmae_tr[s][:, ri], marker="o", ms=4, lw=1.7,
                    color=colors[s[0]], ls=style[s[1]], label=lbl(s))
        ax.set_title(f"{reg_labels[ri]} m/s   ({int(counts[ri])} valid times)", fontsize=10)
        ax.set_xlabel("lead time [h]"); ax.set_ylabel(f"MAE [% of {total_cap:.0f} MW]")
        ax.set_xticks(args.leads); ax.grid(ls="--", alpha=0.5)
    for j in range(R, nrw * ncw):
        axs[j // ncw][j % ncw].axis("off")
    axs[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Total {args.region} power MAE by wind regime — regime set by "
                 f"{'OBSERVED power through the power curve' if args.regime_by == 'obs-cf' else 'FORECAST wind'}",
                 fontsize=11)
    fig.tight_layout()
    p = args.out / f"mae_regimes_{args.region}{sfx}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    # =========================================================================
    # CSVs
    # =========================================================================
    rows = []
    for s in keys:
        for lh in args.leads:
            k = lead_pos[lh]
            rows.append(dict(run=s[0], method=s[1], lead_hours=lh, scope="TOTAL",
                             mae_mw=mae_t[s][k], nmae_pct=nmae_t[s][k], rmse_mw=rmse_t[s][k],
                             bias_mw=bias_t[s][k], n=int(n_t[s][k])))
            for i, farm in enumerate(farms):
                rows.append(dict(run=s[0], method=s[1], lead_hours=lh, scope=farm,
                                 mae_mw=mae[s][i, k], nmae_pct=nmae[s][i, k],
                                 rmse_mw=rmse[s][i, k], bias_mw=bias[s][i, k],
                                 n=int(n[s][i, k])))
    reg_rows = []
    for s in keys:
        for ri, rl in enumerate(reg_labels):
            for lh in args.leads:
                k = lead_pos[lh]
                reg_rows.append(dict(run=s[0], method=s[1], regime=rl, lead_hours=lh,
                                     scope="TOTAL", regime_by=args.regime_by,
                                     mae_mw=sae_tr[s][k, ri] / n_tr[s][k, ri]
                                     if n_tr[s][k, ri] else np.nan,
                                     nmae_pct=nmae_tr[s][k, ri],
                                     rmse_mw=np.sqrt(sse_tr[s][k, ri] / n_tr[s][k, ri])
                                     if n_tr[s][k, ri] else np.nan,
                                     bias_mw=bias_tr[s][k, ri], n=int(n_tr[s][k, ri])))
            for i, farm in enumerate(farms):          # per farm, pooled over leads
                reg_rows.append(dict(run=s[0], method=s[1], regime=rl, lead_hours=-1,
                                     scope=farm, regime_by=args.regime_by,
                                     mae_mw=sae_fr[s][i, ri] / n_fr[s][i, ri]
                                     if n_fr[s][i, ri] else np.nan,
                                     nmae_pct=nmae_fr[s][i, ri], rmse_mw=np.nan, bias_mw=np.nan,
                                     n=int(n_fr[s][i, ri])))
    out_reg = args.out / f"scores_regimes_{args.region}{sfx}.csv"
    pd.DataFrame(reg_rows).to_csv(out_reg, index=False)
    print(f"saved {out_reg}")

    out_csv = args.out / f"scores_{args.region}{sfx}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    if summary_rows:
        out_sum = args.out / f"summary_{args.region}{sfx}.csv"
        pd.DataFrame(summary_rows).to_csv(out_sum, index=False)
        print(f"saved {out_sum}")


if __name__ == "__main__":
    main()
