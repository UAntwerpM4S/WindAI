#!/usr/bin/env python3
"""Score wind power forecasts against the ENTSO-E/Elexon observations, by lead time.

Every run yields up to two power forecasts of the same quantity, scored on one identical sample:

  DIRECT  the model's own `capacityfactor`, mapped back to farms -- the adjoint of
          build_power.py's capacity-weighted distribution:
              P(farm,t) = SUM_cell capacity(farm's turbines in cell) * CF(cell,t)
          Only runs that carry the variable get this line.

  CURVE   the classical baseline: forecast ws100 at the farm (capacity-weighted over its cells)
          pushed through that farm's own power curve. power_obs at t is the MEAN over
          [t, t+3h), and DIRECT is trained on that mean, so a curve must be made to predict the
          same mean or it is charged an error that is bookkeeping rather than skill. HOW depends
          on what the curve is, and CURVE_KIND records it:
              specs      instantaneous, so average the POWERS: 1/2[A(ws_t) + A(ws_t+3h)].
                         Averaging powers, not winds -- the curve is cubic on the ramp, so
                         A(mean ws) != mean A(ws).
              empirical  already measured mean-to-mean, so read it at the mean WIND,
                         g(1/2[ws_t + ws_t+3h]), and do not average again.
          Every run gets these lines, so a weather-only run is scored on equal terms.

PER_FARM chooses what is scored. False: the summed regional total, the operationally relevant
quantity, and a case counts only when EVERY farm reports (a partial sum is not a known total).
True: each farm on its own, scored whenever THAT farm reports -- so the farms have different
sample sizes, which is printed.

METRIC is MAE or RMSE, as % of capacity (the unit's own). The choice is not cosmetic: MAE is
minimised by the conditional MEDIAN and RMSE by the conditional MEAN, and the two sides are fitted
to different functionals -- the measured curve to a median or mean of each bin (farm_curves
BIN_STAT), the model to its own squared-error training loss. A curve summarised by the median and
scored on MAE is fitted to the functional it is graded on, which flatters it for a reason that is
not skill. Score both ways, or pair BIN_STAT="mean" with METRIC="rmse", before reading a margin as
accuracy. BIAS is printed too: the idealised specs curve ignores wake losses so it should
over-predict at high wind, and whether DIRECT removes that is a testable claim neither metric
alone can show.

BINNING splits the sample, on the quantity REGIME_BY names (CERRA truth ws100 at the unit's
cells, or observed power through the unit's own curve). Mutually exclusive by construction:
  "none"       one curve per run/method.
  "regimes"    FIXED wind bands (REGIME_WS_EDGES), labelled in m/s to match verify_weather.py.
               With REGIME_BY="obs-cf" the curve is flat above rated, so the top bin is "at or
               above rated" and cannot be subdivided -- a property of power, not of the binning.
  "quantiles"  EQUAL-COUNT bins, N_QUANT of them, with edges cut PER UNIT so each farm is split
               on its own distribution ("this farm's calmest tenth"). Every bin then carries the
               same number of cases, unlike the fixed bands where the 12+ bin is 22.7% of BE
               hours and 0-4.5 is 15.5%. Edges are printed.

CAVEAT neither binned table can settle anything on its own: truth-conditioned bins reward an
UNDER-dispersive forecast in the middle and punish it in the tails, with no difference in skill.
The direct head sits near sigma_p/sigma_o 0.67 and the un-smoothed specs curve near 1.11, so part
of any middle-bin margin is that gap rather than accuracy. Read it as realised accuracy (which it
is), not as evidence of where skill lives.

Figures: one PNG (one per bin when PER_FARM and a binned mode are both on -- farms x bins does
not fit a readable single figure). Every number plotted is also printed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import farm_curves as fc

# ============================== SETTINGS ==============================
REGION   = "BE"              # "BE" | "UK" | "all"
METRIC   = "mae"             # "mae" | "rmse" -- see the METRIC note in the docstring. RMSE is
                             # aggregated from summed SQUARED errors, so the all-bins number is
                             # the true overall RMSE, not an average of the per-bin ones.
SEASON   = "all"             # "all" | "DJF" | "MAM" | "JJA" | "SON"  -- filters on INIT month
BINNING  = "regimes"            # "none" | "regimes" | "quantiles"   -- mutually exclusive
N_QUANT  = 10                # BINNING="quantiles": equal-count bins, cut per unit
REGIME_BY = "cerra-ws"       # what the bins are cut on, in both binned modes.
                             # "cerra-ws": CERRA truth ws100 at the unit's cells.
                             # "obs-cf"  : OBSERVED power through the unit's own curve.
                             #   obs-cf cannot separate winds above rated -- they all give the
                             #   same power -- so with "regimes" any farm already at rated by the
                             #   top edge gets an EMPTY top bin, and with "quantiles" the upper
                             #   bins are cut on ties. cerra-ws has no such blind spot.
PER_FARM = False            # False: the summed regional total. True: one series per farm.
CURVE_MODES = ["specs","empirical"] #, "empirical"]   # curve baselines, both scored on one sample.
                            # "specs"    : turbine_specs.csv through the cubic law. Nothing
                            #   observed, nothing fitted -- the manufacturer curve.
                            # "empirical": the farm's own MEASURED curve, by the method of bins
                            #   (IEC 61400-12-1) over TRAIN_START..TRAIN_END. Standard practice in
                            #   wind power forecasting, and the strong baseline: it carries the
                            #   farm's real wake, availability and electrical
                            #   losses, which no datasheet does.
TRAIN_START = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")   # where "empirical" is
                            # measured. Must end before INIT_START:
TRAIN_END   = pd.Timestamp("2024-07-31 21:00:00", tz="UTC")
                            # power_obs.csv starts 2020-01-01, and this window is exactly what
                            # the model saw (its training + validation), so both sides learn from
                            # the same history and are judged on the same held-out year.


FORECAST_DIRS = {
    "RegularWeather":     Path("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"),
   #"SH_Finetune":  Path("/mnt/weatherloss/WindPower/inference/WPDistr/SHC_Finetune"),
    #"Vanilla_Finetune":     Path("/mnt/weatherloss/WindPower/inference/WPDistr/Vanilla_Finetune"),
  #  "Vanilla":  Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaCapacityGT"),
   #"H_Finetune": Path("/mnt/weatherloss/WindPower/inference/WPDistr/HC_Finetune"),
 #"VH": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
   # "VH_Finetune": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VHC_Finetune"),
    #"VH_Finetune_7var": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VHC_Finetune_7var"),
   # "VH_Finetune_Half": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VHC_Half_Finetune"),
       "VH_Finetune_5k": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VHC_5k_Finetune"),
      # "VH_Finetune_10k": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VHC_10k_Finetune"),


}

WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")   # farms/turbines/obs/specs live here
TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
OUT_DIR    = Path("DistrFigures")

CF_VAR = "capacityfactor"
WS_VAR = "ws100"

INIT_START = pd.Timestamp("2024-8-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-7-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))
OBS_STEP_H = 3               # the observation window, and the forecast step

REGIME_WS_EDGES = [4.5, 8.0, 12.0]               # m/s; converted to CF through the unit's curve
REGIME_LABELS   = ["0-4.5", "4.5-8", "8-12", "12+"]
# ======================================================================

SEASONS = {"all": None, "DJF": {12, 1, 2}, "MAM": {3, 4, 5},
           "JJA": {6, 7, 8}, "SON": {9, 10, 11}}
FORECAST_RE = re.compile(r"forecast_(\d{14})")
FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")
CURVE_NAME = {"specs": "specs power curve", "empirical": "measured power curve"}
CURVE_STYLE = {"specs": "--", "empirical": ":"}
# How each curve must be READ to predict the observation's 3h mean. They differ, and using the
# wrong one silently costs the curve accuracy that is bookkeeping rather than skill:
#   "instant" the curve maps a SNAPSHOT wind to a snapshot power, so the POWERS are averaged:
#             0.5[A(ws_t) + A(ws_t+3h)]. Correct for specs -- a physical curve is instantaneous.
#   "window"  the curve was MEASURED as window-mean wind -> window-mean power (the IEC convention
#             bins averaged wind against averaged power over the same window), so the smoothing
#             is already inside it. It is read at the mean WIND, g(0.5[ws_t + ws_t+3h]), and the
#             powers are NOT averaged again -- doing both would smooth the ramp twice.
CURVE_KIND = {"specs": "instant", "empirical": "window"}


def mlabel(m):
    return ("direct (capacity factor)" if m == "direct"
            else f"{CURVE_NAME[m.split(':')[1]]} (window mean)")


def mstyle(m):
    return "-" if m == "direct" else CURVE_STYLE[m.split(":")[1]]

# Okabe-Ito: distinguishable under deuteranopia, protanopia and tritanopia. Ordered for contrast
# on white; yellow last because it washes out in a thin line. RUNS differ by colour AND marker,
# METHODS by linestyle, so the figure survives both colour-vision deficiency and greyscale.
CB_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
             "#56B4E9", "#000000", "#F0E442"]
CB_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def parse_init(path):
    return pd.to_datetime(FORECAST_RE.search(path.name).group(1),
                          format="%Y%m%d%H%M%S", utc=True)


def build_reconstruction(fc_lat, fc_lon, turbines, farms):
    """cell_idx (forecast cells holding turbines), G[f,j] = capacity of farm f in cell j."""
    coslat = np.cos(np.radians(float(fc_lat.mean())))
    tree = cKDTree(np.c_[to_180(fc_lon) * coslat, fc_lat])
    _, cell = tree.query(np.c_[to_180(turbines["longitude"]) * coslat,
                               turbines["latitude"].to_numpy()], k=1)
    t = turbines.assign(cell=cell.astype(int))
    cell_idx = np.sort(t["cell"].unique())
    cpos = {int(c): j for j, c in enumerate(cell_idx)}
    fpos = {f: i for i, f in enumerate(farms)}
    G = np.zeros((len(farms), cell_idx.size))
    for (farm, c), cap in t.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[fpos[farm], cpos[int(c)]] = cap
    return cell_idx, G


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    specs = pd.read_csv(WPOWER_DIR / "turbine_specs.csv", index_col=0)
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")

    farms = (farms_df.farm.tolist() if REGION == "all"
             else farms_df[farms_df.region.str.upper() == REGION].farm.tolist())
    if not farms:
        raise SystemExit(f"no farms for REGION={REGION!r}; have {sorted(farms_df.region.unique())}")
    turbines = turbines[turbines.farm.isin(farms)]
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    # turbines.csv supplies the reconstruction weights; farms.csv supplies the capacity every
    # percentage is divided by. farm_metadata.py keeps them consistent -- but if they ever drift
    # the MW would be built from one and normalised by the other, silently.
    tsum = turbines.groupby("farm")["capacity_mw"].sum().reindex(farms)
    off = (tsum - cap).abs() / cap
    if (off > 0.001).any():
        bad = ", ".join(f"{f}: turbines.csv {tsum[f]:.1f} MW vs farms.csv {cap[f]:.1f} MW"
                        for f in farms if off[f] > 0.001)
        raise SystemExit(f"capacity mismatch >0.1% -- rerun farm_metadata.py\n  {bad}")

    # counts and capacities must agree before a baseline built on them means anything
    fc.validate(farms, farms_df, turbines, specs)
    cset = {}
    for m in CURVE_MODES:
        if m == "specs":
            cset[m] = fc.build_specs(farms, farms_df, specs)
        else:
            # measuring the curve on the scored period would be a fake baseline, not a strong one
            if TRAIN_END >= INIT_START:
                raise SystemExit(f"curve measured to {TRAIN_END} but scoring starts {INIT_START}"
                                 f" -- overlapping, so the baseline would be in-sample")
            cset[m] = fc.empirical(farms, farms_df, turbines, obs, TRUTH_ZARR,
                                   TRAIN_START, TRAIN_END)
    print("\nCURVE baselines: " + ", ".join(
        CURVE_NAME[m] + (f" (measured {TRAIN_START.date()}..{TRAIN_END.date()})"
                         if m == "empirical" else "") for m in CURVE_MODES))
    print(f"Region {REGION}: {len(farms)} farms, {float(cap.sum()):.0f} MW")

    # what gets scored: the summed total, or each farm on its own
    if PER_FARM:
        units = [(f, float(cap[f]), np.array([i])) for i, f in enumerate(farms)]
    else:
        units = [(f"TOTAL {REGION}", float(cap.sum()), np.arange(len(farms)))]
    U = len(units)
    print(f"Scoring {'each farm separately' if PER_FARM else 'the summed regional total'}"
          f" ({U} series)")

    # leads that can form the window; every method is held to the same set
    dt = pd.Timedelta(hours=OBS_STEP_H)
    leads = [lh for lh in LEAD_HOURS if lh + OBS_STEP_H <= max(LEAD_HOURS)]
    dropped = sorted(set(LEAD_HOURS) - set(leads))
    if dropped:
        print(f"Leads dropped (no window available): {dropped}")
    lpos = {lh: k for k, lh in enumerate(leads)}
    L = len(leads)

    months = SEASONS[SEASON]
    fmaps = {}
    for label, d in FORECAST_DIRS.items():
        m = {parse_init(f): f for f in sorted(d.glob("forecast_*.nc"))
             if INIT_START <= parse_init(f) <= INIT_END}
        if months:
            m = {i: f for i, f in m.items() if i.month in months}
        print(f"{label}: {len(m)} files")
        fmaps[label] = m
    inits = sorted(set.intersection(*(set(m) for m in fmaps.values())))
    if not inits:
        raise SystemExit("no init times common to all runs")
    print(f"Common inits: {len(inits)} | season {SEASON}")

    if METRIC not in ("mae", "rmse"):
        raise SystemExit(f"METRIC must be 'mae' or 'rmse', got {METRIC!r}")
    if BINNING not in ("none", "regimes", "quantiles"):
        raise SystemExit(f"BINNING must be 'none', 'regimes' or 'quantiles', got {BINNING!r}")
    binned = BINNING != "none"
    nbin = len(REGIME_LABELS) if BINNING == "regimes" else (N_QUANT if binned else 1)
    probs = np.arange(1, N_QUANT) / N_QUANT
    bin_labels = (list(REGIME_LABELS) if BINNING == "regimes" else
                  [f"{100*i/N_QUANT:.0f}-{100*(i+1)/N_QUANT:.0f}%" for i in range(N_QUANT)]
                  if binned else [""])
    ws_truth = {}                 # (valid time -> per-unit CERRA ws100), for REGIME_BY=cerra-ws
    edges = {}                    # per unit: the nbin-1 interior thresholds np.digitize cuts on
    if binned and REGIME_BY == "cerra-ws":
        # capacity-weighted CERRA ws100 over each unit's cells -- always defined, unlike a
        # capacity-factor threshold, which saturates above rated
        dz = xr.open_zarr(TRUTH_ZARR, consolidated=False)
        tv = list(dz.attrs["variables"])
        td = pd.to_datetime(dz["dates"].values).tz_localize("UTC")
        glat = np.asarray(dz["latitudes"]).ravel()
        glon = to_180(np.asarray(dz["longitudes"]).ravel())
        ck = np.cos(np.radians(float(glat.mean())))
        _, tc = cKDTree(np.c_[glon * ck, glat]).query(
            np.c_[to_180(turbines.longitude) * ck, turbines.latitude.to_numpy()], k=1)
        tt = turbines.assign(cell=tc.astype(int))
        tcells = np.sort(tt.cell.unique())
        tpos = {int(c): j for j, c in enumerate(tcells)}
        keep = np.where((td >= INIT_START) & (td <= INIT_END + pd.Timedelta(hours=max(LEAD_HOURS))))[0]
        wsc = dz["data"].isel(time=keep, variable=tv.index(WS_VAR),
                              ensemble=0).values[:, tcells].astype(np.float64)
        dz.close()
        Wt = np.zeros((U, tcells.size))
        for u, (uname, ucap, sel) in enumerate(units):
            sub = tt[tt.farm.isin([farms[i] for i in sel])]
            for (fm, c), mw in sub.groupby(["farm", "cell"])["capacity_mw"].sum().items():
                Wt[u, tpos[int(c)]] += mw
        Wt = Wt / Wt.sum(1, keepdims=True)
        wsu = wsc @ Wt.T                                   # (T, U) capacity-weighted truth wind
        ws_truth = {t: v for t, v in zip(td[keep], wsu)}
        # quantile edges are cut on every truth time in the window, not only the scored cases --
        # a close enough approximation and it keeps the edges independent of which run is loaded
        for u, (uname, ucap, sel) in enumerate(units):
            edges[uname] = (np.asarray(REGIME_WS_EDGES, float) if BINNING == "regimes"
                            else np.nanquantile(wsu[:, u], probs))
        print(f"Binning on CERRA ws100 at each unit's cells "
              f"({len(ws_truth)} truth times loaded)")
    elif binned:
        arr = obs.loc[(obs.index >= INIT_START) &
                      (obs.index <= INIT_END + pd.Timedelta(hours=max(LEAD_HOURS))),
                      farms].to_numpy(float)
        for u, (uname, ucap, sel) in enumerate(units):
            if BINNING == "regimes":
                e = np.array([float(sum(cset[CURVE_MODES[0]][farms[i]](np.array([v]))
                                for i in sel)[0]) / ucap
                              for v in REGIME_WS_EDGES])
                for i in range(1, len(e)):    # a flat curve would tie two thresholds
                    e[i] = max(e[i], e[i - 1] + 1e-9)
                if e[-1] >= 0.999:
                    print(f"  WARNING {uname}: already at rated by {REGIME_WS_EDGES[-1]} m/s, so "
                          f"the top bin needs CF >= {e[-1]:.4f} and will be nearly empty.")
            else:
                ok = np.isfinite(arr[:, sel]).all(1)   # a partial sum is not a known total
                e = np.nanquantile(arr[ok][:, sel].sum(1) / ucap, probs)
            edges[uname] = e
        print("Binning on observed power through the unit's own curve")
    if binned:
        print(f"Bin edges ({units[0][0]}, {BINNING}): " +
              ", ".join(f"{v:.4f}" for v in edges[units[0][0]]))

    methods = ["direct"] + [f"curve:{m}" for m in CURVE_MODES]
    serr = {(r, m): np.zeros((U, L, nbin)) for r in fmaps for m in methods}
    sbias = {k: np.zeros((U, L, nbin)) for k in serr}
    n = {k: np.zeros((U, L, nbin)) for k in serr}
    sq = METRIC == "rmse"
    has_direct = {r: False for r in fmaps}
    n_nan = {r: 0 for r in fmaps}
    recon = {}

    for label, fmap in fmaps.items():
        print(f"\nScoring {label} ...")
        for c, init in enumerate(inits):
            if c % 500 == 0:
                print(f"  {c}/{len(inits)}", flush=True)
            with xr.open_dataset(fmap[init]) as ds:
                la_, lo_ = ds["latitude"].values, ds["longitude"].values
                key = (la_.size, round(float(la_[0]), 4), round(float(la_[-1]), 4),
                       round(float(lo_[0]), 4), round(float(lo_[-1]), 4))
                if key not in recon:
                    recon[key] = build_reconstruction(la_, lo_, turbines, farms)
                    print(f"  grid {key[0]} cells -> {recon[key][0].size} farm cells")
                cell_idx, G = recon[key]
                ftimes = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
                ws = ds[WS_VAR].values[:, cell_idx]
                cf = ds[CF_VAR].values[:, cell_idx] if CF_VAR in ds else None

            has_direct[label] |= cf is not None
            t2i = {t: j for j, t in enumerate(ftimes)}
            w = G / G.sum(1, keepdims=True)
            ws_farm = ws @ w.T                                        # (T, F) capacity-weighted
            # row j of ws_win is the mean wind over the window [t_j, t_j+3h) that the
            # observation at t_j averages over; the forecast steps at OBS_STEP_H, so that
            # window is the pair (j, j+1) and the guard below holds it to that.
            ws_win = 0.5 * (ws_farm[:-1] + ws_farm[1:])
            p_curve = {m: np.column_stack([cset[m][f](w[:, i])          # (T or T-1, F) MW
                                           for i, f in enumerate(farms)])
                       for m in CURVE_MODES
                       for w in [ws_farm if CURVE_KIND[m] == "instant" else ws_win]}
            p_direct = cf @ G.T if cf is not None else None

            for lh in leads:
                vt = init + pd.Timedelta(hours=lh)
                nxt = t2i.get(vt + dt)
                # nxt must be the NEXT step: a window curve is indexed by the pair (j, j+1)
                if (vt not in t2i or nxt is None or vt not in obs.index
                        or nxt != t2i[vt] + 1):
                    continue
                ptrue = obs.loc[vt, farms].to_numpy(float)
                pred = {f"curve:{m}":
                        (0.5 * (p_curve[m][t2i[vt]] + p_curve[m][nxt])
                         if CURVE_KIND[m] == "instant" else p_curve[m][t2i[vt]])
                        for m in CURVE_MODES}
                if p_direct is not None:
                    pred["direct"] = p_direct[t2i[vt]]

                k = lpos[lh]
                nan_here = False
                for u, (uname, ucap, sel) in enumerate(units):
                    pt = ptrue[sel]
                    if not np.isfinite(pt).all():   # a partial sum is not a known total
                        continue
                    # one sample for every method: a NaN anywhere drops the case from all of them
                    if not all(np.isfinite(pp[sel]).all() for pp in pred.values()):
                        nan_here = True          # count the CASE once, not once per unit
                        continue
                    if not binned:
                        r = 0
                    elif REGIME_BY == "cerra-ws":
                        wv = ws_truth.get(vt)
                        if wv is None:
                            continue
                        r = int(np.digitize(wv[u], edges[uname]))
                    else:
                        r = int(np.digitize(pt.sum() / ucap, edges[uname]))
                    for m, pp in pred.items():
                        e = pp[sel].sum() - pt.sum()
                        serr[(label, m)][u, k, r] += e * e if sq else abs(e)
                        sbias[(label, m)][u, k, r] += e
                        n[(label, m)][u, k, r] += 1
                n_nan[label] += nan_here

    series = [(r, m) for r in fmaps for m in methods
              if not (m == "direct" and not has_direct[r]) and n[(r, m)].sum() > 0]
    if not series:
        raise SystemExit("nothing scored -- check that forecast valid times overlap power_obs")

    def _mean(acc, key, u, r):
        s = acc[key][u, :, r] if r is not None else acc[key][u].sum(1)
        c = n[key][u, :, r] if r is not None else n[key][u].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s / c

    def score(key, u, r=None):
        """The metric in MW. Squared errors ADD across bins, so the r=None aggregate is the true
        overall RMSE and not a mean of the per-bin ones."""
        v = _mean(serr, key, u, r)
        return np.sqrt(v) if sq else v

    def bias(key, u, r=None):
        return _mean(sbias, key, u, r)

    print("\nScored cases per lead (methods within a series must tie exactly):")
    for r in fmaps:
        got = [n[(r, m)] for m in methods if (r, m) in series]
        if got:
            tied = len({tuple(g.sum(2).ravel()) for g in got}) == 1
            tot = got[0].sum(2)
            print(f"  {r:22s} {int(tot.min()):5d}-{int(tot.max()):<5d} per series-lead | "
                  f"methods tied: {tied}"
                  + (f" | {n_nan[r]} case(s) dropped on NaN" if n_nan[r] else ""))

    def share(u, r_i):
        """(cases, % of that unit's record) in regime r_i -- how much of the sample this bin is."""
        k0 = series[0]
        tot = n[k0][u].sum()
        c = n[k0][u, :, r_i].sum()
        return c, (100.0 * c / tot if tot else np.nan)

    if binned:
        by = "CERRA ws100 at the cells" if REGIME_BY == "cerra-ws" else "observed power via curve"
        print(f"\nHOW THE SAMPLE SPLITS ACROSS BINS   ({BINNING}, binned on {by})")
        print(f"{'unit':22s} " + " ".join(f"{l:>16s}" for l in bin_labels))
        for u, (uname, ucap, sel) in enumerate(units):
            print(f"{uname:22s} " + " ".join(f"{int(c):7d} ({pc:4.1f}%)"
                                             for c, pc in (share(u, i) for i in range(nbin))))
        if BINNING == "quantiles":
            print("  Bins are equal-count BY CONSTRUCTION, so these shares should all read "
                  f"~{100/N_QUANT:.0f}%.")
            print("  A share far off that means the edges were cut on a different sample than "
                  "the one scored")
            print("  (ties above rated with REGIME_BY='obs-cf' are the usual cause).")
        else:
            print("  With REGIME_BY='obs-cf' a 0.0% top bin is expected wherever a farm is "
                  "already at")
            print("  rated by the top edge: all winds above rated give the same power, so no "
                  "observed")
            print("  value can land there. Switch to 'cerra-ws' to bin on the wind itself.")

    lab = {(r, m): f"{r} / {mlabel(m)}" for r, m in series}
    wid = max(len(v) for v in lab.values()) + 1
    hdr = f"{'run / method':{wid}s} " + " ".join(f"{lh:>6d}h" for lh in leads)
    MET = METRIC.upper()
    reg_range = range(nbin) if binned else [None]
    bin_unit = "m/s" if BINNING == "regimes" else ""

    for u, (uname, ucap, sel) in enumerate(units):
        print(f"\n{'='*len(hdr)}\n{uname} — {MET} as % of {ucap:.0f} MW  (season {SEASON})"
              f"\n{'='*len(hdr)}")
        for r_i in reg_range:
            if r_i is not None:
                cnt = max(n[k][u, :, r_i].max() for k in series)
                print(f"\n  bin {bin_labels[r_i]} {bin_unit} (up to {cnt:.0f} cases per lead)")
            print(hdr)
            for k in series:
                print(f"{lab[k]:{wid}s} " +
                      " ".join(f"{v:7.2f}" for v in 100.0 * score(k, u, r_i) / ucap))
            print(f"{'  bias [MW]':{wid}s}")
            for k in series:
                print(f"{lab[k]:{wid}s} " +
                      " ".join(f"{v:+7.1f}" for v in bias(k, u, r_i)))

    # ---------------- figures ----------------
    colors = {r: CB_COLORS[i % len(CB_COLORS)] for i, r in enumerate(fmaps)}
    marks = {r: CB_MARKERS[i % len(CB_MARKERS)] for i, r in enumerate(fmaps)}
    # runs x methods would be one entry per combination -- 12 lines at fontsize 7. Split it:
    # colour and marker identify the RUN, linestyle identifies the METHOD, so the legend is
    # runs + methods rather than runs * methods, and it states the encoding instead of listing it.
    runs_in = [r for r in fmaps if any(k[0] == r for k in series)]
    meth_in = [m for m in methods if any(k[1] == m for k in series)]
    handles = ([plt.Line2D([], [], color=colors[r], marker=marks[r], ls="-", ms=5, label=r)
                for r in runs_in]
               + [plt.Line2D([], [], color="0.35", ls=mstyle(m), lw=1.8, label=mlabel(m))
                  for m in meth_in])
    NLEG = len(handles)

    def put_legend(fig):
        """One legend for the whole figure, under it -- not squeezed into the first panel."""
        fig.legend(handles=handles, loc="lower center", ncol=min(NLEG, 4), fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
    base = f"{REGION}_{SEASON}" + ("_perfarm" if PER_FARM else "") + \
           (f"_{REGIME_BY}" if binned else "")

    def panel(ax, u, ucap, r_i, title):
        for k in series:
            ax.plot(leads, 100.0 * score(k, u, r_i) / ucap, mstyle(k[1]),
                    color=colors[k[0]], lw=1.5, marker=marks[k[0]], ms=4,
                    markerfacecolor=colors[k[0]] if k[1] == "direct" else "none")
        ax.set_title(title, fontsize=10)
        ax.grid(True, ls="--", alpha=0.5)
        ax.set_xticks(leads)

    def grid_fig(r_i, suffix, sup):
        ncol = int(np.ceil(np.sqrt(U)))
        nrow = int(np.ceil(U / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                                 sharex=True, squeeze=False)
        for u, (uname, ucap, sel) in enumerate(units):
            t_ = f"{uname} ({ucap:.0f} MW)"
            if r_i is not None:
                c_, pc_ = share(u, r_i)
                t_ += f"\n{int(c_)} cases — {pc_:.1f}% of its record"
            panel(axes[u // ncol][u % ncol], u, ucap, r_i, t_)
        for ax in axes.ravel()[U:]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Lead time [h]")
        for row in axes:
            row[0].set_ylabel(f"{MET} [% of capacity]")
        fig.suptitle(sup, fontsize=12)
        fig.tight_layout(rect=(0, 0.04 + 0.03 * (NLEG > 4), 1, 1))
        put_legend(fig)
        out = OUT_DIR / f"power_{METRIC}_{base}{suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")

    stamp = f"{len(inits)} inits, season {SEASON}"
    byl = "CERRA wind" if REGIME_BY == "cerra-ws" else "observed power"
    print()
    if PER_FARM and binned:
        # farms x bins does not fit one readable figure: one PNG per bin
        for r_i, rlab in enumerate(bin_labels):
            grid_fig(r_i, f"_bin{r_i}",
                     f"{REGION} per-farm power {MET} — {rlab} {bin_unit} by {byl} ({stamp})")
    elif PER_FARM:
        grid_fig(None, "", f"{REGION} per-farm power {MET} ({stamp})")
    elif binned:
        ncol = 2 if nbin <= 4 else 5
        nrow = int(np.ceil(nbin / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow),
                                 sharex=True, squeeze=False)
        axl = axes.ravel()
        for r_i in range(nbin):
            c_, pc_ = share(0, r_i)
            panel(axl[r_i], 0, units[0][1], r_i,
                  f"{bin_labels[r_i]} {bin_unit} — {int(c_)} cases ({pc_:.1f}%)")
        for ax in axl[nbin:]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Lead time [h]")
        for row in axes:
            row[0].set_ylabel(f"{MET} [% of capacity]")
        by = ("wind regime" if BINNING == "regimes"
              else f"{N_QUANT} equal-count bins, cut per unit")
        fig.suptitle(f"{units[0][0]} power {MET} by {by}, binned on {byl} ({stamp})", fontsize=12)
        fig.tight_layout(rect=(0, 0.04 + 0.03 * (NLEG > 4), 1, 1))
        put_legend(fig)
        out = OUT_DIR / f"power_{METRIC}_{base}_{BINNING}.png"
        fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved: {out}")
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        panel(ax, 0, units[0][1], None, "")
        ax.set(xlabel="Lead time [h]", ylabel=f"{MET} [% of capacity]")
        ax.set_title(f"{units[0][0]} power {MET} — {units[0][1]:.0f} MW, {stamp}", fontsize=12)
        fig.tight_layout(rect=(0, 0.04 + 0.03 * (NLEG > 4), 1, 1))
        put_legend(fig)
        out = OUT_DIR / f"power_{METRIC}_{base}.png"
        fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved: {out}")


if __name__ == "__main__":
    main()

