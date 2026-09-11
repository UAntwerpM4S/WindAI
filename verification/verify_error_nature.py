#!/usr/bin/env python3
"""What is the error that remains, and is any of it still reachable?

Three questions, and the first two are answerable from forecasts that already exist.

1. IS IT THE WIND?
   Insert the truth wind into the curve path and the error splits exactly:

       curve(model wind) - obs  =  [curve(model wind) - curve(truth wind)]  +  [curve(truth wind) - obs]
                                            WIND                                    IRREDUCIBLE

   The first term is the model's wind error, carried through a fixed conversion. The second is
   what is left when the wind is PERFECT: the curve's own misfit, plus availability, curtailment
   and every non-stationarity no forecast can see. That second number is a hard floor for any
   system built on a power curve, and it is the number that says whether there is room left.

2. IS THE HEAD DOING SOMETHING DIFFERENT FROM THE CURVE?
   Correlate their errors case by case. Highly correlated means both are just tracking the same
   wind error and the head's -0.29 pp is a small refinement of a shared signal. Weakly correlated
   means they fail on different cases -- and then the average of the two should beat both, which
   is also reported. That is a real test of whether "learned conversion" is a distinct capability
   or a better-tuned version of the same one.

3. CAN IT BE IMPROVED?
   The phase test. For each case, allow the truth wind to be taken at t-6h .. t+6h and keep the
   shift that fits best. If the error collapses under a shift, the residual is TIMING -- the model
   has the right weather at the wrong hour -- and nothing in the decoder, the head, or the loss
   can fix that, because the timing is set by the frozen trunk's dynamics. If it barely moves, the
   error is local and amplitude-like, and a readout has something to bite on.

Everything is farm-aggregated exactly as verify_power.py does it, so the numbers are on the same
scale as the 6.94 headline. Prints tables; saves one PNG.
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
REGION   = "BE"
RUN      = ("MixedRollout", Path("/mnt/weatherloss/WindPower/inference/WPDistr/MixedRollout"))

TRAIN_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")   # the curve's fit window,
TRAIN_END   = pd.Timestamp("2024-01-31 21:00:00", tz="UTC")   # matched to the head's training
INIT_START  = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END    = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEADS       = tuple(range(3, 34, 3))
SHIFTS      = (-6, -3, 0, 3, 6)      # hours, for the phase test
WS_EDGES    = (4.5, 8.0, 12.0)       # the same regime bins verify_power reports

WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
OUT_DIR    = Path("DistrFigures")

WS_VAR, CF_VAR = "ws100", "capacityfactor"
OBS_STEP_H = 3
# ======================================================================

FORECAST_RE = re.compile(r"forecast_(\d{14})")


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def parse_init(p):
    return pd.to_datetime(FORECAST_RE.search(p.name).group(1), format="%Y%m%d%H%M%S", utc=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag, fdir = RUN

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")
    farms = farms_df[farms_df.region.str.upper() == REGION].farm.tolist()
    turbines = turbines[turbines.farm.isin(farms)]
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    total_cap = float(cap.sum())

    # ---- farm cells and the reconstruction weights, as verify_power builds them ----
    ds = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    glat = np.asarray(ds["latitudes"]).ravel()
    glon = to_180(np.asarray(ds["longitudes"]).ravel())
    ds.close()
    ck = np.cos(np.radians(float(glat.mean())))
    _, tc = cKDTree(np.c_[glon * ck, glat]).query(
        np.c_[to_180(turbines.longitude) * ck, turbines.latitude.to_numpy()], k=1)
    turbines = turbines.assign(cell=tc.astype(int))
    cells = np.sort(turbines.cell.unique())
    cpos = {int(c): j for j, c in enumerate(cells)}
    G = np.zeros((len(farms), cells.size))
    for (fm, c), mw in turbines.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[farms.index(fm), cpos[int(c)]] = mw
    Wn = G / G.sum(1, keepdims=True)

    # ---- the measured curve, and the CERRA truth wind over a span wide enough to shift ----
    curve = fc.empirical(farms, farms_df, turbines, obs, TRUTH_ZARR, TRAIN_START, TRAIN_END)
    ttimes, tws = fc.farm_truth_wind(farms, turbines, TRUTH_ZARR,
                                     INIT_START - pd.Timedelta(hours=12),
                                     INIT_END + pd.Timedelta(hours=48))
    truth_ws = pd.DataFrame(tws, index=ttimes, columns=farms)

    def curve_mw(ws_row):
        """Farm winds -> total MW through each farm's own measured curve."""
        return float(sum(curve[f](np.array([ws_row[i]]))[0] for i, f in enumerate(farms)))

    # ---- read the forecasts ----
    files = {parse_init(p): p for p in sorted(fdir.glob("forecast_*.nc"))
             if INIT_START <= parse_init(p) <= INIT_END}
    if not files:
        raise SystemExit(f"no forecasts under {fdir}")
    print(f"\n{tag}: {len(files)} forecasts | region {REGION}, {len(farms)} farms, "
          f"{total_cap:.0f} MW\n")

    fcells = None
    rows = []
    for k, (init, fp) in enumerate(sorted(files.items())):
        if k % 500 == 0:
            print(f"  {k}/{len(files)}", flush=True)
        with xr.open_dataset(fp) as fx:
            if CF_VAR not in fx:
                raise SystemExit(f"{fp.name} has no {CF_VAR}")
            if fcells is None:
                fl = np.asarray(fx["latitude"].values)
                fo = to_180(np.asarray(fx["longitude"].values))
                fk = np.cos(np.radians(float(fl.mean())))
                _, fcells = cKDTree(np.c_[fo * fk, fl]).query(
                    np.c_[glon[cells] * fk, glat[cells]], k=1)
            ft = pd.DatetimeIndex(fx["time"].values).tz_localize("UTC")
            pos = {t: q for q, t in enumerate(ft)}
            cf_all = fx[CF_VAR].values[:, fcells]
            ws_all = fx[WS_VAR].values[:, fcells]

        for lead in LEADS:
            vt = init + pd.Timedelta(hours=lead)
            j, j2 = pos.get(vt), pos.get(vt + pd.Timedelta(hours=OBS_STEP_H))
            if j is None or j2 is None or vt not in truth_ws.index:
                continue
            o = obs.reindex([vt])[farms].to_numpy(float).ravel()
            if not np.isfinite(o).all():
                continue
            # the curve methods use the WINDOW-MEAN wind, matching verify_power.py
            wsm = 0.5 * (ws_all[j] @ Wn.T + ws_all[j2] @ Wn.T)
            wst = truth_ws.loc[vt].to_numpy(float)
            if not np.isfinite(wst).all():
                continue
            rec = {"init": init, "lead": lead,
                   "obs": float(o.sum()),
                   "direct": float(cf_all[j] @ G.T @ np.ones(len(farms))),   # not "head": DataFrame.head is a method
                   "curve_model": curve_mw(wsm),
                   "curve_truth": curve_mw(wst),
                   "ws_truth": float(wst @ (cap.to_numpy() / total_cap))}
            for s in SHIFTS:
                st = vt + pd.Timedelta(hours=s)
                rec[f"shift{s}"] = (curve_mw(truth_ws.loc[st].to_numpy(float))
                                    if st in truth_ws.index else np.nan)
            rows.append(rec)

    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("nothing scored")
    pc = lambda x: 100.0 * np.mean(np.abs(x)) / total_cap

    # ---- 1. is it the wind? ----
    e_total = d.curve_model - d.obs
    e_wind = d.curve_model - d.curve_truth
    e_irred = d.curve_truth - d.obs
    print(f"\n{'='*78}\n1. IS IT THE WIND?   n = {len(d):,} (init, lead) cases\n{'='*78}")
    print(f"  curve on MODEL wind   vs obs      {pc(e_total):6.2f} %cap   the baseline you score")
    print(f"    = wind error        carried     {pc(e_wind):6.2f} %cap   model wind vs CERRA truth")
    print(f"    + irreducible       remainder   {pc(e_irred):6.2f} %cap   PERFECT wind, still wrong")
    e_direct = d["direct"] - d.obs
    print(f"\n  the head                          {pc(e_direct):6.2f} %cap")
    print(f"\n  The irreducible term is the floor for anything built on a power curve: the curve's")
    print(f"  own misfit plus availability, curtailment and non-stationarity. A direct head is not")
    print(f"  bound by the curve's misfit, but it IS bound by the availability part.")

    # ---- 2. is the head doing something different? ----
    eh, ec = e_direct.to_numpy(), (d.curve_model - d.obs).to_numpy()
    r = float(np.corrcoef(eh, ec)[0, 1])
    print(f"\n{'='*78}\n2. IS THE HEAD DIFFERENT FROM THE CURVE?\n{'='*78}")
    print(f"  corr(head error, curve error)     {r:+6.3f}")
    print(f"  head MAE                          {pc(eh):6.2f} %cap")
    print(f"  curve MAE                         {pc(ec):6.2f} %cap")
    print(f"  50/50 blend of the two            {pc(0.5*(eh+ec)):6.2f} %cap")
    print(f"\n  A blend beating both means they fail on DIFFERENT cases and the head is a distinct")
    print(f"  capability. A blend landing between them means they are tracking the same wind error")
    print(f"  and the head is a better-tuned version of the same thing.")

    # ---- 3. can it be improved? the phase test ----
    sh = np.stack([np.abs(d[f"shift{s}"] - d.obs) for s in SHIFTS])
    best = np.nanmin(sh, 0)
    which = np.array(SHIFTS)[np.nanargmin(sh, 0)]
    print(f"\n{'='*78}\n3. IS THE RESIDUAL TIMING?\n{'='*78}")
    print(f"  curve on truth wind at the right hour       {pc(e_irred):6.2f} %cap")
    print(f"  ... allowing the BEST shift in {SHIFTS} h    "
          f"{100*np.nanmean(best)/total_cap:6.2f} %cap")
    print(f"  share of cases where 0h is NOT the best shift: "
          f"{100*np.mean(which != 0):.1f}%")
    print(f"\n  This bounds how much of even the IRREDUCIBLE term is really timing. If the shifted")
    print(f"  number collapses, then much of what looks like availability is the observation and")
    print(f"  the wind being out of phase, and no local correction reaches it.")

    # ---- by regime ----
    b = np.digitize(d.ws_truth, WS_EDGES)
    names = ["0-4.5", "4.5-8", "8-12", "12+"]
    print(f"\n{'='*78}\nBY WIND REGIME (binned on CERRA truth wind)\n{'='*78}")
    print(f"{'bin':>8}{'share':>8}{'head':>8}{'curve':>8}{'wind':>8}{'irreducible':>13}")
    for i, nm in enumerate(names):
        m = b == i
        if not m.any():
            continue
        print(f"{nm:>8}{100*m.mean():7.1f}%{pc(eh[m]):8.2f}{pc(ec[m]):8.2f}"
              f"{pc(e_wind[m]):8.2f}{pc(e_irred[m]):13.2f}")

    # ---- figure ----
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
    L = sorted(d.lead.unique())
    for lbl, s in (("head", eh), ("curve on model wind", ec),
                   ("wind error alone", e_wind.to_numpy()),
                   ("irreducible (perfect wind)", e_irred.to_numpy())):
        axs[0].plot(L, [pc(s[d.lead == l]) for l in L], "o-", ms=4, lw=1.4, label=lbl)
    axs[0].set_xlabel("lead [h]"); axs[0].set_ylabel("MAE [% of capacity]")
    axs[0].set_title("Where the error comes from, by lead", fontsize=10)
    axs[0].legend(fontsize=7)

    axs[1].scatter(ec, eh, s=3, alpha=0.15, color="C0")
    lim = np.nanpercentile(np.abs(np.r_[eh, ec]), 99)
    axs[1].plot([-lim, lim], [-lim, lim], color="0.5", lw=1, ls="--")
    axs[1].set_xlim(-lim, lim); axs[1].set_ylim(-lim, lim)
    axs[1].set_xlabel("curve error [MW]"); axs[1].set_ylabel("head error [MW]")
    axs[1].set_title(f"Same mistakes?   corr {r:+.3f}", fontsize=10)

    x = np.arange(len(names))
    for i, nm in enumerate(names):
        m = b == i
        if not m.any():
            continue
        axs[2].bar(i - 0.2, pc(e_wind[m]), 0.4, color="C0",
                   label="wind" if i == 0 else None)
        axs[2].bar(i + 0.2, pc(e_irred[m]), 0.4, color="C3",
                   label="irreducible" if i == 0 else None)
    axs[2].set_xticks(x); axs[2].set_xticklabels(names)
    axs[2].set_xlabel("CERRA truth wind [m/s]"); axs[2].set_ylabel("MAE [% of capacity]")
    axs[2].set_title("How much of each bin is even reachable", fontsize=10)
    axs[2].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{tag} — the nature of the remaining error, {INIT_START.date()}"
                 f"..{INIT_END.date()}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / f"error_nature_{REGION}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
