#!/usr/bin/env python3
"""Availability audit: how often does a farm underproduce when the wind says it shouldn't?

Above its rated wind a farm should sit at its plateau. Every hour it does not is an outage, a
curtailment, a derate or a turbine out -- something that is NOT in the weather and that no
forecast can see. This counts those hours, per farm, and asks whether the test year differs from
the period the baseline power curve was measured on.

That question is load-bearing. verify_power.py scores the direct head against a measured curve
fitted on TRAIN_START..TRAIN_END and applied to TEST_START..TEST_END. If a farm behaved
differently in the test year, the baseline carries a handicap that has nothing to do with either
model, and any margin over it has to be read with that in mind.

The rated-wind threshold is derived PER FARM rather than hardcoded: it is the first wind bin at
which that farm's own measured curve reaches RATED_FRAC of its plateau. A single constant would
mean different things for a 90 m rotor and a 164 m one, and would put some farms still on their
ramp -- where a shortfall is ordinary physics, not a fault.

Everything is read against CERRA truth wind, never a forecast: this audits the FARM, not the
model. power_obs at t is the mean over [t, t+3h), so the wind is the window mean over the same
interval (farm_curves.farm_truth_wind does this).

Prints, per farm and period: hours above rated, median output as a fraction of the historical
plateau, the share of hours below each SHORTFALL level, and the mean energy shortfall. Then a
monthly series so a step change is visible as a step rather than a shifted average.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import farm_curves as fc

# ============================== SETTINGS ==============================
REGION      = "BE"

TRAIN_START = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")   # where the baseline curve is fitted
TRAIN_END   = pd.Timestamp("2024-07-31 21:00:00", tz="UTC")
TEST_START  = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")   # where verify_power.py scores
TEST_END    = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")

RATED_FRAC  = 0.98        # "at rated" = the farm's measured curve reaches this share of plateau
WS_FLOOR    = 11.0        # never call anything below this rated, whatever the fitted curve says
SHORTFALL   = (0.90, 0.50, 0.10)   # report the share of above-rated hours below these fractions
MIN_MONTH   = 8           # months with fewer above-rated hours are not plotted

WPOWER_DIR  = Path("/mnt/weatherloss/WindPower/data/WPDistr")
TRUTH_ZARR  = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
OUT_DIR     = Path("DistrFigures")
# ======================================================================


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")

    farms = (farms_df.farm.tolist() if REGION == "all"
             else farms_df[farms_df.region.str.upper() == REGION].farm.tolist())
    turbines = turbines[turbines.farm.isin(farms)]
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]

    # the historical curve -- the same object verify_power.py scores against
    curve = fc.empirical(farms, farms_df, turbines, obs, TRUTH_ZARR, TRAIN_START, TRAIN_END)

    # per-farm plateau and rated wind, read off that curve
    probe = np.arange(0.25, 25.0, 0.25)
    plateau, ws_rated = {}, {}
    for f in farms:
        y = curve[f](probe)
        plateau[f] = float(y.max())
        hit = probe[y >= RATED_FRAC * plateau[f]]
        ws_rated[f] = max(float(hit[0]), WS_FLOOR)

    # CERRA window-mean wind at the farms, covering both periods in one read
    times, ws_win = fc.farm_truth_wind(farms, turbines, TRUTH_ZARR, TRAIN_START, TEST_END)
    periods = {"train": (times >= TRAIN_START) & (times <= TRAIN_END),
               "test":  (times >= TEST_START) & (times <= TEST_END)}

    print(f"\nRegion {REGION}: {len(farms)} farms, {float(cap.sum()):.0f} MW")
    print(f"baseline curve fitted {TRAIN_START.date()}..{TRAIN_END.date()}, "
          f"scored {TEST_START.date()}..{TEST_END.date()}\n")

    hdr = (f"{'farm':14s} {'plateau':>8s} {'rated':>6s} {'period':>6s} {'hours':>6s} "
           f"{'median':>7s}" + "".join(f"{'<'+str(int(100*s))+'%':>7s}" for s in SHORTFALL)
           + f" {'lost MW':>8s}")
    print("=" * len(hdr)); print(hdr); print("=" * len(hdr))

    rows, monthly = [], {}
    for f in farms:
        o_all = obs[f].reindex(times).to_numpy(float)
        for pname, pmask in periods.items():
            m = pmask & np.isfinite(o_all) & (ws_win[:, farms.index(f)] >= ws_rated[f])
            if m.sum() == 0:
                continue
            r = o_all[m] / plateau[f]                       # output as a share of the plateau
            shares = [float((r < s).mean()) for s in SHORTFALL]
            lost = float(np.mean(np.clip(plateau[f] - o_all[m], 0, None)))
            print(f"{f:14s} {plateau[f]:7.1f}MW {ws_rated[f]:5.1f} {pname:>6s} {int(m.sum()):6d} "
                  f"{np.median(r):7.3f}" + "".join(f"{100*s:6.1f}%" for s in shares)
                  + f" {lost:7.1f}")
            rows.append((f, pname, np.median(r), shares, lost))
        # monthly median of the same ratio, whole span, for the step-change plot
        mm = np.isfinite(o_all) & (ws_win[:, farms.index(f)] >= ws_rated[f])
        s = pd.Series(o_all[mm] / plateau[f], index=times[mm].tz_localize(None))
        g = s.groupby(s.index.to_period("M"))
        monthly[f] = g.median().where(g.size() >= MIN_MONTH).dropna()
        print()

    print("  plateau  : top of the farm's measured curve over the TRAIN window, in MW")
    print("  rated    : wind at which that curve first reaches "
          f"{100*RATED_FRAC:.0f}% of plateau (floor {WS_FLOOR:.0f} m/s)")
    print("  median   : median output above rated, as a fraction of that plateau. 1.00 means the")
    print("             farm behaved exactly as its historical curve says it should.")
    print("  <90/50/10: share of above-rated hours below that fraction of plateau. <10% is")
    print("             effectively a full stop -- outage, or curtailed to nothing.")
    print("  lost MW  : mean shortfall below plateau over above-rated hours.")

    # ---- what changed between the two periods ----
    print(f"\n{'='*72}\nCHANGE FROM THE CURVE'S OWN WINDOW TO THE SCORED YEAR\n{'='*72}")
    print(f"{'farm':14s} {'median train':>13s} {'median test':>12s} {'change':>8s} "
          f"{'<50% train':>11s} {'<50% test':>10s}")
    d = {(f, p): (med, sh) for f, p, med, sh, _ in rows}
    delta = {}
    for f in farms:
        if (f, "train") not in d or (f, "test") not in d:
            continue
        mtr, mte = d[(f, "train")][0], d[(f, "test")][0]
        delta[f] = mte - mtr
        print(f"{f:14s} {mtr:13.3f} {mte:12.3f} {mte-mtr:+8.3f} "
              f"{100*d[(f,'train')][1][1]:10.1f}% {100*d[(f,'test')][1][1]:9.1f}%")
    worst = sorted(delta, key=delta.get)[:3]
    print(f"\n  largest drops: " + ", ".join(f"{f} ({delta[f]:+.3f})" for f in worst))
    print("  A negative change means the farm produced LESS above rated during the scored year")
    print("  than during the window its baseline curve was fitted on. The baseline then over-")
    print("  predicts it there, through no fault of any forecast -- and a model trained on the")
    print("  earlier period inherits the same error unless it can see the shortfall coming.")

    # ---- figure ----
    ncol = int(np.ceil(np.sqrt(len(farms)))); nrow = int(np.ceil(len(farms) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 2.8 * nrow), squeeze=False,
                            sharex=True, sharey=True)
    for i, f in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        s = monthly[f]
        if len(s):
            ax.plot(s.index.to_timestamp(), s.values, "o-", ms=3, lw=1.2, color="C0")
        ax.axhline(1.0, color="0.4", lw=1, ls="--")
        ax.axvline(TEST_START.tz_localize(None), color="C3", lw=1.5)
        ax.set_title(f"{f}  ({plateau[f]:.0f} MW plateau, rated {ws_rated[f]:.1f} m/s)", fontsize=9)
        ax.grid(alpha=0.3); ax.set_ylim(0, 1.15)
        if i % ncol == 0: ax.set_ylabel("output / plateau", fontsize=8)
    for ax in axs.ravel()[len(farms):]:
        ax.axis("off")
    fig.suptitle("Monthly median output above rated wind, as a fraction of the historical "
                 "plateau\n(dashed = as expected, red line = start of the scored year)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / f"availability_{REGION}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
