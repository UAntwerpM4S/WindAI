#!/usr/bin/env python3
"""Has the power head learned a power curve, or does it see something the curve cannot?

The head now scores level with a method-of-bins curve fitted on its OWN forecast wind (gap
~0.0 pp of capacity). Equal error MAGNITUDE admits two readings and cannot separate them:

  IT IS A CURVE   the head has effectively learned a 1-D function of its own wind. Then its
                  errors are the curve's errors, blending the two buys nothing, and the 512-dim
                  latent contributed nothing beyond the wind direction of it.
  IT IS NOT       the head reads stability, shear, direction and upstream state out of the latent
                  and is limited elsewhere, landing at the same MAE by coincidence. Then its
                  errors are partly INDEPENDENT of the curve's, and a blend beats both.

Error CORRELATION separates them, so that is what this measures. For every (init, lead, farm-sum)
it forms three predictions of the same quantity from the same forecast, and correlates their
errors against the observation:

  head       the model's own `capacityfactor`, mapped to farms (the adjoint of the distribution)
  measured   this run's forecast wind through the farm's measured curve
  specs      this run's forecast wind through the datasheet curve

The (measured, specs) pair is the CONTROL and the reason this test can be read at all. Both are
1-D functions of one wind field, differing only in the curve shape, so their error correlation is
what "two methods of the same kind" looks like in this data -- near 1. Judge the head against that
number, not against 1.0: if corr(head, measured) sits at the control, the head is a curve; if it
sits well below, the head carries information the wind alone does not.

BLEND is the same question asked in units that matter. The blend error is exactly
alpha*e_head + (1-alpha)*e_curve, so the variance-minimising alpha and its gain follow in closed
form. A blend that beats both components is a forecast improvement you can have for free, and it
is only possible when the errors are less than perfectly correlated.

BINNING is on this run's OWN forecast wind, not CERRA truth. The question here is conditional on
the input the curve was given -- "handed the same wind, do the two make the same mistakes" -- so
conditioning on that shared input is the right cut, and it keeps the truth zarr out of the loop.
That makes these bins NOT comparable to verify_power.py's truth-binned tables; they answer a
different question.

Only the summed regional total is scored: the blend is a forecasting claim about the operational
quantity, and per-farm errors are dominated by noise that cancels in the sum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import farm_curves as fc
from verify_power import build_reconstruction, parse_init

# ============================== SETTINGS ==============================
REGION = "BE"
EXCLUDE_FARMS = []      # keep identical to verify_power.py -- see the note there
RUN    = ("Huber_Head1", Path("/mnt/weatherloss/WindPower/inference/WPDistr/HuberCFHead1"))

TRAIN_START = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")   # where the measured curve is fitted
TRAIN_END   = pd.Timestamp("2024-07-31 21:00:00", tz="UTC")   # must end before INIT_START
INIT_START  = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END    = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS  = list(range(3, 37, 3))
OBS_STEP_H  = 3

WS_EDGES   = [4.5, 8.0, 12.0]                      # m/s, on THIS RUN's forecast farm wind
BIN_LABELS = ["0-4.5", "4.5-8", "8-12", "12+"]

WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
OUT_DIR    = Path("DistrFigures")
CF_VAR, WS_VAR = "capacityfactor", "ws100"
# ======================================================================

OKABE = {"head": "#0072B2", "measured": "#D55E00", "specs": "#009E73", "blend": "#CC79A7"}


def blend(a, b):
    """Variance-minimising mix of two error series: alpha, its RMSE, and the gain over the best.

    The blended PREDICTION is alpha*p_a + (1-alpha)*p_b, so the blended ERROR is exactly
    alpha*e_a + (1-alpha)*e_b -- linear, hence the closed form. alpha is not clipped to [0,1]:
    a value outside it says one series is best used as a correction to the other, which is itself
    worth seeing.
    """
    A, B, C = np.mean(a * a), np.mean(b * b), np.mean(a * b)
    alpha = (B - C) / (A + B - 2 * C)
    e = alpha * a + (1 - alpha) * b
    rmse = np.sqrt(np.mean(e * e))
    return alpha, rmse, np.sqrt(min(A, B)) - rmse, np.mean(np.abs(e))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label, fdir = RUN

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    specs = pd.read_csv(WPOWER_DIR / "turbine_specs.csv", index_col=0)
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")

    farms = farms_df[farms_df.region.str.upper() == REGION].farm.tolist()
    if EXCLUDE_FARMS:
        unknown = set(EXCLUDE_FARMS) - set(farms)
        if unknown:
            raise SystemExit(f"EXCLUDE_FARMS names farms not in REGION={REGION!r}: "
                             f"{sorted(unknown)}")
        farms = [f for f in farms if f not in EXCLUDE_FARMS]
        print(f"EXCLUDED: {', '.join(EXCLUDE_FARMS)} -- {len(farms)} farms remain")
    turbines = turbines[turbines.farm.isin(farms)]
    cap = float(farms_df.set_index("farm").loc[farms, "capacity_mw"].sum())

    fc.validate(farms, farms_df, turbines, specs)
    if TRAIN_END >= INIT_START:
        raise SystemExit("curve window overlaps the scored period -- the baseline would be in-sample")
    curves = {"measured": fc.empirical(farms, farms_df, turbines, obs, TRUTH_ZARR,
                                       TRAIN_START, TRAIN_END),
              "specs": fc.build_specs(farms, farms_df, specs)}

    dt = pd.Timedelta(hours=OBS_STEP_H)
    leads = [lh for lh in LEAD_HOURS if lh + OBS_STEP_H <= max(LEAD_HOURS)]
    fmap = {parse_init(f): f for f in sorted(fdir.glob("forecast_*.nc"))
            if INIT_START <= parse_init(f) <= INIT_END}
    inits = sorted(fmap)
    print(f"{label}: {len(inits)} inits, {len(leads)} leads, {len(farms)} farms, {cap:.0f} MW")
    print(f"measured curve fitted {TRAIN_START.date()}..{TRAIN_END.date()}\n")

    rows = []          # (lead, bin, e_head, e_measured, e_specs) -- all MW
    recon = {}
    for c, init in enumerate(inits):
        if c % 200 == 0:
            print(f"  {c}/{len(inits)}", flush=True)
        with xr.open_dataset(fmap[init]) as ds:
            la_, lo_ = ds["latitude"].values, ds["longitude"].values
            key = (la_.size, round(float(la_[0]), 4), round(float(la_[-1]), 4))
            if key not in recon:
                recon[key] = build_reconstruction(la_, lo_, turbines, farms)
            cell_idx, G = recon[key]
            ftimes = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
            ws = ds[WS_VAR].values[:, cell_idx]
            cf = ds[CF_VAR].values[:, cell_idx]

        t2i = {t: j for j, t in enumerate(ftimes)}
        w = G / G.sum(1, keepdims=True)
        ws_farm = ws @ w.T
        ws_win = 0.5 * (ws_farm[:-1] + ws_farm[1:])       # the window the observation averages
        # measured is fitted mean-to-mean, so read it AT the mean wind; specs is instantaneous,
        # so average the POWERS over the adjacent pair -- the curve is cubic on the ramp
        p_meas = np.column_stack([curves["measured"][f](ws_win[:, i]) for i, f in enumerate(farms)])
        p_spec = np.column_stack([curves["specs"][f](ws_farm[:, i]) for i, f in enumerate(farms)])
        p_head = cf @ G.T

        for lh in leads:
            vt = init + pd.Timedelta(hours=lh)
            j = t2i.get(vt)
            nxt = t2i.get(vt + dt)
            if j is None or nxt is None or nxt != j + 1 or vt not in obs.index:
                continue
            truth = obs.loc[vt, farms].to_numpy(float)
            if not np.isfinite(truth).all():          # a partial sum is not a known total
                continue
            o = truth.sum()
            b = int(np.digitize(float(ws_win[j] @ (G.sum(1) / G.sum())), WS_EDGES))
            rows.append((lh, b,
                         p_head[j].sum() - o,
                         p_meas[j].sum() - o,
                         0.5 * (p_spec[j].sum() + p_spec[nxt].sum()) - o))

    R = np.array(rows, float)
    lead, bin_, eh, em, es = R[:, 0], R[:, 1].astype(int), R[:, 2], R[:, 3], R[:, 4]
    pc = 100.0 / cap                                   # MW -> % of capacity

    def line(name, a, b):
        r = np.corrcoef(a, b)[0, 1]
        al, rmse, gain, mae = blend(a, b)
        return (f"  {name:22s} r={r:6.3f}   unexplained var {100*(1-r*r):5.1f}%   "
                f"alpha={al:5.2f}  blend RMSE {rmse*pc:5.2f}  gain {gain*pc:+5.2f} pp")

    print(f"\n{'='*100}\nERROR CORRELATION, all leads pooled ({len(R)} cases)\n{'='*100}")
    print(f"  {'series':22s} {'MAE':>6s} {'RMSE':>7s}   [% of {cap:.0f} MW]")
    for nm, e in (("head", eh), ("measured curve", em), ("specs curve", es)):
        print(f"  {nm:22s} {np.mean(np.abs(e))*pc:6.2f} {np.sqrt(np.mean(e*e))*pc:7.2f}")
    print()
    print(line("head vs measured", eh, em))
    print(line("head vs specs", eh, es))
    print(line("measured vs specs", em, es) + "   <-- CONTROL: two 1-D curves, same wind")

    print(f"\n{'='*100}\nBY LEAD\n{'='*100}")
    print(f"  {'lead':>5s} {'r(head,meas)':>13s} {'r(meas,specs)':>14s} "
          f"{'head MAE':>9s} {'meas MAE':>9s} {'blend MAE':>10s} {'blend gain':>11s}")
    rl, rc = [], []
    for lh in leads:
        s = lead == lh
        r = np.corrcoef(eh[s], em[s])[0, 1]
        rctl = np.corrcoef(em[s], es[s])[0, 1]
        al, _, _, bmae = blend(eh[s], em[s])
        best = min(np.mean(np.abs(eh[s])), np.mean(np.abs(em[s])))
        rl.append(r); rc.append(rctl)
        print(f"  {lh:4d}h {r:13.3f} {rctl:14.3f} {np.mean(np.abs(eh[s]))*pc:9.2f} "
              f"{np.mean(np.abs(em[s]))*pc:9.2f} {bmae*pc:10.2f} {(best-bmae)*pc:+11.2f}")

    print(f"\n{'='*100}\nBY FORECAST WIND BIN (this run's own ws100, NOT truth)\n{'='*100}")
    print(f"  {'bin':>8s} {'n':>6s} {'r(head,meas)':>13s} {'r(meas,specs)':>14s} "
          f"{'head MAE':>9s} {'meas MAE':>9s} {'blend gain':>11s}")
    for b, bl in enumerate(BIN_LABELS):
        s = bin_ == b
        if s.sum() < 30:
            continue
        r = np.corrcoef(eh[s], em[s])[0, 1]
        rctl = np.corrcoef(em[s], es[s])[0, 1]
        _, _, _, bmae = blend(eh[s], em[s])
        best = min(np.mean(np.abs(eh[s])), np.mean(np.abs(em[s])))
        print(f"  {bl:>8s} {s.sum():6d} {r:13.3f} {rctl:14.3f} "
              f"{np.mean(np.abs(eh[s]))*pc:9.2f} {np.mean(np.abs(em[s]))*pc:9.2f} "
              f"{(best-bmae)*pc:+11.2f}")

    r_all = np.corrcoef(eh, em)[0, 1]
    r_ctl = np.corrcoef(em, es)[0, 1]
    al, _, gain, _ = blend(eh, em)
    print(f"\n{'='*100}\nREAD IT LIKE THIS\n{'='*100}")
    print(f"  control (measured vs specs)     r = {r_ctl:.3f}  <- two 1-D curves on one wind")
    print(f"  head vs measured                r = {r_all:.3f}")
    _, _, _, bmae = blend(eh, em)
    bestmae = min(np.mean(np.abs(eh)), np.mean(np.abs(em)))
    print(f"  blend gain over the better one    {gain*pc:+.2f} pp RMSE, {(bestmae-bmae)*pc:+.2f} pp MAE, at alpha={al:.2f}")
    print("\n  r at or above the control, gain ~0  -> the head IS a power curve on its own wind;")
    print("     the latent bought nothing beyond the wind. Report it as a negative result.")
    print("  r well below the control, gain > 0  -> the head reads the latent for something the")
    print("     wind alone does not carry, and the blend is a better forecast for free.")
    print("\n  scale, from synthetic pairs at this sample size and error magnitude: a blend pays")
    print("  nothing until the errors decorrelate appreciably -- r=0.99 and r=0.90 both give")
    print("  ~0.00 pp, r=0.70 gives ~0.5 pp, r=0.00 gives ~2.1 pp. Read r<~0.85 as the threshold")
    print("  where 'the head is doing something else' stops being a claim and starts being MW.")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].hexbin(em * pc, eh * pc, gridsize=45, cmap="Blues", mincnt=1)
    lim = np.percentile(np.abs(np.r_[eh, em]) * pc, 99.5)
    ax[0].plot([-lim, lim], [-lim, lim], color="0.4", lw=1, ls="--")
    ax[0].set(xlabel="measured-curve error [% of capacity]",
              ylabel="head error [% of capacity]", xlim=(-lim, lim), ylim=(-lim, lim),
              title=f"paired errors, r = {r_all:.3f} (control {r_ctl:.3f})")
    ax[1].plot(leads, rl, "o-", color=OKABE["head"], label="head vs measured")
    ax[1].plot(leads, rc, "s--", color=OKABE["specs"], label="measured vs specs (control)")
    ax[1].set(xlabel="lead time [h]", ylabel="error correlation", ylim=(0, 1.02),
              title="is the head a curve?")
    ax[1].legend(frameon=False, fontsize=9)
    for nm, e, k in (("head", eh, "head"), ("measured curve", em, "measured")):
        ax[2].plot(leads, [np.mean(np.abs(e[lead == lh])) * pc for lh in leads],
                   "o-", color=OKABE[k], label=nm)
    ax[2].plot(leads, [blend(eh[lead == lh], em[lead == lh])[3] * pc for lh in leads],
               "^-", color=OKABE["blend"], label="optimal blend")
    ax[2].set(xlabel="lead time [h]", ylabel="MAE [% of capacity]", title="what a blend buys")
    ax[2].legend(frameon=False, fontsize=9)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{label} -- power head vs a curve on its own wind, {REGION}", y=1.01)
    fig.tight_layout()
    out = OUT_DIR / f"head_independence_{label}_{REGION}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
