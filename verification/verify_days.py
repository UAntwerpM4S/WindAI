#!/usr/bin/env python3
"""Case-level verification: WHEN does the direct head win, not by how much on average.

A mean over ~2900 inits hides everything. This scores the same forecasts verify_power.py does,
but keeps every (init, lead) error instead of collapsing it, then asks three questions the mean
cannot answer:

  WIN RATE    per init, is the head better than the baseline? "Better on 78 % of days" is a
              paired statement, immune to a handful of outliers carrying an average.
  TAIL        P50/P90/P99 of per-init error. MAE is dominated by its tail; a method that improves
              the median while worsening P99 is a different product from one that improves both.
  STRATA      MAE inside truth-selected event classes: ramps, above-rated, storms.

THE RULE THIS SCRIPT OBEYS: every stratum is cut on OBSERVATIONS or CERRA truth, never on
forecast error. Selecting days by where a model did badly conditions on the outcome -- that is
what produced the phantom "12+ anomaly" before the regime bins were re-cut on CERRA wind
(see MODELS.md, regime binning). If you add a stratum, cut it on truth.

Reuses verify_power's loaders by import, so the reconstruction, the backward-window rule and the
curve construction are identical by definition rather than by copy. verify_power itself is not
modified and not run (its main() is under a __main__ guard).

Outputs PNGs only, into DistrFigures/, plus every number printed for interpretation.
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
from verify_power import to_180, parse_init, build_reconstruction

# ================================ SETTINGS ================================
# The systems to compare. Keys are labels; the first one is treated as THE CANDIDATE and every
# win-rate / scatter is drawn against the others. Update the paths to the current inference dirs.
FORECAST_DIRS = {
    "Wx25CF300_20":     Path("/mnt/weatherloss/WindPower/inference/WPDistr/Wx25CF300_20"),
    "Transformer":      Path("/mnt/weatherloss/WindPower/inference/WPDistr/CERRATransformerMAENoRated"),
    "RegularWeather_20K": Path("/mnt/weatherloss/WindPower/inference/WPDistr/RegularWeather_20K"),
}
CANDIDATE   = "Wx25CF300_20"        # the direct head being tested
CURVE_RUN   = "RegularWeather_20K"  # whose wind feeds the measured-curve baseline
DIRECT_RUNS = ["Wx25CF300_20", "Transformer"]   # runs read through their capacityfactor channel
BACKWARD_WINDOW_RUNS = ["Wx25CF300_20"]         # EXACT label match, as in verify_power

REGION     = "BE"
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))
OBS_STEP_H = 3

# the measured curve is fitted BEFORE the scored period, or it is not a baseline
TRAIN_START = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")
TRAIN_END   = pd.Timestamp("2024-01-31 21:00:00", tz="UTC")

# ---- event strata, all cut on TRUTH ----
RAMP_H    = 6      # hours over which a ramp is measured, on observed total CF
RAMP_PCT  = 0.20   # |Δ CF| over RAMP_H at or above this = a ramp
RATED_CF  = 0.90   # observed total CF at or above this = above-rated
STORM_WS  = 22.0   # capacity-weighted CERRA ws100 at or above this = storm

WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
OUT_DIR    = Path("DistrFigures")
CF_VAR, WS_VAR = "capacityfactor", "ws100"
N_CASES    = 3     # named case-study days drawn per stratum
# ==========================================================================

SYS_COLOR = {"Wx25CF300_20": "tab:blue", "Transformer": "tab:red",
             "curve": "tab:green", "RegularWeather_20K": "black"}


def load_truth(farms, turbines, caps, obs):
    """Per valid time: observed total CF, the RAMP_H change in it, and CERRA farm wind."""
    tw_times, tw = fc.farm_truth_wind(farms, turbines, TRUTH_ZARR, INIT_START,
                                      INIT_END + pd.Timedelta(hours=max(LEAD_HOURS)))
    ws_series = pd.Series((tw * caps).sum(1) / caps.sum(), index=tw_times)

    ok = np.isfinite(obs[farms]).all(1)          # a partial sum is not a known total
    cf = (obs.loc[ok, farms].sum(1) / caps.sum())
    ramp = cf - cf.shift(RAMP_H // OBS_STEP_H)   # change over RAMP_H, truth-side
    return cf, ramp, ws_series


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    specs = pd.read_csv(WPOWER_DIR / "turbine_specs.csv", index_col=0)
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")

    farms = farms_df[farms_df.region.str.upper() == REGION].farm.tolist()
    turbines = turbines[turbines.farm.isin(farms)]
    caps = farms_df.set_index("farm").loc[farms, "capacity_mw"].to_numpy(float)
    cap_total = float(caps.sum())
    fc.validate(farms, farms_df, turbines, specs)
    print(f"Region {REGION}: {len(farms)} farms, {cap_total:.0f} MW")

    if TRAIN_END >= INIT_START:
        raise SystemExit("the measured curve overlaps the scored period -- it would be in-sample")
    curve = fc.empirical(farms, farms_df, turbines, obs, TRUTH_ZARR, TRAIN_START, TRAIN_END)
    print(f"Measured curve fitted {TRAIN_START.date()}..{TRAIN_END.date()}")

    cf_true, ramp_true, ws_true = load_truth(farms, turbines, caps, obs)

    fmaps = {}
    for label, d in FORECAST_DIRS.items():
        m = {parse_init(f): f for f in sorted(d.glob("forecast_*.nc"))
             if INIT_START <= parse_init(f) <= INIT_END}
        print(f"{label}: {len(m)} files")
        fmaps[label] = m
    inits = sorted(set.intersection(*(set(m) for m in fmaps.values())))
    if not inits:
        raise SystemExit("no init times common to all runs")
    print(f"Common inits: {len(inits)}")

    dt = pd.Timedelta(hours=OBS_STEP_H)
    leads = [lh for lh in LEAD_HOURS if lh + OBS_STEP_H <= max(LEAD_HOURS)]
    print(f"Leads scored: {leads}")

    systems = DIRECT_RUNS + ["curve"]
    rows = []                       # one row per (init, lead), all systems side by side
    recon = {}

    for c, init in enumerate(inits):
        if c % 250 == 0:
            print(f"  {c}/{len(inits)}", flush=True)
        pred_at = {}                # system -> {valid time -> predicted total MW}
        for label, fmap in fmaps.items():
            with xr.open_dataset(fmap[init]) as ds:
                la_, lo_ = ds["latitude"].values, ds["longitude"].values
                key = (la_.size, round(float(la_[0]), 4), round(float(la_[-1]), 4),
                       round(float(lo_[0]), 4), round(float(lo_[-1]), 4))
                if key not in recon:
                    recon[key] = build_reconstruction(la_, lo_, turbines, farms)
                cell_idx, G = recon[key]
                ftimes = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
                ws = ds[WS_VAR].values[:, cell_idx]
                cfp = ds[CF_VAR].values[:, cell_idx] if CF_VAR in ds else None
            t2i = {t: j for j, t in enumerate(ftimes)}
            w = G / G.sum(1, keepdims=True)
            ws_farm = ws @ w.T
            ws_win = 0.5 * (ws_farm[:-1] + ws_farm[1:])      # the window the obs averages over

            if label in DIRECT_RUNS and cfp is not None:
                p = cfp @ G.T
                pred_at[label] = (p, t2i, label in BACKWARD_WINDOW_RUNS)
            if label == CURVE_RUN:
                pc = np.column_stack([curve[f](ws_win[:, i]) for i, f in enumerate(farms)])
                pred_at["curve"] = (pc, t2i, False)

        for lh in leads:
            vt = init + pd.Timedelta(hours=lh)
            if vt not in obs.index or vt not in cf_true.index:
                continue
            ptrue = obs.loc[vt, farms].to_numpy(float)
            if not np.isfinite(ptrue).all():
                continue
            row = {"init": init, "lead": lh, "vt": vt, "obs": ptrue.sum()}
            ok = True
            for s in systems:
                if s not in pred_at:
                    ok = False
                    break
                p, t2i, backward = pred_at[s]
                j = t2i.get(vt)
                nxt = t2i.get(vt + dt)
                if j is None or nxt is None or nxt != j + 1:
                    ok = False
                    break
                v = p[nxt if backward else j]
                if not np.isfinite(v).all():
                    ok = False
                    break
                row[s] = float(v.sum())
            if ok:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("nothing scored -- check the forecast dirs and power_obs overlap")
    for s in systems:
        df[f"e_{s}"] = (df[s] - df["obs"]) / cap_total * 100.0     # signed error, % of capacity

    # ---- truth-side strata, joined on valid time ----
    df["cf_obs"] = df["vt"].map(cf_true)
    df["ramp"] = df["vt"].map(ramp_true)
    df["ws"] = df["vt"].map(ws_true)
    df["stratum"] = "normal"
    df.loc[df["ramp"] >= RAMP_PCT, "stratum"] = f"ramp up >={RAMP_PCT:.0%}/{RAMP_H}h"
    df.loc[df["ramp"] <= -RAMP_PCT, "stratum"] = f"ramp down >={RAMP_PCT:.0%}/{RAMP_H}h"
    df.loc[df["cf_obs"] >= RATED_CF, "stratum"] = f"above rated (CF>={RATED_CF})"
    df.loc[df["ws"] >= STORM_WS, "stratum"] = f"storm (ws100>={STORM_WS})"

    print(f"\nScored cases: {len(df)}  ({df['init'].nunique()} inits x {len(leads)} leads)")

    # ============================ 1. OVERALL + STRATA ============================
    order = ["ALL", "normal"] + [s for s in df["stratum"].unique() if s != "normal"]
    print("\n" + "=" * 96)
    print("MAE by stratum, % of capacity. Strata are cut on OBSERVED power / CERRA wind, never "
          "on error.")
    print("=" * 96)
    print(f"{'stratum':34s} {'n':>7s} " + " ".join(f"{s:>16s}" for s in systems))
    strata_mae = {}
    for st in order:
        sub = df if st == "ALL" else df[df["stratum"] == st]
        if sub.empty:
            continue
        vals = [sub[f"e_{s}"].abs().mean() for s in systems]
        strata_mae[st] = vals
        print(f"{st:34s} {len(sub):7d} " + " ".join(f"{v:16.2f}" for v in vals))

    fig, ax = plt.subplots(figsize=(11, 5))
    sts = list(strata_mae)
    x = np.arange(len(sts))
    for i, s in enumerate(systems):
        ax.bar(x + (i - 1) * 0.26, [strata_mae[t][i] for t in sts], 0.26,
               label=s, color=SYS_COLOR.get(s, f"C{i}"))
    for k, t in enumerate(sts):
        sub = df if t == "ALL" else df[df["stratum"] == t]
        ax.annotate(f"n={len(sub)}", (k, 0), textcoords="offset points", xytext=(0, -28),
                    ha="center", fontsize=7, color="0.35")
    ax.set_xticks(x); ax.set_xticklabels(sts, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("MAE [% of capacity]"); ax.legend(fontsize=8)
    ax.set_title(f"Error by truth-selected event class — {REGION} total, {df['init'].nunique()} inits")
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT_DIR / "days_strata.png", dpi=150); plt.close(fig)

    # ============================ 2. WIN RATE PER INIT ============================
    per_init = df.groupby("init")[[f"e_{s}" for s in systems]].apply(
        lambda g: g.abs().mean())
    print("\n" + "=" * 96)
    print(f"WIN RATE: share of inits where {CANDIDATE} has the lower per-init MAE")
    print("=" * 96)
    for s in systems:
        if s == CANDIDATE:
            continue
        wr = float((per_init[f"e_{CANDIDATE}"] < per_init[f"e_{s}"]).mean())
        med = float((per_init[f"e_{s}"] - per_init[f"e_{CANDIDATE}"]).median())
        print(f"  vs {s:22s} {100*wr:5.1f} %   median gain {med:+.2f} pp")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    others = [s for s in systems if s != CANDIDATE]
    for i, s in enumerate(others):
        axs[0].plot([100 * float((df[df.lead == lh][f"e_{CANDIDATE}"].abs().values <
                                  df[df.lead == lh][f"e_{s}"].abs().values).mean())
                     for lh in leads], marker="o", label=f"vs {s}",
                    color=SYS_COLOR.get(s, f"C{i}"))
    axs[0].axhline(50, color="0.5", ls="--", lw=1)
    axs[0].set_xticks(range(len(leads))); axs[0].set_xticklabels(leads)
    axs[0].set_xlabel("lead time [h]"); axs[0].set_ylabel(f"% of cases {CANDIDATE} wins")
    axs[0].set_title("Win rate by lead"); axs[0].legend(fontsize=8); axs[0].grid(ls=":", alpha=0.5)

    s0 = others[0]
    lim = float(max(per_init[f"e_{CANDIDATE}"].max(), per_init[f"e_{s0}"].max())) * 1.05
    axs[1].scatter(per_init[f"e_{s0}"], per_init[f"e_{CANDIDATE}"], s=6, alpha=0.3,
                   color=SYS_COLOR.get(CANDIDATE, "tab:blue"))
    axs[1].plot([0, lim], [0, lim], color="0.4", lw=1)
    axs[1].set_xlim(0, lim); axs[1].set_ylim(0, lim)
    axs[1].set_xlabel(f"{s0} per-init MAE [% cap]")
    axs[1].set_ylabel(f"{CANDIDATE} per-init MAE [% cap]")
    axs[1].set_title("Below the line = the head wins that day")
    axs[1].grid(ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT_DIR / "days_winrate.png", dpi=150); plt.close(fig)

    # ============================ 3. THE TAIL ============================
    print("\n" + "=" * 96)
    print("PER-INIT MAE DISTRIBUTION [% of capacity] -- the mean is carried by the tail")
    print("=" * 96)
    qs = [50, 75, 90, 95, 99]
    print(f"{'system':22s} {'mean':>7s} " + " ".join(f"{'P'+str(q):>7s}" for q in qs) + f" {'max':>7s}")
    for s in systems:
        v = per_init[f"e_{s}"]
        print(f"{s:22s} {v.mean():7.2f} " + " ".join(f"{np.percentile(v, q):7.2f}" for q in qs)
              + f" {v.max():7.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(systems):
        v = np.sort(per_init[f"e_{s}"].values)
        ax.plot(100 * np.arange(v.size) / v.size, v, label=s, color=SYS_COLOR.get(s, f"C{i}"))
    ax.set_xlabel("percentile of inits"); ax.set_ylabel("per-init MAE [% of capacity]")
    ax.set_title("Where the average comes from"); ax.legend(fontsize=8); ax.grid(ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT_DIR / "days_tail.png", dpi=150); plt.close(fig)

    # ============================ 4. CASE STUDIES ============================
    # Chosen by TRUTH: the largest observed ramp, the longest above-rated spell, the calmest day.
    cand = df.groupby("init").agg(ramp=("ramp", lambda x: np.nanmax(np.abs(x))),
                                  cf=("cf_obs", "mean"))
    picks = {
        "largest observed ramp": cand["ramp"].idxmax(),
        "highest observed output": cand["cf"].idxmax(),
        "lowest observed output": cand["cf"].idxmin(),
    }
    print("\n" + "=" * 96)
    print("CASE STUDIES -- inits chosen on OBSERVED power alone, not on any model's error")
    print("=" * 96)
    fig, axs = plt.subplots(len(picks), 1, figsize=(10, 3.1 * len(picks)), sharex=False)
    for ax, (why, init) in zip(np.atleast_1d(axs), picks.items()):
        g = df[df["init"] == init].sort_values("lead")
        ax.plot(g["lead"], g["obs"] / cap_total * 100, color="k", lw=2, marker="o",
                label="observed")
        for i, s in enumerate(systems):
            ax.plot(g["lead"], g[s] / cap_total * 100, marker=".",
                    color=SYS_COLOR.get(s, f"C{i}"), label=s)
        maes = "  ".join(f"{s} {g[f'e_{s}'].abs().mean():.1f}" for s in systems)
        ax.set_title(f"{why} — init {init:%Y-%m-%d %H}Z   MAE: {maes}", fontsize=9)
        ax.set_ylabel("% of capacity"); ax.grid(ls=":", alpha=0.5)
        print(f"  {why:26s} init {init:%Y-%m-%d %H}Z   " + maes)
    np.atleast_1d(axs)[-1].set_xlabel("lead time [h]")
    np.atleast_1d(axs)[0].legend(fontsize=7, ncol=4)
    fig.tight_layout(); fig.savefig(OUT_DIR / "days_cases.png", dpi=150); plt.close(fig)

    print(f"\nSaved: {OUT_DIR}/days_strata.png, days_winrate.png, days_tail.png, days_cases.png")


if __name__ == "__main__":
    main()
