#!/usr/bin/env python3
"""What conversion did the model actually learn? The implied power curve, per farm.

The head predicts power. It could have learned either of two very different things:

  A CONVERSION -- a mapping from wind to power that knows this farm's cut-in, its ramp, its
      rated plateau AND its real losses. Then plotting the model's own power against the
      model's own wind should trace a power-curve shape, and it should sit near the OBSERVED
      curve rather than the idealised spec-sheet one.

  A CLIMATOLOGY -- each farm's typical output level, conditioned loosely on the weather. That
      also removes a mean bias, and would pass every test in verify_power.py, but the implied
      curve would be flat-ish and carry little of the model's own wind signal.

Four curves per farm, on the same axes:

  SPEC      the manufacturer curve from turbine_specs.csv -- an ideal turbine in clean air.
  OBSERVED  binned median of observed power against CERRA wind OVER THE SCORED PERIOD -- what
            the farm really did while the forecasts were being scored. This is the target.
  MEASURED  the BASELINE the head is compared against in verify_power.py: the same method of
            bins, but fitted on TRAIN_START..TRAIN_END, a window that ends before scoring
            begins. Built by calling farm_curves.empirical with the same arguments verify_power
            uses, so it is the identical object, not a re-implementation.
  IMPLIED   binned median of the model's power against the model's own wind, at LEAD.

If IMPLIED tracks OBSERVED, the head learned the conversion including the losses. If it tracks
SPEC, it learned the textbook curve and not the losses. If it is flat, it learned a level.

MEASURED vs OBSERVED is the baseline's own handicap, and it is worth reading before crediting
the head with anything. The two are the same farm measured in two different windows, so any gap
between them is NON-STATIONARITY -- availability, curtailment, derates, turbines swapped -- not
model error. The head beats this curve in verify_power.py; the `MEAS->OBS` block below says how
much of that win is simply the curve being stale. Note the head is if anything staler: its
training data ends 2024-01-31, the curve is fitted through 2024-07-31.

Four numbers decide it rather than eyeballing:
  * to OBSERVED -- occupancy-weighted RMS distance from IMPLIED to the real farm's curve, in
    % of capacity. THE headline number: it says how far the learned conversion is from the
    farm's actual one, and it needs no reference to the spec sheet.
  * to MEASURED -- the same distance to the baseline curve. Small means the head reproduced the
    baseline; the interesting case is `to OBSERVED` small while `to MEASURED` is not, which
    says the head tracked the farm's CURRENT behaviour rather than its historical curve.
  * noise floor -- bootstrap SE of the observed binned median, weighted the same way. The
    distance two independent estimates of the SAME farm's curve would sit apart. A model
    cannot get closer to the observed curve than the observed curve is to itself, so
    "to OBSERVED / floor" says how much room is actually left.
  * wind-explained -- how much of the model's power variance its own wind accounts for. A
    conversion is a function of wind, so this must be high; a climatology need not be.

The spec-sheet ratio is kept as CONTEXT, not as a verdict. d_obs turns out to be roughly
constant across farms while d_spec varies by a factor of seven, so the ratio mostly describes
how wrong the spec sheet is for that farm rather than how good the model is.

CAVEAT: power_obs at t is the mean over [t, t+3h), so OBSERVED is binned on the window-mean
wind. The model's power is trained on that same window mean while its ws100 is instantaneous,
so IMPLIED inherits a half-step offset. It is the mapping the model actually internalised,
which is the thing being asked about, but do not read the cut-in position to the metre.
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

import farm_curves as fc          # same module verify_power.py builds its baselines with

# ============================== SETTINGS ==============================
REGION   = "BE"
EXCLUDE_FARMS = []             # keep identical to verify_power.py -- see the note there
LEAD     = 33                  # lead hour the implied curve is read at
WS_EDGES = np.arange(0.0, 25.1, 0.5)     # wind bins for all three curves
MIN_BIN  = 30                 # a bin with fewer cases is not plotted
RAMP     = (3.0, 13.0)        # m/s window the scores are computed over
N_BOOT   = 200                # resamples for the observed curve's noise floor


FORECAST_DIRS = {
    "MixedRollout": Path("/mnt/weatherloss/WindPower/inference/WPDistr/MixedRollout"),
    "VH":           Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
    # earlier runs, kept for reference:
    # "HuberCFHead5Mixed": Path("/mnt/weatherloss/WindPower/inference/WPDistr/HuberCFHead5Mixed"),
    # "Huber_Head1":       Path("/mnt/weatherloss/WindPower/inference/WPDistr/HuberCFHead1"),
    # "SH_Finetune":       Path("/mnt/weatherloss/WindPower/inference/WPDistr/SHC_Finetune"),
    # "Vanilla_Finetune":  Path("/mnt/weatherloss/WindPower/inference/WPDistr/Vanilla_Finetune"),
    # "H_Finetune":        Path("/mnt/weatherloss/WindPower/inference/WPDistr/HC_Finetune"),
    # "VH_Finetune":       Path("/mnt/weatherloss/WindPower/inference/WPDistr/VHC_Finetune"),
}

TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
OUT_DIR    = Path("DistrFigures")

WS_VAR, CF_VAR = "ws100", "capacityfactor"
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
OBS_STEP_H = 3

# The MEASURED baseline curve. Keep these identical to verify_power.py or the curve drawn here
# is not the curve that was scored there. The guard below is the same one verify_power applies:
# fitting the baseline on the scored period would make it in-sample and the comparison fake.
TRAIN_START = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")
TRAIN_END   = pd.Timestamp("2024-07-31 21:00:00", tz="UTC")
# ======================================================================

FORECAST_RE = re.compile(r"forecast_(\d{14})")
FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def parse_init(p):
    return pd.to_datetime(FORECAST_RE.search(p.name).group(1), format="%Y%m%d%H%M%S", utc=True)


def turbine_power(ws, cut_in, rated_ws, cut_out, rated_mw):
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws)
    ramp = (ws >= cut_in) & (ws < rated_ws)
    out[ramp] = rated_mw * (ws[ramp] ** 3 - cut_in ** 3) / (rated_ws ** 3 - cut_in ** 3)
    out[(ws >= rated_ws) & (ws < cut_out)] = rated_mw
    return out


def build_farm_curves(farms_df, specs, farms):
    curves, meta = {}, farms_df.set_index("farm")
    for farm in farms:
        parts = []
        for chunk in str(meta.loc[farm, "fleet"]).split(";"):
            m = FLEET_RE.match(chunk)
            if not m:
                raise SystemExit(f"{farm}: cannot parse fleet entry {chunk!r}")
            parts.append((int(m.group(1)), specs.loc[m.group(2)]))
        scale = float(meta.loc[farm, "capacity_mw"]) / \
            sum(c * float(s["rated_power_mw"]) for c, s in parts)

        def curve(ws, parts=parts, scale=scale):
            tot = np.zeros_like(np.asarray(ws, dtype=float))
            for count, s in parts:
                tot += count * turbine_power(ws, float(s["cut_in_ms"]), float(s["rated_ws_ms"]),
                                             float(s["cut_out_ms"]), float(s["rated_power_mw"]))
            return tot * scale
        curves[farm] = curve
    return curves


def binned(x, y, edges, min_n):
    """Median of y in each x bin, plus the bin occupancy. Median, not mean: the tails are
    skewed and a couple of curtailed hours should not drag a whole bin."""
    idx = np.digitize(x, edges) - 1
    mid = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(mid.size, np.nan)
    cnt = np.zeros(mid.size)
    for b in range(mid.size):
        m = idx == b
        cnt[b] = m.sum()
        if cnt[b] >= min_n:
            med[b] = np.median(y[m])
    return mid, med, cnt


def boot_floor(x, y, edges, min_n, n_boot, rng):
    """Bootstrap SE of the binned median, per bin. This is the floor `to OBSERVED` is measured
    against: the observed curve is itself an estimate, and a model cannot sit closer to it than
    two independent estimates of it would sit to each other."""
    idx = np.digitize(x, edges) - 1
    se = np.full(edges.size - 1, np.nan)
    for b in range(se.size):
        yb = y[idx == b]
        if yb.size >= min_n:
            se[b] = np.std([np.median(rng.choice(yb, yb.size)) for _ in range(n_boot)])
    return se


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
    if EXCLUDE_FARMS:
        unknown = set(EXCLUDE_FARMS) - set(farms)
        if unknown:
            raise SystemExit(f"EXCLUDE_FARMS names farms not in REGION={REGION!r}: "
                             f"{sorted(unknown)}")
        farms = [f for f in farms if f not in EXCLUDE_FARMS]
        print(f"EXCLUDED: {', '.join(EXCLUDE_FARMS)} -- {len(farms)} farms remain")
    turbines = turbines[turbines.farm.isin(farms)]
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    tsum = turbines.groupby("farm")["capacity_mw"].sum().reindex(farms)
    if ((tsum - cap).abs() / cap > 0.001).any():
        raise SystemExit("turbines.csv and farms.csv capacities disagree -- rerun farm_metadata.py")
    curves = build_farm_curves(farms_df, specs, farms)

    if TRAIN_END >= INIT_START:
        raise SystemExit(f"baseline curve measured to {TRAIN_END} but scoring starts "
                         f"{INIT_START} -- overlapping, so the baseline would be in-sample")
    measured = fc.empirical(farms, farms_df, turbines, obs, TRUTH_ZARR, TRAIN_START, TRAIN_END)

    print(f"\nRegion {REGION}: {len(farms)} farms, {float(cap.sum()):.0f} MW | lead +{LEAD}h\n")

    # ---- CERRA wind at the farm cells, and the observed curve ----
    ds = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    tvars = list(ds.attrs["variables"])
    tdates = pd.to_datetime(ds["dates"].values).tz_localize("UTC")
    glat = np.asarray(ds["latitudes"]).ravel()
    glon = to_180(np.asarray(ds["longitudes"]).ravel())
    ck = np.cos(np.radians(float(glat.mean())))
    _, tc = cKDTree(np.c_[glon * ck, glat]).query(
        np.c_[to_180(turbines.longitude) * ck, turbines.latitude.to_numpy()], k=1)
    turbines = turbines.assign(cell=tc.astype(int))
    cells = np.sort(turbines.cell.unique())
    cpos = {int(c): j for j, c in enumerate(cells)}
    tsel = np.where((tdates >= INIT_START) & (tdates <= INIT_END))[0]
    ws_c = ds["data"].isel(time=tsel, variable=tvars.index(WS_VAR),
                           ensemble=0).values[:, cells].astype(np.float64)
    ds.close()
    times = tdates[tsel]

    G = np.zeros((len(farms), cells.size))
    for (fm, c), mw in turbines.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[farms.index(fm), cpos[int(c)]] = mw
    Wn = G / G.sum(1, keepdims=True)
    ws_f = ws_c @ Wn.T
    # the observation is a 3h mean, so bin it on the 3h-mean wind
    nxt = np.searchsorted(times, times + pd.Timedelta(hours=OBS_STEP_H))
    ok = (nxt < len(times)) & (np.abs((times[np.minimum(nxt, len(times) - 1)] - times)
                                      .total_seconds() - 3600 * OBS_STEP_H) < 1)
    ws_win = np.full_like(ws_f, np.nan)
    ws_win[ok] = 0.5 * (ws_f[ok] + ws_f[nxt[ok]])

    observed, floor = {}, {}
    rng = np.random.default_rng(0)
    for i, f in enumerate(farms):
        o = obs[f].reindex(times).to_numpy(float)
        m = np.isfinite(o) & np.isfinite(ws_win[:, i])
        observed[f] = binned(ws_win[m, i], 100.0 * o[m] / cap[f], WS_EDGES, MIN_BIN)
        floor[f] = boot_floor(ws_win[m, i], 100.0 * o[m] / cap[f], WS_EDGES, MIN_BIN, N_BOOT, rng)
        print(f"  observed curve {f:14s} {int(m.sum()):6d} cases")

    # ---- the model's implied curve ----
    implied, wind_expl = {}, {}
    for label, d in FORECAST_DIRS.items():
        files = {parse_init(p): p for p in sorted(d.glob("forecast_*.nc"))
                 if INIT_START <= parse_init(p) <= INIT_END}
        if not files:
            print(f"\n{label}: no forecast files -- skipped"); continue
        with xr.open_dataset(sorted(files.values())[0]) as f0:
            if CF_VAR not in f0:
                print(f"\n{label}: no {CF_VAR!r} -- skipped (weather-only run)"); continue
            fl, fo = np.asarray(f0["latitude"].values), to_180(np.asarray(f0["longitude"].values))
            nfc = int(f0[CF_VAR].shape[1])
        fk = np.cos(np.radians(float(fl.mean())))
        _, fcells = cKDTree(np.c_[fo * fk, fl]).query(np.c_[glon[cells] * fk, glat[cells]], k=1)

        print(f"\nReading {label}: {len(files)} files at +{LEAD}h ...")
        MW, MP = [], []
        for k, (init, fp) in enumerate(sorted(files.items())):
            if k % 500 == 0:
                print(f"  {k}/{len(files)}", flush=True)
            vt = init + pd.Timedelta(hours=LEAD)
            with xr.open_dataset(fp) as fx:
                ft = pd.DatetimeIndex(fx["time"].values).tz_localize("UTC")
                j = {t: q for q, t in enumerate(ft)}.get(vt)
                if j is None:
                    continue
                if fx[CF_VAR].shape[1] != nfc:
                    raise SystemExit(f"{fp.name}: grid is not constant across this run")
                MW.append(fx[WS_VAR].values[j, fcells] @ Wn.T)     # model wind  per farm
                MP.append(fx[CF_VAR].values[j, fcells] @ G.T)      # model power per farm
        MW, MP = np.asarray(MW), np.asarray(MP)
        implied[label], wind_expl[label] = {}, {}
        for i, f in enumerate(farms):
            m = np.isfinite(MW[:, i]) & np.isfinite(MP[:, i])
            implied[label][f] = binned(MW[m, i], 100.0 * MP[m, i] / cap[f], WS_EDGES, MIN_BIN)
            wind_expl[label][f] = float(np.corrcoef(MW[m, i], MP[m, i])[0, 1] ** 2)

    # ---- the two numbers that decide it ----
    mid = 0.5 * (WS_EDGES[:-1] + WS_EDGES[1:])
    band = (mid >= RAMP[0]) & (mid <= RAMP[1])
    # Bins scored: in the ramp band, with an observed median, and populated by EVERY run. With a
    # per-run mask the SPEC distance changed by up to 40% between runs for the same farm -- it
    # involves only the farm, so it must not depend on which run is being scored.
    keep = {}
    for f in farms:
        ok = band & np.isfinite(observed[f][1])
        for label in implied:
            ok = ok & np.isfinite(implied[label][f][1])
        keep[f] = ok
    thin = [f for f in farms if keep[f].sum() < 3]
    if thin:
        print(f"\n  WARNING too few shared bins to score: {', '.join(thin)}")

    # ---- how stale is the baseline? model-independent, so it gets its own block ----
    # MEASURED and OBSERVED are the same farm binned the same way in two different windows, so
    # the gap between them is non-stationarity, not model error. It is the handicap the baseline
    # carries into verify_power.py, and the head has to beat only what is left after it.
    print(f"\n{'='*90}\nBASELINE STALENESS -- the measured curve ({TRAIN_START.date()}.."
          f"{TRAIN_END.date()}) against what the farm actually did\n"
          f"while the forecasts were scored ({INIT_START.date()}..{INIT_END.date()})\n{'='*90}")
    print(f"{'farm':14s} {'MEAS->OBS':>10s} {'floor':>7s} {'x floor':>8s} {'plateau shift':>14s}")
    d_meas_obs = {}
    for f in farms:
        om, oc = observed[f][1], observed[f][2]
        me = 100.0 * measured[f](mid) / cap[f]
        w = np.where(keep[f], oc, 0.0)
        if w.sum() == 0:
            continue
        rms = lambda v, w=w: float(np.sqrt(np.nansum(w * v ** 2) / w.sum()))
        d_meas_obs[f] = rms(me - om)
        flo = rms(floor[f])
        top = mid >= RAMP[1]
        shift = float(np.nanmax(om[top]) - np.nanmax(me[top])) if top.any() else np.nan
        print(f"{f:14s} {d_meas_obs[f]:9.2f}% {flo:6.2f}% {d_meas_obs[f]/flo:8.1f} "
              f"{shift:+13.2f}%")
    print(f"{'MEAN':14s} {np.mean(list(d_meas_obs.values())):9.2f}%")
    print("  plateau shift : observed top minus measured top above the ramp band. Positive means")
    print("                  the farm ran HIGHER than its historical curve during the test year,")
    print("                  so the baseline under-predicts there through no fault of the model.")

    print(f"\n{'='*124}\nDID IT LEARN A CONVERSION OR A LEVEL?   scored over {RAMP[0]}-{RAMP[1]} m/s"
          f", on bins populated by all {len(implied)} runs\n{'='*124}")
    print(f"{'run':20s} {'farm':14s} {'to OBSERVED':>12s} {'to MEASURED':>12s} {'floor':>7s} "
          f"{'x floor':>8s} {'to SPEC':>9s} {'ratio':>7s} {'wind-expl':>10s} {'range':>7s}")
    for label in implied:
        col = {k: [] for k in ("obs", "mea", "flo", "rat")}
        for f in farms:
            om, oc = observed[f][1], observed[f][2]
            im, sp = implied[label][f][1], 100.0 * curves[f](mid) / cap[f]
            me = 100.0 * measured[f](mid) / cap[f]
            w = np.where(keep[f], oc, 0.0)
            if w.sum() == 0:
                continue
            rms = lambda v: float(np.sqrt(np.nansum(w * v ** 2) / w.sum()))
            d_obs, d_mea, d_spec = rms(im - om), rms(im - me), rms(sp - om)
            flo = rms(floor[f])
            col["obs"].append(d_obs); col["mea"].append(d_mea)
            col["flo"].append(d_obs / flo); col["rat"].append(d_obs / d_spec)
            print(f"{label:20s} {f:14s} {d_obs:11.2f}% {d_mea:11.2f}% {flo:6.2f}% "
                  f"{d_obs/flo:8.1f} {d_spec:8.2f}% {d_obs/d_spec:7.2f} "
                  f"{100*wind_expl[label][f]:9.1f}% {np.nanmax(im) - np.nanmin(im):6.1f}%")
        print(f"{label:20s} {'MEAN':14s} {np.mean(col['obs']):11.2f}% {np.mean(col['mea']):11.2f}% "
              f"{'':6s} {np.mean(col['flo']):8.1f} {'':9s} {np.mean(col['rat']):7.2f}\n")
    print("  to OBSERVED : the headline. How far the learned conversion sits from the farm's")
    print("                real one, in % of capacity. No spec sheet involved.")
    print("  to MEASURED : same distance to the BASELINE curve verify_power.py scores against.")
    print("                Compare it with MEAS->OBS above: if the head is closer to OBSERVED")
    print("                than the baseline is, it tracked the farm's current behaviour rather")
    print("                than reproducing a historical curve -- which is the whole claim.")
    print("  floor       : bootstrap SE of the observed curve itself. x floor = how many times")
    print("                the irreducible noise the model is away -- near 1 is as good as the")
    print("                data can show, 10+ means real room left.")
    print("  ratio       : CONTEXT only. d_obs is near-constant across farms while d_spec varies")
    print("                sevenfold, so a low ratio mostly says the spec sheet was badly wrong")
    print("                for that farm, not that the model was especially good there.")
    print("  wind-expl   : share of the model's power variance explained by its own wind. A level")
    print("                would be near zero. It cannot reach 100%: the head's power is a 3h")
    print("                mean while its ws100 is instantaneous, which caps the correlation.")
    print("  range       : span of the implied curve -- but a narrow one can mean an under-")
    print("                dispersed WIND (fewer extreme bins populated), not a flat power")
    print("                response. Read it with sigma_p/sigma_o from verify_weather.py.")

    # ---- figure ----
    ncol = int(np.ceil(np.sqrt(len(farms)))); nrow = int(np.ceil(len(farms) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.1 * nrow), squeeze=False,
                            sharex=True, sharey=True)
    for i, f in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        ax.plot(mid, 100.0 * curves[f](mid) / cap[f], color="0.55", lw=2, ls="--",
                label="spec sheet (ideal)")
        ax.plot(mid, 100.0 * measured[f](mid) / cap[f], color="0.25", lw=2, ls=":",
                label=f"measured baseline ({TRAIN_START.year}-{TRAIN_END.year})")
        ax.plot(mid, observed[f][1], "k-", lw=2.5, label="observed (real farm, scored period)")
        for j, label in enumerate(implied):
            ax.plot(mid, implied[label][f][1], lw=1.8,
                    color=plt.cm.tab10.colors[j], label=f"implied — {label}")
        ax.set_title(f"{f} ({cap[f]:.0f} MW)", fontsize=10)
        ax.grid(alpha=0.3); ax.set_xlim(0, 25); ax.set_ylim(-5, 105)
        if i % ncol == 0: ax.set_ylabel("power [% of capacity]", fontsize=8)
        if i // ncol == nrow - 1: ax.set_xlabel("wind speed [m/s]", fontsize=8)
    for ax in axs.ravel()[len(farms):]:
        ax.axis("off")
    axs[0][0].legend(fontsize=7, framealpha=0.85)
    fig.suptitle("Implied power curve — what the head learned (colour) vs the real farm "
                 "(black) and the spec sheet (grey)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / f"implied_curve_{REGION}_lead{LEAD}h.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
