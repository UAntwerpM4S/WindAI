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

Three curves per farm, on the same axes:

  SPEC      the manufacturer curve from turbine_specs.csv -- an ideal turbine in clean air.
  OBSERVED  binned median of observed power against CERRA wind -- what the farm really does,
            losses and all. This is the target the model should be matching.
  IMPLIED   binned median of the model's power against the model's own wind, at LEAD.

If IMPLIED tracks OBSERVED, the head learned the conversion including the losses. If it tracks
SPEC, it learned the textbook curve and not the losses. If it is flat, it learned a level.

Two numbers decide it rather than eyeballing:
  * closeness -- occupancy-weighted RMS distance from IMPLIED to OBSERVED vs to SPEC. Below 1
    means the model is nearer the real farm than the spec sheet is.
  * wind-explained -- how much of the model's power variance its own wind accounts for. A
    conversion is a function of wind, so this must be high; a climatology need not be.

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

# ============================== SETTINGS ==============================
REGION   = "BE"
LEAD     = 33                  # lead hour the implied curve is read at
WS_EDGES = np.arange(0.0, 25.1, 0.5)     # wind bins for all three curves
MIN_BIN  = 30                 # a bin with fewer cases is not plotted
RAMP     = (3.0, 13.0)        # m/s window the closeness score is computed over

FORECAST_DIRS = {
    "HighCapacityGT":     Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
    "HighPowerGTFinetune": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacity_Finetune"),

}

TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
OUT_DIR    = Path("DistrFigures")

WS_VAR, CF_VAR = "ws100", "capacityfactor"
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
OBS_STEP_H = 3
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
    turbines = turbines[turbines.farm.isin(farms)]
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    tsum = turbines.groupby("farm")["capacity_mw"].sum().reindex(farms)
    if ((tsum - cap).abs() / cap > 0.001).any():
        raise SystemExit("turbines.csv and farms.csv capacities disagree -- rerun farm_metadata.py")
    curves = build_farm_curves(farms_df, specs, farms)
    print(f"Region {REGION}: {len(farms)} farms, {float(cap.sum()):.0f} MW | lead +{LEAD}h\n")

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

    observed = {}
    for i, f in enumerate(farms):
        o = obs[f].reindex(times).to_numpy(float)
        m = np.isfinite(o) & np.isfinite(ws_win[:, i])
        observed[f] = binned(ws_win[m, i], 100.0 * o[m] / cap[f], WS_EDGES, MIN_BIN)
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
    print(f"\n{'='*100}\nDID IT LEARN A CONVERSION OR A LEVEL?   scored over {RAMP[0]}-{RAMP[1]} m/s"
          f"\n{'='*100}")
    print(f"{'run':20s} {'farm':14s} {'to OBSERVED':>12s} {'to SPEC':>9s} {'ratio':>7s} "
          f"{'wind-expl':>10s} {'implied range':>14s}")
    for label in implied:
        rat = []
        for f in farms:
            om, oc = observed[f][1], observed[f][2]
            im = implied[label][f][1]
            sp = 100.0 * curves[f](mid) / cap[f]
            w = np.where(band & np.isfinite(om) & np.isfinite(im), oc, 0.0)
            if w.sum() == 0:
                continue
            d_obs = np.sqrt(np.nansum(w * (im - om) ** 2) / w.sum())
            d_spec = np.sqrt(np.nansum(w * (sp - om) ** 2) / w.sum())
            rat.append(d_obs / d_spec)
            rng_ = np.nanmax(im) - np.nanmin(im)
            print(f"{label:20s} {f:14s} {d_obs:11.2f}% {d_spec:8.2f}% {d_obs/d_spec:7.2f} "
                  f"{100*wind_expl[label][f]:9.1f}% {rng_:13.1f}%")
        print(f"{label:20s} {'MEAN':14s} {'':11s}  {'':8s} {np.mean(rat):7.2f}\n")
    print("  ratio < 1  : the model's implied curve is closer to the real farm than the spec")
    print("               sheet is -- it learned the losses, not just the textbook shape.")
    print("  ratio > 1  : it is no better than the spec curve.")
    print("  wind-expl  : share of the model's power variance explained by its own wind. A")
    print("               conversion is a function of wind, so this should be high (>90%).")
    print("               A low value with a flat implied range means it learned a LEVEL.")

    # ---- figure ----
    ncol = int(np.ceil(np.sqrt(len(farms)))); nrow = int(np.ceil(len(farms) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.1 * nrow), squeeze=False,
                            sharex=True, sharey=True)
    for i, f in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        ax.plot(mid, 100.0 * curves[f](mid) / cap[f], color="0.55", lw=2, ls="--",
                label="spec sheet (ideal)")
        ax.plot(mid, observed[f][1], "k-", lw=2.5, label="observed (real farm)")
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
