#!/usr/bin/env python3
"""How much of the "wake loss" is real, once the power curve and the hub height are right?

verify_curve_params.py showed the spec-vs-observed gap is 87% explained by refitting three
parameters of a hard-cornered cubic, and that its apparent link to turbine spacing vanishes under
partial correlation (rho -0.15) because nn_D carries rotor diameter and so does the datasheet
rated wind speed. That leaves the question open rather than answered: the cubic was never a
credible turbine model.

This builds the credible one from what the data actually says, and reports the gap at three
increasingly honest baselines:

  CUBIC     turbine_specs.csv through the hard-cornered cubic, at CERRA ws100.  The old baseline.
  CURVE     the real turbine's power curve (powercurves/, PyWake GenericWindTurbine) for the
            type the EWW database records at that turbine's coordinates, still at ws100.
  HUB       the same curve, with ws100 extrapolated to that turbine's OWN hub height. Your farms
            sit at 71-109 m and every one of them has been fed 100 m wind: Northwind at 71 m has
            been given ~4% too much, Mermaid at 109 m ~1% too little. On a cubic ramp 4% of wind
            is ~12% of power, so this is not a rounding error.

What survives HUB is then fitted with two parameters, on the real curve's shape rather than a
cubic's:

  alpha   a wind multiplier: the farm behaves like its turbines seeing alpha * ws. alpha < 1 is
          a wind deficit -- wake, or a shear exponent that is wrong for that site.
  scale   the plateau as a fraction of nameplate -- availability, derating, curtailment.

alpha is the clean wake test the last script could not run: it is fitted against a real curve at
the real hub height, so it carries no datasheet number and no rotor diameter, and correlating it
with nn_D is therefore not circular.

Turbine type and hub height come from the EWW open database, matched to turbines.csv by nearest
coordinate (median 8 m, max 348 m over the 399 BE turbines). turbine_specs.csv is used only for
the CUBIC baseline it is being compared against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy import stats

import farm_curves as fc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== SETTINGS ==============================
REGION      = "BE"
WS_EDGES    = np.arange(0.0, 25.1, 0.5)
MIN_BIN     = 30
FIT_RANGE   = (0.0, 22.0)     # include the plateau, or `scale` is not identifiable
SHEAR       = 0.11            # offshore power-law exponent: ws(z) = ws100 * (z/100)**SHEAR
BOUNDS      = ([0.70, 0.50],  # alpha, scale
               [1.15, 1.15])

WPOWER_DIR  = Path("/mnt/weatherloss/WindPower/data/WPDistr")
CURVE_DIR   = WPOWER_DIR / "powercurves"
EWW_CSV     = WPOWER_DIR / "farmdetails.csv"
TRUTH_ZARR  = WPOWER_DIR / "Anemoidatasets/power_cerra_A.zarr"
OUT_DIR     = Path("DistrFigures")

WS_VAR      = "ws100"
INIT_START  = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END    = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
OBS_STEP_H  = 3
MAX_MATCH_KM = 1.0            # a turbine further than this from any EWW row is a mismatch
# ======================================================================


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def farm_curve(sub, curves, use_hub):
    """callable(ws100) -> % of nameplate for one farm: every turbine on its own curve, at its own
    hub height, summed and renormalised so the plateau is 100%."""
    parts = []
    for (tt, hub), g in sub.groupby(["ttype", "hub"]):
        key = str(tt).replace(" ", "_").replace("/", "_")
        if key not in curves:
            raise SystemExit(f"no power curve for {tt!r} (looked for {key}.csv)")
        f = (hub / 100.0) ** SHEAR if use_hub else 1.0
        parts.append((len(g), curves[key], f))

    def c(ws100, parts=parts):
        ws100 = np.asarray(ws100, float)
        return sum(n * fn(ws100 * f) for n, fn, f in parts)
    top = float(np.max(c(np.arange(4.0, 25.0, 0.1))))
    return lambda ws, c=c, top=top: 100.0 * c(ws) / top


def cubic_curve(sub, specs, farms_df, farm):
    """The old baseline: turbine_specs.csv through a hard-cornered cubic, at ws100."""
    import re
    meta = farms_df.set_index("farm")
    fl = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")
    parts = []
    for chunk in str(meta.loc[farm, "fleet"]).split(";"):
        m = fl.match(chunk)
        parts.append((int(m.group(1)), specs.loc[m.group(2)]))

    def c(ws, parts=parts):
        ws = np.asarray(ws, float)
        tot = np.zeros_like(ws)
        for n, s in parts:
            ci, rw, co, pw = (float(s[k]) for k in
                              ("cut_in_ms", "rated_ws_ms", "cut_out_ms", "rated_power_mw"))
            r = (ws >= ci) & (ws < rw)
            out = np.zeros_like(ws)
            out[r] = pw * (ws[r] ** 3 - ci ** 3) / (rw ** 3 - ci ** 3)
            out[(ws >= rw) & (ws < co)] = pw
            tot += n * out
        return tot
    top = float(np.max(c(np.arange(3.0, 25.0, 0.1))))
    return lambda ws, c=c, top=top: 100.0 * c(ws) / top


def binned(x, y, edges, min_n):
    idx = np.digitize(x, edges) - 1
    mid = 0.5 * (edges[:-1] + edges[1:])
    med, cnt = np.full(mid.size, np.nan), np.zeros(mid.size)
    for b in range(mid.size):
        m = idx == b
        cnt[b] = m.sum()
        if cnt[b] >= min_n:
            med[b] = np.median(y[m])
    return mid, med, cnt


def nn_rotor(sub, e_rot):
    """Median nearest-neighbour spacing in rotor diameters -- what wake loss scales with."""
    lat, lon = sub.latitude.to_numpy(float), to_180(sub.longitude.to_numpy(float))
    ck = np.cos(np.radians(float(lat.mean())))
    xy = np.c_[lon * ck, lat] * 111_320.0
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]) / e_rot)


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
    turbines = fc.attach_eww(turbines[turbines.farm.isin(farms)], EWW_CSV,
                             max_km=MAX_MATCH_KM)
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    curves = fc.load_curves(CURVE_DIR)

    eww = pd.read_csv(EWW_CSV, low_memory=False)
    rot = eww.set_index("turbine_type").rotor_diameter.groupby(level=0).median()
    print(f"\n{'farm':14s} {'turbines':>9s} {'hub m':>7s} {'type':>22s} {'nn_D':>6s} {'comm':>9s}")
    geo = {}
    for f in farms:
        s = turbines[turbines.farm == f]
        r = float(np.average([rot.get(t, np.nan) for t in s.ttype],
                             weights=s.capacity_mw.to_numpy(float)))
        geo[f] = dict(hub_m=float(np.average(s.hub, weights=s.capacity_mw)),
                      nn=nn_rotor(s, r),
                      year=int(str(sorted(s.comm)[0])[:4]))
        print(f"{f:14s} {len(s):9d} {geo[f]['hub_m']:7.0f} "
              f"{s.ttype.mode()[0]:>22s} {geo[f]['nn']:6.1f} {geo[f]['year']:9d}")

    # ---- observed curve on the window-mean wind ----
    ds = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    tv = list(ds.attrs["variables"])
    td = pd.to_datetime(ds["dates"].values).tz_localize("UTC")
    glat = np.asarray(ds["latitudes"]).ravel(); glon = to_180(np.asarray(ds["longitudes"]).ravel())
    ck = np.cos(np.radians(float(glat.mean())))
    _, tc = cKDTree(np.c_[glon * ck, glat]).query(
        np.c_[to_180(turbines.longitude) * ck, turbines.latitude.to_numpy()], k=1)
    turbines = turbines.assign(cell=tc.astype(int))
    cells = np.sort(turbines.cell.unique()); cpos = {int(c): j for j, c in enumerate(cells)}
    sel = np.where((td >= INIT_START) & (td <= INIT_END))[0]
    ws_c = ds["data"].isel(time=sel, variable=tv.index(WS_VAR),
                           ensemble=0).values[:, cells].astype(np.float64)
    ds.close()
    times = td[sel]
    G = np.zeros((len(farms), cells.size))
    for (fm, c), mw in turbines.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[farms.index(fm), cpos[int(c)]] = mw
    ws_f = ws_c @ (G / G.sum(1, keepdims=True)).T
    nxt = np.searchsorted(times, times + pd.Timedelta(hours=OBS_STEP_H))
    ok = (nxt < len(times)) & (np.abs((times[np.minimum(nxt, len(times) - 1)] - times)
                                      .total_seconds() - 3600 * OBS_STEP_H) < 1)
    ws_win = np.full_like(ws_f, np.nan); ws_win[ok] = 0.5 * (ws_f[ok] + ws_f[nxt[ok]])

    # ---- three baselines, then fit what survives ----
    mid = 0.5 * (WS_EDGES[:-1] + WS_EDGES[1:])
    res = {}
    print(f"\n{'':14s} {'---- RMS to observed, % of capacity ----':>44s}")
    print(f"{'farm':14s} {'CUBIC':>9s} {'CURVE':>9s} {'HUB':>9s} {'fitted':>9s} "
          f"{'alpha':>7s} {'scale':>7s}")
    for i, f in enumerate(farms):
        o = obs[f].reindex(times).to_numpy(float)
        m = np.isfinite(o) & np.isfinite(ws_win[:, i])
        _, om, oc = binned(ws_win[m, i], 100.0 * o[m] / cap[f], WS_EDGES, MIN_BIN)
        use = np.isfinite(om) & (mid >= FIT_RANGE[0]) & (mid <= FIT_RANGE[1])
        if use.sum() < 6:
            print(f"{f:14s} too few bins -- skipped"); continue
        x, y, w = mid[use], om[use], oc[use]
        rms = lambda v: float(np.sqrt(np.sum(w * v ** 2) / w.sum()))

        sub = turbines[turbines.farm == f]
        c_cub = cubic_curve(sub, specs, farms_df, f)
        c_ws = farm_curve(sub, curves, use_hub=False)
        c_hub = farm_curve(sub, curves, use_hub=True)
        sw = np.sqrt(w / w.sum())
        fit = least_squares(lambda p: sw * (p[1] * c_hub(p[0] * x) - y), x0=[1.0, 1.0],
                            bounds=BOUNDS)
        a, sc = fit.x
        res[f] = dict(cubic=rms(c_cub(x) - y), curve=rms(c_ws(x) - y), hub=rms(c_hub(x) - y),
                      after=rms(sc * c_hub(a * x) - y), alpha=a, scale=sc,
                      curves=(x, y, c_cub, c_hub, a, sc), **geo[f])
        r = res[f]
        print(f"{f:14s} {r['cubic']:8.2f}% {r['curve']:8.2f}% {r['hub']:8.2f}% "
              f"{r['after']:8.2f}% {a:7.3f} {sc:7.3f}")

    k = list(res)
    A = {n: np.array([res[f][n] for f in k])
         for n in ("cubic", "curve", "hub", "after", "alpha", "scale", "nn")}
    A["hubm"] = np.array([geo[f]["hub_m"] for f in k], float)  # metres, not the RMS column
    A["year"] = np.array([geo[f]["year"] for f in k], float)

    print(f"\n{'='*88}\nWHAT EACH FIX BUYS   (mean RMS to the observed curve)\n{'='*88}")
    for n, lab in (("cubic", "cubic + turbine_specs, ws100"), ("curve", "real curve, ws100"),
                   ("hub", "real curve, own hub height"), ("after", "+ fitted alpha and scale")):
        print(f"  {lab:34s} {A[n].mean():6.2f}%")
    print(f"\n  alpha  {A['alpha'].min():.3f}-{A['alpha'].max():.3f}  mean {A['alpha'].mean():.3f}"
          f"   (below 1 = the farm needs MORE wind than its turbines' curves say)")
    print(f"  scale  {A['scale'].min():.3f}-{A['scale'].max():.3f}  mean {A['scale'].mean():.3f}")

    print(f"\n{'='*88}\nIS THE RESIDUAL DEFICIT WAKES?\n{'='*88}")
    for xn, xl in (("nn", "nn_D (rotor diameters)"), ("hubm", "hub height [m]"),
                   ("year", "commissioning year")):
        for yn, yl in (("alpha", "alpha"), ("scale", "scale")):
            r_, p_ = stats.pearsonr(A[xn], A[yn]); rs, ps = stats.spearmanr(A[xn], A[yn])
            print(f"  {xl:24s} vs {yl:6s}  r = {r_:+.3f} (p={p_:.3f})   rho = {rs:+.3f} (p={ps:.3f})")
    print("\n  alpha rising with nn_D is the wake signature: widely spaced farms need less of a")
    print("  wind correction. Unlike the datasheet shift in verify_curve_params.py, alpha carries")
    print("  no rotor diameter, so this correlation is not circular.")
    print("  alpha against HUB HEIGHT tests the other reading: if it tracks hub height instead,")
    print("  SHEAR is simply wrong for these sites and alpha is absorbing that, not wakes.")

    # ---- figure ----
    ncol = int(np.ceil(np.sqrt(len(k)))); nrow = int(np.ceil(len(k) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.1 * nrow), squeeze=False,
                            sharex=True, sharey=True)
    for i, f in enumerate(k):
        ax = axs[i // ncol][i % ncol]
        x, y, c_cub, c_hub, a, sc = res[f]["curves"]
        ax.plot(mid, c_cub(mid), color="0.7", lw=1.6, ls=":", label="cubic + specs")
        ax.plot(mid, c_hub(mid), color="0.35", lw=2, ls="--", label="real curve @ hub")
        ax.plot(x, y, "k-", lw=2.5, label="observed")
        ax.plot(mid, sc * c_hub(a * mid), color="#D55E00", lw=1.8, label="fitted")
        ax.set_title(f"{f}  hub {res[f]['hub_m']:.0f} m, a={a:.3f}, s={sc:.2f}", fontsize=9)
        ax.grid(alpha=0.3); ax.set_xlim(0, 25); ax.set_ylim(-5, 105)
        if i % ncol == 0: ax.set_ylabel("power [% of capacity]", fontsize=8)
        if i // ncol == nrow - 1: ax.set_xlabel("CERRA ws100 [m/s]", fontsize=8)
    for ax in axs.ravel()[len(k):]:
        ax.axis("off")
    axs[0][0].legend(fontsize=7, framealpha=0.85)
    fig.suptitle("Real turbine curves at real hub heights vs what the farms actually produce",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / f"curve_physical_{REGION}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
