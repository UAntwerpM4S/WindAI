#!/usr/bin/env python3
"""Is the spec-vs-observed gap a real loss, or just the numbers in turbine_specs.csv?

verify_implied_curve.py reports how far each farm's SPEC curve sits from what the farm really
does. That gap has been read as wake and availability loss. But across the ten BE farms it
correlates with the ASSUMED rated wind speed at r = -0.83 (R^2 = 0.70): farms whose turbines are
entered at 12 m/s show a 16-24% gap, the farm entered at 14.9 m/s shows 3%. A datasheet number
has no business predicting how much wake a farm suffers, so most of the between-farm variation
may be an artefact of the CSV rather than physics.

This separates the two. Per farm it fits the SAME functional form the spec curve uses -- cubic
from cut-in to rated, flat above -- to the farm's own observed curve, with three free parameters:

  cut_in      where production starts. A NUISANCE PARAMETER, not a result: power_obs is a 3h
              mean, so a window straddling cut-in has half its hours producing and half not,
              and mean power is positive at mean winds below cut-in. That drags the fitted
              value left -- 0.3 m/s at a 1.2 m/s within-window wind spread, 1.1 m/s at 2.0,
              pinned to the bound at 3.0. It is left free so it ABSORBS that smearing rather
              than pushing it into rated_ws; do not read it.
  rated_ws    where it reaches the plateau. Its SHIFT from the datasheet value is an effective
              wind offset: wake deficit, hub-height mismatch and a wrong datasheet all look
              identical here, so the shift bounds them jointly rather than isolating wakes.
  scale       the plateau as a fraction of nameplate. A vertical deficit: availability,
              derating, curtailment. Independent of the horizontal shift above. Biased LOW by
              the same window averaging -- about 1% at a realistic wind spread -- so the
              availability deficit it reports is an overestimate by roughly a point.

Simulation with known parameters says rated_ws survives all of this: fitting window-averaged
power binned on window-mean wind recovers it to within 0.06 m/s even at a 3 m/s within-window
spread. That is why the shift is the number to read and the other two carry caveats.

Three numbers come out:
  * shift and scale, the physically interpretable decomposition of the gap
  * residual AFTER the fit -- what the parametric shape cannot explain at all. If this is near
    the observed curve's own noise, the form is adequate and the gap was parameters. If it stays
    large, the cubic-with-a-corner is the wrong model and neither reading is safe.
  * the correlation of shift and scale against nn_D, nearest-neighbour spacing in ROTOR
    DIAMETERS. Wake loss scales with the dimensionless spacing, so if the shift tracks nn_D
    after the datasheet value is fitted out, that residual shift is wakes. If it does not, the
    gap was the CSV.

Prints every number; writes one PNG (observed, spec as given, spec fitted).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== SETTINGS ==============================
REGION    = "BE"
WS_EDGES  = np.arange(0.0, 25.1, 0.5)
MIN_BIN   = 30                 # a bin with fewer cases is not fitted
FIT_RANGE = (0.0, 22.0)        # fit over the plateau too, or `scale` is not identifiable
CUT_OUT   = 25.0               # held fixed: winds above it are too rare to constrain
BOUNDS    = ([1.0, 9.0, 0.50],  # cut_in, rated_ws, scale
             [6.0, 18.0, 1.15])

WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
OUT_DIR    = Path("DistrFigures")

WS_VAR     = "ws100"
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
OBS_STEP_H = 3                 # power_obs at t is the mean over [t, t+3h)
# ======================================================================

FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def fleet(spec_row, specs):
    """[(count, spec)] for a farm's fleet string, e.g. '55x Vestas-3-V90; 1x Haliade-6-150'."""
    out = []
    for chunk in str(spec_row).split(";"):
        m = FLEET_RE.match(chunk)
        if not m or m.group(2) not in specs.index:
            raise SystemExit(f"cannot resolve fleet entry {chunk!r} against turbine_specs.csv")
        out.append((int(m.group(1)), specs.loc[m.group(2)]))
    return out


def weighted(parts, col):
    """Capacity-weighted mean of a spec column over a fleet -- the datasheet value to beat."""
    num = sum(n * float(s["rated_power_mw"]) * float(s[col]) for n, s in parts)
    return num / sum(n * float(s["rated_power_mw"]) for n, s in parts)


def shape(ws, cut_in, rated_ws, scale):
    """The spec curve's own functional form, as % of capacity."""
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws)
    r = (ws >= cut_in) & (ws < rated_ws)
    out[r] = (ws[r] ** 3 - cut_in ** 3) / (rated_ws ** 3 - cut_in ** 3)
    out[(ws >= rated_ws) & (ws < CUT_OUT)] = 1.0
    return 100.0 * scale * out


def binned(x, y, edges, min_n):
    """Median of y per x bin, and the occupancy. Median: a few curtailed hours must not drag
    a whole bin."""
    idx = np.digitize(x, edges) - 1
    mid = 0.5 * (edges[:-1] + edges[1:])
    med, cnt = np.full(mid.size, np.nan), np.zeros(mid.size)
    for b in range(mid.size):
        m = idx == b
        cnt[b] = m.sum()
        if cnt[b] >= min_n:
            med[b] = np.median(y[m])
    return mid, med, cnt


def nn_rotor_spacing(t, rotor_m):
    """Median nearest-neighbour turbine spacing in ROTOR DIAMETERS. Dimensionless, which is what
    wake loss actually scales with -- metres alone does not, as the density analysis showed."""
    lat, lon = t.latitude.to_numpy(float), to_180(t.longitude.to_numpy(float))
    ck = np.cos(np.radians(float(lat.mean())))
    xy = np.c_[lon * ck, lat] * 111_320.0                       # degrees -> metres
    d, _ = cKDTree(xy).query(xy, k=2)                           # k=2: self, then nearest
    return float(np.median(d[:, 1]) / rotor_m)


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
    meta = farms_df.set_index("farm")
    cap = meta.loc[farms, "capacity_mw"]

    # datasheet values and geometry, per farm
    book, nn_D = {}, {}
    for f in farms:
        parts = fleet(meta.loc[f, "fleet"], specs)
        book[f] = (weighted(parts, "cut_in_ms"), weighted(parts, "rated_ws_ms"))
        nn_D[f] = nn_rotor_spacing(turbines[turbines.farm == f], weighted(parts, "rotor_diameter_m"))
    print(f"Region {REGION}: {len(farms)} farms, {float(cap.sum()):.0f} MW\n")

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
    ws_f = ws_c @ (G / G.sum(1, keepdims=True)).T
    # the observation is a 3h mean, so bin it on the 3h-mean wind
    nxt = np.searchsorted(times, times + pd.Timedelta(hours=OBS_STEP_H))
    ok = (nxt < len(times)) & (np.abs((times[np.minimum(nxt, len(times) - 1)] - times)
                                      .total_seconds() - 3600 * OBS_STEP_H) < 1)
    ws_win = np.full_like(ws_f, np.nan)
    ws_win[ok] = 0.5 * (ws_f[ok] + ws_f[nxt[ok]])

    # ---- fit the spec form to each farm's own curve ----
    mid = 0.5 * (WS_EDGES[:-1] + WS_EDGES[1:])
    res = {}
    print(f"{'farm':14s} {'cut-in':>14s} {'rated ws':>16s} {'plateau':>9s} "
          f"{'RMS as given':>13s} {'RMS fitted':>11s} {'nn_D':>7s}")
    print(f"{'':14s} {'book':>6s}{'fit':>8s} {'book':>7s}{'fit':>7s}{'shift':>6s} "
          f"{'scale':>9s} {'%cap':>13s} {'%cap':>11s}")
    for i, f in enumerate(farms):
        o = obs[f].reindex(times).to_numpy(float)
        m = np.isfinite(o) & np.isfinite(ws_win[:, i])
        _, om, oc = binned(ws_win[m, i], 100.0 * o[m] / cap[f], WS_EDGES, MIN_BIN)
        use = np.isfinite(om) & (mid >= FIT_RANGE[0]) & (mid <= FIT_RANGE[1])
        if use.sum() < 6:
            print(f"{f:14s} too few bins to fit -- skipped"); continue
        x, y, w = mid[use], om[use], oc[use]
        sw = np.sqrt(w / w.sum())

        ci0, rw0 = book[f]
        fit = least_squares(lambda p: sw * (shape(x, *p) - y), x0=[ci0, rw0, 1.0], bounds=BOUNDS)
        ci, rw, sc = fit.x
        rms = lambda p: float(np.sqrt(np.sum(w * (shape(x, *p) - y) ** 2) / w.sum()))
        res[f] = dict(cut_in=ci, rated=rw, shift=rw - rw0, scale=sc,
                      before=rms([ci0, rw0, 1.0]), after=rms(fit.x), nn=nn_D[f],
                      book=(ci0, rw0), curve=(x, y, w))
        r = res[f]
        print(f"{f:14s} {ci0:6.1f}{ci:8.1f} {rw0:7.1f}{rw:7.1f}{r['shift']:+6.1f} "
              f"{sc:9.3f} {r['before']:12.2f}% {r['after']:10.2f}% {r['nn']:7.1f}")

    # ---- does what is left track the geometry? ----
    k = list(res)
    A = {n: np.array([res[f][n] for f in k]) for n in ("shift", "scale", "before", "after", "nn")}
    print(f"\n{'='*84}\nWHAT SURVIVES THE FIT\n{'='*84}")
    print(f"  RMS to the observed curve: {A['before'].mean():.2f}% as given "
          f"-> {A['after'].mean():.2f}% fitted   ({100*(1-A['after'].mean()/A['before'].mean()):.0f}% removed)")
    for name, lab in (("shift", "rated-ws shift [m/s]"), ("scale", "plateau scale"),
                      ("after", "residual after fit [%cap]"), ("before", "gap as given [%cap]")):
        r_, p_ = stats.pearsonr(A["nn"], A[name])
        rs, ps = stats.spearmanr(A["nn"], A[name])
        print(f"  nn_D vs {lab:26s} r = {r_:+.3f} (p={p_:.3f})   rho = {rs:+.3f} (p={ps:.3f})")
    print("\n  The gap AS GIVEN correlating with nn_D is weak evidence: it also correlates with")
    print("  the datasheet rated wind speed. The SHIFT correlating with nn_D after the datasheet")
    print("  value is fitted out is the wake claim -- that is the line to read.")
    print("  A shift is not wakes on its own: a hub-height mismatch (CERRA ws100 against hubs at")
    print("  100-150 m) and a wrong datasheet produce the same horizontal offset.")
    print("  If `residual after fit` stays near `as given`, the cubic-with-a-corner is simply the")
    print("  wrong shape for a farm-aggregate curve and neither reading holds. Expect some of it")
    print("  at the KNEE regardless: a real farm's curve is smoothed twice over -- its turbines")
    print("  see different winds, and the 3h observation window averages a changing one -- so no")
    print("  hard-cornered form can sit on it there.")
    print("  cut-in is a nuisance parameter (window smearing drags it left); scale is ~1% low for")
    print("  the same reason. rated_ws is unbiased -- read the shift, caveat the rest.")

    # ---- figure ----
    ncol = int(np.ceil(np.sqrt(len(k)))); nrow = int(np.ceil(len(k) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.1 * nrow), squeeze=False,
                            sharex=True, sharey=True)
    for i, f in enumerate(k):
        ax = axs[i // ncol][i % ncol]
        r = res[f]
        ax.plot(mid, shape(mid, *r["book"], 1.0), color="0.55", lw=2, ls="--", label="spec as given")
        ax.plot(*r["curve"][:2], "k-", lw=2.5, label="observed")
        ax.plot(mid, shape(mid, r["cut_in"], r["rated"], r["scale"]), color="#D55E00", lw=1.8,
                label="spec re-fitted")
        ax.set_title(f"{f}  shift {r['shift']:+.1f} m/s, scale {r['scale']:.2f}", fontsize=9)
        ax.grid(alpha=0.3); ax.set_xlim(0, 25); ax.set_ylim(-5, 105)
        if i % ncol == 0: ax.set_ylabel("power [% of capacity]", fontsize=8)
        if i // ncol == nrow - 1: ax.set_xlabel("wind speed [m/s]", fontsize=8)
    for ax in axs.ravel()[len(k):]:
        ax.axis("off")
    axs[0][0].legend(fontsize=7, framealpha=0.85)
    fig.suptitle("Spec curve as given vs re-fitted to each farm's own production", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / f"curve_params_{REGION}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
