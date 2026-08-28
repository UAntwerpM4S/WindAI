#!/usr/bin/env python3
"""Is the specs power curve's per-farm bias WAKE-SHAPED? A directional test.

Wake loss is the one component of the gap between an idealised specs curve and observed
production that depends on WIND DIRECTION: a turbine standing downstream of its neighbour loses
output, so an array loses most when the wind blows ALONG its tightest packing axis and least
across it. Availability, curtailment, hub-height mismatch and curve-shape error have no reason to
vary with direction. So a directional signal that lines up with array geometry attributes the
bias to wakes; one that does not, does not.

Two halves:

  CONVERSION BIAS (primary) -- CERRA analysis ws100 pushed through each farm's specs curve and
      compared with observed power. No forecast error at all, so this isolates the conversion.
      Binned on CERRA's own wdir100, recovered from wdir100_cos / wdir100_sin.

  HEAD BIAS -- the same, for each run's `capacityfactor` forecast at LEAD. If the head has
      absorbed the wake loss its directional amplitude should be far smaller than the curve's.

The test is CONVENTION-FREE. We never assume how wdir100 is signed or referenced. Wake loss is
180-degree periodic (a row wakes both ways), so we fit the second harmonic

    bias(theta) = a + b*cos(2 theta) + c*sin(2 theta)   [+ nuisance terms in ws]

per farm, take its peak direction phi, and ask whether (phi - pack_axis) comes out the SAME for
every farm. Any fixed convention error is a constant rotation and cancels in that difference.
Farms whose layouts point in different directions -- Northwester2 is rotated ~50 deg from the
rest -- are what make this decisive: a directional bias in the WIND (easterlies being harder,
say) would peak at the same absolute angle everywhere, while wakes peak at each farm's own axis.

ws and ws^2 enter the fit as nuisance covariates, so a sector that happens to be windier cannot
manufacture a directional signal on its own.

Restricted to the ramp regime (RAMP_WS): above rated a waked turbine still reaches rated so the
loss vanishes, below cut-in there is no power to lose. The ramp is where wakes show, and it is
also where the head's censoring failure is inactive.
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
REGION     = "BE"
N_SECTORS  = 12               # direction bins for the plot (the fit uses raw angles)
RAMP_WS    = (4.5, 8.0)       # m/s, on CERRA truth: where wake loss actually shows
LEAD       = 3                # lead hour for the forecast half
INCLUDE_HEAD = True           # also test each run's direct capacityfactor forecast

FORECAST_DIRS = {
    "HighCapacityGT":     Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
}

TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
OUT_DIR    = Path("DistrFigures")

WS_VAR, COS_VAR, SIN_VAR, CF_VAR = "ws100", "wdir100_cos", "wdir100_sin", "capacityfactor"
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
OBS_STEP_H = 3
N_PERM     = 20000
MIN_CASES  = 200              # a farm with fewer cases in the ramp band is not fitted
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


def pack_axis(lon, lat):
    """Bearing (deg from north, undirected mod 180) along which turbines sit closest together.

    Each turbine's vector to its nearest neighbour is an axial datum; the circular mean of the
    DOUBLED angles is the standard estimator for such data. `conc` is the resultant length:
    1 = perfectly rowed, 0 = isotropic, so it says how much directional signal to expect.
    """
    n = (lat - lat.mean()) * 111320.0
    e = (lon - lon.mean()) * 111320.0 * np.cos(np.radians(lat.mean()))
    p = np.c_[e, n]
    _, idx = cKDTree(p).query(p, k=2)
    v = p[idx[:, 1]] - p
    brg = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 180.0
    a = np.radians(2 * brg)
    C, S = np.cos(a).mean(), np.sin(a).mean()
    return (np.degrees(np.arctan2(S, C)) / 2) % 180.0, float(np.hypot(C, S)), p


def axial_spread(deg):
    """Circular sd of axial (mod-180) data, in degrees. A plain std() would call 179 and 1
    degrees far apart when they are 2 degrees apart."""
    a = np.radians(2 * np.asarray(deg, dtype=float))
    R = np.hypot(np.cos(a).mean(), np.sin(a).mean())
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12)))) / 2)


def nuisance(ws):
    """Orthonormal basis for [1, ws, ws^2]. Projecting it out once makes every later fit a 2x2
    solve, which is what keeps the permutation tests to seconds rather than an hour."""
    Q, _ = np.linalg.qr(np.c_[np.ones_like(ws), ws, ws ** 2])
    return Q


def harmonic_fit(theta_deg, y, ws, Q=None, y_r=None):
    """Fit  y ~ a + b cos2t + c sin2t + (ws, ws^2)  and return the second harmonic.

    Returns (peak direction mod 180, amplitude, fraction of the ws-adjusted variance explained).
    ws enters as a nuisance covariate so a sector that happens to be windier cannot manufacture
    a directional signal on its own.
    """
    if Q is None:
        Q = nuisance(ws)
    if y_r is None:
        y_r = y - Q @ (Q.T @ y)
    t = np.radians(theta_deg)
    C = np.c_[np.cos(2 * t), np.sin(2 * t)]
    C = C - Q @ (Q.T @ C)
    beta, *_ = np.linalg.lstsq(C, y_r, rcond=None)
    amp = float(np.hypot(beta[0], beta[1]))
    phi = (np.degrees(np.arctan2(beta[1], beta[0])) / 2) % 180.0
    ss0 = float((y_r ** 2).sum())
    ve = float(1 - ((y_r - C @ beta) ** 2).sum() / ss0) if ss0 > 0 else np.nan
    return phi, amp, ve


def perm_amp(theta_deg, y, ws, rng):
    """p for the harmonic amplitude, shuffling direction against the (bias, ws) pairs."""
    Q = nuisance(ws)
    y_r = y - Q @ (Q.T @ y)
    _, a0, _ = harmonic_fit(theta_deg, y, ws, Q, y_r)
    hits = sum(harmonic_fit(rng.permutation(theta_deg), y, ws, Q, y_r)[1] >= a0
               for _ in range(N_PERM))
    return a0, (hits + 1) / (N_PERM + 1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

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
    curves = build_farm_curves(farms_df, specs, farms)
    print(f"Region {REGION}: {len(farms)} farms, {float(cap.sum()):.0f} MW\n")

    # ---- array geometry, per farm ----
    axes, concs = {}, {}
    print(f"{'farm':14s} {'turbines':>9s} {'pack axis':>10s} {'row alignment':>14s}")
    for f in farms:
        t = turbines[turbines.farm == f]
        ax, cc, _ = pack_axis(t.longitude.to_numpy(), t.latitude.to_numpy())
        axes[f], concs[f] = ax, cc
        print(f"{f:14s} {len(t):9d} {ax:9.0f}° {cc:14.2f}")

    # ---- CERRA truth at the farm cells ----
    ds = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    tvars = list(ds.attrs["variables"])
    for v in (WS_VAR, COS_VAR, SIN_VAR):
        if v not in tvars:
            raise SystemExit(f"{v!r} not in truth zarr; have {tvars}")
    tdates = pd.to_datetime(ds["dates"].values).tz_localize("UTC")
    glat = np.asarray(ds["latitudes"]).ravel()
    glon = to_180(np.asarray(ds["longitudes"]).ravel())

    coslat = np.cos(np.radians(float(glat.mean())))
    tree = cKDTree(np.c_[glon * coslat, glat])
    _, tcell = tree.query(np.c_[to_180(turbines.longitude) * coslat,
                                turbines.latitude.to_numpy()], k=1)
    turbines = turbines.assign(cell=tcell.astype(int))
    cells = np.sort(turbines.cell.unique())
    cpos = {int(c): j for j, c in enumerate(cells)}

    keep = (tdates >= INIT_START) & (tdates <= INIT_END)
    tsel = np.where(keep)[0]
    print(f"\nLoading CERRA: {tsel.size} times x {cells.size} farm cells ...")
    grab = lambda v: ds["data"].isel(time=tsel, variable=tvars.index(v),
                                     ensemble=0).values[:, cells].astype(np.float64)
    ws_c, cos_c, sin_c = grab(WS_VAR), grab(COS_VAR), grab(SIN_VAR)
    ds.close()
    times = tdates[tsel]

    # capacity-weighted per farm; direction by VECTOR mean of the cos/sin pair
    W = np.zeros((len(farms), cells.size))
    for (fm, c), mw in turbines.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        W[farms.index(fm), cpos[int(c)]] = mw
    W = W / W.sum(1, keepdims=True)
    ws_f = ws_c @ W.T                                              # (T, F)
    dir_f = np.degrees(np.arctan2(sin_c @ W.T, cos_c @ W.T)) % 360.0
    print(f"Mean wind direction over all farms: {dir_f.mean():.0f}° "
          f"(expect a SW-ish prevailing value for the North Sea -- sanity check the convention)")

    # ---- primary: the CONVERSION bias, no forecast involved ----
    rows, panels = [], {}
    print(f"\n{'='*96}\nCONVERSION BIAS — CERRA analysis wind through the specs curve, "
          f"ramp {RAMP_WS[0]}-{RAMP_WS[1]} m/s\n{'='*96}")
    print(f"{'farm':14s} {'n':>7s} {'mean bias':>10s} {'peak dir':>9s} {'axis':>6s} "
          f"{'delta':>7s} {'amp':>7s} {'p':>8s} {'var expl':>9s}")
    for i, f in enumerate(farms):
        o = obs[f].reindex(times).to_numpy(float)
        w = ws_f[:, i]
        m = np.isfinite(o) & np.isfinite(w) & (w >= RAMP_WS[0]) & (w < RAMP_WS[1])
        if m.sum() < MIN_CASES:
            print(f"{f:14s} {m.sum():7d}   SKIPPED: under MIN_CASES={MIN_CASES} in the ramp band")
            continue
        pred = curves[f](w[m])
        bias = 100.0 * (pred - o[m]) / cap[f]
        th, wsm = dir_f[m, i], w[m]
        phi, amp, ve = harmonic_fit(th, bias, wsm)
        _, p = perm_amp(th, bias, wsm, rng)
        d = (phi - axes[f]) % 180.0
        d = d - 180.0 if d > 90.0 else d          # report in [-90, 90]: 178 deg IS -2 deg
        print(f"{f:14s} {m.sum():7d} {bias.mean():+9.2f}% {phi:8.0f}° {axes[f]:5.0f}° "
              f"{d:+6.0f}° {amp:7.2f} {p:8.4f} {100*ve:8.1f}%")
        rows.append((f, phi, axes[f], d, amp, concs[f]))
        panels[f] = (th, bias, wsm, phi, amp)

    # ---- the decisive part: do the peaks follow each farm's OWN axis? ----
    #
    # R alone is NOT enough. Seven of the ten BE farms are laid out within ~17 deg of each
    # other, so a bias peaking at one fixed ABSOLUTE angle (a directional error in the wind,
    # say) also produces a high resultant in (peak - axis). Verified on synthetic data: a pure
    # wind bias scores R = 0.87 against wakes' 1.00 -- nowhere near separable.
    #
    # The test that does separate them asks whether pairing each farm's peak with ITS OWN axis
    # beats pairing the peaks and axes at random. A wind bias is indifferent to which farm owns
    # which layout, so it fails; wakes cannot fail. On synthetic data this gives p = 0.0002 for
    # wakes, p = 0.60 for a pure wind bias and p = 0.58 for no signal, holding power down to a
    # 2%-of-capacity amplitude.
    def resultant(x):
        a = np.radians(2 * np.asarray(x))
        return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))

    phis = np.array([r[1] for r in rows])
    axs_ = np.array([r[2] for r in rows])
    R_wake = resultant((phis - axs_) % 180.0)          # peaks track each farm's own axis
    R_wind = resultant(phis)                            # peaks all at one absolute angle
    null = np.array([resultant((phis - rng.permutation(axs_)) % 180.0) for _ in range(N_PERM)])
    p_wake = (np.sum(null >= R_wake) + 1) / (N_PERM + 1)
    mean_d = (np.degrees(np.arctan2(np.sin(np.radians(2 * ((phis - axs_) % 180))).mean(),
                                    np.cos(np.radians(2 * ((phis - axs_) % 180))).mean())) / 2) % 180
    mean_d = mean_d - 180.0 if mean_d > 90.0 else mean_d

    print(f"\n{'-'*96}\nDOES THE BIAS FOLLOW ARRAY GEOMETRY?   n = {len(phis)} farms")
    print(f"  R_wake = {R_wake:.3f}   peaks at a fixed offset from each farm's OWN axis "
          f"(mean offset {mean_d:+.0f} deg)")
    print(f"  R_wind = {R_wind:.3f}   peaks all at one ABSOLUTE angle, ignoring layout")
    print(f"  p      = {p_wake:.4f}   against axes shuffled between farms")
    print(f"  spread of peak directions {axial_spread(phis):.0f} deg | spread of pack axes "
          f"{axial_spread(axs_):.0f} deg   (circular)")
    print()
    if p_wake < 0.05 and R_wake > R_wind:
        print("  -> The bias tracks each farm's own layout. That is the wake signature, and no")
        print("     confound in the list (availability, curtailment, hub height, curve shape)")
        print("     has any reason to produce it.")
    elif R_wind > R_wake:
        print("  -> The peaks cluster at one ABSOLUTE angle rather than following the layouts.")
        print("     That points at a directional error in the WIND, not at wakes. Do not claim")
        print("     wake losses on this evidence.")
    else:
        print("  -> No directional structure tied to geometry. The bias may still be wakes, but")
        print("     this test does not show it; report the null rather than the density result")
        print("     alone.")
    print()
    # how much does the test rest on a single oddly-oriented farm?
    names = [r[0] for r in rows]
    a2 = np.radians(2 * axs_)
    mu = np.arctan2(np.sin(a2).mean(), np.cos(a2).mean())
    off = np.degrees(np.abs(np.arctan2(np.sin(a2 - mu), np.cos(a2 - mu)))) / 2
    odd = np.argsort(-off)
    print(f"  CAVEAT: the layouts are clustered -- {int((off < 20).sum())} of {len(off)} farms "
          f"lie within 20 deg of the mean orientation, so the farms that differ carry the")
    print(f"  discriminating power. Most distinct: " +
          ", ".join(f"{names[i]} ({off[i]:.0f} deg off)" for i in odd[:3]))
    lo = [resultant((np.delete(phis, i) - np.delete(axs_, i)) % 180.0) for i in range(len(phis))]
    print(f"  Leave-one-farm-out R_wake spans {min(lo):.3f} to {max(lo):.3f}"
          f" (dropping {names[int(np.argmin(lo))]} costs the most).")
    print("  If the verdict rests on one farm, say so in the paper rather than leaning on it.")

    # ---- second half: does the head remove the directional structure? ----
    if INCLUDE_HEAD:
        G = np.zeros((len(farms), cells.size))
        for (fm, c), mw in turbines.groupby(["farm", "cell"])["capacity_mw"].sum().items():
            G[farms.index(fm), cpos[int(c)]] = mw
        t2row = {t: j for j, t in enumerate(times)}
        for label, d in FORECAST_DIRS.items():
            files = {parse_init(p): p for p in sorted(d.glob("forecast_*.nc"))
                     if INIT_START <= parse_init(p) <= INIT_END}
            if not files:
                print(f"\n{label}: no forecast files in range -- skipped")
                continue
            # map the truth cells onto THIS run's grid rather than assuming the two are
            # index-for-index (verify_weather.py and verify_power.py both remap; so must this)
            with xr.open_dataset(sorted(files.values())[0]) as f0:
                if CF_VAR not in f0:
                    print(f"\n{label}: no {CF_VAR!r} in its forecasts -- skipped "
                          f"(weather-only run, nothing to test)")
                    continue
                fcl = np.asarray(f0["latitude"].values)
                fco = to_180(np.asarray(f0["longitude"].values))
                nfc = int(f0[CF_VAR].shape[1])
            ck = np.cos(np.radians(float(fcl.mean())))
            dist, fcells = cKDTree(np.c_[fco * ck, fcl]).query(
                np.c_[glon[cells] * ck, glat[cells]], k=1)
            if dist.max() > 1e-6:
                print(f"  {label}: grid differs from CERRA, remapped "
                      f"(max offset {dist.max()*111:.3f} km)")
            print(f"\n{'='*96}\nHEAD BIAS — {label}, direct capacityfactor at +{LEAD}h, "
                  f"same ramp band\n{'='*96}")
            acc = {f: ([], [], []) for f in farms}
            for k, (init, fp) in enumerate(sorted(files.items())):
                if k % 500 == 0:
                    print(f"  {k}/{len(files)}", flush=True)
                vt = init + pd.Timedelta(hours=LEAD)
                if vt not in t2row or vt not in obs.index:
                    continue
                with xr.open_dataset(fp) as fx:
                    ft = pd.DatetimeIndex(fx["time"].values).tz_localize("UTC")
                    j = {t: q for q, t in enumerate(ft)}.get(vt)
                    if j is None:
                        continue
                    if fx[CF_VAR].shape[1] != nfc:
                        raise SystemExit(f"{fp.name}: {fx[CF_VAR].shape[1]} cells, expected "
                                         f"{nfc} -- this run's grid is not constant")
                    cf = fx[CF_VAR].values[j, fcells]
                p = cf @ G.T
                r = t2row[vt]
                for i, f in enumerate(farms):
                    ov, wv = obs.at[vt, f], ws_f[r, i]
                    if np.isfinite(ov) and np.isfinite(p[i]) and RAMP_WS[0] <= wv < RAMP_WS[1]:
                        acc[f][0].append(dir_f[r, i])
                        acc[f][1].append(100.0 * (p[i] - ov) / cap[f])
                        acc[f][2].append(wv)
            print(f"{'farm':14s} {'n':>7s} {'mean bias':>10s} {'peak dir':>9s} {'axis':>6s} "
                  f"{'amp':>7s} {'p':>8s}")
            for f in farms:
                th, bs, wm = map(np.asarray, acc[f])
                if th.size < MIN_CASES:
                    print(f"{f:14s} {th.size:7d}   SKIPPED: under MIN_CASES={MIN_CASES} "
                          f"(only ~1/3 of inits land in the ramp band, so you need roughly "
                          f"{3*MIN_CASES} forecast files)")
                    continue
                phi, amp, _ = harmonic_fit(th, bs, wm)
                _, pv = perm_amp(th, bs, wm, rng)
                print(f"{f:14s} {th.size:7d} {bs.mean():+9.2f}% {phi:8.0f}° {axes[f]:5.0f}° "
                      f"{amp:7.2f} {pv:8.4f}")
            print("  Compare `amp` with the conversion-bias table: if the head has absorbed the")
            print("  wake loss, its directional amplitude should be markedly smaller.")

    # ---- figure ----
    edges = np.linspace(0, 360, N_SECTORS + 1)
    ncol = int(np.ceil(np.sqrt(len(panels))))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), squeeze=False)
    for k, (f, (th, bias, wsm, phi, amp)) in enumerate(panels.items()):
        ax = axs[k // ncol][k % ncol]
        mids = 0.5 * (edges[:-1] + edges[1:])
        sel = [(th >= edges[q]) & (th < edges[q + 1]) for q in range(N_SECTORS)]
        mean = np.array([bias[m].mean() if m.any() else np.nan for m in sel])
        if np.isnan(mean).any():
            print(f"  {f}: {int(np.isnan(mean).sum())} of {N_SECTORS} direction sectors empty")
        ax.bar(mids, mean, width=360 / N_SECTORS * 0.85, color="#4C78A8")
        g = np.linspace(0, 360, 361)
        ax.plot(g, np.nanmean(mean) + amp * np.cos(2 * np.radians(g - phi)), "r-", lw=1.5)
        for a in (axes[f], axes[f] + 180):
            ax.axvline(a, color="k", ls="--", lw=1)
        ax.set_title(f"{f}  (axis {axes[f]:.0f}°, align {concs[f]:.2f})", fontsize=9)
        ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
        ax.grid(alpha=0.3)
        if k % ncol == 0:
            ax.set_ylabel("curve bias [% cap]", fontsize=8)
        if k // ncol == nrow - 1:
            ax.set_xlabel("CERRA wind direction [°]", fontsize=8)
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Specs-curve conversion bias vs wind direction, {RAMP_WS[0]}-{RAMP_WS[1]} m/s "
                 f"(dashed = each farm's own packing axis; red = fitted second harmonic)",
                 fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / f"wake_direction_{REGION}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
