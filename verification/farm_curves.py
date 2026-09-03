#!/usr/bin/env python3
"""One farm power-curve builder, shared by every verify_*.py, in three flavours.

The baseline every script scored against used to be a hard-cornered cubic built from
turbine_specs.csv at CERRA ws100. Two things were wrong with it:

  * the cubic is not a turbine. Refitting three of its parameters to the observed curves removed
    87% of the spec-vs-observed gap, and the fitted rated wind speeds clustered at 13.1-14.6 m/s
    against a datasheet spread of 12.0-14.9 -- so most of the between-farm "wake loss" was the CSV.
  * every farm was fed 100 m wind. Hub heights run 71-109 m. Northwind at 71 m was handed ~4% too
    much wind, which on a cubic ramp is ~11% too much power.

MODES
  "cubic"  turbine_specs.csv through the cubic, at ws100.  The old baseline, kept so the change
                                                           can be quantified rather than assumed.
  "curve"  the real turbine's power curve at ws100.
  "hub"    the real curve, ws100 sheared to that turbine's OWN hub height.
  "calib"  the hub curve with a per-farm wind multiplier and plateau scale FITTED on a training
           period -- what an operator with production history would actually deploy. It is the
           strongest curve baseline available and the one a referee will ask for: raw curves
           over-predict these farms by ~20% in wind terms, and calibration removes that. Fit it
           on data BEFORE the scoring window or you are reporting an in-sample number.

Turbine type and hub height come from the EWW open database, matched to turbines.csv by nearest
coordinate. validate() checks every count and capacity against farms.csv before anything is built:
a silently mismatched fleet would move the baseline without moving any error message.

The curve files are PyWake GenericWindTurbine output, NOT manufacturer datasheets -- cut-in is
forced to 4.0 m/s for every turbine and TI=0.05 smoothing is baked in. Better than a cubic, still
a model. Say so in the methods.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")
SHEAR = 0.11               # offshore power-law exponent: ws(z) = ws100 * (z/100)**SHEAR
CAP_TOL = 0.001            # capacity agreement required between turbines.csv and farms.csv


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def load_curves(curve_dir):
    """file stem -> callable(ws) -> kW. Zero outside the tabulated range (cut-in and cut-out)."""
    out = {}
    for p in sorted(Path(curve_dir).glob("*.csv")):
        d = pd.read_csv(p)
        if not {"ws", "power"} <= set(d.columns):
            continue
        ws, kw = d["ws"].to_numpy(float), d["power"].to_numpy(float)
        out[p.stem] = lambda v, ws=ws, kw=kw: np.interp(
            np.asarray(v, float), ws, kw, left=0.0, right=0.0)
    if not out:
        raise SystemExit(f"no power-curve CSVs with (ws,power) columns found in {curve_dir}")
    return out


def attach_eww(turbines, eww_csv, country="Belg", max_km=1.0, quiet=False):
    """Add ttype / hub / comm to turbines.csv rows by nearest coordinate in the EWW database."""
    e = pd.read_csv(eww_csv, low_memory=False)
    need = {"latitude", "longitude", "country", "turbine_type", "hub_height"}
    if not need <= set(e.columns):
        raise SystemExit(f"{eww_csv}: missing {sorted(need - set(e.columns))} -- is this the EWW "
                         f"open database? columns are {sorted(e.columns)}")
    e = e[e.country.astype(str).str.startswith(country)].dropna(subset=["latitude", "longitude"])
    if e.empty:
        raise SystemExit(f"{eww_csv}: no rows with country starting {country!r}")
    ck = np.cos(np.radians(float(turbines.latitude.mean())))
    d, j = cKDTree(np.c_[to_180(e.longitude) * ck, e.latitude]).query(
        np.c_[to_180(turbines.longitude) * ck, turbines.latitude.to_numpy()], k=1)
    km = d * 111.32
    if (km > max_km).any():
        bad = turbines.assign(km=km).nlargest(3, "km")[["farm", "turbine", "km"]]
        raise SystemExit(f"{(km > max_km).sum()} turbine(s) more than {max_km} km from any EWW "
                         f"row -- wrong coordinates, wrong country, or a farm the database is "
                         f"missing:\n{bad.to_string(index=False)}")
    if not quiet:
        print(f"EWW match: {len(turbines)} turbines, median {np.median(km)*1000:.0f} m, "
              f"max {km.max()*1000:.0f} m")
    return turbines.assign(
        ttype=e.turbine_type.to_numpy()[j], hub=e.hub_height.to_numpy()[j].astype(float),
        comm=(e.commissioning_date.to_numpy()[j] if "commissioning_date" in e else ""))


def validate(farms, farms_df, turbines, specs, curves=None, strict=True):
    """Every count and capacity must line up before a baseline built on them means anything.

    Checks per farm: turbine count against farms.csv n_turbines, summed turbine capacity against
    farms.csv capacity_mw, the fleet string resolving against turbine_specs.csv, and -- when the
    EWW columns are present -- that every matched turbine type has a power curve.
    """
    meta = farms_df.set_index("farm")
    have_eww = "ttype" in turbines
    bad = []
    print(f"{'farm':14s} {'turbines.csv':>12s} {'farms.csv':>10s} {'MW turb':>9s} "
          f"{'MW farms':>9s}" + (f" {'hub m':>6s} {'types':>34s}" if have_eww else ""))
    for f in farms:
        s = turbines[turbines.farm == f]
        n_t, n_f = len(s), int(meta.loc[f, "n_turbines"])
        mw_t, mw_f = float(s.capacity_mw.sum()), float(meta.loc[f, "capacity_mw"])
        line = f"{f:14s} {n_t:12d} {n_f:10d} {mw_t:8.1f} {mw_f:9.1f}"
        if have_eww:
            tv = s.ttype.value_counts()
            line += (f" {float(np.average(s.hub, weights=s.capacity_mw)):6.0f} "
                     + " ".join(f"{k}x{v}" for k, v in tv.items())[:34])
        print(line)
        if n_t != n_f:
            bad.append(f"{f}: {n_t} turbines in turbines.csv, {n_f} in farms.csv")
        if abs(mw_t - mw_f) / mw_f > CAP_TOL:
            bad.append(f"{f}: {mw_t:.1f} MW in turbines.csv, {mw_f:.1f} MW in farms.csv")
        for chunk in str(meta.loc[f, "fleet"]).split(";"):
            m = FLEET_RE.match(chunk)
            if not m:
                bad.append(f"{f}: unparseable fleet entry {chunk!r}")
            elif m.group(2) not in specs.index:
                bad.append(f"{f}: {m.group(2)!r} not in turbine_specs.csv")
        if have_eww:
            n_fleet = sum(int(FLEET_RE.match(c).group(1))
                          for c in str(meta.loc[f, "fleet"]).split(";") if FLEET_RE.match(c))
            if n_fleet != n_t:
                bad.append(f"{f}: fleet string totals {n_fleet} turbines, turbines.csv has {n_t}")
            if curves is not None:
                for tt in s.ttype.unique():
                    k = str(tt).replace(" ", "_").replace("/", "_")
                    if k not in curves:
                        bad.append(f"{f}: no power curve for {tt!r} (expected {k}.csv)")
    tot_t, tot_f = len(turbines[turbines.farm.isin(farms)]), int(meta.loc[farms, "n_turbines"].sum())
    print(f"{'TOTAL':14s} {tot_t:12d} {tot_f:10d} "
          f"{float(turbines[turbines.farm.isin(farms)].capacity_mw.sum()):8.1f} "
          f"{float(meta.loc[farms, 'capacity_mw'].sum()):9.1f}")
    if bad:
        msg = "fleet/capacity mismatches:\n  " + "\n  ".join(bad)
        if strict:
            raise SystemExit(msg + "\n  rerun farm_metadata.py, or set strict=False to proceed")
        print("  WARNING " + msg)
    return not bad


def _cubic(ws, cut_in, rated_ws, cut_out, rated_mw):
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws)
    r = (ws >= cut_in) & (ws < rated_ws)
    out[r] = rated_mw * (ws[r] ** 3 - cut_in ** 3) / (rated_ws ** 3 - cut_in ** 3)
    out[(ws >= rated_ws) & (ws < cut_out)] = rated_mw
    return out


def build(farms, farms_df, turbines, specs, mode, curves=None, shear=SHEAR, calib=None):
    """farm -> callable(ws100) -> MW, rescaled so the plateau is the farm's nameplate."""
    if mode not in ("cubic", "curve", "hub", "calib"):
        raise SystemExit(f"mode must be cubic|curve|hub|calib, got {mode!r}")
    if mode != "cubic" and ("ttype" not in turbines or curves is None):
        raise SystemExit("modes 'curve', 'hub' and 'calib' need attach_eww() and load_curves()")
    if mode == "calib":
        if calib is None:
            raise SystemExit("mode 'calib' needs calib={farm: (alpha, scale)} from calibrate()")
        base = build(farms, farms_df, turbines, specs, "hub", curves, shear)
        return {f: (lambda ws, b=base[f], a=calib[f][0], s=calib[f][1]:
                    s * b(a * np.asarray(ws, float))) for f in farms}

    meta, out = farms_df.set_index("farm"), {}
    probe = np.arange(0.0, 25.01, 0.05)
    for f in farms:
        if mode == "cubic":
            parts = [(int(m.group(1)), specs.loc[m.group(2)])
                     for m in (FLEET_RE.match(c) for c in str(meta.loc[f, "fleet"]).split(";"))]

            def raw(ws, parts=parts):
                return sum(n * _cubic(ws, float(s["cut_in_ms"]), float(s["rated_ws_ms"]),
                                      float(s["cut_out_ms"]), float(s["rated_power_mw"]))
                           for n, s in parts)
        else:
            parts = []
            for (tt, hub), g in turbines[turbines.farm == f].groupby(["ttype", "hub"]):
                key = str(tt).replace(" ", "_").replace("/", "_")
                z = (float(hub) / 100.0) ** shear if mode == "hub" else 1.0
                parts.append((len(g), curves[key], z))

            def raw(ws, parts=parts):                       # kW
                ws = np.asarray(ws, float)
                return sum(n * fn(ws * z) for n, fn, z in parts)

        top = float(np.max(raw(probe)))
        if top <= 0:
            raise SystemExit(f"{f}: curve is zero everywhere -- check the fleet or curve files")
        out[f] = lambda ws, raw=raw, k=float(meta.loc[f, "capacity_mw"]) / top: raw(ws) * k
    return out


def rated_ws(curve, frac=0.99, lo=4.0, hi=25.0, step=0.05):
    """Wind at which a farm curve first reaches `frac` of its plateau. Replaces the datasheet
    rated_ws_ms as a clamp trigger: it is what the curve does, not what a CSV claims."""
    ws = np.arange(lo, hi, step)
    p = np.asarray(curve(ws), float)
    return float(ws[np.argmax(p >= frac * p.max())])


def farm_truth_wind(farms, turbines, truth_zarr, start, end, ws_var="ws100", obs_step_h=3):
    """(times, window-mean CERRA wind per farm). The observation at t is the mean over
    [t, t+obs_step_h), so the curve must be read at the mean wind over the same window."""
    ds = xr.open_zarr(truth_zarr, consolidated=False)
    tvars = list(ds.attrs["variables"])
    td = pd.to_datetime(ds["dates"].values).tz_localize("UTC")
    glat = np.asarray(ds["latitudes"]).ravel()
    glon = to_180(np.asarray(ds["longitudes"]).ravel())
    ck = np.cos(np.radians(float(glat.mean())))
    _, tc = cKDTree(np.c_[glon * ck, glat]).query(
        np.c_[to_180(turbines.longitude) * ck, turbines.latitude.to_numpy()], k=1)
    t = turbines.assign(cell=tc.astype(int))
    cells = np.sort(t.cell.unique())
    cpos = {int(c): j for j, c in enumerate(cells)}
    sel = np.where((td >= start) & (td <= end))[0]
    if sel.size == 0:
        raise SystemExit(f"no truth times in {start}..{end}")
    ws_c = ds["data"].isel(time=sel, variable=tvars.index(ws_var),
                           ensemble=0).values[:, cells].astype(np.float64)
    ds.close()
    times = td[sel]
    G = np.zeros((len(farms), cells.size))
    for (fm, c), mw in t.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[farms.index(fm), cpos[int(c)]] = mw
    ws_f = ws_c @ (G / G.sum(1, keepdims=True)).T
    nxt = np.searchsorted(times, times + pd.Timedelta(hours=obs_step_h))
    ok = (nxt < len(times)) & (np.abs((times[np.minimum(nxt, len(times) - 1)] - times)
                                      .total_seconds() - 3600 * obs_step_h) < 1)
    out = np.full_like(ws_f, np.nan)
    out[ok] = 0.5 * (ws_f[ok] + ws_f[nxt[ok]])
    return times, out


def calibrate(farms, farms_df, turbines, specs, curves, obs, truth_zarr, start, end,
              ws_edges=None, min_bin=30, fit_range=(0.0, 22.0),
              bounds=((0.70, 0.50), (1.15, 1.15)), shear=SHEAR, quiet=False):
    """Per-farm (alpha, scale) for the "calib" mode, fitted on start..end.

    alpha is a wind multiplier: the farm behaves like its turbines seeing alpha*ws. scale is the
    plateau as a fraction of nameplate. Both are fitted to the farm's OWN binned observed curve
    against CERRA truth wind -- the conversion an operator would calibrate from production
    history, then apply to forecast wind. Fitting on forecast wind instead would absorb forecast
    bias and make this a statistical post-processor rather than a power curve.
    """
    if ws_edges is None:
        ws_edges = np.arange(0.0, 25.1, 0.5)
    mid = 0.5 * (ws_edges[:-1] + ws_edges[1:])
    base = build(farms, farms_df, turbines, specs, "hub", curves, shear)
    times, ws_win = farm_truth_wind(farms, turbines, truth_zarr, start, end)
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]

    out = {}
    if not quiet:
        print(f"\nCalibrating on {start.date()}..{end.date()}  ({len(times)} truth times)")
        print(f"{'farm':14s} {'alpha':>7s} {'scale':>7s} {'bins':>5s} {'RMS raw':>9s} "
              f"{'RMS calib':>10s}")
    for i, f in enumerate(farms):
        o = obs[f].reindex(times).to_numpy(float)
        m = np.isfinite(o) & np.isfinite(ws_win[:, i])
        idx = np.digitize(ws_win[m, i], ws_edges) - 1
        y = 100.0 * o[m] / float(cap[f])
        med = np.full(mid.size, np.nan); cnt = np.zeros(mid.size)
        for b in range(mid.size):
            k = idx == b
            cnt[b] = k.sum()
            if cnt[b] >= min_bin:
                med[b] = np.median(y[k])
        use = np.isfinite(med) & (mid >= fit_range[0]) & (mid <= fit_range[1])
        if use.sum() < 6:
            raise SystemExit(f"{f}: only {use.sum()} usable bins in {start.date()}..{end.date()} "
                             f"-- widen the calibration window")
        x, yy, w = mid[use], med[use], cnt[use]
        pct = lambda ws, a=1.0, s=1.0: 100.0 * s * base[f](a * ws) / float(cap[f])
        sw = np.sqrt(w / w.sum())
        fit = least_squares(lambda p: sw * (pct(x, *p) - yy), x0=[1.0, 1.0],
                            bounds=(list(bounds[0]), list(bounds[1])))
        a, s_ = fit.x
        out[f] = (float(a), float(s_))
        if not quiet:
            rms = lambda v: float(np.sqrt(np.sum(w * v ** 2) / w.sum()))
            print(f"{f:14s} {a:7.3f} {s_:7.3f} {int(use.sum()):5d} "
                  f"{rms(pct(x) - yy):8.2f}% {rms(pct(x, a, s_) - yy):9.2f}%")
    return out
