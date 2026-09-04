#!/usr/bin/env python3
"""Farm power curves: the manufacturer curve, and the farm's own measured one.

Two modes, both returning callable(ws) -> MW for each farm:

  "specs"      the manufacturer curve. Each turbine is 0 below cut-in, a cubic ramp to rated,
               flat to cut-out, summed over the fleet in farms.csv and rescaled to nameplate.
               Built from turbine_specs.csv alone -- nothing observed, nothing fitted.

  "empirical"  the farm's MEASURED power curve, derived from its own production history by the
               METHOD OF BINS: bin observed power on wind speed, take the central value of each
               bin, and interpolate between bin centres. This is the standard procedure for
               obtaining a measured power curve (IEC 61400-12-1), applied to a whole farm rather
               than one turbine, and using reanalysis wind in place of a met mast. Deriving a
               site-specific curve this way instead of trusting the manufacturer's is routine in
               wind power forecasting, because the measured curve carries the farm's real wake,
               availability and electrical losses -- which no datasheet does.

               Two departures from the standard worth stating in a methods section: bins are
               summarised by the MEDIAN by default (BIN_STAT), so a handful of curtailed hours
               cannot drag a bin, and the wind is CERRA ws100 at the farm's cells rather than a
               hub-height mast measurement.

Derive it on data BEFORE the scoring window or the baseline is in-sample and means nothing.
empirical() uses CERRA truth wind, not forecast wind: it is measuring the farm's CONVERSION, to
be applied to forecasts afterwards. Using forecast wind would fold forecast bias into the curve
and make it a statistical post-processor rather than a power curve.

validate() checks every count and capacity against farms.csv before anything is built -- a
silently mismatched fleet would move the baseline without moving any error message.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")
CAP_TOL = 0.001            # capacity agreement required between turbines.csv and farms.csv
BIN_STAT = "median"        # how each wind bin is summarised: "median" or "mean"


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def fleet(farms_df, farm, specs):
    """[(count, spec row)] for a farm, e.g. '55x Vestas-3-V90; 1x AlstomHaliade-6-150'."""
    out = []
    for chunk in str(farms_df.set_index("farm").loc[farm, "fleet"]).split(";"):
        m = FLEET_RE.match(chunk)
        if not m:
            raise SystemExit(f"{farm}: cannot parse fleet entry {chunk!r}")
        if m.group(2) not in specs.index:
            raise SystemExit(f"{farm}: turbine type {m.group(2)!r} not in turbine_specs.csv")
        out.append((int(m.group(1)), specs.loc[m.group(2)]))
    return out


def validate(farms, farms_df, turbines, specs, quiet=False):
    """Counts and capacities must line up before a baseline built on them means anything."""
    meta, bad = farms_df.set_index("farm"), []
    if not quiet:
        print(f"{'farm':14s} {'turbines.csv':>12s} {'farms.csv':>10s} {'MW turb':>9s} "
              f"{'MW farms':>9s}")
    for f in farms:
        s = turbines[turbines.farm == f]
        n_t, n_f = len(s), int(meta.loc[f, "n_turbines"])
        mw_t, mw_f = float(s.capacity_mw.sum()), float(meta.loc[f, "capacity_mw"])
        if not quiet:
            print(f"{f:14s} {n_t:12d} {n_f:10d} {mw_t:8.1f} {mw_f:9.1f}")
        if n_t != n_f:
            bad.append(f"{f}: {n_t} turbines in turbines.csv, {n_f} in farms.csv")
        if abs(mw_t - mw_f) / mw_f > CAP_TOL:
            bad.append(f"{f}: {mw_t:.1f} MW in turbines.csv, {mw_f:.1f} MW in farms.csv")
        n_fleet = sum(c for c, _ in fleet(farms_df, f, specs))
        if n_fleet != n_t:
            bad.append(f"{f}: fleet string totals {n_fleet} turbines, turbines.csv has {n_t}")
    if not quiet:
        sub = turbines[turbines.farm.isin(farms)]
        print(f"{'TOTAL':14s} {len(sub):12d} {int(meta.loc[farms, 'n_turbines'].sum()):10d} "
              f"{float(sub.capacity_mw.sum()):8.1f} "
              f"{float(meta.loc[farms, 'capacity_mw'].sum()):9.1f}")
    if bad:
        raise SystemExit("fleet/capacity mismatches -- rerun farm_metadata.py:\n  "
                         + "\n  ".join(bad))
    return True


def _turbine(ws, cut_in, rated_ws, cut_out, rated_mw):
    """One turbine: 0 below cut-in, cubic ramp to rated, flat at rated, 0 above cut-out."""
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws)
    r = (ws >= cut_in) & (ws < rated_ws)
    out[r] = rated_mw * (ws[r] ** 3 - cut_in ** 3) / (rated_ws ** 3 - cut_in ** 3)
    out[(ws >= rated_ws) & (ws < cut_out)] = rated_mw
    return out


def build_specs(farms, farms_df, specs):
    """farm -> callable(ws) -> MW, the fleet's summed manufacturer curve, scaled to nameplate."""
    meta, out = farms_df.set_index("farm"), {}
    for f in farms:
        parts = fleet(farms_df, f, specs)
        k = float(meta.loc[f, "capacity_mw"]) / sum(c * float(s["rated_power_mw"])
                                                    for c, s in parts)

        def raw(ws, parts=parts, k=k):
            return k * sum(c * _turbine(ws, float(s["cut_in_ms"]), float(s["rated_ws_ms"]),
                                        float(s["cut_out_ms"]), float(s["rated_power_mw"]))
                           for c, s in parts)
        out[f] = raw
    return out


def farm_truth_wind(farms, turbines, truth_zarr, start, end, ws_var="ws100", obs_step_h=3):
    """(times, window-mean CERRA wind per farm). power_obs at t is the mean over [t, t+3h), so
    the curve has to be read at the mean wind over that same window."""
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


def empirical(farms, farms_df, turbines, obs, truth_zarr, start, end,
              ws_edges=None, min_bin=30, stat=None, quiet=False):
    """The farm's MEASURED power curve, by the method of bins, over start..end.

    Returns farm -> callable(ws) -> MW. Bins with fewer than min_bin cases are not used; the
    curve interpolates between the centres of the bins that survive, is zero below the lowest,
    and holds the last value above the highest (a farm at rated stays at rated).
    """
    stat = stat or BIN_STAT
    if stat not in ("median", "mean"):
        raise SystemExit(f"stat must be median|mean, got {stat!r}")
    if ws_edges is None:
        ws_edges = np.arange(0.0, 25.1, 0.5)
    mid = 0.5 * (ws_edges[:-1] + ws_edges[1:])
    times, ws_win = farm_truth_wind(farms, turbines, truth_zarr, start, end)
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    agg = np.median if stat == "median" else np.mean

    out = {}
    if not quiet:
        print(f"\nMeasured power curves, method of bins on {start.date()}..{end.date()} "
              f"({len(times)} truth times, {stat} of each {ws_edges[1]-ws_edges[0]:.1f} m/s bin)")
        print(f"{'farm':14s} {'cases':>8s} {'bins':>5s} {'ws range':>13s} {'plateau':>9s} "
              f"{'of nameplate':>13s}")
    for i, f in enumerate(farms):
        o = obs[f].reindex(times).to_numpy(float)
        m = np.isfinite(o) & np.isfinite(ws_win[:, i])
        idx = np.digitize(ws_win[m, i], ws_edges) - 1
        y = o[m]
        val, cnt = np.full(mid.size, np.nan), np.zeros(mid.size)
        for b in range(mid.size):
            k = idx == b
            cnt[b] = k.sum()
            if cnt[b] >= min_bin:
                val[b] = agg(y[k])
        use = np.isfinite(val)
        if use.sum() < 6:
            raise SystemExit(f"{f}: only {use.sum()} bins with >= {min_bin} cases in "
                             f"{start.date()}..{end.date()} -- widen the window")
        xs, ys = mid[use], val[use]
        # below the first populated bin the farm is not producing; above the last it stays where
        # it was -- extrapolating a trend into winds the record does not cover would be invented
        out[f] = lambda ws, xs=xs, ys=ys: np.interp(
            np.asarray(ws, float), xs, ys, left=0.0, right=ys[-1])
        if not quiet:
            print(f"{f:14s} {int(m.sum()):8d} {int(use.sum()):5d} "
                  f"{xs[0]:5.1f}-{xs[-1]:5.1f} {ys.max():8.1f}MW "
                  f"{100*ys.max()/float(cap[f]):12.1f}%")
    return out
