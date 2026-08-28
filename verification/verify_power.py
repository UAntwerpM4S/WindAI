#!/usr/bin/env python3
"""Score wind power forecasts against the ENTSO-E/Elexon observations, by lead time.

Every run yields up to two power forecasts of the same quantity, scored on one identical sample:

  DIRECT  the model's own `capacityfactor`, mapped back to farms -- the adjoint of
          build_power.py's capacity-weighted distribution:
              P(farm,t) = SUM_cell capacity(farm's turbines in cell) * CF(cell,t)
          Only runs that carry the variable get this line.

  CURVE   the classical baseline: forecast ws100 at the farm (capacity-weighted over its cells)
          pushed through that farm's own aggregate specs power curve, then AVERAGED over the
          observation's own window:
              P(farm,t) = 1/2 [ A(ws100(t)) + A(ws100(t+3h)) ]
          power_obs at t is the MEAN over [t, t+3h), so DIRECT is trained on a window mean while
          an instantaneous curve is not -- grading a snapshot against an average charges an error
          that is bookkeeping, not skill. Averaging the POWERS (not the winds) removes it: the
          curve is cubic on the ramp, so A(mean ws) != mean A(ws). Every run gets this line, so a
          weather-only run is scored on equal terms.

PER_FARM chooses what is scored. False: the summed regional total, the operationally relevant
quantity, and a case counts only when EVERY farm reports (a partial sum is not a known total).
True: each farm on its own, scored whenever THAT farm reports -- so the farms have different
sample sizes, which is printed.

Metric is MAE as % of capacity (the unit's own). BIAS is printed too: the idealised specs curve
ignores wake losses so it should over-predict at high wind, and whether DIRECT removes that is a
testable claim MAE alone cannot show.

REGIMES splits by wind regime, binning on the OBSERVED power converted to an equivalent wind
speed through the unit's own power curve -- truth-conditioned, labelled in m/s to match
verify_weather.py. Above rated the curve is flat, so the top bin is "at or above rated" and
cannot be subdivided; that is a property of power, not of the binning.

CAVEAT the regime table cannot settle on its own: truth-conditioned bins reward an
UNDER-dispersive forecast in the middle bins and punish it in the tails, with no difference in
skill. The direct head sits near sigma_p/sigma_o 0.67 and the un-smoothed specs curve near 1.11,
so part of any middle-bin margin is that gap rather than accuracy. Read it as realised accuracy
(which it is), not as evidence of where skill lives.

Figures: one PNG (one per regime when PER_FARM and REGIMES are both on -- farms x regimes does
not fit a readable single figure). Every number plotted is also printed.
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
REGION   = "BE"              # "BE" | "UK" | "all"
SEASON   = "all"             # "all" | "DJF" | "MAM" | "JJA" | "SON"  -- filters on INIT month
REGIMES  = False             # split by wind regime
REGIME_BY = "cerra-ws"       # "cerra-ws": bin on CERRA truth ws100 at the unit's cells.
                             # "obs-cf"  : bin on OBSERVED power through the unit's own curve.
                             #   obs-cf cannot separate winds above rated -- they all give the
                             #   same power -- so any farm already at rated by the top edge gets
                             #   an EMPTY top bin. cerra-ws has no such blind spot.
PER_FARM = False             # False: the summed regional total. True: one series per farm.

FORECAST_DIRS = {
    "RegularWeather":     Path("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"),
    "VanillaCapacityGT":  Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaCapacityGT"),
    "HighCapacityGT":     Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
}

WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")   # farms/turbines/obs/specs live here
TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
OUT_DIR    = Path("DistrFigures")

CF_VAR = "capacityfactor"
WS_VAR = "ws100"

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))
OBS_STEP_H = 3               # the observation window, and the forecast step

REGIME_WS_EDGES = [4.5, 8.0, 12.0]               # m/s; converted to CF through the unit's curve
REGIME_LABELS   = ["0-4.5", "4.5-8", "8-12", "12+"]
# ======================================================================

SEASONS = {"all": None, "DJF": {12, 1, 2}, "MAM": {3, 4, 5},
           "JJA": {6, 7, 8}, "SON": {9, 10, 11}}
FORECAST_RE = re.compile(r"forecast_(\d{14})")
FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")
METHOD_LABEL = {"direct": "direct (capacity factor)", "curve": "power curve (window mean)"}
STYLE = {"direct": "-", "curve": "--"}


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def parse_init(path):
    return pd.to_datetime(FORECAST_RE.search(path.name).group(1),
                          format="%Y%m%d%H%M%S", utc=True)


def build_reconstruction(fc_lat, fc_lon, turbines, farms):
    """cell_idx (forecast cells holding turbines), G[f,j] = capacity of farm f in cell j."""
    coslat = np.cos(np.radians(float(fc_lat.mean())))
    tree = cKDTree(np.c_[to_180(fc_lon) * coslat, fc_lat])
    _, cell = tree.query(np.c_[to_180(turbines["longitude"]) * coslat,
                               turbines["latitude"].to_numpy()], k=1)
    t = turbines.assign(cell=cell.astype(int))
    cell_idx = np.sort(t["cell"].unique())
    cpos = {int(c): j for j, c in enumerate(cell_idx)}
    fpos = {f: i for i, f in enumerate(farms)}
    G = np.zeros((len(farms), cell_idx.size))
    for (farm, c), cap in t.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[fpos[farm], cpos[int(c)]] = cap
    return cell_idx, G


def turbine_power(ws, cut_in, rated_ws, cut_out, rated_mw):
    """One turbine: 0 below cut-in, cubic ramp to rated, flat at rated, 0 above cut-out."""
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws)
    ramp = (ws >= cut_in) & (ws < rated_ws)
    out[ramp] = rated_mw * (ws[ramp] ** 3 - cut_in ** 3) / (rated_ws ** 3 - cut_in ** 3)
    out[(ws >= rated_ws) & (ws < cut_out)] = rated_mw
    return out


def build_farm_curves(farms_df, specs, farms):
    """farm -> callable(ws) -> MW, the fleet's summed curve rescaled to the nameplate."""
    curves, meta = {}, farms_df.set_index("farm")
    for farm in farms:
        parts = []
        for chunk in str(meta.loc[farm, "fleet"]).split(";"):
            m = FLEET_RE.match(chunk)
            if not m:
                raise SystemExit(f"{farm}: cannot parse fleet entry {chunk!r}")
            if m.group(2) not in specs.index:
                raise SystemExit(f"{farm}: turbine type {m.group(2)!r} not in turbine_specs.csv")
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
    if not farms:
        raise SystemExit(f"no farms for REGION={REGION!r}; have {sorted(farms_df.region.unique())}")
    turbines = turbines[turbines.farm.isin(farms)]
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    curves = build_farm_curves(farms_df, specs, farms)
    print(f"Region {REGION}: {len(farms)} farms, {float(cap.sum()):.0f} MW")

    # what gets scored: the summed total, or each farm on its own
    if PER_FARM:
        units = [(f, float(cap[f]), np.array([i])) for i, f in enumerate(farms)]
    else:
        units = [(f"TOTAL {REGION}", float(cap.sum()), np.arange(len(farms)))]
    U = len(units)
    print(f"Scoring {'each farm separately' if PER_FARM else 'the summed regional total'}"
          f" ({U} series)")

    # leads that can form the window; every method is held to the same set
    dt = pd.Timedelta(hours=OBS_STEP_H)
    leads = [lh for lh in LEAD_HOURS if lh + OBS_STEP_H <= max(LEAD_HOURS)]
    dropped = sorted(set(LEAD_HOURS) - set(leads))
    if dropped:
        print(f"Leads dropped (no window available): {dropped}")
    lpos = {lh: k for k, lh in enumerate(leads)}
    L = len(leads)

    months = SEASONS[SEASON]
    fmaps = {}
    for label, d in FORECAST_DIRS.items():
        m = {parse_init(f): f for f in sorted(d.glob("forecast_*.nc"))
             if INIT_START <= parse_init(f) <= INIT_END}
        if months:
            m = {i: f for i, f in m.items() if i.month in months}
        print(f"{label}: {len(m)} files")
        fmaps[label] = m
    inits = sorted(set.intersection(*(set(m) for m in fmaps.values())))
    if not inits:
        raise SystemExit("no init times common to all runs")
    print(f"Common inits: {len(inits)} | season {SEASON}")

    nreg = len(REGIME_LABELS) if REGIMES else 1
    ws_truth = {}                 # (valid time -> per-unit CERRA ws100), for REGIME_BY=cerra-ws
    edges = {}
    if REGIMES and REGIME_BY == "cerra-ws":
        # capacity-weighted CERRA ws100 over each unit's cells -- always defined, unlike a
        # capacity-factor threshold, which saturates above rated
        dz = xr.open_zarr(TRUTH_ZARR, consolidated=False)
        tv = list(dz.attrs["variables"])
        td = pd.to_datetime(dz["dates"].values).tz_localize("UTC")
        glat = np.asarray(dz["latitudes"]).ravel()
        glon = to_180(np.asarray(dz["longitudes"]).ravel())
        ck = np.cos(np.radians(float(glat.mean())))
        _, tc = cKDTree(np.c_[glon * ck, glat]).query(
            np.c_[to_180(turbines.longitude) * ck, turbines.latitude.to_numpy()], k=1)
        tt = turbines.assign(cell=tc.astype(int))
        tcells = np.sort(tt.cell.unique())
        tpos = {int(c): j for j, c in enumerate(tcells)}
        keep = np.where((td >= INIT_START) & (td <= INIT_END + pd.Timedelta(hours=max(LEAD_HOURS))))[0]
        wsc = dz["data"].isel(time=keep, variable=tv.index(WS_VAR),
                              ensemble=0).values[:, tcells].astype(np.float64)
        dz.close()
        Wt = np.zeros((U, tcells.size))
        for u, (uname, ucap, sel) in enumerate(units):
            sub = tt[tt.farm.isin([farms[i] for i in sel])]
            for (fm, c), mw in sub.groupby(["farm", "cell"])["capacity_mw"].sum().items():
                Wt[u, tpos[int(c)]] += mw
        Wt = Wt / Wt.sum(1, keepdims=True)
        ws_truth = {t: v for t, v in zip(td[keep], wsc @ Wt.T)}
        print(f"Regime binning: CERRA ws100 at each unit's cells, edges "
              f"{REGIME_WS_EDGES} m/s ({len(ws_truth)} truth times loaded)")
    elif REGIMES:
        for uname, ucap, sel in units:
            e = np.array([float(sum(curves[farms[i]](np.array([v])) for i in sel)[0]) / ucap
                          for v in REGIME_WS_EDGES])
            for i in range(1, len(e)):        # a flat curve would tie two thresholds
                e[i] = max(e[i], e[i - 1] + 1e-9)
            edges[uname] = e
            if e[-1] >= 0.999:
                print(f"  WARNING {uname}: already at rated by {REGIME_WS_EDGES[-1]} m/s, so the "
                      f"top bin needs CF >= {e[-1]:.4f} and will be nearly empty.")
        print("Regime edges (first unit): " +
              ", ".join(f"{v} m/s -> CF {c:.4f}"
                        for v, c in zip(REGIME_WS_EDGES, edges[units[0][0]])))

    methods = ["direct", "curve"]
    sae = {(r, m): np.zeros((U, L, nreg)) for r in fmaps for m in methods}
    sbias = {k: np.zeros((U, L, nreg)) for k in sae}
    n = {k: np.zeros((U, L, nreg)) for k in sae}
    has_direct = {r: False for r in fmaps}
    n_nan = {r: 0 for r in fmaps}
    recon = {}

    for label, fmap in fmaps.items():
        print(f"\nScoring {label} ...")
        for c, init in enumerate(inits):
            if c % 500 == 0:
                print(f"  {c}/{len(inits)}", flush=True)
            with xr.open_dataset(fmap[init]) as ds:
                la_, lo_ = ds["latitude"].values, ds["longitude"].values
                key = (la_.size, round(float(la_[0]), 4), round(float(la_[-1]), 4),
                       round(float(lo_[0]), 4), round(float(lo_[-1]), 4))
                if key not in recon:
                    recon[key] = build_reconstruction(la_, lo_, turbines, farms)
                    print(f"  grid {key[0]} cells -> {recon[key][0].size} farm cells")
                cell_idx, G = recon[key]
                ftimes = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
                ws = ds[WS_VAR].values[:, cell_idx]
                cf = ds[CF_VAR].values[:, cell_idx] if CF_VAR in ds else None

            has_direct[label] |= cf is not None
            t2i = {t: j for j, t in enumerate(ftimes)}
            w = G / G.sum(1, keepdims=True)
            ws_farm = ws @ w.T                                        # (T, F) capacity-weighted
            p_curve = np.column_stack([curves[f](ws_farm[:, i])       # (T, F) MW
                                       for i, f in enumerate(farms)])
            p_direct = cf @ G.T if cf is not None else None

            for lh in leads:
                vt = init + pd.Timedelta(hours=lh)
                nxt = t2i.get(vt + dt)
                if vt not in t2i or nxt is None or vt not in obs.index:
                    continue
                ptrue = obs.loc[vt, farms].to_numpy(float)

                pred = {"curve": 0.5 * (p_curve[t2i[vt]] + p_curve[nxt])}
                if p_direct is not None:
                    pred["direct"] = p_direct[t2i[vt]]

                k = lpos[lh]
                nan_here = False
                for u, (uname, ucap, sel) in enumerate(units):
                    pt = ptrue[sel]
                    if not np.isfinite(pt).all():   # a partial sum is not a known total
                        continue
                    # one sample for every method: a NaN anywhere drops the case from all of them
                    if not all(np.isfinite(pp[sel]).all() for pp in pred.values()):
                        nan_here = True          # count the CASE once, not once per unit
                        continue
                    if not REGIMES:
                        r = 0
                    elif REGIME_BY == "cerra-ws":
                        wv = ws_truth.get(vt)
                        if wv is None:
                            continue
                        r = int(np.digitize(wv[u], REGIME_WS_EDGES))
                    else:
                        r = int(np.digitize(pt.sum() / ucap, edges[uname]))
                    for m, pp in pred.items():
                        e = pp[sel].sum() - pt.sum()
                        sae[(label, m)][u, k, r] += abs(e)
                        sbias[(label, m)][u, k, r] += e
                        n[(label, m)][u, k, r] += 1
                n_nan[label] += nan_here

    series = [(r, m) for r in fmaps for m in methods
              if not (m == "direct" and not has_direct[r]) and n[(r, m)].sum() > 0]
    if not series:
        raise SystemExit("nothing scored -- check that forecast valid times overlap power_obs")

    def stat(acc, key, u, r=None):
        s = acc[key][u, :, r] if r is not None else acc[key][u].sum(1)
        c = n[key][u, :, r] if r is not None else n[key][u].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s / c

    print("\nScored cases per lead (methods within a series must tie exactly):")
    for r in fmaps:
        got = [n[(r, m)] for m in methods if (r, m) in series]
        if got:
            tied = len({tuple(g.sum(2).ravel()) for g in got}) == 1
            tot = got[0].sum(2)
            print(f"  {r:22s} {int(tot.min()):5d}-{int(tot.max()):<5d} per series-lead | "
                  f"methods tied: {tied}"
                  + (f" | {n_nan[r]} case(s) dropped on NaN" if n_nan[r] else ""))

    def share(u, r_i):
        """(cases, % of that unit's record) in regime r_i -- how much of the sample this bin is."""
        k0 = series[0]
        tot = n[k0][u].sum()
        c = n[k0][u, :, r_i].sum()
        return c, (100.0 * c / tot if tot else np.nan)

    if REGIMES:
        by = "CERRA ws100 at the cells" if REGIME_BY == "cerra-ws" else "observed power via curve"
        print(f"\nHOW THE SAMPLE SPLITS ACROSS REGIMES   (binned on {by})")
        print(f"{'unit':22s} " + " ".join(f"{l:>16s}" for l in REGIME_LABELS))
        for u, (uname, ucap, sel) in enumerate(units):
            print(f"{uname:22s} " + " ".join(f"{int(c):7d} ({pc:4.1f}%)"
                                             for c, pc in (share(u, i) for i in range(nreg))))
        print("  With REGIME_BY='obs-cf' a 0.0% top bin is expected wherever a farm is already at")
        print("  rated by the top edge: all winds above rated give the same power, so no observed")
        print("  value can land there. Switch to 'cerra-ws' to bin on the wind itself.")

    lab = {(r, m): f"{r} / {METHOD_LABEL[m]}" for r, m in series}
    wid = max(len(v) for v in lab.values()) + 1
    hdr = f"{'run / method':{wid}s} " + " ".join(f"{lh:>6d}h" for lh in leads)
    reg_range = range(nreg) if REGIMES else [None]

    for u, (uname, ucap, sel) in enumerate(units):
        print(f"\n{'='*len(hdr)}\n{uname} — MAE as % of {ucap:.0f} MW  (season {SEASON})"
              f"\n{'='*len(hdr)}")
        for r_i in reg_range:
            if r_i is not None:
                cnt = max(n[k][u, :, r_i].max() for k in series)
                print(f"\n  regime {REGIME_LABELS[r_i]} m/s  (up to {cnt:.0f} cases per lead)")
            print(hdr)
            for k in series:
                print(f"{lab[k]:{wid}s} " +
                      " ".join(f"{v:7.2f}" for v in 100.0 * stat(sae, k, u, r_i) / ucap))
            print(f"{'  bias [MW]':{wid}s}")
            for k in series:
                print(f"{lab[k]:{wid}s} " +
                      " ".join(f"{v:+7.1f}" for v in stat(sbias, k, u, r_i)))

    # ---------------- figures ----------------
    colors = {r: plt.cm.tab10.colors[i % 10] for i, r in enumerate(fmaps)}
    handles = [plt.Line2D([], [], color=colors[r], ls=STYLE[m], marker="o", ms=4,
                          label=lab[(r, m)]) for r, m in series]
    base = f"{REGION}_{SEASON}" + ("_perfarm" if PER_FARM else "") + \
           (f"_{REGIME_BY}" if REGIMES else "")

    def panel(ax, u, ucap, r_i, title):
        for k in series:
            ax.plot(leads, 100.0 * stat(sae, k, u, r_i) / ucap, STYLE[k[1]],
                    color=colors[k[0]], lw=1.5, marker="o", ms=3)
        ax.set_title(title, fontsize=10)
        ax.grid(True, ls="--", alpha=0.5)
        ax.set_xticks(leads)

    def grid_fig(r_i, suffix, sup):
        ncol = int(np.ceil(np.sqrt(U)))
        nrow = int(np.ceil(U / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                                 sharex=True, squeeze=False)
        for u, (uname, ucap, sel) in enumerate(units):
            t_ = f"{uname} ({ucap:.0f} MW)"
            if r_i is not None:
                c_, pc_ = share(u, r_i)
                t_ += f"\n{int(c_)} cases — {pc_:.1f}% of its record"
            panel(axes[u // ncol][u % ncol], u, ucap, r_i, t_)
        for ax in axes.ravel()[U:]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Lead time [h]")
        for row in axes:
            row[0].set_ylabel("MAE [% of capacity]")
        axes[0][0].legend(handles=handles, fontsize=7, framealpha=0.8)
        fig.suptitle(sup, fontsize=12)
        fig.tight_layout()
        out = OUT_DIR / f"power_mae_{base}{suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")

    stamp = f"{len(inits)} inits, season {SEASON}"
    print()
    if PER_FARM and REGIMES:
        # farms x regimes does not fit one readable figure: one PNG per regime
        for r_i, rlab in enumerate(REGIME_LABELS):
            byl = "CERRA wind" if REGIME_BY == "cerra-ws" else "observed power"
            grid_fig(r_i, f"_regime{r_i}",
                     f"{REGION} per-farm power MAE — {rlab} m/s by {byl} ({stamp})")
    elif PER_FARM:
        grid_fig(None, "", f"{REGION} per-farm power MAE ({stamp})")
    elif REGIMES:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for r_i, ax in enumerate(axes.ravel()):
            c_, pc_ = share(0, r_i)
            panel(ax, 0, units[0][1], r_i,
                  f"{REGIME_LABELS[r_i]} m/s — {int(c_)} cases ({pc_:.1f}% of the record)")
        for ax in axes[1]:
            ax.set_xlabel("Lead time [h]")
        for ax in axes[:, 0]:
            ax.set_ylabel("MAE [% of capacity]")
        axes[0, 0].legend(handles=handles, fontsize=7, framealpha=0.8)
        byl = "CERRA wind" if REGIME_BY == "cerra-ws" else "observed power"
        fig.suptitle(f"{units[0][0]} power MAE by wind regime, binned on {byl} ({stamp})",
                     fontsize=12)
        fig.tight_layout()
        out = OUT_DIR / f"power_mae_{base}_regimes.png"
        fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved: {out}")
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        panel(ax, 0, units[0][1], None, "")
        ax.set(xlabel="Lead time [h]", ylabel="MAE [% of capacity]")
        ax.set_title(f"{units[0][0]} power MAE — {units[0][1]:.0f} MW, {stamp}", fontsize=12)
        ax.legend(handles=handles, fontsize=8, framealpha=0.8)
        fig.tight_layout()
        out = OUT_DIR / f"power_mae_{base}.png"
        fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved: {out}")


if __name__ == "__main__":
    main()
