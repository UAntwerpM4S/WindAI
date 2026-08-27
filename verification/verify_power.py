#!/usr/bin/env python3
"""Score regional wind power forecasts against the ENTSO-E/Elexon observations, by lead time.

Every run yields up to two power forecasts of the same regional total, scored on one identical
sample:

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

Metric is MAE as % of total capacity. BIAS is printed too: the idealised specs curve ignores wake
losses so it should over-predict at high wind, and whether DIRECT removes that is a testable
claim MAE alone cannot show.

REGIMES splits by wind regime, binning on the OBSERVED total power converted to an equivalent
wind speed through the fleet curve -- truth-conditioned, labelled in m/s to match
verify_weather.py. Above rated the curve is flat, so the top bin is "at or above rated" and
cannot be subdivided; that is a property of power, not of the binning.

CURVE_CONTROL adds the BACKWARD window 1/2[A(ws(t-3h)) + A(ws(t))] on the same leads as the
forward one. It separates the two things the window does -- aligning with the observation, and
smoothing two forecasts together. If the backward control gains as much as the forward window,
the gain was smoothing, not alignment.

Writes one PNG; prints every number it plots.
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
REGION  = "BE"               # "BE" | "UK" | "all"
SEASON  = "all"              # "all" | "DJF" | "MAM" | "JJA" | "SON"  -- filters on INIT month
REGIMES = False              # split by observed wind regime
CURVE_CONTROL = False        # also score the BACKWARD window as an alignment control

FORECAST_DIRS = {
    "RegularWeather":     Path("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"),
    "VanillaCapacityGT":  Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaCapacityGT"),
    "HighCapacityGT":     Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
}

WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")   # farms/turbines/obs/specs live here
OUT_DIR    = Path("figures")

CF_VAR = "capacityfactor"
WS_VAR = "ws100"

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))
OBS_STEP_H = 3               # the observation window, and the forecast step

REGIME_WS_EDGES = [4.5, 8.0, 12.0]               # m/s; converted to CF through the fleet curve
REGIME_LABELS   = ["0-4.5", "4.5-8", "8-12", "12+"]
# ======================================================================

SEASONS = {"all": None, "DJF": {12, 1, 2}, "MAM": {3, 4, 5},
           "JJA": {6, 7, 8}, "SON": {9, 10, 11}}
FORECAST_RE = re.compile(r"forecast_(\d{14})")
FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")
METHOD_LABEL = {"direct": "direct (capacity factor)",
                "curve": "power curve (window mean)",
                "curve_back": "power curve (BACKWARD control)"}
STYLE = {"direct": "-", "curve": "--", "curve_back": ":"}


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
    cap_total = float(cap.sum())
    curves = build_farm_curves(farms_df, specs, farms)
    print(f"Region {REGION}: {len(farms)} farms, {cap_total:.0f} MW")

    # leads that can form the window(s); every method is held to the same set
    dt = pd.Timedelta(hours=OBS_STEP_H)
    leads = [lh for lh in LEAD_HOURS if lh + OBS_STEP_H <= max(LEAD_HOURS)]
    if CURVE_CONTROL:
        leads = [lh for lh in leads if lh - OBS_STEP_H >= min(LEAD_HOURS)]
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

    # regime thresholds on the OBSERVED total capacity factor, via the fleet curve
    nreg = len(REGIME_LABELS) if REGIMES else 1
    if REGIMES:
        fleet = lambda ws: sum(curves[f](ws) for f in farms)
        cf_edges = np.array([float(fleet(np.array([e]))[0]) / cap_total
                             for e in REGIME_WS_EDGES])
        for i in range(1, len(cf_edges)):          # a flat curve would tie two thresholds
            cf_edges[i] = max(cf_edges[i], cf_edges[i - 1] + 1e-9)
        print("Regime edges: " + ", ".join(f"{e} m/s -> CF {c:.4f}"
                                           for e, c in zip(REGIME_WS_EDGES, cf_edges)))
        if cf_edges[-1] >= 0.999:
            print(f"  WARNING: the fleet is already at rated by {REGIME_WS_EDGES[-1]} m/s, so the "
                  f"top bin needs CF >= {cf_edges[-1]:.4f} and will be nearly empty.")

    methods = ["direct", "curve"] + (["curve_back"] if CURVE_CONTROL else [])
    keys = [(r, m) for r in fmaps for m in methods]
    sae = {k: np.zeros((L, nreg)) for k in keys}
    sbias = {k: np.zeros((L, nreg)) for k in keys}
    n = {k: np.zeros((L, nreg)) for k in keys}
    farm_sae = {k: np.zeros((len(farms), L)) for k in keys}
    farm_n = {k: np.zeros((len(farms), L)) for k in keys}
    has_direct = {r: False for r in fmaps}
    n_nan = {r: 0 for r in fmaps}
    recon = {}

    for label, fmap in fmaps.items():
        print(f"\nScoring {label} ...")
        for c, init in enumerate(inits):
            if c % 500 == 0:
                print(f"  {c}/{len(inits)}", flush=True)
            with xr.open_dataset(fmap[init]) as ds:
                key = (ds.sizes.get("values", ds["latitude"].size),)
                if key not in recon:
                    recon[key] = build_reconstruction(ds["latitude"].values,
                                                      ds["longitude"].values, turbines, farms)
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
                if vt not in t2i or vt not in obs.index:
                    continue
                ptrue = obs.loc[vt, farms].to_numpy(float)
                if not np.isfinite(ptrue).all():          # a partial sum is not a known total
                    continue
                nxt, prv = t2i.get(vt + dt), t2i.get(vt - dt)
                if nxt is None or (CURVE_CONTROL and prv is None):
                    continue

                pred = {"curve": 0.5 * (p_curve[t2i[vt]] + p_curve[nxt])}
                if CURVE_CONTROL:
                    pred["curve_back"] = 0.5 * (p_curve[prv] + p_curve[t2i[vt]])
                if p_direct is not None:
                    pred["direct"] = p_direct[t2i[vt]]

                # one sample for every method: a NaN anywhere drops the case from all of them
                if not all(np.isfinite(pp).all() for pp in pred.values()):
                    n_nan[label] += 1
                    continue

                r = (int(np.digitize(ptrue.sum() / cap_total, cf_edges)) if REGIMES else 0)
                k = lpos[lh]
                for m, pp in pred.items():
                    e = pp.sum() - ptrue.sum()
                    sae[(label, m)][k, r] += abs(e)
                    sbias[(label, m)][k, r] += e
                    n[(label, m)][k, r] += 1
                    farm_sae[(label, m)][:, k] += np.abs(pp - ptrue)
                    farm_n[(label, m)][:, k] += 1

    series = [(r, m) for r in fmaps for m in methods
              if not (m == "direct" and not has_direct[r]) and n[(r, m)].sum() > 0]
    if not series:
        raise SystemExit("nothing scored -- check that forecast valid times overlap power_obs")

    def mae_pct(key, r=None):
        s = sae[key][:, r] if r is not None else sae[key].sum(1)
        c = n[key][:, r] if r is not None else n[key].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100.0 * (s / c) / cap_total

    def bias_mw(key, r=None):
        s = sbias[key][:, r] if r is not None else sbias[key].sum(1)
        c = n[key][:, r] if r is not None else n[key].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s / c

    print("\nScored cases per lead (methods within a run must tie exactly):")
    for r in fmaps:
        got = {m: n[(r, m)].sum(1) for m in methods if (r, m) in dict.fromkeys(series)}
        if got:
            same = len({tuple(v) for v in got.values()}) == 1
            print(f"  {r:22s} {int(list(got.values())[0].min()):5d}-"
                  f"{int(list(got.values())[0].max()):<5d} per lead | methods tied: {same}"
                  + (f" | {n_nan[r]} case(s) dropped on NaN" if n_nan[r] else ""))

    hdr = f"{'run / method':44s} " + " ".join(f"{lh:>6d}h" for lh in leads)
    print(f"\n{'='*len(hdr)}\nTOTAL {REGION} POWER — MAE as % of {cap_total:.0f} MW capacity"
          f"  (season {SEASON})\n{'='*len(hdr)}")
    print(hdr)
    for r, m in series:
        print(f"{r + ' / ' + METHOD_LABEL[m]:44s} " +
              " ".join(f"{v:7.2f}" for v in mae_pct((r, m))))
    print(f"\nBIAS [MW]  (specs curve ignores wakes -> expect it positive at high wind)")
    print(hdr)
    for r, m in series:
        print(f"{r + ' / ' + METHOD_LABEL[m]:44s} " +
              " ".join(f"{v:+7.1f}" for v in bias_mw((r, m))))

    if REGIMES:
        for r_i, rlab in enumerate(REGIME_LABELS):
            cnt = max(n[k][:, r_i].max() for k in series)
            print(f"\nMAE % of capacity  |  regime {rlab} m/s  (up to {cnt:.0f} cases per lead)")
            print(hdr)
            for r, m in series:
                print(f"{r + ' / ' + METHOD_LABEL[m]:44s} " +
                      " ".join(f"{v:7.2f}" for v in mae_pct((r, m), r_i)))
            print(f"{'  bias [MW]':44s}")
            for r, m in series:
                print(f"{r + ' / ' + METHOD_LABEL[m]:44s} " +
                      " ".join(f"{v:+7.1f}" for v in bias_mw((r, m), r_i)))

    print(f"\nPer-farm MAE as % of that farm's capacity (mean over leads)")
    print(f"{'farm':16s} {'cap MW':>8s} " +
          " ".join(f"{r[:10] + '/' + m[:4]:>16s}" for r, m in series))
    for i, f in enumerate(farms):
        cells = []
        for r, m in series:
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.nanmean(farm_sae[(r, m)][i] / farm_n[(r, m)][i])
            cells.append(f"{100.0 * v / cap[f]:16.2f}")
        print(f"{f:16s} {cap[f]:8.0f} " + " ".join(cells))

    # ---------------- figure ----------------
    colors = {r: plt.cm.tab10.colors[i % 10] for i, r in enumerate(fmaps)}
    tag = f"{REGION}_{SEASON}" + ("_regimes" if REGIMES else "") + \
          ("_control" if CURVE_CONTROL else "")
    if REGIMES:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for r_i, (ax, rlab) in enumerate(zip(axes.ravel(), REGIME_LABELS)):
            for r, m in series:
                ax.plot(leads, mae_pct((r, m), r_i), STYLE[m], color=colors[r],
                        lw=1.5, marker="o", ms=3)
            ax.set_title(f"{rlab} m/s", fontsize=11)
            ax.grid(True, ls="--", alpha=0.5)
            ax.set_xticks(leads)
        for ax in axes[1]:
            ax.set_xlabel("Lead time [h]")
        for ax in axes[:, 0]:
            ax.set_ylabel("MAE [% of capacity]")
        fig.suptitle(f"{REGION} total power MAE by observed wind regime "
                     f"({len(inits)} inits, season {SEASON})", fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for r, m in series:
            ax.plot(leads, mae_pct((r, m)), STYLE[m], color=colors[r], lw=1.6,
                    marker="o", ms=4, label=f"{r} — {METHOD_LABEL[m]}")
        ax.set(xlabel="Lead time [h]", ylabel="MAE [% of capacity]")
        ax.set_xticks(leads)
        ax.set_title(f"{REGION} total power MAE — {cap_total:.0f} MW, "
                     f"{len(inits)} inits, season {SEASON}", fontsize=12)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8, framealpha=0.8)
    handles = [plt.Line2D([], [], color=colors[r], ls=STYLE[m], marker="o", ms=4,
                          label=f"{r} — {METHOD_LABEL[m]}") for r, m in series]
    if REGIMES:
        axes[0, 0].legend(handles=handles, fontsize=7, framealpha=0.8)
    fig.tight_layout()
    out = OUT_DIR / f"power_mae_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
