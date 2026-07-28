#!/usr/bin/env python3
"""Score wind-farm power forecasts against the observations, comparing runs and methods.

Self-contained: reconstruction, power curves, scoring and plotting all live here.

For every run, up to TWO power forecasts are produced, summed to the regional total, and scored
against the ENTSO-E/Elexon observations in power_obs.csv:

  DIRECT      the model's own `capacityfactor` prediction, mapped back to farms:
                  P_direct(farm,t) = SUM_cell capacity(farm's turbines in cell) * CF_pred(cell,t)
              the adjoint of build_power.py's capacity-weighted distribution (exact for a farm's
              un-shared cells: a perfect CF field reproduces P_obs).

  POWERCURVE  the classical baseline: the model's forecast ws100 at the farm, pushed through that
              farm's own aggregate power curve, built from turbine_specs.csv (cut-in, rated wind,
              cut-out, rated MW) and the farm's fleet in farms.csv:
                  P_curve(farm,t) = A_farm( ws100_forecast(farm,t) )
              A_farm is rescaled so its rated plateau equals the farm's nameplate capacity.

A run with only ws100 and no `capacityfactor` (e.g. RegularWeather) still gets a POWERCURVE
forecast and is scored on equal terms -- it simply has no DIRECT line. That comparison is the
point: does predicting power directly beat predicting wind and applying the power curve?

PERSISTENCE (P_obs held flat from the init time) is scored too, as the honest floor for a 3-36 h
horizon: a power forecast that cannot beat it is not useful. Skill = 1 - MAE/MAE_persistence.

Metrics per farm and for the regional total, by lead time: MAE (MW and % of capacity), RMSE,
BIAS, and skill vs persistence. Bias matters here -- the idealised specs power curve ignores wake
losses, so it should over-predict at high wind; whether the direct forecast removes that is a
concrete, testable claim MAE alone cannot show.

Figures:
  1. mae_total_<region>.png     MAE of the TOTAL summed power vs lead, % of total capacity
  2. mae_per_farm_<region>.png  MAE vs lead per farm, % of that farm's capacity
Solid = DIRECT, dashed = POWERCURVE, dotted black = persistence, colour = run. Every run is
scored on the init times common to all runs, so the comparison uses an identical sample.

Usage:
  python score_power_configs.py                     # uses the SETTINGS below
  python score_power_configs.py --region UK
  python score_power_configs.py --runs A=/path/one B=/path/two
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# -------------------- SETTINGS --------------------
# Where farms.csv / turbines.csv / power_obs.csv / turbine_specs.csv live (NOT where this script
# lives, so it can sit in verification/ while the metadata stays with the data).
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")

# One entry per run; the key is the legend label. A run WITHOUT the power variable but WITH ws100
# (e.g. RegularWeather) is fine -- it gets the POWERCURVE forecast only.
FORECAST_DIRS = {
    "HighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaPowerGT"),
    "RegularWeather": Path("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
  #  "SemiHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/SemiHighCapacity"),
}
VAR        = "capacityfactor"        # absent in a run -> POWERCURVE only
REGION     = "BE"                    # BE | UK | all
LEAD_HOURS = list(range(3, 37, 3))
OUT_DIR    = Path("WPDistr_scores")  # relative to cwd
# --------------------------------------------------

FORECAST_RE = re.compile(r"forecast_(\d{14})")
FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")


# =============================================================================
# reconstruction: forecast cells -> per-farm power
# =============================================================================
def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(FORECAST_RE.search(path.name).group(1),
                          format="%Y%m%d%H%M%S", utc=True)


def build_reconstruction(fc_lat, fc_lon, turbines, farms):
    """Assign turbines to forecast cells and build the reconstruction operators.

    Returns:
      cell_idx : forecast-cell indices that hold turbines
      G        : (n_farms, n_cells) with G[f,j] = capacity of farm f in cell_idx[j]
      cap_cell : (n_cells,) total capacity per cell (for a raw-power forecast -> CF)
    """
    coslat = np.cos(np.radians(float(fc_lat.mean())))
    tree = cKDTree(np.c_[to_180(fc_lon) * coslat, fc_lat])
    _, cell = tree.query(np.c_[to_180(turbines["longitude"]) * coslat,
                               turbines["latitude"].to_numpy()], k=1)
    t = turbines.assign(cell=cell.astype(int))

    cell_idx = np.sort(t["cell"].unique())
    cpos = {int(c): j for j, c in enumerate(cell_idx)}
    fpos = {f: i for i, f in enumerate(farms)}

    G = np.zeros((len(farms), cell_idx.size), dtype=np.float64)
    for (farm, c), cap in t.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[fpos[farm], cpos[int(c)]] = cap
    cap_cell = t.groupby("cell")["capacity_mw"].sum().reindex(cell_idx).to_numpy()
    return cell_idx, G, cap_cell


def farm_wind(ws_cells: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Capacity-weighted mean wind per farm. ws_cells (T, C), G (F, C) -> (T, F)."""
    w = G / G.sum(1, keepdims=True)
    return ws_cells @ w.T


# =============================================================================
# per-farm aggregate power curve  (specs -> P(v))
# =============================================================================
def turbine_power(ws, cut_in, rated_ws, cut_out, rated_mw) -> np.ndarray:
    """One turbine: 0 below cut-in, cubic ramp to rated, flat at rated, 0 above cut-out."""
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws)
    ramp = (ws >= cut_in) & (ws < rated_ws)
    out[ramp] = rated_mw * (ws[ramp] ** 3 - cut_in ** 3) / (rated_ws ** 3 - cut_in ** 3)
    out[(ws >= rated_ws) & (ws < cut_out)] = rated_mw
    return out


def build_farm_curves(farms_df, specs, farms) -> dict:
    """farm -> callable(ws) -> MW, the fleet's summed power curve scaled to the nameplate."""
    curves = {}
    meta = farms_df.set_index("farm")
    for farm in farms:
        fleet, cap = meta.loc[farm, "fleet"], float(meta.loc[farm, "capacity_mw"])
        parts = []
        for chunk in str(fleet).split(";"):
            m = FLEET_RE.match(chunk)
            if not m:
                raise SystemExit(f"{farm}: cannot parse fleet entry {chunk!r}")
            count, ttype = int(m.group(1)), m.group(2)
            if ttype not in specs.index:
                raise SystemExit(f"{farm}: turbine type {ttype!r} not in turbine_specs.csv")
            parts.append((count, specs.loc[ttype]))

        scale = cap / sum(c * float(s["rated_power_mw"]) for c, s in parts)   # nameplate wins

        def curve(ws, parts=parts, scale=scale):
            tot = np.zeros_like(np.asarray(ws, dtype=float))
            for count, s in parts:
                tot += count * turbine_power(ws, float(s["cut_in_ms"]), float(s["rated_ws_ms"]),
                                             float(s["cut_out_ms"]), float(s["rated_power_mw"]))
            return tot * scale
        curves[farm] = curve
    return curves


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None, metavar="LABEL=DIR",
                    help="override FORECAST_DIRS: one or more LABEL=/path/to/inference/dir")
    ap.add_argument("--region", default=REGION, choices=["BE", "UK", "all"])
    ap.add_argument("--var", default=VAR, help="the power variable (absent -> POWERCURVE only)")
    ap.add_argument("--leads", type=int, nargs="+", default=LEAD_HOURS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.runs:
        runs: dict[str, Path] = {}
        for r in args.runs:
            if "=" not in r:
                raise SystemExit(f"--runs takes LABEL=DIR, got {r!r}")
            label, d = r.split("=", 1)
            runs[label] = Path(d)
    else:
        runs = dict(FORECAST_DIRS)
    args.out.mkdir(parents=True, exist_ok=True)

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    specs = pd.read_csv(WPOWER_DIR / "turbine_specs.csv")
    specs = specs.rename(columns={specs.columns[0]: "turbine_type"}).set_index("turbine_type")
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")

    farms = (farms_df.farm.tolist() if args.region == "all"
             else farms_df[farms_df.region == args.region].farm.tolist())
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    turbines = turbines[turbines.farm.isin(farms)]
    L = len(args.leads)
    lead_pos = {lh: k for k, lh in enumerate(args.leads)}
    print(f"{args.region}: {len(farms)} farms, {cap.sum():.0f} MW\n")

    # ---- file maps, restricted to init times present in EVERY run ----
    fmaps: dict[str, dict] = {}
    has_power: dict[str, bool] = {}
    for label, d in runs.items():
        files = sorted(d.glob("forecast_*.nc"))
        if not files:
            print(f"{label:16s} NO forecast_*.nc in {d} -- skipping")
            continue
        with xr.open_dataset(files[0]) as ds0:
            hp, hw = args.var in ds0, "ws100" in ds0
        if not hw:
            print(f"{label:16s} no ws100 -- skipping (needed for the power curve)")
            continue
        has_power[label] = hp
        print(f"{label:16s} {len(files):4d} files   "
              f"{'DIRECT + POWERCURVE' if hp else 'POWERCURVE only (no %s)' % args.var}")
        fmaps[label] = {parse_init(f): f for f in files}

    if not fmaps:
        raise SystemExit("no usable runs -- check the dirs")
    common = sorted(set.intersection(*(set(m) for m in fmaps.values())))
    print(f"\ncommon init times across {len(fmaps)} run(s): {len(common)}")
    if not common:
        raise SystemExit("no init times common to all runs")
    for label, fm in fmaps.items():
        if len(fm) > len(common):
            print(f"  note: {label} has {len(fm)} inits, {len(fm) - len(common)} dropped")

    # An inference directory can be HETEROGENEOUS: an earlier pass may have written files without
    # the power variable and a later pass added it. Deciding from files[0] alone then blows up
    # mid-run. Verify, and drop inits where a direct-capable run lacks it, so every series is
    # scored on one identical sample.
    direct_runs = [r for r in fmaps if has_power[r]]
    if direct_runs:
        print(f"\nverifying '{args.var}' is present in all {len(common)} candidate inits "
              f"({len(direct_runs)} run(s) with a direct forecast)...")
        good, missing = [], {r: 0 for r in direct_runs}
        for i, init in enumerate(common):
            ok_all = True
            for r in direct_runs:
                with xr.open_dataset(fmaps[r][init]) as ds:
                    if args.var not in ds:
                        missing[r] += 1
                        ok_all = False
                        break
            if ok_all:
                good.append(init)
            if i and i % 500 == 0:
                print(f"  checked {i}/{len(common)}", flush=True)
        if len(good) < len(common):
            print(f"  DROPPED {len(common) - len(good)} init(s) missing '{args.var}':")
            for r, c in missing.items():
                if c:
                    print(f"    {r}: {c} file(s) without it  <-- that inference run is incomplete")
        common = good
        print(f"  usable init times: {len(common)}")
        if not common:
            raise SystemExit(f"no init has '{args.var}' in every direct run")

    # ---- drop farms with NO observations in the scored window ----
    # UK observations end in 2023 (ALLOWED_YEARS = Nost's full-capacity periods), so a 2024-25
    # inference window has nothing to score most UK farms against, and the regional TOTAL (which
    # needs EVERY farm reporting) would come out silently all-NaN.
    valid_times = pd.DatetimeIndex(sorted({i + pd.Timedelta(hours=lh)
                                           for i in common for lh in args.leads}))
    valid_times = valid_times.intersection(obs.index)
    if len(valid_times) == 0:
        raise SystemExit("no forecast valid time overlaps power_obs.csv -- check the periods")
    counts = obs.loc[valid_times, farms].notna().sum()
    dead = [f for f in farms if counts[f] == 0]
    if dead:
        print(f"\n{len(dead)} of {len(farms)} {args.region} farm(s) have NO observations in the "
              f"scored window ({valid_times.min():%Y-%m-%d} .. {valid_times.max():%Y-%m-%d}):")
        print("   " + ", ".join(dead))
        farms = [f for f in farms if counts[f] > 0]
        if not farms:
            raise SystemExit(
                f"no {args.region} farm has observations in the forecast window -- nothing to "
                f"score. UK observations end in 2023, so a 2024-25 window can only score BE.")
        cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
        turbines = turbines[turbines.farm.isin(farms)]
        print(f"   -> scoring the remaining {len(farms)} farm(s), total capacity {cap.sum():.0f} MW")
        print("      (the TOTAL below is that reduced fleet, NOT the full region)")

    cap_np = cap.to_numpy()
    total_cap = float(cap.sum())
    curves = build_farm_curves(farms_df, specs, farms)
    F = len(farms)

    # ---- accumulators: keyed (run, method), plus persistence (run-independent) ----
    series = [(r, "direct") for r in fmaps if has_power[r]] + [(r, "curve") for r in fmaps]
    keys = series + [("persistence", "persist")]
    z2 = lambda: np.zeros((F, L))                                          # noqa: E731
    sae = {s: z2() for s in keys}      # sum |err|      -> MAE
    sse = {s: z2() for s in keys}      # sum err^2      -> RMSE
    sbe = {s: z2() for s in keys}      # sum err        -> bias
    n = {s: z2() for s in keys}
    sae_t = {s: np.zeros(L) for s in keys}   # regional total (all farms reporting)
    sse_t = {s: np.zeros(L) for s in keys}
    sbe_t = {s: np.zeros(L) for s in keys}
    n_t = {s: np.zeros(L) for s in keys}
    # cross-moments of the regional total, for the MSE decomposition and the variance ratio
    sp_t = {s: np.zeros(L) for s in keys}    # sum pred
    so_t = {s: np.zeros(L) for s in keys}    # sum obs
    spp_t = {s: np.zeros(L) for s in keys}   # sum pred^2
    soo_t = {s: np.zeros(L) for s in keys}   # sum obs^2
    spo_t = {s: np.zeros(L) for s in keys}   # sum pred*obs

    def accumulate(s, ppred, ptrue, k):
        ok = np.isfinite(ptrue) & np.isfinite(ppred)
        if ok.any():
            e = ppred[ok] - ptrue[ok]
            sae[s][ok, k] += np.abs(e); sse[s][ok, k] += e * e
            sbe[s][ok, k] += e;         n[s][ok, k] += 1
        if ok.all():                      # total only when every farm reports
            pt, ot = ppred.sum(), ptrue.sum()
            et = pt - ot
            sae_t[s][k] += abs(et); sse_t[s][k] += et * et
            sbe_t[s][k] += et;      n_t[s][k] += 1
            sp_t[s][k] += pt;   so_t[s][k] += ot
            spp_t[s][k] += pt * pt; soo_t[s][k] += ot * ot; spo_t[s][k] += pt * ot

    recon_cache: dict = {}

    def get_recon(lat, lon):
        key = (lat.size, round(float(lat[0]), 4), round(float(lon[-1]), 4))
        if key not in recon_cache:
            recon_cache[key] = build_reconstruction(lat, lon, turbines, farms)
        return recon_cache[key]

    # ---- persistence: run-independent, so score it once over the same inits ----
    for init in common:
        if init not in obs.index:
            continue
        pref = obs.loc[init, farms].to_numpy(float)
        for lh in args.leads:
            vt = init + pd.Timedelta(hours=lh)
            if vt in obs.index:
                accumulate(("persistence", "persist"), pref,
                           obs.loc[vt, farms].to_numpy(float), lead_pos[lh])

    # ---- the runs ----
    for label, fmap in fmaps.items():
        hp = has_power[label]
        print(f"\nscoring {label} ({len(common)} inits)...")
        for init in common:
            with xr.open_dataset(fmap[init]) as ds:
                cell_idx, G, cap_cell = get_recon(ds["latitude"].values, ds["longitude"].values)
                fc_times = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
                ws_farm = farm_wind(ds["ws100"].values[:, cell_idx], G)      # (T, F)
                have_var = hp and args.var in ds        # defensive: never KeyError mid-run
                if have_var:
                    field = ds[args.var].values[:, cell_idx]
                    # a raw-power forecast is MW/cell -> to CF first, so the reconstruction is
                    # identical in both cases
                    cf = field if args.var == "capacityfactor" else np.divide(
                        field, cap_cell[None, :], out=np.full_like(field, np.nan),
                        where=cap_cell[None, :] > 0)
                    p_direct_all = cf @ G.T

            p_curve_all = np.column_stack([curves[f](ws_farm[:, i]) for i, f in enumerate(farms)])

            t2i = {t: j for j, t in enumerate(fc_times)}
            for lh in args.leads:
                vt = init + pd.Timedelta(hours=lh)
                if vt not in t2i or vt not in obs.index:
                    continue
                j, k = t2i[vt], lead_pos[lh]
                ptrue = obs.loc[vt, farms].to_numpy(float)
                accumulate((label, "curve"), p_curve_all[j], ptrue, k)
                if have_var:
                    accumulate((label, "direct"), p_direct_all[j], ptrue, k)

    # =========================================================================
    # metrics
    # =========================================================================
    with np.errstate(invalid="ignore", divide="ignore"):
        mae   = {s: sae[s] / n[s] for s in keys}
        rmse  = {s: np.sqrt(sse[s] / n[s]) for s in keys}
        bias  = {s: sbe[s] / n[s] for s in keys}
        nmae  = {s: 100.0 * mae[s] / cap_np[:, None] for s in keys}
        mae_t   = {s: sae_t[s] / n_t[s] for s in keys}
        rmse_t  = {s: np.sqrt(sse_t[s] / n_t[s]) for s in keys}
        bias_t  = {s: sbe_t[s] / n_t[s] for s in keys}
        nmae_t  = {s: 100.0 * mae_t[s] / total_cap for s in keys}

    PERS = ("persistence", "persist")
    lbl = lambda s: (f"{s[0]} · {'direct' if s[1] == 'direct' else 'power curve'}"      # noqa: E731
                     if s != PERS else "persistence")

    def decompose(s, k=None):
        """MSE decomposition of the regional total.

            MSE = (mean_p - mean_o)^2  +  (sd_p - sd_o)^2  +  2*sd_p*sd_o*(1 - r)
                   bias^2 (systematic)     amplitude          phase / timing

        The first two are calibratable (a constant offset and a gain would remove them); the
        third is genuine timing error and is the only part a better forecast must earn.
        Pass k for one lead, or None to pool over all leads.
        """
        sl = slice(None) if k is None else slice(k, k + 1)
        N = float(np.nansum(n_t[s][sl]))
        if N == 0:
            return dict(mse=np.nan, bias2=np.nan, amp=np.nan, phase=np.nan,
                        sd_p=np.nan, sd_o=np.nan, r=np.nan, var_ratio=np.nan)
        mp = np.nansum(sp_t[s][sl]) / N
        mo = np.nansum(so_t[s][sl]) / N
        vp = max(np.nansum(spp_t[s][sl]) / N - mp * mp, 0.0)
        vo = max(np.nansum(soo_t[s][sl]) / N - mo * mo, 0.0)
        cov = np.nansum(spo_t[s][sl]) / N - mp * mo
        sd_p, sd_o = np.sqrt(vp), np.sqrt(vo)
        r = cov / (sd_p * sd_o) if sd_p > 0 and sd_o > 0 else np.nan
        return dict(mse=np.nansum(sse_t[s][sl]) / N,
                    bias2=(mp - mo) ** 2,
                    amp=(sd_p - sd_o) ** 2,
                    phase=2 * sd_p * sd_o * (1 - r) if np.isfinite(r) else np.nan,
                    sd_p=sd_p, sd_o=sd_o, r=r,
                    var_ratio=sd_p / sd_o if sd_o > 0 else np.nan)

    # ---- 1. total MAE by lead ----
    print(f"\nTOTAL {args.region} power — MAE as % of {total_cap:.0f} MW")
    hdr = "lead  " + "".join(f"{lbl(s):>26s}" for s in series + [PERS])
    print(hdr); print("-" * len(hdr))
    for lh in args.leads:
        k = lead_pos[lh]
        print(f"+{lh:2d}h  " + "".join(f"{nmae_t[s][k]:25.2f}%" for s in series + [PERS]))

    # ---- 2. summary pooled over leads ----
    print(f"\nSUMMARY — {args.region} total, pooled over all leads")
    print(f"{'series':34s} {'MAE%':>7s} {'RMSE MW':>9s} {'bias MW':>9s} {'skill vs persist':>17s}")
    print("-" * 80)
    pers_mae = np.nansum(sae_t[PERS]) / max(np.nansum(n_t[PERS]), 1)
    summary_rows = []
    for s in series + [PERS]:
        N = np.nansum(n_t[s])
        if N == 0:
            continue
        m = np.nansum(sae_t[s]) / N
        r = np.sqrt(np.nansum(sse_t[s]) / N)
        b = np.nansum(sbe_t[s]) / N
        sk = 1 - m / pers_mae if pers_mae > 0 else np.nan
        print(f"{lbl(s):34s} {100*m/total_cap:6.2f}% {r:9.1f} {b:+9.1f} "
              f"{('' if s == PERS else f'{sk:+.3f}'):>17s}")
        d = decompose(s)
        summary_rows.append(dict(series=lbl(s), run=s[0], method=s[1], mae_mw=m,
                                 nmae_pct=100 * m / total_cap, rmse_mw=r, bias_mw=b,
                                 skill_vs_persistence=np.nan if s == PERS else sk,
                                 mse=d["mse"], bias2=d["bias2"], amplitude=d["amp"],
                                 phase=d["phase"], sd_pred=d["sd_p"], sd_obs=d["sd_o"],
                                 corr=d["r"], var_ratio=d["var_ratio"], n=int(N)))

    # =========================================================================
    # plots
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {r: c for r, c in zip(fmaps, plt.cm.tab10.colors)}
    style = {"direct": "-", "curve": "--"}

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for s in series:
        ax.plot(args.leads, nmae_t[s], marker="o", ms=4, lw=1.8,
                color=colors[s[0]], ls=style[s[1]], label=lbl(s))
    ax.set(xlabel="lead time [h]", ylabel="MAE [% of total capacity]",
           title=f"Total {args.region} power — MAE vs observations ({F} farms, {total_cap:.0f} MW)\n"
                 f"solid = direct CF forecast, dashed = ws100 -> power curve")
    ax.set_xticks(args.leads); ax.grid(ls="--", alpha=0.5); ax.legend(fontsize=8)
    fig.tight_layout()
    p = args.out / f"mae_total_{args.region}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"\nsaved {p}")

    ncol = 5 if F > 6 else 3
    nrow = int(np.ceil(F / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                            sharex=True, sharey=True, squeeze=False)
    for i, farm in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        for s in series:
            ax.plot(args.leads, nmae[s][i], marker="o", ms=3, lw=1.2,
                    color=colors[s[0]], ls=style[s[1]], label=lbl(s))
        ax.set_title(f"{farm}\n{cap[farm]:.0f} MW", fontsize=8)
        ax.grid(ls="--", alpha=0.4)
    for j in range(F, nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    axs[0][0].legend(fontsize=6)
    fig.supxlabel("lead time [h]"); fig.supylabel("MAE [% of farm capacity]")
    fig.suptitle(f"Per-farm power MAE vs observations — {args.region}  "
                 f"(solid = direct, dashed = power curve)")
    fig.tight_layout()
    p = args.out / f"mae_per_farm_{args.region}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    # ---- 3. per-farm BIAS ----
    # Sign matters: the idealised specs power curve ignores wake losses, so it should sit ABOVE
    # zero (over-predicting). Whether the direct forecast removes that -- or overshoots into
    # under-prediction -- is invisible in MAE.
    with np.errstate(invalid="ignore", divide="ignore"):
        nbias = {s: 100.0 * bias[s] / cap_np[:, None] for s in series}
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                            sharex=True, sharey=True, squeeze=False)
    for i, farm in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        for s in series:
            ax.plot(args.leads, nbias[s][i], marker="o", ms=3, lw=1.2,
                    color=colors[s[0]], ls=style[s[1]], label=lbl(s))
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_title(f"{farm}\n{cap[farm]:.0f} MW", fontsize=8)
        ax.grid(ls="--", alpha=0.4)
    for j in range(F, nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    axs[0][0].legend(fontsize=6)
    fig.supxlabel("lead time [h]"); fig.supylabel("bias [% of farm capacity]")
    fig.suptitle(f"Per-farm power BIAS (forecast − observed) — {args.region}  "
                 f"(solid = direct, dashed = power curve; above 0 = over-predicting)")
    fig.tight_layout()
    p = args.out / f"bias_per_farm_{args.region}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    # ---- 4. MSE decomposition + variance ratio ----
    # persistence is computed for the printed table (it is the reference floor, and its
    # bias^2 ~ 0 / all-phase signature validates the decomposition) but is NOT plotted.
    dec_all = {s: decompose(s) for s in series + [PERS]}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.8))

    labs = [lbl(s) for s in series]
    b2 = np.array([dec_all[s]["bias2"] for s in series])
    am = np.array([dec_all[s]["amp"] for s in series])
    ph = np.array([dec_all[s]["phase"] for s in series])
    x = np.arange(len(labs))
    axL.bar(x, b2, label="bias²  (systematic offset)", color="tab:red")
    axL.bar(x, am, bottom=b2, label="amplitude  (σ mismatch)", color="tab:orange")
    axL.bar(x, ph, bottom=b2 + am, label="phase  (timing, irreducible)", color="tab:blue")
    axL.set_xticks(x); axL.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    axL.set_ylabel("MSE of the regional total [MW²]")
    axL.set_title("MSE decomposition, pooled over leads\n"
                  "red+orange are calibratable; blue must be earned")
    axL.grid(ls="--", alpha=0.4, axis="y"); axL.legend(fontsize=8)

    for s in series:
        vr = [decompose(s, k)["var_ratio"] for k in range(L)]
        axR.plot(args.leads, vr, marker="o", ms=4, lw=1.8,
                 color=colors[s[0]], ls=style[s[1]], label=lbl(s))
    axR.axhline(1.0, color="grey", ls="--", lw=1.2)
    axR.annotate("perfectly dispersed", (args.leads[-1], 1.0), ha="right", va="bottom",
                 fontsize=8, color="grey")
    axR.set(xlabel="lead time [h]", ylabel=r"$\sigma_{pred}\,/\,\sigma_{obs}$",
            title="Variance ratio of the regional total\nbelow 1 = under-dispersive (too smooth)")
    axR.set_xticks(args.leads); axR.grid(ls="--", alpha=0.5); axR.legend(fontsize=7)
    fig.tight_layout()
    p = args.out / f"decomposition_{args.region}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    # printed decomposition table
    print(f"\nMSE DECOMPOSITION — {args.region} total, pooled over leads  "
          f"(MSE = bias² + amplitude + phase)")
    print(f"{'series':34s} {'RMSE':>8s} {'bias²':>9s} {'ampl':>9s} {'phase':>9s} "
          f"{'σp/σo':>7s} {'r':>6s}")
    print("-" * 88)
    for s in series + [PERS]:
        d = dec_all[s]
        if not np.isfinite(d["mse"]):
            continue
        tot = d["bias2"] + d["amp"] + d["phase"]
        print(f"{lbl(s):34s} {np.sqrt(d['mse']):8.1f} "
              f"{d['bias2']:9.0f} {d['amp']:9.0f} {d['phase']:9.0f} "
              f"{d['var_ratio']:7.3f} {d['r']:6.3f}"
              f"{'   (check: sum/MSE = %.3f)' % (tot / d['mse']) if d['mse'] > 0 else ''}")

    # =========================================================================
    # CSVs
    # =========================================================================
    rows = []
    for s in keys:
        for lh in args.leads:
            k = lead_pos[lh]
            rows.append(dict(run=s[0], method=s[1], lead_hours=lh, scope="TOTAL",
                             mae_mw=mae_t[s][k], nmae_pct=nmae_t[s][k], rmse_mw=rmse_t[s][k],
                             bias_mw=bias_t[s][k],
                             skill_vs_persistence=(np.nan if s == PERS else
                                                   1 - mae_t[s][k] / mae_t[PERS][k]),
                             n=int(n_t[s][k])))
            for i, farm in enumerate(farms):
                rows.append(dict(run=s[0], method=s[1], lead_hours=lh, scope=farm,
                                 mae_mw=mae[s][i, k], nmae_pct=nmae[s][i, k],
                                 rmse_mw=rmse[s][i, k], bias_mw=bias[s][i, k],
                                 skill_vs_persistence=(np.nan if s == PERS else
                                                       1 - mae[s][i, k] / mae[PERS][i, k]),
                                 n=int(n[s][i, k])))
    out_csv = args.out / f"scores_{args.region}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    if summary_rows:
        out_sum = args.out / f"summary_{args.region}.csv"
        pd.DataFrame(summary_rows).to_csv(out_sum, index=False)
        print(f"saved {out_sum}")


if __name__ == "__main__":
    main()
