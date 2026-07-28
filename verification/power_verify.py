#!/usr/bin/env python3
"""Score forecast wind-farm power against observations.

The model predicts `capacityfactor` on the CERRA grid. This reverses the build's distribution
to recover per-FARM power and scores it against the ENTSO-E/Elexon observations in power_obs.csv.

Reconstruction (adjoint of build_power.py's capacity-weighted distribution):

    P_pred(farm, t) = SUM_cell  capacity(farm's turbines in cell) * CF_pred(cell, t)

This is exact for a farm's un-shared cells: if CF_pred equals the true field, P_pred == P_obs.
In shared cells it inherits the same mild "farms in one cell see the same wind" approximation the
forward build makes. (A raw-`power` forecast is handled too: it is divided by the cell capacity to
get CF first, then reconstructed identically.)

Forecasts are anemoi inference NetCDFs, one per init time (`forecast_YYYYMMDDHHMMSS.nc`), each a
rollout of lead times on the inner cutout grid, carrying latitude/longitude. Cells are matched to
turbines by lat/lon (a cos-lat KD-tree), so this is robust to the inner-vs-full grid subsetting.

Scores per farm and per lead time, aggregated over all init files: RMSE, bias, and nRMSE
(RMSE / farm capacity). Belgium is the evaluation target (UK obs end 2023); a persistence
baseline (P_obs at init, held flat) is reported for context. Domain-wide weather scores are
useless here -- 172 of 72,668 cells -- which is the whole reason this farm-space scoring exists.

Usage:
  python score_farm_power.py --forecasts DIR [--var capacityfactor] [--region Belgium] [--plot]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

WPOWER_DIR = Path(__file__).resolve().parent
FORECAST_RE = re.compile(r"forecast_(\d{14})")
EARTH_KM_PER_DEG = 111.0


def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(FORECAST_RE.search(path.name).group(1),
                          format="%Y%m%d%H%M%S", utc=True)


def build_reconstruction(fc_lat, fc_lon, turbines, farms):
    """Assign turbines to forecast cells and build the reconstruction operators.

    Returns:
      cell_idx   : forecast-cell indices that hold turbines
      G          : (n_farms, n_cells) with G[f,j] = capacity of farm f in cell_idx[j]
      cap_cell   : (n_cells,) total capacity per cell (for a raw-power forecast -> CF)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", type=Path, required=True, help="dir of forecast_*.nc")
    ap.add_argument("--var", default="capacityfactor", help="forecast variable (capacityfactor or power)")
    ap.add_argument("--region", default="Belgium", help="Belgium | UK | all")
    ap.add_argument("--leads", type=int, nargs="+", default=list(range(3, 37, 3)),
                    help="lead times in hours")
    ap.add_argument("--out", type=Path, default=WPOWER_DIR / "scores_farm_power.csv")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")

    reg = {"belgium": "be", "be": "be", "uk": "uk", "all": "all"}.get(args.region.lower())
    if reg is None:
        raise SystemExit(f"region must be Belgium/UK/all, got {args.region!r}")
    farms = farms_df.farm.tolist() if reg == "all" else \
        farms_df[farms_df.region.str.lower() == reg].farm.tolist()
    if not farms:
        raise SystemExit(f"no farms for region {args.region!r} "
                         f"(available: {sorted(farms_df.region.unique())})")
    cap_total = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    turbines = turbines[turbines.farm.isin(farms)]

    files = sorted(args.forecasts.glob("forecast_*.nc"))
    if not files:
        raise SystemExit(f"no forecast_*.nc in {args.forecasts}")
    print(f"{len(files)} forecast files | region {args.region} | {len(farms)} farms | var {args.var}")

    # Reconstruction operators depend only on the forecast grid; cache per distinct grid so
    # heterogeneous inference outputs (different connected-node sets) are each handled correctly.
    recon_cache: dict = {}

    def get_recon(lat, lon):
        key = (lat.size, round(float(lat[0]), 4), round(float(lat[-1]), 4),
               round(float(lon[0]), 4), round(float(lon[-1]), 4))
        if key not in recon_cache:
            r = build_reconstruction(lat, lon, turbines, farms)
            recon_cache[key] = r
            print(f"  grid {lat.size} cells -> {r[0].size} farm cells, {r[1].sum():.0f} MW")
        return recon_cache[key]

    # accumulate squared error / abs error / signed error and count per (farm, lead)
    F, L = len(farms), len(args.leads)
    lead_pos = {lh: k for k, lh in enumerate(args.leads)}
    sse = np.zeros((F, L)); sae = np.zeros((F, L)); sse_ref = np.zeros((F, L))
    sbias = np.zeros((F, L)); n = np.zeros((F, L))
    fleet_sse = np.zeros(L); fleet_n = np.zeros(L)      # error on total (summed) generation

    for fp in files:
        init = parse_init(fp)
        with xr.open_dataset(fp) as ds:
            if args.var not in ds:
                print(f"  skip {fp.name}: no '{args.var}'"); continue
            cell_idx, G, cap_cell = get_recon(ds["latitude"].values, ds["longitude"].values)
            fc_times = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
            field = ds[args.var].values[:, cell_idx]                 # (n_time, n_cells)

        cf = field if args.var == "capacityfactor" else np.divide(
            field, cap_cell[None, :], out=np.full_like(field, np.nan), where=cap_cell[None, :] > 0)
        ppred_all = cf @ G.T                                          # (n_time, n_farms)

        t2idx = {t: j for j, t in enumerate(fc_times)}
        for lh in args.leads:
            vt = init + pd.Timedelta(hours=lh)
            if vt not in t2idx or vt not in obs.index or init not in obs.index:
                continue
            ppred = ppred_all[t2idx[vt]]                             # (n_farms,)
            ptrue = obs.loc[vt, farms].to_numpy(float)
            pref = obs.loc[init, farms].to_numpy(float)              # persistence
            k = lead_pos[lh]
            for i in range(F):
                if np.isfinite(ptrue[i]) and np.isfinite(ppred[i]):
                    e = ppred[i] - ptrue[i]
                    sse[i, k] += e * e; sae[i, k] += abs(e); sbias[i, k] += e; n[i, k] += 1
                    if np.isfinite(pref[i]):
                        sse_ref[i, k] += (pref[i] - ptrue[i]) ** 2
            # total generation: only when every farm reports (a known sum, not a partial one)
            if np.isfinite(ptrue).all() and np.isfinite(ppred).all():
                fe = ppred.sum() - ptrue.sum()
                fleet_sse[k] += fe * fe; fleet_n[k] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(sse / n); mae = sae / n; bias = sbias / n
        rmse_ref = np.sqrt(sse_ref / n)
        nrmse = 100.0 * rmse / cap_total.to_numpy()[:, None]

    # ---- per-farm table (averaged over lead) ----
    print(f"\n{'farm':16s} {'n':>6s} {'RMSE':>7s} {'nRMSE%':>7s} {'bias':>7s} {'persist':>8s}  skill")
    print("-" * 66)
    rows = []
    for i, farm in enumerate(farms):
        N = n[i].sum()
        if N == 0:
            continue
        r = np.nansum(sse[i]) ** 0.5 / max(np.sqrt(N), 1)   # pooled RMSE over leads
        rr = (np.nansum(sse_ref[i]) / N) ** 0.5
        b = np.nansum(sbias[i]) / N
        nr = 100.0 * r / cap_total[farm]
        skill = 1 - r / rr if rr > 0 else np.nan
        rows.append(dict(farm=farm, region=farms_df.set_index("farm").loc[farm, "region"],
                         n=int(N), rmse_mw=r, nrmse_pct=nr, bias_mw=b,
                         persist_rmse_mw=rr, skill_vs_persist=skill))
        print(f"{farm:16s} {int(N):6d} {r:7.1f} {nr:7.1f} {b:+7.1f} {rr:8.1f}  {skill:+.2f}")

    per_farm = pd.DataFrame(rows)

    # ---- total generation error vs lead (sum over farms, only when all report) ----
    total_cap = float(cap_total.sum())
    print(f"\ntotal {args.region} generation forecast error by lead "
          f"(sum of {F} farms, capacity {total_cap:.0f} MW):")
    print(f"{'lead':>5s} {'n':>5s} {'RMSE MW':>8s} {'nRMSE%':>7s} | {'mean per-farm RMSE':>18s}")
    for lh in args.leads:
        k = lead_pos[lh]
        fr = (fleet_sse[k] / fleet_n[k]) ** 0.5 if fleet_n[k] else np.nan
        print(f"  +{lh:2d}h {int(fleet_n[k]):5d} {fr:8.1f} {100*fr/total_cap:7.1f} | "
              f"{np.nanmean(rmse[:, k]):18.1f}")

    per_farm.to_csv(args.out, index=False)
    # also dump the full lead x farm RMSE grid
    grid = pd.DataFrame(rmse, index=farms, columns=[f"lead_{lh}h" for lh in args.leads])
    grid.to_csv(args.out.with_name(args.out.stem + "_by_lead.csv"))
    print(f"\nwrote {args.out.name} and {args.out.stem}_by_lead.csv")

    if per_farm.empty:
        raise SystemExit("no scored farm-leads -- check that forecast valid times overlap the obs")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for i, farm in enumerate(farms):
            if n[i].sum() > 0:
                ax.plot(args.leads, nrmse[i], marker="o", ms=3, lw=1, alpha=0.7, label=farm)
        ax.plot(args.leads, np.nanmean(nrmse, 0), "k--", lw=2, label="mean")
        ax.set(xlabel="lead time (h)", ylabel="nRMSE (% of capacity)",
               title=f"farm power forecast skill — {args.region}")
        ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=7)
        p = args.out.with_suffix(".png")
        fig.tight_layout(); fig.savefig(p, dpi=140); print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
