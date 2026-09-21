#!/usr/bin/env python3
"""GraphCast-style scorecard: RMSE skill of RUN against REFERENCE, every weather variable x lead.

    skill = RMSE_run / RMSE_ref - 1        (negative = RUN better)

per variable and lead time, on one identical sample: the inits common to both runs, the same
cells, the same valid times, CERRA as truth. Blue = RUN better, red = RUN worse, white = equal.
The colour scale is clipped at +-CLIP so one badly drifting field cannot wash out the rest; the
annotated numbers are the true, unclipped values. Pressure-level variables are drawn as
level x lead panels, surface variables as one strip each (GraphCast Fig. 2D).

Weather fields are never window-shifted -- only capacityfactor is, in backward-window runs -- so
every run is scored as is. DOMAIN "BE" = the 15 BE farm cells, i.e. where the power target lives.

Prints every number it plots; writes one PNG.
"""

from __future__ import annotations

from pathlib import Path
from multiprocessing import Pool
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr
import h5py
import netCDF4 as nc4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_weather import TRUTH_ZARR, forecast_cell_map, parse_init, select_cells, to_180

# ============================== SETTINGS ==============================
RUN       = ("FinetunedBack",  Path("/mnt/weatherloss/WindPower/inference/WPDistr/unfreeze_backwin"))
REFERENCE = ("RegularWeather", Path("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"))

DOMAIN   = "BE"              # "BE" (15 farm cells) | "BE+UK" (all 172 farm cells)
SEASON   = "all"             # "all" | "DJF" | "MAM" | "JJA" | "SON"  -- filters on INIT month

PL_VARS  = ["u", "v", "z", "t", "q"]
LEVELS   = [500, 600, 700, 750, 800, 850, 900, 950, 1000]      # drawn top -> bottom
SFC_VARS = ["t2m", "msl", "ws10", "ws100", "mcc"]

CLIP     = 0.30              # colour-scale limit on the skill score (+-30 %)
ANNOTATE = True              # write the skill (%) in every cell

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))

N_WORKERS = 8
OUT_DIR   = Path("DistrFigures")
# ======================================================================

SEASONS = {"all": None, "DJF": {12, 1, 2}, "MAM": {3, 4, 5},
           "JJA": {6, 7, 8}, "SON": {9, 10, 11}}

_W = {}           # per-worker globals, filled once by the Pool initializer


def _init_worker(truth, t_index, fc_sorted, inv, varnames, leads):
    _W.update(truth=truth, t_index=t_index, fc_sorted=fc_sorted, inv=inv,
              varnames=varnames, leads=leads)


def _score_file(args):
    """One forecast file -> (V, L) sum of squared errors and (V, L) counts."""
    path, init_iso = args
    V, L = len(_W["varnames"]), len(_W["leads"])
    sse, n = np.zeros((V, L)), np.zeros((V, L))
    init = pd.Timestamp(init_iso)

    with h5py.File(path, "r") as f:
        tv = f["time"]
        raw = nc4.num2date(tv[:], tv.attrs["units"].decode(),
                           tv.attrs.get("calendar", b"standard").decode())
        fmap = {pd.Timestamp(str(t)).tz_localize("UTC").isoformat(): j for j, t in enumerate(raw)}
        rows = []                                   # (lead k, forecast row, truth row)
        for k, lh in enumerate(_W["leads"]):
            vt = (init + pd.Timedelta(hours=lh)).isoformat()
            if vt in fmap and vt in _W["t_index"]:
                rows.append((k, fmap[vt], _W["t_index"][vt]))
        if not rows:
            return sse, n
        ks, fj, tj = (np.array(c) for c in zip(*rows))

        for v, name in enumerate(_W["varnames"]):
            # h5py wants increasing indices: read the cells sorted, then restore truth's order
            y = f[name][:, _W["fc_sorted"]][fj][:, _W["inv"]].astype(np.float64)
            x = _W["truth"][tj, v].astype(np.float64)
            d2 = (y - x) ** 2
            ok = np.isfinite(d2)
            sse[v, ks] += np.where(ok, d2, 0.0).sum(1)
            n[v, ks] += ok.sum(1)
    return sse, n


def score_run(label, fmap, inits, truth, t_index, lat, lon, varnames):
    fc_cells = forecast_cell_map(fmap[inits[0]], lat, lon)
    order = np.argsort(fc_cells)
    inv = np.argsort(order)
    tasks = [(str(fmap[i]), i.isoformat()) for i in inits]
    sse = np.zeros((len(varnames), len(LEAD_HOURS)))
    n = np.zeros_like(sse)
    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(truth, t_index, fc_cells[order], inv, varnames, LEAD_HOURS)) as pool:
        for c, (s, m) in enumerate(pool.imap_unordered(_score_file, tasks, chunksize=4)):
            sse += s
            n += m
            if c % 500 == 0:
                print(f"  {label}: {c}/{len(tasks)}", flush=True)
    print(f"  {label}: done")
    return np.sqrt(sse / n), n


def draw(skill, varnames, title, out):
    """GraphCast-style: one level x lead panel per pressure variable, one strip per surface one."""
    vi = {v: i for i, v in enumerate(varnames)}
    kw = dict(cmap="RdBu_r", vmin=-CLIP, vmax=CLIP, aspect="auto", interpolation="nearest")
    fig = plt.figure(figsize=(3.3 * (len(PL_VARS) + 1) + 0.8, 3.6), layout="constrained")
    gs = fig.add_gridspec(1, len(PL_VARS) + 2, width_ratios=[1] * (len(PL_VARS) + 1) + [0.06])

    def annotate(ax, m):
        if not ANNOTATE:
            return
        for (r, c), val in np.ndenumerate(m):
            ax.text(c, r, f"{100 * val:.0f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(val) > 0.6 * CLIP else "black")

    im = None
    for p, var in enumerate(PL_VARS):
        ax = fig.add_subplot(gs[0, p])
        m = skill[[vi[f"{var}_{lev}"] for lev in LEVELS]]
        im = ax.imshow(m, **kw)
        annotate(ax, m)
        ax.set_title(var)
        ax.set_yticks(range(len(LEVELS)), LEVELS if p == 0 else [""] * len(LEVELS))
        ax.set_xticks(range(len(LEAD_HOURS))[1::2], LEAD_HOURS[1::2])
        ax.set_xlabel("Lead time (h)")
        if p == 0:
            ax.set_ylabel("Level (hPa)")

    sub = gs[0, len(PL_VARS)].subgridspec(len(SFC_VARS), 1)
    for s, var in enumerate(SFC_VARS):
        ax = fig.add_subplot(sub[s, 0])
        m = skill[[vi[var]]]
        ax.imshow(m, **kw)
        annotate(ax, m)
        ax.set_yticks([])
        ax.set_title(var, fontsize=8, pad=2)
        last = s == len(SFC_VARS) - 1
        ax.set_xticks(range(len(LEAD_HOURS))[1::2], LEAD_HOURS[1::2] if last else [])
        if last:
            ax.set_xlabel("Lead time (h)")

    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, -1]), extend="both")
    cb.set_label(f"RMSE skill score\n(blue = {RUN[0]} better)", fontsize=8)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {out}")


def main():
    mp.set_start_method("spawn", force=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DOMAIN not in ("BE", "BE+UK"):
        raise SystemExit(f"DOMAIN must be 'BE' or 'BE+UK', got {DOMAIN!r} -- the full domain x "
                         f"every variable does not fit in memory")
    varnames = [f"{p}_{lev}" for p in PL_VARS for lev in LEVELS] + SFC_VARS

    months = SEASONS[SEASON]
    fmaps = {}
    for label, d in (RUN, REFERENCE):
        m = {parse_init(f): f for f in sorted(d.glob("forecast_*.nc"))
             if INIT_START <= parse_init(f) <= INIT_END}
        if months:
            m = {i: f for i, f in m.items() if i.month in months}
        print(f"{label}: {len(m)} files")
        fmaps[label] = m
    inits = sorted(set(fmaps[RUN[0]]) & set(fmaps[REFERENCE[0]]))
    if not inits:
        raise SystemExit("no init times common to both runs")
    print(f"Common inits: {len(inits)} | season {SEASON}")

    for label, fm in fmaps.items():
        with h5py.File(str(fm[inits[0]]), "r") as fh:
            gone = [v for v in varnames if v not in fh]
        if gone:
            raise SystemExit(f"{label} forecast files lack {gone} -- drop them from the settings")

    cells = select_cells(DOMAIN)
    ds = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    tvars = list(ds.attrs["variables"])
    gone = [v for v in varnames if v not in tvars]
    if gone:
        raise SystemExit(f"truth zarr lacks {gone}")
    tdates = pd.to_datetime(ds["dates"].values).tz_localize("UTC")
    lat = np.asarray(ds["latitudes"]).ravel()[cells]
    lon = to_180(np.asarray(ds["longitudes"]).ravel())[cells]

    d2i = {d: i for i, d in enumerate(tdates)}
    vtimes = sorted({i + pd.Timedelta(hours=lh) for i in inits for lh in LEAD_HOURS} & set(d2i))
    print(f"Loading truth: {len(vtimes)} times x {len(varnames)} variables x {cells.size} cells ...")
    da = ds["data"].isel(time=[d2i[t] for t in vtimes],
                         variable=[tvars.index(v) for v in varnames], ensemble=0)
    truth = da.isel({da.dims[-1]: cells}).values.astype(np.float32)       # (T, V, C)
    ds.close()
    t_index = {t.isoformat(): i for i, t in enumerate(vtimes)}

    rmse, cnt = {}, {}
    for label, _ in (RUN, REFERENCE):
        rmse[label], cnt[label] = score_run(label, fmaps[label], inits, truth, t_index,
                                            lat, lon, varnames)
    same = np.array_equal(cnt[RUN[0]], cnt[REFERENCE[0]])
    print(f"\nIdentical sample in both runs (every variable x lead): {same}")

    skill = rmse[RUN[0]] / rmse[REFERENCE[0]] - 1.0

    hdr = f"{'variable':10s} " + " ".join(f"{lh:>6d}h" for lh in LEAD_HOURS)
    print(f"\nRMSE SKILL SCORE (%)  {RUN[0]} vs {REFERENCE[0]}  -- negative = {RUN[0]} better")
    print(hdr)
    for v, name in enumerate(varnames):
        print(f"{name:10s} " + " ".join(f"{100 * s:+7.1f}" for s in skill[v]))

    print(f"\nRMSE  (first line {REFERENCE[0]}, second {RUN[0]})")
    print(hdr)
    for v, name in enumerate(varnames):
        print(f"{name:10s} " + " ".join(f"{r:7.3g}" for r in rmse[REFERENCE[0]][v]))
        print(f"{'':10s} " + " ".join(f"{r:7.3g}" for r in rmse[RUN[0]][v]))

    dom = {"BE": "BE farm cells", "BE+UK": "BE+UK farm cells"}[DOMAIN]
    draw(skill, varnames,
         f"RMSE skill: {RUN[0]} vs {REFERENCE[0]} — {dom} ({cells.size} cells, "
         f"{len(inits)} inits, season {SEASON})",
         OUT_DIR / f"scorecard_{RUN[0]}_vs_{REFERENCE[0]}_{DOMAIN}_{SEASON}.png")


if __name__ == "__main__":
    main()
