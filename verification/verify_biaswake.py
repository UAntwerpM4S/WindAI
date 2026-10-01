#!/usr/bin/env python3
"""Does the hub-height wind adjustment scale with turbine density? A wake fingerprint test.

verify_biasring.py established that Wx25CF300_20K carries a -0.13 m/s ws100 bias against a
window-matched weather model, confined to the trained cells, reversing within 25 km, and 8x
stronger at 100 m than at 10 m despite identical loss weight. That is consistent with the model
learning an EFFECTIVE, wake-reduced wind -- but curtailment, availability and electrical losses
would all produce the same signature, so "consistent with" is as far as that evidence goes.

THIS test discriminates. Wake losses grow with the number of upstream turbines, so if the
adjustment is wakes it should be LARGER at densely packed cells and smaller at sparse ones.
Curtailment and availability are farm-level and carry no reason to scale with per-cell turbine
density. A monotone relationship is therefore a wake fingerprint; a flat one is not.

Scored on the 15 BE farm cells only, one group per CELL (no pooling), so the per-cell adjustment
can be regressed on that cell's turbine count. 15 points is few -- Spearman is reported rather
than a fit, and the per-cell table is printed so the relationship can be eyeballed rather than
taken on trust.

Same loaders as verify_scorecard / verify_biasring; neither is modified. PNG + printed numbers.
"""

from __future__ import annotations

import multiprocessing as mp
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_weather import TRUTH_ZARR, forecast_cell_map, parse_init, select_cells, to_180

# ============================== SETTINGS ==============================
RUN       = ("Wx25CF300_20K", Path("/mnt/weatherloss/WindPower/inference/WPDistr/Wx25CF300_20K"))
REFERENCE = ("Regular2020",   Path("/mnt/weatherloss/WindPower/inference/WPDistr/Regular2020"))

VARS  = ["ws100", "ws10"]          # ws10 is the control: power does not depend on it
DENSITY_VAR = "turbinecount"       # per-cell, a forcing the model sees
CAP_VAR     = "capacity"

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))
SHOW_LEADS = [12, 24, 36]
FIT_LEAD   = 24                    # the lead the correlation and the scatter use

BLOCK_INITS = 100
TMP_DIR   = None
N_WORKERS = 8
OUT_DIR   = Path("DistrFigures")
# ======================================================================

_W = {}


def _init_worker(fc_by_run, varnames, leads, onehot):
    _W.update(fc=fc_by_run, varnames=varnames, leads=leads, oh=onehot)


def _score_file(args):
    """One forecast file -> (V,L,G) squared-error sum, signed sum and count, by group."""
    path, init_iso, label, tpath, t_index = args
    oh = _W["oh"]
    V, L, G = len(_W["varnames"]), len(_W["leads"]), oh.shape[1]
    sse, se, n = (np.zeros((V, L, G)) for _ in range(3))
    init = pd.Timestamp(init_iso)
    fc = _W["fc"][label]
    truth = np.load(tpath, mmap_mode="r")

    with h5py.File(path, "r") as f:
        tv = f["time"]
        raw = nc4.num2date(tv[:], tv.attrs["units"].decode(),
                           tv.attrs.get("calendar", b"standard").decode())
        fmap = {pd.Timestamp(str(t)).tz_localize("UTC").isoformat(): j for j, t in enumerate(raw)}
        rows = [(k, fmap[vt], t_index[vt]) for k, lh in enumerate(_W["leads"])
                for vt in [(init + pd.Timedelta(hours=lh)).isoformat()]
                if vt in fmap and vt in t_index]
        if not rows:
            return sse, se, n
        ks, fj, tj = (np.array(c) for c in zip(*rows))
        for v, name in enumerate(_W["varnames"]):
            y = f[name][:][fj][:, fc].astype(np.float64)
            x = truth[tj, v].astype(np.float64)
            d = y - x
            ok = np.isfinite(d)
            dz = np.where(ok, d, 0.0)
            sse[v, ks] += (dz * dz) @ oh
            se[v, ks] += dz @ oh
            n[v, ks] += ok.astype(np.float64) @ oh
    return sse, se, n


def main():
    mp.set_start_method("spawn", force=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = (RUN, REFERENCE)

    fmaps = {}
    for label, d in runs:
        m = {parse_init(f): f for f in sorted(d.glob("forecast_*.nc"))
             if INIT_START <= parse_init(f) <= INIT_END}
        print(f"{label}: {len(m)} files")
        fmaps[label] = m
    inits = sorted(set(fmaps[RUN[0]]) & set(fmaps[REFERENCE[0]]))
    if not inits:
        raise SystemExit("no init times common to both runs")
    print(f"Common inits: {len(inits)}")

    cells = select_cells("BE")
    C = cells.size
    ds = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    tvars = list(ds.attrs["variables"])
    for v in VARS + [DENSITY_VAR, CAP_VAR]:
        if v not in tvars:
            raise SystemExit(f"truth zarr lacks {v}")
    tdates = pd.to_datetime(ds["dates"].values).tz_localize("UTC")
    d2i = {d: i for i, d in enumerate(tdates)}
    vidx = [tvars.index(v) for v in VARS]
    lat = np.asarray(ds["latitudes"]).ravel()
    lon = to_180(np.asarray(ds["longitudes"]).ravel())

    # per-cell turbine count and capacity, read inside the scored period so late farms are counted
    mid = tdates[min(range(len(tdates)), key=lambda i: abs(tdates[i] - INIT_END))]
    stat = ds["data"].isel(time=d2i[mid], variable=[tvars.index(DENSITY_VAR), tvars.index(CAP_VAR)],
                           ensemble=0).values[:, cells]
    nturb, cap = np.nan_to_num(stat[0]), np.nan_to_num(stat[1])
    print(f"\nPer-cell statics read at {mid:%Y-%m-%d %H}Z")

    fc_by_run = {label: forecast_cell_map(fmaps[label][inits[0]], lat[cells], lon[cells])
                 for label, _ in runs}
    onehot = np.eye(C)                       # one group per cell: no pooling

    blocks = [inits[i:i + BLOCK_INITS] for i in range(0, len(inits), BLOCK_INITS)]
    tmp = Path(tempfile.mkdtemp(prefix="biaswake_", dir=TMP_DIR))
    V, L = len(VARS), len(LEAD_HOURS)
    acc = {lab: {k: np.zeros((V, L, C)) for k in ("sse", "se", "n")} for lab, _ in runs}

    t0 = time.time()
    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(fc_by_run, VARS, LEAD_HOURS, onehot)) as pool:
        for b, blk in enumerate(blocks):
            vtimes = sorted({i + pd.Timedelta(hours=lh) for i in blk for lh in LEAD_HOURS}
                            & set(d2i))
            da = ds["data"].isel(time=[d2i[t] for t in vtimes], variable=vidx,
                                 ensemble=0).isel({ds["data"].dims[-1]: cells})
            tpath = tmp / f"truth_block{b:03d}.npy"
            mm = np.lib.format.open_memmap(tpath, mode="w+", dtype=np.float32,
                                           shape=(len(vtimes), V, C))
            mm[:] = da.values
            mm.flush()
            del mm
            t_index = {t.isoformat(): r for r, t in enumerate(vtimes)}
            for label, _ in runs:
                tasks = [(str(fmaps[label][i]), i.isoformat(), label, str(tpath), t_index)
                         for i in blk]
                for s, sg, m in pool.imap_unordered(_score_file, tasks, chunksize=2):
                    acc[label]["sse"] += s
                    acc[label]["se"] += sg
                    acc[label]["n"] += m
            tpath.unlink()
            done = sum(len(x) for x in blocks[:b + 1])
            el = time.time() - t0
            print(f"  block {b + 1}/{len(blocks)}: {done}/{len(inits)} | {el/60:.1f} min",
                  flush=True)
    ds.close()
    tmp.rmdir()

    bias = {lab: acc[lab]["se"] / acc[lab]["n"] for lab, _ in runs}
    r, q = RUN[0], REFERENCE[0]
    diff = bias[r] - bias[q]                      # (V, L, C): the adjustment, per cell
    print(f"\nIdentical sample in both runs: "
          f"{np.array_equal(acc[r]['n'], acc[q]['n'])}")

    ki = [LEAD_HOURS.index(h) for h in SHOW_LEADS]
    kf = LEAD_HOURS.index(FIT_LEAD)
    order = np.argsort(-nturb)                    # densest first

    for v, name in enumerate(VARS):
        print("\n" + "=" * 96)
        print(f"{name}: per-cell bias adjustment ({r} minus {q}), BE farm cells, densest first")
        print("=" * 96)
        print(f"{'cell':>7s} {'turbines':>9s} {'MW':>7s} "
              + "".join(f"{str(h) + 'h':>10s}" for h in SHOW_LEADS))
        for i in order:
            print(f"{cells[i]:7d} {nturb[i]:9.0f} {cap[i]:7.1f} "
                  + "".join(f"{diff[v, k, i]:10.4f}" for k in ki))

        d = diff[v, kf]
        for lab, x in (("turbines", nturb), ("capacity", cap)):
            rho, p = spearmanr(x, d)
            print(f"  Spearman({lab}, adjustment) at +{FIT_LEAD}h: rho = {rho:+.2f}, p = {p:.3f}"
                  f"   (wake predicts rho < 0: denser -> more negative)")
        t = np.array_split(order, 3)
        print("  terciles by turbine count (densest -> sparsest), mean adjustment at "
              f"+{FIT_LEAD}h: "
              + "  ".join(f"{nturb[g].mean():.0f} turb -> {d[g].mean():+.4f}" for g in t))

    fig, axs = plt.subplots(1, len(VARS), figsize=(5.5 * len(VARS), 4.5), squeeze=False)
    for ax, (v, name) in zip(axs[0], enumerate(VARS)):
        ax.scatter(nturb, diff[v, kf], s=40, c=cap, cmap="viridis")
        for i in range(C):
            ax.annotate(f"{cells[i]}", (nturb[i], diff[v, kf, i]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="0.5", lw=1)
        ax.set_xlabel("turbines in cell")
        ax.set_ylabel(f"{name} bias adjustment [m/s]")
        rho, p = spearmanr(nturb, diff[v, kf])
        ax.set_title(f"{name} at +{FIT_LEAD}h   rho={rho:+.2f}, p={p:.3f}", fontsize=10)
        ax.grid(ls=":", alpha=0.5)
    fig.suptitle(f"{r} minus {q}: does the adjustment scale with turbine density? "
                 f"({len(inits)} inits, {C} BE cells)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / f"biaswake_{r}_vs_{q}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
