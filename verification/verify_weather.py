#!/usr/bin/env python3
"""RMSE of ONE weather variable against the CERRA truth, by lead time.

Two modes, set by REGIMES:
  REGIMES = False   one RMSE curve per run over the cells DOMAIN selects.
  REGIMES = True    ws100 only, BE farm cells, one panel per wind regime. The regime of each
                    (cell, valid time) is set by the TRUTH ws100 there, so a curve reads "when
                    the true wind was in band B, this run's RMSE was R" -- a conditional on truth.

Truth-conditioned bins are NOT neutral to dispersion: shrinking a forecast's variance adds no
information yet mechanically helps the middle bins and hurts both tails. So "better at 4.5-12,
worse at 0-4.5 and 12+" is the signature of an under-dispersive forecast, not of skill. Set
MATCH_VARIANCE to rescale every run to the truth spread first (one linear map per run+lead,
fitted pooled over regimes -- never per regime). A linear rescale cannot change correlation, so
whatever survives is skill and whatever vanishes was dispersion. sigma_p/sigma_o and r are
printed either way.

Writes one PNG; prints every number it plots. Runs share identical init times and cells.
"""

from pathlib import Path
from multiprocessing import Pool
import multiprocessing as mp
import re
import numpy as np
import pandas as pd
import xarray as xr
import h5py
import netCDF4 as nc4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ============================== SETTINGS ==============================
VARIABLE = "ws100"           # any variable in the truth zarr; regimes need ws100
DOMAIN   = "BE"              # "all" | "BE" | "BE+UK"   (forced to BE when REGIMES)
SEASON   = "all"             # "all" | "DJF" | "MAM" | "JJA" | "SON"  -- filters on INIT month
REGIMES  = False             # ws100 only: split by truth wind regime
MATCH_VARIANCE = False       # regime mode: rescale each run to the truth spread before scoring

FORECAST_DIRS = {
    "RegularWeather":     Path("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"),
    "VanillaCapacityGT":  Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaCapacityGT"),
    "HighCapacityGT":     Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
}

TRUTH_ZARR   = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
TURBMASK_SRC = Path("/mnt/weatherloss/WindPower/data/WPDistr/power_cerra_src.zarr")
TURBINES_CSV = Path("/mnt/weatherloss/WindPower/data/WPDistr/turbines.csv")
OUT_DIR      = Path("DistrFigures")

INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
LEAD_HOURS = list(range(3, 37, 3))

REGIME_EDGES  = [0.0, 4.5, 8.0, 12.0, np.inf]    # m/s, [lo, hi), last open-ended
REGIME_LABELS = ["0-4.5", "4.5-8", "8-12", "12+"]

N_WORKERS = 8
# ======================================================================

SEASONS = {"all": None, "DJF": {12, 1, 2}, "MAM": {3, 4, 5},
           "JJA": {6, 7, 8}, "SON": {9, 10, 11}}
FORECAST_RE = re.compile(r"forecast_(\d{14})")
NMOM = 6          # n, Sy, Syy, Sx, Sxx, Sxy   with y = forecast, x = truth

_W = {}           # per-worker globals, filled once by the Pool initializer


def parse_init(path):
    return pd.to_datetime(FORECAST_RE.search(path.name).group(1),
                          format="%Y%m%d%H%M%S", utc=True)


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def select_cells(domain):
    """CERRA cell indices to score. 'all' = every cell; otherwise the distributed farm cells.

    turbmask is 1 wherever turbines sit and NaN elsewhere, so it is exactly "where the power
    target lives". For BE we intersect it with the BE turbine cells, assigned with the same
    cos-lat KD-tree build_power.py used, so the cells match the target's cells exactly.
    """
    ds = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    n_all = int(np.asarray(ds["latitudes"]).size)
    lat = np.asarray(ds["latitudes"]).ravel()
    lon = to_180(np.asarray(ds["longitudes"]).ravel())
    ds.close()

    if domain == "all":
        print(f"Domain: all | {n_all} cells")
        return np.arange(n_all)

    dsm = xr.open_zarr(TURBMASK_SRC, consolidated=True)
    farm = np.where(np.isfinite(dsm["turbmask"].isel(time=0).values))[0]
    dsm.close()

    if domain == "BE+UK":
        print(f"Domain: BE+UK | {farm.size} farm cells")
        return np.sort(farm)

    t = pd.read_csv(TURBINES_CSV)
    t = t[t["region"].str.upper() == "BE"]
    coslat = np.cos(np.radians(float(lat.mean())))
    tree = cKDTree(np.c_[lon * coslat, lat])
    _, cell = tree.query(np.c_[to_180(t["longitude"]) * coslat, t["latitude"].to_numpy()], k=1)
    sel = np.sort(np.intersect1d(farm, np.unique(cell.astype(int))))
    print(f"Domain: BE | {t.farm.nunique()} farms | {sel.size} of {farm.size} farm cells")
    return sel


def forecast_cell_map(sample_file, lat, lon):
    """Map the selected CERRA cells onto the forecast file's own grid (identity if they match)."""
    with h5py.File(str(sample_file), "r") as f:
        flat = f["latitude"][:]
        flon = to_180(f["longitude"][:])
    coslat = np.cos(np.radians(float(lat.mean())))
    d, j = cKDTree(np.c_[flon * coslat, flat]).query(np.c_[lon * coslat, lat], k=1)
    if d.max() > 1e-6:
        print(f"  grid differs from CERRA: remapped, max offset {d.max()*111:.3f} km")
    return j.astype(int)


def _init_worker(truth, t_index, fc_cells, bins, leads, nreg):
    _W.update(truth=truth, t_index=t_index, fc_cells=fc_cells,
              bins=bins, leads=leads, nreg=nreg)


def _score_file(args):
    """One forecast file -> (L, R, 6) moment sums accumulated over cells and lead times."""
    nc_path, init_iso = args
    leads, nreg = _W["leads"], _W["nreg"]
    acc = np.zeros((len(leads), nreg, NMOM))
    init = pd.Timestamp(init_iso)

    with h5py.File(str(nc_path), "r") as f:
        tv = f["time"]
        raw = nc4.num2date(tv[:], tv.attrs["units"].decode(),
                           tv.attrs.get("calendar", b"standard").decode())
        fmap = {pd.Timestamp(str(t)).tz_localize("UTC").isoformat(): j for j, t in enumerate(raw)}
        var = f[VARIABLE][:, :]

    for k, lh in enumerate(leads):
        vt = (init + pd.Timedelta(hours=lh)).isoformat()
        if vt not in fmap or vt not in _W["t_index"]:
            continue
        y = var[fmap[vt]][_W["fc_cells"]].astype(np.float64)
        x = _W["truth"][_W["t_index"][vt]].astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        reg = np.digitize(x, _W["bins"]) if nreg > 1 else np.zeros(x.size, dtype=int)
        for r in range(nreg):
            m = reg == r
            if not m.any():
                continue
            xs, ys = x[m], y[m]
            acc[k, r] += [xs.size, ys.sum(), (ys * ys).sum(),
                          xs.sum(), (xs * xs).sum(), (xs * ys).sum()]
    return acc


def rmse_from(mom, a=0.0, b=1.0):
    """RMSE of (a + b*forecast) vs truth, straight from the moment sums."""
    n, sy, syy, sx, sxx, sxy = mom
    if n == 0:
        return np.nan
    mse = (a * a * n + b * b * syy + sxx + 2 * a * b * sy - 2 * a * sx - 2 * b * sxy) / n
    return float(np.sqrt(max(mse, 0.0)))


def dispersion(mom):
    """(sigma_p/sigma_o, correlation) from the moment sums."""
    n, sy, syy, sx, sxx, sxy = mom
    if n < 2:
        return np.nan, np.nan
    vy, vx = syy / n - (sy / n) ** 2, sxx / n - (sx / n) ** 2
    cov = sxy / n - (sy / n) * (sx / n)
    return float(np.sqrt(vy / vx)), float(cov / np.sqrt(vy * vx))


def variance_map(mom):
    """b = s_o/s_p, a = m_p(1-b): match the spread, leave the mean bias alone."""
    n, sy, syy, sx, sxx, sxy = mom
    vy, vx = syy / n - (sy / n) ** 2, sxx / n - (sx / n) ** 2
    b = np.sqrt(vx / vy)
    return (sy / n) * (1 - b), b


def main():
    mp.set_start_method("spawn", force=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    regimes = REGIMES
    domain = DOMAIN
    if regimes:
        if VARIABLE != "ws100":
            raise SystemExit(f"REGIMES needs VARIABLE='ws100', got {VARIABLE!r}")
        if domain != "BE":
            print(f"REGIMES: domain forced from {domain!r} to 'BE'")
            domain = "BE"
    nreg = len(REGIME_LABELS) if regimes else 1
    months = SEASONS[SEASON]

    cells = select_cells(domain)

    ds = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    tvars = list(ds.attrs["variables"])
    if VARIABLE not in tvars:
        raise SystemExit(f"{VARIABLE!r} not in truth zarr; have: {tvars}")
    tdates = pd.to_datetime(ds["dates"].values).tz_localize("UTC")
    lat = np.asarray(ds["latitudes"]).ravel()[cells]
    lon = to_180(np.asarray(ds["longitudes"]).ravel())[cells]

    # runs, restricted to the common init times and the season
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
    print(f"Common inits: {len(inits)} | season {SEASON} | variable {VARIABLE}")

    # truth, once, for the selected cells and the valid times actually needed
    d2i = {d: i for i, d in enumerate(tdates)}
    vtimes = sorted({i + pd.Timedelta(hours=lh) for i in inits for lh in LEAD_HOURS} & set(d2i))
    print(f"Loading truth: {len(vtimes)} times x {cells.size} cells ...")
    truth = ds["data"].isel(time=[d2i[t] for t in vtimes],
                            variable=tvars.index(VARIABLE),
                            ensemble=0).values[:, cells].astype(np.float32)
    ds.close()
    t_index = {t.isoformat(): i for i, t in enumerate(vtimes)}
    bins = np.asarray(REGIME_EDGES[1:-1]) if regimes else np.zeros(0)

    results = {}
    for label in fmaps:
        fc_cells = forecast_cell_map(fmaps[label][inits[0]], lat, lon)
        tasks = [(str(fmaps[label][i]), i.isoformat()) for i in inits]
        acc = np.zeros((len(LEAD_HOURS), nreg, NMOM))
        with Pool(N_WORKERS, initializer=_init_worker,
                  initargs=(truth, t_index, fc_cells, bins, LEAD_HOURS, nreg)) as pool:
            for k, a in enumerate(pool.imap_unordered(_score_file, tasks, chunksize=4)):
                acc += a
                if k % 500 == 0:
                    print(f"  {label}: {k}/{len(tasks)}", flush=True)
        results[label] = acc
        print(f"  {label}: done")

    # ---------------- report ----------------
    tag = f"{VARIABLE}_{domain}_{SEASON}" + ("_regimes" if regimes else "")
    if MATCH_VARIANCE and regimes:
        tag += "_matched"

    print(f"\nsigma_p/sigma_o and correlation r (pooled over regimes, per lead)")
    print(f"{'run':22s} " + " ".join(f"{lh:>5d}h" for lh in LEAD_HOURS))
    for label, acc in results.items():
        sd = [dispersion(acc[k].sum(0))[0] for k in range(len(LEAD_HOURS))]
        rr = [dispersion(acc[k].sum(0))[1] for k in range(len(LEAD_HOURS))]
        print(f"{label:22s} " + " ".join(f"{v:6.3f}" for v in sd) + "   sigma_p/sigma_o")
        print(f"{'':22s} " + " ".join(f"{v:6.3f}" for v in rr) + "   r")

    maps = {}
    if MATCH_VARIANCE and regimes:
        for label, acc in results.items():
            maps[label] = [variance_map(acc[k].sum(0)) for k in range(len(LEAD_HOURS))]
        print("\nMATCH_VARIANCE on: each run rescaled to the truth spread per lead "
              "(fitted pooled over regimes).")

    colors, markers = plt.cm.tab10.colors, ["o", "s", "^", "D", "v"]
    if regimes:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for r, (ax, rlab) in enumerate(zip(axes.ravel(), REGIME_LABELS)):
            print(f"\nRMSE  |  regime {rlab} m/s")
            print(f"{'run':22s} " + " ".join(f"{lh:>6d}h" for lh in LEAD_HOURS) + "     n")
            for i, (label, acc) in enumerate(results.items()):
                vals = [rmse_from(acc[k, r], *(maps[label][k] if maps else (0.0, 1.0)))
                        for k in range(len(LEAD_HOURS))]
                n = acc[:, r, 0].sum()
                print(f"{label:22s} " + " ".join(f"{v:7.3f}" for v in vals) + f" {n:10.0f}")
                ax.plot(LEAD_HOURS, vals, marker=markers[i % 5], color=colors[i % 10],
                        lw=1.5, ms=4, label=label)
            ax.set_title(f"{rlab} m/s", fontsize=11)
            ax.grid(True, ls="--", alpha=0.5)
            ax.set_xticks(LEAD_HOURS)
        for ax in axes[1]:
            ax.set_xlabel("Lead time [h]")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"RMSE {VARIABLE} [m/s]")
        axes[0, 0].legend(fontsize=8, framealpha=0.8)
        fig.suptitle(f"{VARIABLE} RMSE by truth wind regime — BE farm cells "
                     f"({cells.size} cells, {len(inits)} inits, season {SEASON})"
                     + (" — variance matched" if maps else ""), fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        print(f"\nRMSE  |  {VARIABLE}, domain {domain}, season {SEASON}")
        print(f"{'run':22s} " + " ".join(f"{lh:>6d}h" for lh in LEAD_HOURS))
        for i, (label, acc) in enumerate(results.items()):
            vals = [rmse_from(acc[k, 0]) for k in range(len(LEAD_HOURS))]
            print(f"{label:22s} " + " ".join(f"{v:7.3f}" for v in vals))
            ax.plot(LEAD_HOURS, vals, marker=markers[i % 5], color=colors[i % 10],
                    lw=1.5, label=label)
        ax.set(xlabel="Lead time [h]", ylabel=f"RMSE {VARIABLE}")
        ax.set_xticks(LEAD_HOURS)
        ax.set_title(f"{VARIABLE} RMSE — {domain} ({cells.size} cells, "
                     f"{len(inits)} inits, season {SEASON})", fontsize=12)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(framealpha=0.8)

    fig.tight_layout()
    out = OUT_DIR / f"rmse_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
