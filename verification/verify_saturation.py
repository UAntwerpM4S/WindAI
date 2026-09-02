#!/usr/bin/env python3
"""Above rated, did the head not KNOW, or know and not COMMIT?

The head never over-predicts at or above rated output. Two explanations:

  IT DID NOT KNOW   its own wind forecast is wrong at those moments, so it had no way to tell
                    it should be at full output.
  IT KNEW           its wind forecast correctly says 18 m/s, but the power head still hedges,
                    because in training every wind above rated produced the same target and it
                    never learned the top of the range is somewhere to go.

The loss-weight invariance and the lead-independence both argue for the second. Neither shows it.
This does, by comparing two clamps that differ ONLY in what triggers them:

  CLAMP-MODEL   wherever the MODEL's own ws100 at the farm is at or above that farm's rated wind
                speed, replace the head's power with the farm's plateau.
  CLAMP-TRUTH   the same, triggered by CERRA's ws100 instead. The oracle: the best any trigger
                could do.

If CLAMP-MODEL scores like CLAMP-TRUTH, the model's own wind was already sufficient and the
failure lives in the power head, not in the wind -- the censoring claim, demonstrated rather than
inferred. If CLAMP-MODEL is much worse, the wind was the problem and the account is incomplete.

This is a DIAGNOSTIC, not a proposed forecast. Clamping is a power curve bolted onto a model meant
to replace power curves; the principled fix is an objective that does not return the conditional
mean of a censored target. The point here is only to locate the failure.

Two quantities come from outside the model, both stated rather than fitted:
  * RATED WIND per farm -- capacity-weighted rated wind speed of its fleet, from
    turbine_specs.csv. Metadata, not tuned.
  * PLATEAU per farm -- median OBSERVED power when CERRA wind is well above rated (PLATEAU_WS).
    One number per farm, from observations the model trained on. NOT nameplate: real farms
    plateau below it, and by how much differs per farm.
We deliberately do not read the plateau off the model's own implied curve: that curve tops out
near 60% of capacity, which IS the failure being measured.

The plateau table also prints the MEAN observed power above rated. MSE fits the conditional mean,
so if above-rated output is left-skewed the head is right to sit below the plateau, and that part
of the under-prediction is a property of the target rather than a defect. Only the remainder is
the head's own compression. mean-minus-median quantifies the split.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# ============================== SETTINGS ==============================
REGION     = "BE"
RATED_BAND = 12.0             # m/s on CERRA truth: the regime scored (matches the paper's)
PLATEAU_WS = 15.0             # m/s on CERRA truth: where the observed plateau is measured
LOW_FRAC   = 0.5              # "low tail" = above-rated hours producing below this x the plateau
LEADS      = list(range(3, 34, 3))

FORECAST_DIRS = {
    "HighCapacityGT":     Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VeryHighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT"),
}
TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")

WS_VAR, CF_VAR = "ws100", "capacityfactor"
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
# ======================================================================

FORECAST_RE = re.compile(r"forecast_(\d{14})")
FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def parse_init(p):
    return pd.to_datetime(FORECAST_RE.search(p.name).group(1), format="%Y%m%d%H%M%S", utc=True)


def fleet_rated_ws(fleet, specs):
    """Capacity-weighted rated wind speed of a farm's fleet. Metadata, not fitted."""
    w = []
    for chunk in str(fleet).split(";"):
        m = FLEET_RE.match(chunk)
        if m and m.group(2) in specs.index:
            n, s = int(m.group(1)), specs.loc[m.group(2)]
            w.append((n * float(s["rated_power_mw"]), float(s["rated_ws_ms"])))
    if not w:
        raise SystemExit(f"cannot resolve fleet {fleet!r} against turbine_specs.csv")
    return sum(c * v for c, v in w) / sum(c for c, _ in w)


def main():
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
    tsum = turbines.groupby("farm")["capacity_mw"].sum().reindex(farms)
    if ((tsum - cap).abs() / cap > 0.001).any():
        raise SystemExit("turbines.csv and farms.csv capacities disagree -- rerun farm_metadata.py")
    rated = {f: fleet_rated_ws(farms_df.set_index("farm").loc[f, "fleet"], specs) for f in farms}
    print(f"Region {REGION}: {len(farms)} farms, {float(cap.sum()):.0f} MW\n")

    # ---- CERRA wind at the farm cells ----
    ds = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    tvars = list(ds.attrs["variables"])
    tdates = pd.to_datetime(ds["dates"].values).tz_localize("UTC")
    glat = np.asarray(ds["latitudes"]).ravel()
    glon = to_180(np.asarray(ds["longitudes"]).ravel())
    ck = np.cos(np.radians(float(glat.mean())))
    _, tc = cKDTree(np.c_[glon * ck, glat]).query(
        np.c_[to_180(turbines.longitude) * ck, turbines.latitude.to_numpy()], k=1)
    turbines = turbines.assign(cell=tc.astype(int))
    cells = np.sort(turbines.cell.unique())
    cpos = {int(c): j for j, c in enumerate(cells)}
    tsel = np.where((tdates >= INIT_START) & (tdates <= INIT_END))[0]
    ws_c = ds["data"].isel(time=tsel, variable=tvars.index(WS_VAR),
                           ensemble=0).values[:, cells].astype(np.float64)
    ds.close()
    times = tdates[tsel]
    trow = {t: j for j, t in enumerate(times)}

    G = np.zeros((len(farms), cells.size))
    for (fm, c), mw in turbines.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[farms.index(fm), cpos[int(c)]] = mw
    Wn = G / G.sum(1, keepdims=True)
    ws_t = ws_c @ Wn.T

    # ---- the plateau: median OBSERVED power well above rated ----
    # MEAN is printed beside it because MSE fits the conditional MEAN, not the median. If
    # above-rated output is left-skewed (curtailment, outages, derating) the mean sits well below
    # the plateau, and a head trained under MSE is CORRECT to aim there. That gap is the part of
    # the under-prediction that is a property of the target, not a failure of the model; whatever
    # remains below the mean is the head's own compression. LOW = share of above-rated hours below
    # LOW_FRAC of the plateau, i.e. how heavy the low tail doing the pulling actually is.
    print(f"{'farm':14s} {'rated ws':>9s} {'plateau':>10s} {'of nameplt':>11s} "
          f"{'mean':>9s} {'mean-med':>9s} {'low tail':>9s} {'n':>7s}")
    plateau, above_mean = {}, {}
    for i, f in enumerate(farms):
        o = obs[f].reindex(times).to_numpy(float)
        m = np.isfinite(o) & (ws_t[:, i] >= PLATEAU_WS)
        plateau[f] = float(np.median(o[m]))
        above_mean[f] = float(np.mean(o[m]))
        low = float(np.mean(o[m] < LOW_FRAC * plateau[f]))
        print(f"{f:14s} {rated[f]:8.1f}m/s {plateau[f]:9.1f}MW {100*plateau[f]/cap[f]:10.1f}% "
              f"{above_mean[f]:8.1f}MW {100*(above_mean[f]-plateau[f])/cap[f]:+8.1f}% "
              f"{100*low:8.1f}% {int(m.sum()):7d}")
    gap = 100 * sum(above_mean[f] - plateau[f] for f in farms) / float(cap.sum())
    print(f"  NB the plateau is NOT nameplate -- that gap is the loss the specs curve misses.")
    print(f"  fleet mean-minus-median above rated: {gap:+.1f}% of capacity. Compare this against")
    print(f"  the head's above-rated bias below: that share of it is the target's skew, not the head.")

    # ---- score the four series in the above-rated regime ----
    print(f"\n{'='*104}\nABOVE RATED (CERRA ws >= {RATED_BAND} m/s), leads "
          f"+{LEADS[0]}..{LEADS[-1]}h\n{'='*104}")
    for label, d in FORECAST_DIRS.items():
        files = {parse_init(p): p for p in sorted(d.glob("forecast_*.nc"))
                 if INIT_START <= parse_init(p) <= INIT_END}
        if not files:
            print(f"\n{label}: no forecast files -- skipped"); continue
        with xr.open_dataset(sorted(files.values())[0]) as f0:
            if CF_VAR not in f0:
                print(f"\n{label}: no {CF_VAR!r} -- skipped"); continue
            fl, fo = np.asarray(f0["latitude"].values), to_180(np.asarray(f0["longitude"].values))
            nfc = int(f0[CF_VAR].shape[1])
        fk = np.cos(np.radians(float(fl.mean())))
        _, fc = cKDTree(np.c_[fo * fk, fl]).query(np.c_[glon[cells] * fk, glat[cells]], k=1)

        print(f"\nReading {label}: {len(files)} files ...")
        acc = {k: np.zeros((len(farms), 4)) for k in ("sae", "sb")}
        cnt = np.zeros(len(farms))
        trig = np.zeros((len(farms), 2))
        for q, (init, fp) in enumerate(sorted(files.items())):
            if q % 500 == 0:
                print(f"  {q}/{len(files)}", flush=True)
            with xr.open_dataset(fp) as fx:
                ft = pd.DatetimeIndex(fx["time"].values).tz_localize("UTC")
                pos = {t: j for j, t in enumerate(ft)}
                if fx[CF_VAR].shape[1] != nfc:
                    raise SystemExit(f"{fp.name}: this run's grid is not constant")
                need = [(lh, pos[init + pd.Timedelta(hours=lh)]) for lh in LEADS
                        if init + pd.Timedelta(hours=lh) in pos]
                if not need:
                    continue
                idx = [j for _, j in need]
                mws = fx[WS_VAR].values[idx][:, fc] @ Wn.T
                mp = fx[CF_VAR].values[idx][:, fc] @ G.T
            for k, (lh, _) in enumerate(need):
                vt = init + pd.Timedelta(hours=lh)
                r = trow.get(vt)
                if r is None or vt not in obs.index:
                    continue
                o = obs.loc[vt, farms].to_numpy(float)
                for i, f in enumerate(farms):
                    if not (np.isfinite(o[i]) and np.isfinite(mp[k, i])
                            and ws_t[r, i] >= RATED_BAND):
                        continue
                    fire_m = mws[k, i] >= rated[f]
                    fire_t = ws_t[r, i] >= rated[f]
                    series = np.array([
                        mp[k, i],
                        plateau[f] if fire_m else mp[k, i],
                        plateau[f] if fire_t else mp[k, i],
                        plateau[f]])
                    e = series - o[i]
                    acc["sae"][i] += np.abs(e); acc["sb"][i] += e
                    cnt[i] += 1
                    trig[i] += (fire_m, fire_t)

        nm = ["as-is", "clamp on MODEL wind", "clamp on TRUTH wind", "always plateau"]
        mae = 100 * acc["sae"] / cnt[:, None] / cap.to_numpy()[:, None]
        bias = 100 * acc["sb"] / cnt[:, None] / cap.to_numpy()[:, None]
        print(f"\n{label} — MAE as % of capacity, above rated")
        print(f"{'farm':14s} {'n':>7s} " + " ".join(f"{s:>21s}" for s in nm) + "   trigger agree")
        for i, f in enumerate(farms):
            ag = 100 * min(trig[i]) / max(trig[i]) if max(trig[i]) else np.nan
            print(f"{f:14s} {int(cnt[i]):7d} " +
                  " ".join(f"{mae[i,j]:20.2f}%" for j in range(4)) + f"{ag:14.0f}%")
        w = cnt / cnt.sum()
        tot = (mae * w[:, None]).sum(0)
        tb = (bias * w[:, None]).sum(0)
        print(f"{'FLEET':14s} {int(cnt.sum()):7d} " + " ".join(f"{v:20.2f}%" for v in tot))
        print(f"{'  bias':14s} {'':7s} " + " ".join(f"{v:+20.2f}%" for v in tb))
        rec = 100 * (tot[0] - tot[1]) / tot[0]
        head = tot[0] - tot[2]
        print()
        if head <= 0:
            print(f"  NOTHING TO RECOVER: as-is ({tot[0]:.2f}%) is already better than the oracle")
            print(f"  clamp ({tot[2]:.2f}%), so this run does not have the failure being tested.")
            print(f"  The recovery percentages are meaningless here; read `trigger agree` instead.")
        else:
            print(f"  the model-wind clamp removes {rec:.0f}% of the above-rated error,")
            print(f"  which is {100*(tot[0]-tot[1])/head:.0f}% of what the TRUTH-wind clamp achieves.")
        print( "  near 100% of the oracle => the model's own wind was already sufficient, and the")
        print( "  failure sits in the power head rather than in the wind. Below ~70% => the wind")
        print( "  is part of the problem and the censoring account is incomplete.")


if __name__ == "__main__":
    main()
