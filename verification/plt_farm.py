#!/usr/bin/env python3
"""Plot one farm's MEASURED power curve: the production history it is fitted to, the method-of-bins
curve through it, and the manufacturer curve for reference.

Same inputs, window and conventions as verify_power.py's "empirical" baseline (farm_curves.py):
CERRA truth ws100 at the farm's cells, capacity-weighted, averaged over the 3h observation window,
against observed power over that window. Bins summarised by the median; bins under MIN_BIN cases
are dropped, so the drawn curve interpolates only the markers shown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import farm_curves as fc

# ============================== SETTINGS ==============================
FARM       = "Belwind"
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
TRUTH_ZARR = WPOWER_DIR / "Anemoidatasets/power_cerra_A.zarr"
OUT_DIR    = Path("DistrFigures")

TRAIN_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")   # verify_power.py's fit window
TRAIN_END   = pd.Timestamp("2024-01-31 21:00:00", tz="UTC")

WS_EDGES = np.arange(0.0, 25.1, 0.5)   # method-of-bins edges, m/s
MIN_BIN  = 30                          # bins with fewer cases are not used
BIN_STAT = "median"                    # "median" | "mean" -- must match farm_curves.BIN_STAT
SHOW_SPECS = True                      # overlay the manufacturer curve
# ======================================================================

fc.BIN_STAT = BIN_STAT

farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
specs = pd.read_csv(WPOWER_DIR / "turbine_specs.csv", index_col=0)
obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
if obs.index.tz is None:
    obs.index = obs.index.tz_localize("UTC")

farms = [FARM]
turbines = turbines[turbines.farm == FARM]
cap = float(farms_df.set_index("farm").loc[FARM, "capacity_mw"])
fc.validate(farms, farms_df, turbines, specs)

# the curve itself, built exactly as the baseline builds it
curves = fc.empirical(farms, farms_df, turbines, obs, TRUTH_ZARR, TRAIN_START, TRAIN_END,
                      ws_edges=WS_EDGES, min_bin=MIN_BIN, stat=BIN_STAT)
curve = curves[FARM]

# the same wind/power pairs again, to draw the cloud and the surviving bin medians
times, ws_win = fc.farm_truth_wind(farms, turbines, TRUTH_ZARR, TRAIN_START, TRAIN_END)
ws = ws_win[:, 0]
mw = obs[FARM].reindex(times).to_numpy(float)
ok = np.isfinite(ws) & np.isfinite(mw)
ws, mw = ws[ok], mw[ok]

mid = 0.5 * (WS_EDGES[:-1] + WS_EDGES[1:])
idx = np.digitize(ws, WS_EDGES) - 1
agg = np.median if BIN_STAT == "median" else np.mean
val = np.full(mid.size, np.nan)
cnt = np.zeros(mid.size, dtype=int)
for b in range(mid.size):
    k = idx == b
    cnt[b] = int(k.sum())
    if cnt[b] >= MIN_BIN:
        val[b] = agg(mw[k])
use = np.isfinite(val)

print(f"{FARM}: {cap:.1f} MW nameplate, {ok.sum()} cases in "
      f"{TRAIN_START.date()}..{TRAIN_END.date()}, {int(use.sum())} bins of "
      f"{WS_EDGES[1] - WS_EDGES[0]:.1f} m/s with >= {MIN_BIN} cases")
print(f"{'ws bin':>11s} {'cases':>7s} {BIN_STAT + ' MW':>10s} {'% cap':>7s}")
for b in np.where(cnt > 0)[0]:
    flag = "" if use[b] else "   (dropped)"
    v = f"{val[b]:10.1f}" if use[b] else f"{'-':>10s}"
    p = f"{100 * val[b] / cap:6.1f}%" if use[b] else f"{'-':>7s}"
    print(f"{WS_EDGES[b]:5.1f}-{WS_EDGES[b+1]:4.1f} {cnt[b]:7d} {v} {p}{flag}")
print(f"plateau {np.nanmax(val):.1f} MW = {100 * np.nanmax(val) / cap:.1f}% of nameplate")

grid = np.linspace(0.0, 25.0, 501)
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.scatter(ws, mw, s=2, alpha=0.06, color="0.45", linewidths=0, rasterized=True,
           label=f"observed, {ok.sum()} 3h windows")
ax.plot(grid, curve(grid), "-", lw=2.2, color="#0072B2",
        label=f"measured curve (method of bins, {BIN_STAT})")
ax.plot(mid[use], val[use], "o", ms=4, color="#0072B2", label=f"bin {BIN_STAT} (>= {MIN_BIN} cases)")
if SHOW_SPECS:
    spec_curve = fc.build_specs(farms, farms_df, specs)[FARM]
    ax.plot(grid, spec_curve(grid), "--", lw=1.6, color="#D55E00", label="manufacturer curve")
ax.axhline(cap, color="0.3", lw=0.8, ls=":")
ax.text(0.3, cap, f"nameplate {cap:.0f} MW", va="bottom", fontsize=8, color="0.3")

ax.set_xlim(0, 25)
ax.set_ylim(0, cap * 1.12)
ax.set_xlabel("CERRA ws100 at the farm's cells, 3h window mean [m/s]")
ax.set_ylabel("power [MW]")
ax.set_title(f"{FARM} power curve, fitted on {TRAIN_START.date()}..{TRAIN_END.date()}")
ax.grid(alpha=0.25)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
fig.tight_layout()

OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / f"curve_{FARM}.png"
fig.savefig(out, dpi=160)
print(f"wrote {out}")
