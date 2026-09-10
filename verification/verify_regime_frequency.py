#!/usr/bin/env python3
"""How often each wind regime occurs, by season, from CERRA truth ws100 at the farms' cells.

Binned exactly the way verify_power.py bins with REGIME_BY="cerra-ws": the capacity-weighted
mean ws100 over the region's turbine cells, cut on REGIME_WS_EDGES. So the shares printed here
are the ones the power tables are conditioned on, and a regime that looks rare in a seasonal
table is rare in that table's sample for the reason shown here.

Counts are CERRA timesteps. At a 3 h step each one stands for 3 hours, so both the raw count and
the implied hours are printed -- "absolute frequency" means the count in one field and the hours
in the other, and mixing them up is the usual way these tables get misread.

WINDOW picks what is counted: "all" every truth timestep in START..END, "scored" only the
timesteps a forecast is verified at (INIT_START..INIT_END + max lead). Use "scored" when the
numbers have to line up with a verify_power.py run, "all" for the climatology.

Figure: one PNG, grouped bars of absolute counts per season. Every number plotted is printed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import farm_curves as fc

# ============================== SETTINGS ==============================
REGION = "BE"                # "BE" | "UK" | "all"
WINDOW = "all"               # "all" | "scored"  -- see the docstring
START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")   # WINDOW="all" counts this range
END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")
INIT_START = pd.Timestamp("2024-08-01 00:00:00", tz="UTC")   # WINDOW="scored" uses these,
INIT_END   = pd.Timestamp("2025-07-31 21:00:00", tz="UTC")   # to match verify_power.py
MAX_LEAD_H = 36

WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
TRUTH_ZARR = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
OUT_DIR    = Path("DistrFigures")

WS_VAR = "ws100"
STEP_H = 3                   # CERRA timestep, for converting counts to hours

REGIME_WS_EDGES = [4.5, 8.0, 12.0]
REGIME_LABELS   = ["0-4.5", "4.5-8", "8-12", "12+"]
SEASONS = {"DJF": {12, 1, 2}, "MAM": {3, 4, 5}, "JJA": {6, 7, 8}, "SON": {9, 10, 11}}
# ======================================================================

CB_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    farms = (farms_df.farm.tolist() if REGION == "all"
             else farms_df[farms_df.region.str.upper() == REGION].farm.tolist())
    if not farms:
        raise SystemExit(f"no farms for REGION={REGION!r}; "
                         f"have {sorted(farms_df.region.unique())}")
    turbines = turbines[turbines.farm.isin(farms)]
    cap = float(farms_df.set_index("farm").loc[farms, "capacity_mw"].sum())

    if WINDOW == "scored":
        lo, hi = INIT_START, INIT_END + pd.Timedelta(hours=MAX_LEAD_H)
    elif WINDOW == "all":
        lo, hi = START, END
    else:
        raise SystemExit(f"WINDOW must be 'all' or 'scored', got {WINDOW!r}")

    # capacity-weighted CERRA ws100 over the region's turbine cells -- identical to the
    # REGIME_BY="cerra-ws" binning quantity in verify_power.py
    dz = xr.open_zarr(TRUTH_ZARR, consolidated=False)
    tvars = list(dz.attrs["variables"])
    td = pd.to_datetime(dz["dates"].values).tz_localize("UTC")
    glat = np.asarray(dz["latitudes"]).ravel()
    glon = fc.to_180(np.asarray(dz["longitudes"]).ravel())
    ck = np.cos(np.radians(float(glat.mean())))
    _, tc = cKDTree(np.c_[glon * ck, glat]).query(
        np.c_[fc.to_180(turbines.longitude) * ck, turbines.latitude.to_numpy()], k=1)
    t = turbines.assign(cell=tc.astype(int))
    cells = np.sort(t.cell.unique())
    cpos = {int(c): j for j, c in enumerate(cells)}
    keep = np.where((td >= lo) & (td <= hi))[0]
    if keep.size == 0:
        raise SystemExit(f"no truth times in {lo}..{hi}")
    wsc = dz["data"].isel(time=keep, variable=tvars.index(WS_VAR),
                          ensemble=0).values[:, cells].astype(np.float64)
    dz.close()

    w = np.zeros(cells.size)
    for (_, c), mw in t.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        w[cpos[int(c)]] += mw
    ws = wsc @ (w / w.sum())                       # (T,) capacity-weighted regional wind
    times = td[keep]

    print(f"Region {REGION}: {len(farms)} farms, {cap:.0f} MW, {len(cells)} CERRA cells")
    print(f"Window {WINDOW}: {lo.date()}..{hi.date()}  ->  {len(times)} timesteps "
          f"({STEP_H} h each)")
    print(f"Regime edges (m/s): {REGIME_WS_EDGES}")

    reg = np.digitize(ws, REGIME_WS_EDGES)         # 0..3
    mon = times.month.to_numpy()
    nseason, nreg = len(SEASONS), len(REGIME_LABELS)
    counts = np.zeros((nseason, nreg), dtype=int)
    for i, months in enumerate(SEASONS.values()):
        sel = np.isin(mon, list(months))
        for r in range(nreg):
            counts[i, r] = int(np.sum(sel & (reg == r)))

    snames = list(SEASONS)
    row_tot = counts.sum(1)
    col_tot = counts.sum(0)
    tot = counts.sum()

    wid = 9
    print(f"\nABSOLUTE FREQUENCY  [timesteps]   ({STEP_H} h each)")
    print(f"{'season':8s}" + "".join(f"{l:>{wid}s}" for l in REGIME_LABELS)
          + f"{'total':>{wid}s}")
    for i, s in enumerate(snames):
        print(f"{s:8s}" + "".join(f"{c:{wid}d}" for c in counts[i]) + f"{row_tot[i]:{wid}d}")
    print(f"{'TOTAL':8s}" + "".join(f"{c:{wid}d}" for c in col_tot) + f"{tot:{wid}d}")

    print(f"\nABSOLUTE FREQUENCY  [hours]")
    print(f"{'season':8s}" + "".join(f"{l:>{wid}s}" for l in REGIME_LABELS)
          + f"{'total':>{wid}s}")
    for i, s in enumerate(snames):
        print(f"{s:8s}" + "".join(f"{c*STEP_H:{wid}d}" for c in counts[i])
              + f"{row_tot[i]*STEP_H:{wid}d}")
    print(f"{'TOTAL':8s}" + "".join(f"{c*STEP_H:{wid}d}" for c in col_tot)
          + f"{tot*STEP_H:{wid}d}")

    print(f"\nWITHIN EACH SEASON  [% of that season's timesteps]   -- rows sum to 100")
    print(f"{'season':8s}" + "".join(f"{l:>{wid}s}" for l in REGIME_LABELS))
    for i, s in enumerate(snames):
        print(f"{s:8s}" + "".join(f"{100*c/row_tot[i]:{wid}.1f}" for c in counts[i]))
    print(f"{'ALL':8s}" + "".join(f"{100*c/tot:{wid}.1f}" for c in col_tot))

    print(f"\nWITHIN EACH REGIME  [% of that regime's timesteps]   -- columns sum to 100")
    print(f"{'season':8s}" + "".join(f"{l:>{wid}s}" for l in REGIME_LABELS))
    for i, s in enumerate(snames):
        print(f"{s:8s}" + "".join(f"{100*counts[i, r]/col_tot[r]:{wid}.1f}"
                                  for r in range(nreg)))

    print(f"\nmean ws100 by season [m/s]: " + ", ".join(
        f"{s} {ws[np.isin(mon, list(SEASONS[s]))].mean():.2f}" for s in snames))

    # ---------------- figure ----------------
    x = np.arange(nseason)
    bw = 0.8 / nreg
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in range(nreg):
        b = ax.bar(x + (r - (nreg - 1) / 2) * bw, counts[:, r], bw,
                   color=CB_COLORS[r % len(CB_COLORS)],
                   label=f"{REGIME_LABELS[r]} m/s", edgecolor="black", linewidth=0.4)
        ax.bar_label(b, fmt="%d", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(snames)
    ax.set(xlabel="Season", ylabel=f"Timesteps ({STEP_H} h each)")
    ax.set_ylim(0, counts.max() * 1.10)     # headroom so bar_label clears the legend
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    ax.set_axisbelow(True)
    # legend above the axes: inside it collides with the tallest bar's label
    ax.legend(frameon=False, ncol=nreg, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, 1.005))
    ax.set_title(f"{REGION} wind regime frequency by season — capacity-weighted CERRA ws100, "
                 f"{lo.date()}..{hi.date()}", fontsize=11, pad=26)
    fig.tight_layout()
    out = OUT_DIR / f"regime_frequency_{REGION}_{WINDOW}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
