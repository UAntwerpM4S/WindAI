#!/usr/bin/env python3
"""Rank every checkpoint under a training root by FARM-LEVEL power and wind error.

The stock validation metric cannot rank these runs: capacityfactor is a diagnostic, so the inverse
imputer never restores its NaNs (imputer.py:93-98, 260) and the logged metric averages ~99.8%
fabricated zeros. It moved 0.3% across six lead times and called step 499 better than step 4999 --
the checkpoint that actually scored 6.94. So checkpoint selection has to be done from FORECASTS,
which is what this script does -- and it works on checkpoints that already exist.

It scores two numbers per checkpoint, on the VALIDATION window, at the farm cells only:

  POWER  MAE of the direct head's regional total against the observed total, as % of capacity.
         The same construction verify_power.py uses -- capacityfactor * capacity summed over each
         farm's cells -- so the number is on the same scale as the 6.94 headline, just measured on
         2024-02..2024-07 instead of the scored year.
  WIND   MAE of the capacity-weighted farm ws100 against CERRA truth ws100, m/s. Held out from the
         power number entirely: this is the term that says whether a fine-tune bought its power
         skill by damaging the weather.

Those are the validation-window version of the conversion/wind split. A checkpoint that improves
POWER while WIND holds is the one to keep; one that improves POWER while WIND degrades is trading
the trunk away, which is the trade-off cf_head.py exists to navigate.

Forecasts are produced by `anemoi-inference run`, the same binary Inference_loop.py uses, so the
model is exercised exactly as it will be at scoring time. That is the expensive part: one process
per (checkpoint, date). Keep N_DATES small -- the comparison is PAIRED, every checkpoint sees the
identical dates, so it separates runs far more sharply than the absolute sample size suggests.
Existing .nc files are reused, so a second run only fills in what is missing.

Prints one table, ranked by power. Saves one figure: power against step per run, wind against step
per run, and the power/wind scatter that shows the trade-off directly. The table is printed only.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import tempfile
import zipfile
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
REGION      = "BE"

# Name the checkpoints here and nothing is discovered. Absolute paths or globs, both fine.
# Leave the list empty to sweep CKPT_ROOT/CKPT_GLOB instead. Explicit is the sane default: the
# root holds 134 checkpoints across 11 runs, most of them dead ends, and every one costs N_DATES
# inference subprocesses. Keep the 6.94 reference in each sweep so the ranking has an anchor.
_CKPT_DIR = ("/mnt/weatherloss/WindPower/training/WPDistr/VeryHighCapacityGTFinetuneHuber/"
             "checkpoint/")
CHECKPOINTS = [
    _CKPT_DIR + "MixedRollout/inference-*.ckpt",
    _CKPT_DIR + "HuberCFHead5Mixed/inference-*.ckpt",
]

# Unpickling an inference checkpoint imports the decoder by its module path, so `cf_head` must be
# importable in the SUBPROCESS or torch.load fails. Worse, it fails misleadingly: runner.py:589
# calls validate_environment() from inside the except handler, and that itself dies with
#   TypeError: expected string or bytes-like object, got 'DotDict'
# because this env writes module_versions as {"version": "..."} dicts where packaging.Version
# wants a plain string. The DotDict error is the SECOND error; the real one is above it. Setting
# this here means the sweep does not depend on whether the calling shell happened to export it.
TRAINING_DIR = Path("/mnt/weatherloss/WindPower/training")

CKPT_ROOT   = Path("/mnt/weatherloss/WindPower/training/WPDistr/VeryHighCapacityGTFinetuneHuber")
CKPT_GLOB   = "checkpoint/*/inference-*.ckpt"   # inference checkpoints only; the lightning
                                                # .ckpt files are state dicts and cannot be run
ONLY_RUNS   = []          # when sweeping, run directory names to keep. empty means all
VAL_START   = pd.Timestamp("2024-02-01 00:00:00", tz="UTC")   # dataloader.validation window
VAL_END     = pd.Timestamp("2024-07-31 21:00:00", tz="UTC")
N_DATES     = 16          # initial times, spread evenly over the window. paired across runs.

# MATCH verify_power.py's lead range, and do not shorten it to save time. Measured the hard way:
# with SCORE_LEADS stopping at +18h, MixedRollout and HuberCFHead5Mixed came out TIED at 6.09
# %cap, and over epochs 5-9 MixedRollout looked 0.04 WORSE -- while on the test year, scored to
# +33h, MixedRollout beats it by 0.51. The ranking was inverted, not merely blunt.
# The reason is that those two runs differ only in rollout (max 1 vs max 6), and rollout trades
# short-lead accuracy for long-lead accuracy: at R=6 the +3h step carries 1/6 of the gradient.
# Scoring +3..18h samples the half of the range where rollout is a COST. The wind column agreed --
# MixedRollout was 0.022 m/s worse there, which is precisely that price.
LEAD_HOURS  = 37          # inference lead; gives forecast steps at +3h .. +36h
SCORE_LEADS = tuple(range(3, 34, 3))       # +3h .. +33h, the same 11 leads verify_power scores
LONG_FROM   = 21          # leads >= this also get their own column: where rollout actually pays
DEVICE      = "cuda"
SKIP_RUNS   = []          # run directory names to leave out, e.g. a crashed run

WPOWER_DIR  = Path("/mnt/weatherloss/WindPower/data/WPDistr")
TRUTH_ZARR  = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr")
INNER_ZARR  = TRUTH_ZARR
OUTER_ZARR  = Path("/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_era5_A.zarr")

# anemoi-inference can only deliver a forecast by writing a file, so each one lands as a .nc,
# is read, and is deleted immediately -- peak disk is a single forecast and the only thing this
# script leaves behind is the PNG. Set KEEP_FORECASTS = True to cache them instead, which makes
# a re-run with a larger N_DATES cheap; then they accumulate under WORK_DIR and are yours to
# clean up.
KEEP_FORECASTS = False
WORK_DIR    = Path("/mnt/weatherloss/WindPower/verification/_rank_work")
OUT_DIR     = Path("DistrFigures")

WS_VAR, CF_VAR = "ws100", "capacityfactor"
OBS_STEP_H  = 3

# HOW THE FORECAST IS WRITTEN -- and it is NOT interchangeable, which is what cost three rounds
# of debugging here. Measured on this checkpoint in this env:
#   extract_lam   what WPDistr/Inference_loop.py uses, and what produced every forecast
#                 verify_power.py scores. Emits capacityfactor. USE THIS.
#   extract_mask  what NEWENV_Inference_loop.py uses, written for a weather-only checkpoint.
#                 Runs, writes a correct 72,668-cell LAM grid and all 57 weather fields -- and
#                 silently omits capacityfactor. The netcdf writer itself does not filter
#                 (output.py:82 only skips when an explicit `variables` list is set), so the
#                 diagnostic is already absent from the state that post-processor hands it.
#                 A weather-only script never noticed.
# The earlier `TypeError: ... got 'DotDict'` was NOT this setting. It came from torch.load failing
# because cf_head was not importable, and runner.py:589 then calling validate_environment() from
# inside the except handler, where it died on this env's dict-shaped module_versions. TRAINING_DIR
# on PYTHONPATH fixes the real cause; the DotDict was only ever the second exception.
OUTPUT_STYLE = "extract_lam"
OUTPUT_BLOCKS = {
    "extract_mask": ("output:\n  netcdf:\n    post_processors:\n      - extract_mask:\n"
                     "          mask: lam_0/cutout_mask\n          as_slice: true\n"
                     "    path: {path}\n"),
    "extract_lam":  "output:\n  extract_lam:\n    output:\n      netcdf: {path}\n",
    "plain":        "output:\n  netcdf:\n    path: {path}\n",
}
# ======================================================================

STEP_RE = re.compile(r"epoch_(\d+)-step_(\d+)")
_SHOWN_FAILURE = False    # dump the full traceback once, not once per date


def to_180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def is_runnable(ckpt: Path) -> bool:
    """Does this checkpoint carry the supporting arrays anemoi-inference needs?

    An inference checkpoint is a zip holding the pickled model plus an anemoi-metadata/ tree with
    latitudes/longitudes/cutout_mask etc. Runs that died before save_metadata completed leave the
    model without them, and anemoi-inference then fails with `KeyError: 'latitudes'` -- once per
    date, silently burning a subprocess each time. Checking the zip index costs milliseconds and
    skips those before any of that.
    """
    try:
        with zipfile.ZipFile(ckpt) as z:
            return any(n.endswith("anemoi-metadata/latitudes.numpy") for n in z.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def epoch_step(ckpt: Path) -> tuple[int, int]:
    """Which epoch and training step is this checkpoint?

    by_time/by_epoch/by_step files carry both in the filename. `inference-last.ckpt` does not, and
    reporting it as (-1, -1) put a bogus row in the table that duplicated the final epoch. The
    checkpoint's own metadata has the answer -- training.current_epoch / training.global_step, the
    same fields the run record quotes -- so read it from the zip and let the (run, epoch, step)
    de-duplication collapse it against the epoch checkpoint it is a copy of.
    """
    m = STEP_RE.search(ckpt.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    # anemoi.utils.checkpoints writes "anemoi.json"; "ai-models.json" is its DEPRECATED_NAME and
    # older checkpoints still carry that, so accept either.
    try:
        with zipfile.ZipFile(ckpt) as z:
            name = next(n for n in z.namelist()
                        if n.endswith(("anemoi-metadata/anemoi.json",
                                       "anemoi-metadata/ai-models.json")))
            t = json.loads(z.read(name))["training"]
        return int(t["current_epoch"]), int(t["global_step"])
    except Exception:
        return -1, -1


def find_checkpoints():
    """The checkpoints to score: CHECKPOINTS if given, else a sweep of CKPT_ROOT/CKPT_GLOB."""
    if CHECKPOINTS:
        paths = []
        for entry in CHECKPOINTS:
            hits = sorted(Path(p) for p in glob.glob(entry))
            if not hits:
                raise SystemExit(f"CHECKPOINTS entry matched nothing: {entry}")
            paths += hits
    else:
        paths = sorted(CKPT_ROOT.glob(CKPT_GLOB))

    out, skipped, seen = [], [], {}
    for p in dict.fromkeys(paths):          # de-duplicate paths, keep order
        run_id = p.parent.name
        if run_id in SKIP_RUNS or (ONLY_RUNS and run_id not in ONLY_RUNS):
            continue
        if not is_runnable(p):
            skipped.append(p)
            continue
        epoch, step = epoch_step(p)
        # The every_n_minutes and every_n_epochs callbacks both fire at the end of an epoch, so a
        # run typically holds by_time AND by_epoch files for the SAME (epoch, step) -- the same
        # weights under two names. Scoring both would double the inference bill to plot the point
        # twice. Prefer by_epoch, which is the series that is complete for every epoch.
        key = (run_id, epoch, step)
        if key in seen:
            if "by_epoch" in p.name and "by_epoch" not in seen[key]["path"].name:
                seen[key]["path"] = p
            continue
        rec = {"path": p, "run": run_id, "epoch": epoch, "step": step,
               "tag": f"{run_id}_e{epoch:03d}s{step:06d}"}
        seen[key] = rec
        out.append(rec)

    dups = len([p for p in dict.fromkeys(paths) if is_runnable(p)]) - len(out)
    if dups > 0:
        print(f"  {dups} duplicate (epoch, step) saves collapsed -- by_time and by_epoch hold the "
              f"same weights")
    if skipped:
        runs = sorted({p.parent.name for p in skipped})
        print(f"  {len(skipped)} checkpoints carry no supporting arrays and cannot be run "
              f"(runs {', '.join(runs)}) -- skipped without launching inference")
    return out


def run_inference(ckpt: Path, date: pd.Timestamp, out_nc: Path) -> bool:
    """One anemoi-inference call. Returns False if it failed, so one bad checkpoint is skipped
    rather than killing the sweep."""
    if out_nc.exists():
        return True
    cfg = (f"checkpoint: {ckpt}\n"
           f"lead_time: {LEAD_HOURS}\n"
           f'date: "{date.strftime("%Y-%m-%dT%H:%M:%S")}"\n'
           f"device: {DEVICE}\n"
           "input:\n  dataset:\n    dataset:\n      cutout:\n"
           f"        - dataset: {INNER_ZARR}\n"
           f"        - dataset: {OUTER_ZARR}\n"
           "      min_distance_km: 0\n      adjust: all\n"
           + OUTPUT_BLOCKS[OUTPUT_STYLE].format(path=out_nc))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(cfg)
        tmp = fh.name
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{TRAINING_DIR}:{env.get('PYTHONPATH', '')}".rstrip(":")
    r = subprocess.run(["anemoi-inference", "run", tmp],
                       capture_output=True, text=True, env=env)
    Path(tmp).unlink(missing_ok=True)
    if r.returncode != 0 or not out_nc.exists():
        print(f"    FAILED {ckpt.parent.name}/{ckpt.name} @ {date.date()}: "
              f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no output'}")
        # The FIRST failure is nearly always the config schema rather than the checkpoint, and one
        # sanitised line is not enough to fix it. Dump the config and the real traceback once.
        global _SHOWN_FAILURE
        if not _SHOWN_FAILURE:
            _SHOWN_FAILURE = True
            print("\n    --- the config that was run ---")
            print("".join(f"    | {ln}\n" for ln in cfg.splitlines()))
            # NOT the last N lines. runner.py:589 raises a second exception from inside the
            # handler for the first one, so the tail of the traceback is the decoy and the real
            # cause sits above it. Print the whole thing.
            print("    --- full stderr ---")
            print("".join(f"    | {ln}\n" for ln in r.stderr.strip().splitlines()))
        return False
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")

    farms = (farms_df.farm.tolist() if REGION == "all"
             else farms_df[farms_df.region.str.upper() == REGION].farm.tolist())
    turbines = turbines[turbines.farm.isin(farms)]
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    total_cap = float(cap.sum())

    # ---- farm cells on the truth grid, and the reconstruction weights ----
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
    ds.close()

    G = np.zeros((len(farms), cells.size))          # MW of capacity per (farm, cell)
    for (fm, c), mw in turbines.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[farms.index(fm), cpos[int(c)]] = mw
    Wn = G / G.sum(1, keepdims=True)                # capacity-weighted cell -> farm

    # ---- CERRA truth wind at the farms, over the validation window ----
    ttimes, tws = fc.farm_truth_wind(farms, turbines, TRUTH_ZARR, VAL_START, VAL_END)
    ws_truth = pd.DataFrame(tws, index=ttimes, columns=farms)

    dates = pd.date_range(VAL_START, VAL_END - pd.Timedelta(hours=LEAD_HOURS),
                          periods=N_DATES).round("3h")
    dates = pd.DatetimeIndex(sorted(set(dates)))

    ckpts = find_checkpoints()
    print(f"\nRegion {REGION}: {len(farms)} farms, {total_cap:.0f} MW")
    print(f"validation window {VAL_START.date()}..{VAL_END.date()} | {len(dates)} inits "
          f"| leads {SCORE_LEADS} h")
    print(f"{len(ckpts)} checkpoints under {CKPT_ROOT.name}, "
          f"{len({c['run'] for c in ckpts})} runs\n")
    if not ckpts:
        raise SystemExit(f"no checkpoints matched {CKPT_ROOT / CKPT_GLOB}")

    fcells = None
    rows = []
    for k, c in enumerate(ckpts):
        wdir = WORK_DIR / c["tag"]
        wdir.mkdir(exist_ok=True)
        print(f"[{k+1}/{len(ckpts)}] {c['tag']}", flush=True)

        p_err, w_err, p_long = [], [], []
        for d in dates:
            nc = wdir / f"forecast_{d.strftime('%Y%m%d%H%M%S')}.nc"
            if not run_inference(c["path"], d, nc):
                # a checkpoint that cannot forecast one date cannot forecast any of them; the
                # failure is in the checkpoint, not the date. Give up on it rather than paying
                # N_DATES subprocesses to be told the same thing N_DATES times.
                print("    inference failed on the first date -- abandoning this checkpoint")
                break
            try:
                with xr.open_dataset(nc) as fx:
                    if CF_VAR not in fx or WS_VAR not in fx:
                        # Either a weather-only checkpoint, or the output post-processor did not
                        # write what was expected. Those look identical from here, so print the
                        # file's own contents rather than guessing -- the .nc is deleted straight
                        # after, so this is the only chance to see them.
                        print(f"    {CF_VAR!r}/{WS_VAR!r} not in the forecast. It contains:")
                        print("      data_vars: " + ", ".join(sorted(map(str, fx.data_vars))))
                        print("      coords   : " + ", ".join(sorted(map(str, fx.coords))))
                        print(f"      dims     : {dict(fx.sizes)}")
                        break
                    if fcells is None:      # the LAM grid is identical for every run here
                        fl = np.asarray(fx["latitude"].values)
                        fo = to_180(np.asarray(fx["longitude"].values))
                        fk = np.cos(np.radians(float(fl.mean())))
                        _, fcells = cKDTree(np.c_[fo * fk, fl]).query(
                            np.c_[glon[cells] * fk, glat[cells]], k=1)
                    ft = pd.DatetimeIndex(fx["time"].values).tz_localize("UTC")
                    pos = {t: q for q, t in enumerate(ft)}
                    for lead in SCORE_LEADS:
                        j = pos.get(d + pd.Timedelta(hours=lead))
                        if j is None:
                            continue
                        vt = d + pd.Timedelta(hours=lead)
                        mw = fx[CF_VAR].values[j, fcells] @ G.T          # MW per farm
                        ws = fx[WS_VAR].values[j, fcells] @ Wn.T         # m/s per farm
                        o = obs.reindex([vt])[farms].to_numpy(float).ravel()
                        if np.isfinite(o).all():
                            e = abs(float(mw.sum()) - float(o.sum()))
                            p_err.append(e)
                            if lead >= LONG_FROM:
                                p_long.append(e)
                        if vt in ws_truth.index:
                            t = ws_truth.loc[vt].to_numpy(float)
                            m = np.isfinite(t) & np.isfinite(ws)
                            if m.any():
                                w_err.append(float(np.abs(ws[m] - t[m]).mean()))
            finally:
                # the .nc is scratch: everything wanted from it is already in p_err/w_err. Delete
                # it here rather than at the end, so peak disk is one forecast and not 160.
                if not KEEP_FORECASTS:
                    nc.unlink(missing_ok=True)
        if not p_err:
            print("    no scored cases -- skipped")
            continue
        rows.append({**c,
                     "power": 100.0 * float(np.mean(p_err)) / total_cap,
                     "long": 100.0 * float(np.mean(p_long)) / total_cap if p_long else np.nan,
                     "wind": float(np.mean(w_err)) if w_err else np.nan,
                     "n": len(p_err)})
        print(f"    power {rows[-1]['power']:.2f} | +{LONG_FROM}h {rows[-1]['long']:.2f} %cap "
              f"| wind {rows[-1]['wind']:.3f} m/s | n={rows[-1]['n']}")

    if not KEEP_FORECASTS:
        for p in sorted(WORK_DIR.glob("*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        if WORK_DIR.exists() and not any(WORK_DIR.iterdir()):
            WORK_DIR.rmdir()

    if not rows:
        raise SystemExit("nothing scored -- check that anemoi-inference runs on one checkpoint")

    df = pd.DataFrame(rows).sort_values("power").reset_index(drop=True)
    best_p, best_w = df.power.min(), df.wind.min()

    w = max(df.run.str.len().max(), 3)      # run names here are directories, not 32-char ids
    best_l = df["long"].min()
    hdr = (f"{'#':>3} {'run':{w}s} {'epoch':>6s} {'step':>7s} {'POWER %cap':>11s} "
           f"{'vs best':>8s} {f'+{LONG_FROM}h only':>12s} {'vs best':>8s} "
           f"{'WIND m/s':>9s} {'vs best':>8s} {'n':>5s}")
    print(f"\n{'='*len(hdr)}\nRANKED BY FARM-LEVEL POWER MAE, validation window "
          f"{VAL_START.date()}..{VAL_END.date()}\n{'='*len(hdr)}")
    print(hdr)
    for i, r in df.iterrows():
        print(f"{i+1:3d} {r['run']:{w}s} {r['epoch']:6d} {r['step']:7d} "
              f"{r['power']:10.2f}% {r['power']-best_p:+8.2f} {r['long']:11.2f}% "
              f"{r['long']-best_l:+8.2f} {r['wind']:9.3f} "
              f"{r['wind']-best_w:+8.3f} {int(r['n']):5d}")
    print("\n  POWER : MAE of the regional total from the direct head, % of "
          f"{total_cap:.0f} MW. Same construction as verify_power.py, so it is on the same scale")
    print("          as the 6.94 headline -- but measured on Feb-Jul, a lower-wind half-year, so")
    print("          the LEVEL is not comparable to the scored year. Only the ranking is.")
    print(f"  +{LONG_FROM}h    : the same MAE over leads >= {LONG_FROM}h only. THIS is where rollout "
          f"pays: it")
    print("          trades short-lead accuracy for long-lead accuracy, so two runs differing only")
    print("          in rollout separate here and can look tied or inverted in the all-lead column.")
    print("  WIND  : MAE of capacity-weighted farm ws100 against CERRA truth. Held out from the")
    print("          power number, so a run that improves power while wind degrades is visible")
    print("          as trading the trunk away rather than learning the conversion.")
    print(f"  n     : scored (init, lead) cases. Every checkpoint sees the same {len(dates)} inits,")
    print("          so this is a PAIRED comparison -- more sensitive than n alone suggests.")

    # ---- figure ----
    runs = sorted(df.run.unique())
    colour = {r: plt.cm.tab10.colors[i % 10] for i, r in enumerate(runs)}
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for r in runs:
        s = df[df.run == r].sort_values("step")
        axs[0].plot(s.step, s.power, "o-", ms=4, lw=1.3, color=colour[r], label=r)
        axs[1].plot(s.step, s.wind, "o-", ms=4, lw=1.3, color=colour[r])
        axs[2].scatter(s.wind, s.power, s=28, color=colour[r])
    axs[0].set_xlabel("training step"); axs[0].set_ylabel("farm power MAE [% of capacity]")
    axs[0].set_title("Power at the farms — lower is better", fontsize=10)
    axs[1].set_xlabel("training step"); axs[1].set_ylabel("farm ws100 MAE [m/s]")
    axs[1].set_title("Wind at the farms — the trunk's side of the trade", fontsize=10)
    axs[2].set_xlabel("farm ws100 MAE [m/s]"); axs[2].set_ylabel("farm power MAE [% of capacity]")
    axs[2].set_title("The trade-off: bottom-left wins both", fontsize=10)
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, title="run", title_fontsize=7)
    fig.suptitle(f"{CKPT_ROOT.name} — every checkpoint scored at the farm cells on "
                 f"{VAL_START.date()}..{VAL_END.date()}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / f"rank_checkpoints_{REGION}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved: {out}")
    return df        # training/WPDistr/Sweep/sweep.py keeps the scores; run standalone, unused


if __name__ == "__main__":
    main()
