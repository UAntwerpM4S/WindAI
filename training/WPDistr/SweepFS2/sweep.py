#!/usr/bin/env python3
"""Sweep on top of FS2: how much weather can the power fine-tune keep, and at what power cost?

    python sweep.py dry       write every run's YAML, check the control against FS2's own stored
                              config, print the plan and relative cost. No GPU. Run this first.
    nohup python sweep.py > sweep.out 2>&1 &
                              the sweep. Resumable: start it again and it carries on.
    python sweep.py report    table + figure from whatever has finished. Safe while it runs.

FS2 = the 5k power fine-tune (encoder frozen, Huber on ws10/ws100 at 0.5, MAE on power at 100,
every other weather variable at 0) on the base trained with the backward window from the start
(VHCapacityBackWin, stages 1-2). Test year: 6.04 % MAE on the BE total -- but the scorecard shows
every weather field it gives no weight drifting, z/t/q up to +30..+90 % RMSE at 36 h. This round
gives those fields a weight again, as a FRACTION of their pre-training weights, and measures the
trade: power (farm total, validation window) against weather (domain-wide RMSE vs the stage-2 base).

Each run: train (anemoi-training, both GPUs) -> score SCORE_POINTS of its epoch checkpoints on the
VALIDATION window with rank_checkpoints.py, now also scoring WEATHER_VARS over the whole domain ->
redraw the summary. The stage-2 base is scored once, untrained, as the weather reference: it is
the model before any fine-tune, so "weather vs base" is exactly what the fine-tune cost.

CONTROLS: FS2 itself (its own checkpoints, anchor seed) and ctrl_s2 (same recipe, another seed).
Their spread is the band a run must clear. Every other run uses the anchor seed -> paired with FS2.

SELECTION: a run is interesting if its power is inside or below the control band AND its weather
is clearly closer to the base than FS2's. Rank by the FINAL checkpoint only. Take at most the one
or two best to the test year (verify_power + verify_scorecard); do not pick on validation noise.

Weights note: the weather term is Huber(delta=1), which is 0.5*MSE for normalised errors < 1, so
"10 % of the pre-training weights" here is ~5 % in the MSE units pre-training used.
"""

from __future__ import annotations

import contextlib
import copy
import glob
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ============================== SETTINGS ==============================
REPO         = Path("/mnt/weatherloss/WindPower")
BASE_YAML    = REPO / "training/WPDistr/VHCapacityBackWin/VHCapacityBackWin.yaml"   # = FS2's recipe
TRAINING_DIR = REPO / "training"            # rollout_tasks, cf_head, combined_loss -> PYTHONPATH
VERIF_DIR    = REPO / "verification"        # rank_checkpoints.py, farm_curves.py
SWEEP_ROOT   = REPO / "training/WPDistr/SweepFS2"
ANCHOR_DIR   = REPO / "training/WPDistr/VHCapacityBackWinFinetune/checkpoint/f9ff915ed31f4356b1da9c48217377fc"
ANCHOR_NAME  = "FS2"
# the stage-2 base FS2 was fine-tuned from, scored untrained: the weather reference
BASE_CKPT    = str(REPO / "training/WPDistr/VHCapacityBackWin/checkpoint/"
                   "538e92ed028f46bf9a0826ef7bb3dee3/inference-*step_007500*.ckpt")
BASE_NAME    = "base_7500"

# Domain-wide weather check: RMSE over every WX_STRIDE-th inner cell. The fields the scorecard
# showed drifting most (z, t, q) plus the surface fields and the farm-relevant wind.
WEATHER_VARS = ("z_500", "z_850", "t_850", "q_850", "u_850", "msl", "t2m", "ws100")
WX_STRIDE    = 7

VAL_BATCHES  = 50        # per GPU; validation is a health monitor only (see the old sweep.py)

# Keys that may differ from FS2's stored config without invalidating the control.
IGNORE_DIFF  = ("system.output", "diagnostics", "dataloader.limit_batches.validation",
                "dataloader.num_workers", "dataloader.prefetch_factor", "training.run_id",
                "training.fork_run_id", "graph.overwrite", "defaults", "data.num_features")

SAME = "anchor"          # seed sentinel: FS2's own seed, read from its metadata

# Pre-training general_variable weights (stages 1-2), the reference the fractions scale.
PRETRAIN = {"default": 1.0, "q": 0.6, "t": 6.0, "u": 0.8, "v": 0.5, "w": 0.001, "z": 12.0,
            "mcc": 0.1}
WXKEY = "training.scalers.weather_variable.weights"


def wx(frac):
    """weather_variable weights: every weather variable at `frac` of its pre-training weight;
    ws10/ws100 stay at FS2's 0.5 (they are the wind the power needs), power stays out (0)."""
    w = {k: round(v * frac, 6) for k, v in PRETRAIN.items()}
    w.update(ws10=0.5, ws100=0.5, capacityfactor=0)
    return w


CONTROLS = [ANCHOR_NAME, "ctrl_s2"]

# Priority order: what the question needs most comes first, in case the sweep is cut short.
#   name            seed  overrides on top of BASE_YAML                       description
RUNS = [
    ("wx10",          SAME, {WXKEY: wx(0.10)},                                 "weather x0.10"),
    ("ctrl_s2",       5678, {},                                                "FS2 recipe, seed 5678"),
    ("wx25",          SAME, {WXKEY: wx(0.25)},                                 "weather x0.25"),
    ("wx05",          SAME, {WXKEY: wx(0.05)},                                 "weather x0.05"),
    # the other end of the trade: processor frozen too, so only the decoder can drift the weather
    ("frozen_trunk",  SAME, {"training.submodules_to_freeze": ["encoder", "processor"]},
                                                                               "encoder+processor frozen"),
    ("wx025",         SAME, {WXKEY: wx(0.025)},                                "weather x0.025"),
    # round-3 shape (every weather variable 0.1, not proportional): tied power, forward window
    ("wxuni01",       SAME, {WXKEY: {"default": 0.1, "ws10": 0.5, "ws100": 0.5, "capacityfactor": 0}},
                                                                               "weather uniform 0.1"),
    # keep the power share of the loss what it was in FS2 while the weather term grows
    ("wx10_cf200",    SAME, {WXKEY: wx(0.10),
                             "training.scalers.power_variable.weights.capacityfactor": 200},
                                                                               "weather x0.10, power 200"),
    ("wx25_cf300",    SAME, {WXKEY: wx(0.25),
                             "training.scalers.power_variable.weights.capacityfactor": 300},
                                                                               "weather x0.25, power 300"),
    ("wx10_s2",       5678, {WXKEY: wx(0.10)},                                 "weather x0.10, seed 5678"),
    # a smaller step moves the trunk less for the same power gradient
    ("wx10_lr1.5e-5", SAME, {WXKEY: wx(0.10), "training.lr.rate": 1.5e-5},     "weather x0.10, lr 1.5e-5"),
    # two objectives may need longer to settle than one
    ("wx10_10k",      SAME, {WXKEY: wx(0.10), "training.max_steps": 10000,
                             "training.lr.iterations": 10000},                 "weather x0.10, 10k steps"),
]

SCORE_POINTS = 5         # epoch checkpoints scored per run, evenly spread, always incl. the last
N_DATES      = 48        # validation inits; scores are only comparable at equal N_DATES
TRAIN_TIMEOUT_H = 12     # a hung DDP job must not eat the weekend
RETRY_FAILED = False     # True: re-attempt runs that failed on an earlier start
# ======================================================================

sys.path.insert(0, str(VERIF_DIR))
import rank_checkpoints as rc      # noqa: E402

STATE = SWEEP_ROOT / "logs" / "sweep_state.json"
FIG = SWEEP_ROOT / "sweep_fs2.png"


class _NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):  # write the reorder map out twice rather than as &id001
        return True


def _step(d, k):
    # a numeric segment indexes a list: "training.training_loss.losses.0.delta"
    return d[int(k)] if isinstance(d, list) else d[k]


def dset(d, key, value):
    *head, last = key.split(".")
    for k in head:
        d = _step(d, k)
    if isinstance(d, list):
        assert int(last) < len(d), f"{key}: index out of range in {BASE_YAML.name}"
        d[int(last)] = value
        return
    assert last in d, f"{key}: no such key in {BASE_YAML.name} -- typo?"
    d[last] = value


def leaves(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, f"{prefix}{k}.")
    else:
        yield prefix[:-1], d


def same(a, b):
    # an interpolation on either side cannot be compared -- one file may store it resolved
    if (isinstance(a, str) and "${" in a) or (isinstance(b, str) and "${" in b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)


def anchor_metadata():
    """(stored config, seed, final checkpoint) of FS2."""
    cks = [p for p in ANCHOR_DIR.glob("inference-*.ckpt") if rc.is_runnable(p)]
    if not cks:
        raise SystemExit(f"no runnable inference checkpoint in {ANCHOR_DIR}")
    last = max(cks, key=lambda p: rc.epoch_step(p)[1])
    with zipfile.ZipFile(last) as z:
        name = next(n for n in z.namelist()
                    if n.endswith(("anemoi-metadata/anemoi.json", "anemoi-metadata/ai-models.json")))
        meta = json.loads(z.read(name))
    return meta["config"], int(meta["seed"]), last


def build_config(base, name, overrides):
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        dset(cfg, k, v)
    dset(cfg, "system.output.root", str(SWEEP_ROOT / name))
    dset(cfg, "training.run_id", None)
    dset(cfg, "training.fork_run_id", None)
    dset(cfg, "dataloader.limit_batches.validation", VAL_BATCHES)
    dset(cfg, "diagnostics.checkpoint.every_n_minutes.save_frequency", 5)
    dset(cfg, "diagnostics.log.mlflow.tracking_uri", f"file:{SWEEP_ROOT}/logs/mlflow")
    dset(cfg, "diagnostics.log.mlflow.experiment_name", "SweepFS2")
    dset(cfg, "diagnostics.log.mlflow.project_name", "SweepFS2")
    dset(cfg, "diagnostics.log.mlflow.run_name", name)
    return cfg


def rel_cost(cfg, ref):
    """Training cost in rollout-steps, relative to FS2. Rollout grows at EPOCH end, so epoch e
    runs at min(start + ceil(e/inc), max)."""
    def units(c):
        t, r = c["training"], c["training"]["rollout"]
        spe, n = c["dataloader"]["limit_batches"]["training"], t["max_steps"]
        tot = 0
        for e in range(math.ceil(n / spe)):
            R = min(r["start"] + (math.ceil(e / r["epoch_increment"]) if r["epoch_increment"] else 0),
                    r["max"])
            tot += min(spe, n - e * spe) * R
        return tot
    return units(cfg) / units(ref)


def check_control(ctrl, stored):
    """The control config against FS2's stored one, in BOTH directions."""
    absent = "(absent)"

    def get(d, key):
        for k in key.split("."):
            if isinstance(d, str) and "${" in d:
                return d
            try:
                d = d[k]
            except (KeyError, TypeError):
                return absent
        return d

    diffs = []
    for key in dict.fromkeys([k for k, _ in leaves(ctrl)] + [k for k, _ in leaves(stored)]):
        if key.startswith(IGNORE_DIFF):
            continue
        mine, theirs = get(ctrl, key), get(stored, key)
        if not same(mine, theirs):
            diffs.append((key, mine, theirs))
    return diffs


def pick(ckpt_dir):
    """SCORE_POINTS runnable epoch checkpoints, evenly spread, ending at the last one."""
    by_step = {}
    for p in sorted(Path(ckpt_dir).glob("**/inference-*.ckpt")):
        if not rc.is_runnable(p):
            continue
        step = rc.epoch_step(p)[1]
        if step not in by_step or "by_epoch" in p.name:
            by_step[step] = p
    steps = sorted(by_step)
    if not steps:
        return []
    k = len(steps) / SCORE_POINTS
    idx = sorted({max(0, round(len(steps) - 1 - i * k)) for i in range(SCORE_POINTS)})
    return [by_step[steps[i]] for i in idx]


def patch_checkpoints(paths):
    """Clear dataset.variables_metadata, or anemoi-inference dies with KeyError: 'latitudes'."""
    from anemoi.utils.checkpoints import load_metadata, replace_metadata
    n = 0
    for p in paths:
        meta, arrays = load_metadata(p, supporting_arrays=True)
        if not meta.get("dataset", {}).get("variables_metadata"):
            continue
        meta["dataset"]["variables_metadata"] = {}
        with contextlib.redirect_stderr(io.StringIO()):
            replace_metadata(p, meta, arrays)
        n += 1
    if n:
        print(f"    patched {n} checkpoint(s): cleared dataset.variables_metadata", flush=True)


def score(ckpts, work):
    """rank_checkpoints on these checkpoints, backward window on, weather check on."""
    ckpts = [Path(p) for p in ckpts]
    if not ckpts:
        return [], "no runnable checkpoint"
    patch_checkpoints(ckpts)
    rc.CHECKPOINTS = [str(p) for p in ckpts]
    rc.CKPT_ROOT, rc.OUT_DIR, rc.WORK_DIR = ckpts[0].parent, work, work / "_rank_work"
    rc.N_DATES = N_DATES
    rc.BACKWARD_WINDOW = True            # every run here, and the base, uses the backward window
    rc.WEATHER_VARS, rc.WX_STRIDE = WEATHER_VARS, WX_STRIDE
    try:
        df = rc.main()
    except SystemExit as e:
        return [], f"scoring failed: {e}"
    if df is None:
        raise SystemExit(f"{VERIF_DIR / 'rank_checkpoints.py'}: main() returned None -- it must "
                         f"end with `return df`")
    cols = ["epoch", "step", "power", "long", "wind", "n"] + [f"wx_{v}" for v in WEATHER_VARS]
    return [{c: (int(r[c]) if c in ("epoch", "step", "n") else float(r[c])) for c in cols}
            for _, r in df.sort_values("step").iterrows()], ""


def train(name, cfg, seed):
    """One anemoi-training run in its own process group. Returns (ok, note)."""
    root = SWEEP_ROOT / name
    logdir = root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"{name}.yaml").write_text(yaml.dump(cfg, Dumper=_NoAlias, sort_keys=False))

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{TRAINING_DIR}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["ANEMOI_BASE_SEED"] = str(seed)
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    cmd = ["anemoi-training", "train", f"--config-path={logdir}", f"--config-name={name}"]
    with open(logdir / "train.log", "w") as log:
        p = subprocess.Popen(cmd, cwd=logdir, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rcode = p.wait(timeout=TRAIN_TIMEOUT_H * 3600)
        except subprocess.TimeoutExpired:
            _kill(p)
            return False, f"timed out after {TRAIN_TIMEOUT_H} h"
        except BaseException:
            _kill(p)          # never leave an orphan holding the GPUs for the next start
            raise
    if rcode != 0:
        tail = (logdir / "train.log").read_text(errors="replace").strip().splitlines()[-3:]
        return False, f"exit {rcode}: " + " | ".join(tail)
    steps = [rc.epoch_step(p)[1] for p in (root / "checkpoint").glob("*/inference-*.ckpt")
             if rc.is_runnable(p)]
    want = cfg["training"]["max_steps"]
    if not steps:
        return False, "exit 0 but no runnable inference checkpoint"
    return True, "" if max(steps) >= want else f"last checkpoint at step {max(steps)} < {want}"


def _kill(p):
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 10)):
        try:
            os.killpg(p.pid, sig)
            p.wait(timeout=wait)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            continue


def _raise_interrupt(*_):
    raise KeyboardInterrupt      # SIGTERM -> the same cleanup as Ctrl-C: kill the training job


def wx_skill(final, base_final):
    """Mean over WEATHER_VARS of RMSE_run / RMSE_base - 1, plus the per-variable values."""
    per = {v: final[f"wx_{v}"] / base_final[f"wx_{v}"] - 1.0 for v in WEATHER_VARS}
    return float(np.mean(list(per.values()))), per


def report(state):
    order = [BASE_NAME, ANCHOR_NAME] + [n for n, _, _, _ in RUNS]
    runs = {n: state[n] for n in order
            if state.get(n, {}).get("scores") and state[n].get("n_dates") == N_DATES}
    if BASE_NAME not in runs:
        print("the stage-2 base is not scored yet -- no weather reference")
        return
    final = {n: r["scores"][-1] for n, r in runs.items()}
    base = final[BASE_NAME]
    ctrls = [n for n in CONTROLS if n in runs]
    cp = np.array([final[n]["power"] for n in ctrls]) if ctrls else np.array([np.nan])
    c_lo, c_hi, c_mean = np.nanmin(cp), np.nanmax(cp), np.nanmean(cp)
    wxs = {n: wx_skill(final[n], base) for n in runs}

    short = {"z_500": "z500", "z_850": "z850", "t_850": "t850", "q_850": "q850", "u_850": "u850",
             "msl": "msl", "t2m": "t2m", "ws100": "ws100"}
    hdr = (f"{'run':14s} {'change':26s} {'POWER':>6s} {'vs ctrl':>7s} {'+21h':>6s} {'WEATHER':>8s} "
           + " ".join(f"{short.get(v, v):>6s}" for v in WEATHER_VARS) + f" {'train h':>7s}")
    print(f"\n{'=' * len(hdr)}\nFS2 SWEEP -- FINAL checkpoint, validation window, {N_DATES} inits. "
          f"POWER = BE total MAE % cap, +3..33h.\nWEATHER = mean RMSE change vs the untrained "
          f"stage-2 base over {len(WEATHER_VARS)} variables, whole domain (every {WX_STRIDE}th cell), "
          f"+3..33h. Lower is better for both.\n{'=' * len(hdr)}")
    print(hdr)
    for n in sorted(runs, key=lambda n: final[n]["power"]):
        f, r = final[n], runs[n]
        m, per = wxs[n]
        band = ("" if not ctrls or n in ctrls or n == BASE_NAME else
                "  below ctrl band" if f["power"] < c_lo else
                "  above ctrl band" if f["power"] > c_hi else "  inside ctrl band")
        print(f"{n:14s} {r.get('change', ''):26.26s} {f['power']:6.2f} {f['power'] - c_mean:+7.2f} "
              f"{f['long']:6.2f} {100 * m:+7.1f}% "
              + " ".join(f"{100 * per[v]:+5.1f}%" for v in WEATHER_VARS)
              + f" {r.get('hours_train', float('nan')):7.1f}{band}")
    pending = [(n, state.get(n, {}).get("status", "pending")) for n, _, _, _ in RUNS
               if state.get(n, {}).get("status") != "scored"]
    if pending:
        print("\nnot finished: " + ", ".join(f"{n} ({s})" for n, s in pending))
    if ctrls:
        print(f"\n  ctrl band ({', '.join(ctrls)}): power {c_lo:.2f}..{c_hi:.2f}. Inside it = same "
              f"power as FS2 within seed noise.")
    print(f"  {BASE_NAME} is the model before any fine-tune: its WEATHER is 0 by definition, and its "
          f"POWER is what")
    print("  pre-training alone gives. Wanted: power inside/below the band, weather as close to 0 "
          "as possible.")

    # ---- figure: the trade-off, and the power curves ----
    palette = [c for i, c in enumerate(plt.cm.tab10.colors) if i != 7]
    style, k = {}, 0
    for n in runs:
        if n == BASE_NAME:
            style[n] = dict(color=to_rgba("k"), marker="s")
        elif n in ctrls:
            style[n] = dict(color=to_rgba("0.5"), marker="o")
        else:
            style[n] = dict(color=to_rgba(palette[k % len(palette)]), marker="o")
            k += 1
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
    ax = axs[0]
    for n in runs:
        ax.scatter(100 * wxs[n][0], final[n]["power"], s=55, zorder=3, **style[n])
        ax.annotate(n, (100 * wxs[n][0], final[n]["power"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    if ctrls:
        ax.axhspan(c_lo, c_hi, color="0.88", zorder=0, label="control band (seed noise)")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("weather: mean RMSE change vs stage-2 base [%]")
    ax.set_ylabel("power MAE, BE total [% cap]")
    ax.set_title("The trade-off -- bottom-left wins both", fontsize=10)
    ax.grid(alpha=0.3)
    for n, r in runs.items():
        if n == BASE_NAME:
            continue
        s = r["scores"]
        x = [q["step"] for q in s]
        kw = dict(color=style[n]["color"], marker="o", ms=3, lw=1.3, label=n)
        axs[1].plot(x, [q["power"] for q in s], **kw)
        axs[2].plot(x, [100 * wx_skill(q, base)[0] for q in s], **kw)
    axs[1].axhline(base["power"], color="k", ls="--", lw=1, label=BASE_NAME)
    axs[1].set_title("Power during the fine-tune", fontsize=10)
    axs[1].set_ylabel("power MAE [% cap]")
    axs[2].axhline(0, color="k", ls="--", lw=1)
    axs[2].set_title("Weather drift during the fine-tune (0 = base)", fontsize=10)
    axs[2].set_ylabel("mean RMSE change vs base [%]")
    for a in axs[1:]:
        a.set_xlabel("training step")
        a.grid(alpha=0.3)
    axs[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"FS2 sweep -- validation window, {N_DATES} inits, paired", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {FIG}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    state = load_state()
    if mode == "report":
        report(state)
        return

    base = yaml.safe_load(BASE_YAML.read_text())
    stored, anchor_seed, anchor_ckpt = anchor_metadata()
    print(f"anchor  {ANCHOR_NAME}: {anchor_ckpt.name}  (seed {anchor_seed})")
    base_hits = sorted(glob.glob(BASE_CKPT))
    if not base_hits:
        raise SystemExit(f"stage-2 base checkpoint not found: {BASE_CKPT}")
    print(f"base    {BASE_NAME}: {Path(base_hits[0]).name}")

    ctrl = build_config(base, "ctrl", {})
    diffs = check_control(ctrl, stored)
    if diffs:
        print(f"\n{BASE_YAML.name} DOES NOT REPRODUCE {ANCHOR_NAME}. {len(diffs)} setting(s) differ "
              f"from the config stored in its checkpoint:\n")
        for k, mine, theirs in diffs:
            print(f"  {k:55s} sweep {mine!r:>22}   {ANCHOR_NAME} {theirs!r}")
        print("\nPart of the recipe -> fix the YAML. Pure plumbing -> add the key to IGNORE_DIFF.")
        raise SystemExit(1)
    print(f"{BASE_YAML.name} matches {ANCHOR_NAME}'s stored config on every shared setting\n")

    plan = []
    for name, seed, ov, desc in RUNS:
        cfg = build_config(base, name, ov)
        plan.append((name, anchor_seed if seed == SAME else seed, desc, cfg, rel_cost(cfg, ctrl)))
    print(f"{'run':15s} {'seed':>6s} {'cost':>5s}  change")
    for name, seed, desc, cfg, cost in plan:
        print(f"{name:15s} {seed:6d} {cost:5.2f}  {desc}")
    total = sum(c for *_, c in plan)
    print(f"\ntotal {total:.1f} FS2-equivalents of training, plus scoring "
          f"({SCORE_POINTS} checkpoints x {N_DATES} inference calls per run, and the base/anchor once).")
    done_h = [(state[n]["hours_train"], state[n]["cost"]) for n in state
              if state[n].get("status") in ("trained", "scored")
              and state[n].get("hours_train") and state[n].get("cost")]
    if done_h:
        per = sum(h for h, _ in done_h) / sum(c for _, c in done_h)
        left = sum(c for n, _, _, _, c in plan if state.get(n, {}).get("status") != "scored")
        print(f"measured {per:.2f} h per unit -> ~{per * left:.0f} h of training left")

    if mode == "dry":
        for name, seed, desc, cfg, _ in plan:
            d = SWEEP_ROOT / name / "logs"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.yaml").write_text(yaml.dump(cfg, Dumper=_NoAlias, sort_keys=False))
        print(f"\nYAMLs written under {SWEEP_ROOT}/<run>/logs/. Nothing trained.")
        return

    signal.signal(signal.SIGTERM, _raise_interrupt)

    # the two references, scored once: untrained base (weather = 0) and FS2 itself
    for name, ckpts, change in ((BASE_NAME, base_hits[:1], "stage-2 base, no fine-tune"),
                                (ANCHOR_NAME, pick(ANCHOR_DIR), "FS2 (the current best)")):
        if state.get(name, {}).get("status") == "scored":
            continue
        print(f"\n### scoring {name}", flush=True)
        t0 = time.time()
        scores, note = score(ckpts, SWEEP_ROOT / "logs" / name)
        state[name] = {"status": "scored" if scores else "failed", "scores": scores,
                       "n_dates": N_DATES, "note": note, "change": change,
                       "hours_score": (time.time() - t0) / 3600}
        save_state(state)
        print(f"    {name}: {note or 'scored'} in {state[name]['hours_score']:.2f} h", flush=True)
    report(state)

    for name, seed, desc, cfg, cost in plan:
        st = state.get(name, {})
        if st.get("status") == "scored":
            continue
        if st.get("status") == "failed" and not RETRY_FAILED:
            print(f"\n### {name}: failed on an earlier start ({st.get('note')}) -- skipped")
            continue
        if st.get("status") in ("training", "failed"):
            # interrupted or failed mid-run: its partial output is worthless, start clean
            shutil.rmtree(SWEEP_ROOT / name / "checkpoint", ignore_errors=True)

        if st.get("status") != "trained":
            print(f"\n### {name}: training  [{desc}]  seed {seed}, cost {cost:.2f}  "
                  f"-- log: {SWEEP_ROOT / name / 'logs' / 'train.log'}", flush=True)
            state[name] = {"status": "training", "cost": cost, "seed": seed, "change": desc,
                           "started": time.strftime("%a %H:%M")}
            save_state(state)
            t0 = time.time()
            ok, note = train(name, cfg, seed)
            state[name].update(status="trained" if ok else "failed", note=note,
                               hours_train=(time.time() - t0) / 3600)
            save_state(state)
            print(f"    {name}: {'trained' if ok else 'FAILED'} in {state[name]['hours_train']:.1f} h"
                  f"{'  -- ' + note if note else ''}", flush=True)
            if not ok:
                continue

        print(f"### {name}: scoring", flush=True)
        t0 = time.time()
        scores, note = score(pick(SWEEP_ROOT / name / "checkpoint"), SWEEP_ROOT / name / "logs")
        state[name].update(scores=scores, hours_score=(time.time() - t0) / 3600, n_dates=N_DATES,
                           status="scored" if scores else "trained",
                           note=note or state[name].get("note", ""))
        save_state(state)
        print(f"    {name}: {note or 'scored'} in {state[name]['hours_score']:.2f} h", flush=True)
        report(state)

    print("\nSweep finished.")


if __name__ == "__main__":
    main()
