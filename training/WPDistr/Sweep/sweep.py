#!/usr/bin/env python3
"""Weekend sweep: fine-tune variations of MixedRollout back to back, score each one by FORECASTING.

    python sweep.py dry       write every run's YAML, check the control against MixedRollout's own
                              metadata, print the plan and relative cost. No GPU. Run this first.
    nohup python sweep.py > sweep.out 2>&1 &
                              the sweep. Resumable: start it again and it carries on.
    python sweep.py report    table + figure from whatever has finished. Safe while it runs.

Each run is: train (anemoi-training, both GPUs, exactly as a normal run) -> score SCORE_POINTS of its
epoch checkpoints on the VALIDATION window with rank_checkpoints.py -> redraw the summary figure.

WHY NOT THE MLFLOW VALIDATION CURVES
------------------------------------
They are logged (all runs share one mlflow store, see below) but they cannot pick a winner:
  * validation rolls out with the run's OWN training rollout -- rollout.py `_step` uses
    self.rollout; `dataloader.validation_rollout` only sizes the batch. A roll3 run is validated
    on +3..9h, a roll8 run on +3..24h. Different horizons, not comparable.
  * the capacityfactor metric averages ~99.8% fabricated zeros (see rank_checkpoints.py).
So selection is done the way the headline is measured: real forecasts, +3..+33h, farm totals, the
same inits for every checkpoint (paired). Validation is cut to VAL_BATCHES and kept as a health
monitor only -- it is ~50% of a run's wall time at the full window and buys nothing here.

THE CONTROL IS THE POINT
------------------------
`ctrl_s1` is MixedRollout's recipe with MixedRollout's own seed: if it does not land on
MixedRollout's curve, nothing else in the sweep is interpretable. `ctrl_s2`/`ctrl_s3` change ONLY
the seed. The seed has never been varied before -- it comes from ANEMOI_BASE_SEED, one fixed env
var, which is why Static and Deeper agreed to 0.001 pp -- so the ~0.07 pp "seed noise" was never a
seed measurement. The spread of the controls is the band a treatment must clear. Treatments share
the control seed, so each is a PAIRED comparison against ctrl_s1 (same data order, same init).

`Optimize.yaml` itself is NOT the MixedRollout recipe: it has 100 batches/epoch and rollout
starting at 6, and 40 epochs x 100 = 4000 steps stops it before max_steps. MIXEDROLLOUT below
restores what RUN_MixedRollout.md records, and startup compares the result against the config
stored in MixedRollout's checkpoint. Any difference aborts the sweep before a GPU is touched.

SELECTION RULE -- decide it now, not after seeing the numbers
--------------------------------------------------------------
Rank runs by their FINAL checkpoint (MixedRollout's 6.94 is its final checkpoint too). Picking
each run's best epoch on 16 inits is selection on noise. Then take the ONE winner to the test year
with verify_power.py. Taking the top three to the test year and reporting the best is selection on
the test set.

Outputs, all under SWEEP_ROOT (git-ignored): one directory per run (checkpoints, train log, its
YAML), logs/mlflow (one store for every run: `mlflow ui --backend-store-uri <SWEEP_ROOT>/logs/mlflow`),
logs/sweep_state.json (the job queue that makes this resumable), sweep_validation.png.
"""

from __future__ import annotations

import contextlib
import copy
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
BASE_YAML    = REPO / "training/WPDistr/Optimize/Optimize.yaml"
TRAINING_DIR = REPO / "training"            # cf_head.py, combined_loss.py -> PYTHONPATH
VERIF_DIR    = REPO / "verification"        # rank_checkpoints.py, farm_curves.py
SWEEP_ROOT   = REPO / "training/WPDistr/Sweep"
ANCHOR_DIR   = REPO / "training/WPDistr/VeryHighCapacityGTFinetuneHuber/checkpoint/MixedRollout"
ANCHOR_NAME  = "MixedRollout"

# Optimize.yaml -> MixedRollout, as recorded in RUN_MixedRollout.md. Checked against the anchor's
# stored config at startup.
MIXEDROLLOUT = {
    "training.rollout.start": 1,
    "training.rollout.epoch_increment": 1,
    "training.rollout.max": 6,
    "dataloader.limit_batches.training": 500,
    "training.max_steps": 5000,
    "training.lr.iterations": 5000,
    "training.lr.rate": 3.0e-5,
    "training.lr.warmup": 200,
}

VAL_BATCHES  = 50        # per GPU. Health monitor only -- see the docstring.

# Keys that may differ from the anchor's stored config without invalidating the control: where
# output goes, logging, and the validation cut above. Anything else that differs aborts.
# `defaults` is consumed by hydra and never stored; `data.num_features` is null in the YAML and
# filled in by anemoi at runtime (67).
IGNORE_DIFF  = ("system.output", "diagnostics", "dataloader.limit_batches.validation",
                "dataloader.num_workers", "dataloader.prefetch_factor", "training.run_id",
                "training.fork_run_id", "graph.overwrite", "defaults", "data.num_features")

SAME = "anchor"          # seed sentinel: MixedRollout's own seed, read from its metadata

# The runs that form the control band in the report. Rounds 1-2 were measured against the
# MixedRollout recipe (ctrl_s1..3). From round 3 the baseline is the unfrozen processor, and every
# new run changes one thing on top of IT -- so the band must be its seeds, or each new run is
# graded against a baseline it already beats by 0.7 pp and everything looks like a candidate.
CONTROLS = ["unfreeze_proc", "unfreeze_s2", "unfreeze_s3"]

# Priority order: if the weekend runs out, what is left undone is the least informative. Controls
# first; the rollout ends, then LR, which bracket the two things most likely to move the number;
# 2x budget last because it costs 2.3 runs.
#   name            seed   overrides on top of MIXEDROLLOUT (dotted config keys; must exist)
RUNS = [
    ("ctrl_s1",       SAME, {}),
    ("ctrl_s2",       5678, {}),
    ("roll8",         SAME, {"training.rollout.max": 8}),     # the 7.5k base was trained to 8
    ("roll4",         SAME, {"training.rollout.max": 4}),
    ("lr1e-4",        SAME, {"training.lr.rate": 1.0e-4}),    # x2 GPUs -> effective 2e-4
    ("lr1e-5",        SAME, {"training.lr.rate": 1.0e-5}),
    ("ctrl_s3",       9012, {}),
    ("roll3",         SAME, {"training.rollout.max": 3}),
    ("roll5",         SAME, {"training.rollout.max": 5}),
    ("roll7",         SAME, {"training.rollout.max": 7}),
    ("cf200",         SAME, {"training.scalers.power_variable.weights.capacityfactor": 200}),
    ("cf50",          SAME, {"training.scalers.power_variable.weights.capacityfactor": 50}),
    ("rollfull6",     SAME, {"training.rollout.start": 6}),   # what Optimize.yaml does now: no ramp
    ("unfreeze_proc", SAME, {"training.submodules_to_freeze": ["encoder"]}),  # attacks the WIND term
    ("steps10k",      SAME, {"training.max_steps": 10000, "training.lr.iterations": 10000}),
    # ROUND 2. unfreeze_proc won round 1 outright -- 5.21 against a 5.95-6.16 control band, and
    # the farm wind IMPROVED (1.019 vs 1.071), so it is not the usual power-for-wind trade. These
    # three ask whether that survives a different seed, and whether it stacks with the other two
    # things that worked, both of which amount to "the fine-tune was under-trained".
    ("unfreeze_s2",     5678, {"training.submodules_to_freeze": ["encoder"]}),
    ("unfreeze_lr1e-4", SAME, {"training.submodules_to_freeze": ["encoder"],
                               "training.lr.rate": 1.0e-4}),
    ("unfreeze_10k",    SAME, {"training.submodules_to_freeze": ["encoder"],
                               "training.max_steps": 10000, "training.lr.iterations": 10000}),
    # ROUND 3. Baseline = unfreeze_proc (processor trainable, lr 3e-5, 5k). Test year, 730 inits:
    # direct 6.35 / 6.28 on two seeds vs MixedRollout 6.84; the gain is CONVERSION at short leads
    # and WIND at long leads. Each run below changes one thing on top of that, seed paired with
    # unfreeze_proc. The report's band is CONTROLS -- see above.
    ("unfreeze_s3",       9012, {"training.submodules_to_freeze": ["encoder"]}),
    # the encoder too: farm identity/capacity can now reach the latent at all
    ("unfreeze_all",      SAME, {"training.submodules_to_freeze": []}),
    # insurance for the weather claim: a small weight on every non-wind variable, so an unfrozen
    # processor cannot drift t/q/z for free. Expect a small power cost; the question is how small.
    ("unfreeze_anchor",   SAME, {"training.submodules_to_freeze": ["encoder"],
                                 "training.scalers.weather_variable.weights.default": 0.01}),
    # MEASURED on unfreeze_proc (verify_weather, 2916 inits): z_500 RMSE +75% at 36h with
    # sigma_p/sigma_o decaying 0.997 -> 0.958 -- the unweighted large-scale field is damped step by
    # step. 0.01 x the 0.7 pressure-level factor may be far too weak against 0.5 on each wind
    # variable, so bracket it. 1.0 is every variable at full weight, as in pre-training.
    ("unfreeze_anchor01", SAME, {"training.submodules_to_freeze": ["encoder"],
                                 "training.scalers.weather_variable.weights.default": 0.1}),
    ("unfreeze_anchor1",  SAME, {"training.submodules_to_freeze": ["encoder"],
                                 "training.scalers.weather_variable.weights.default": 1.0}),
    # 1e-4 broke the unfrozen run (wind gain lost); is 3e-5 even the optimum, or is lower better?
    ("unfreeze_lr1.5e-5", SAME, {"training.submodules_to_freeze": ["encoder"],
                                 "training.lr.rate": 1.5e-5}),
    # rollout was inert with a frozen trunk because only the decoder could use the multi-step
    # signal; with trainable dynamics, training out to the scored horizon may finally matter
    ("unfreeze_roll11",   SAME, {"training.submodules_to_freeze": ["encoder"],
                                 "training.rollout.start": 6, "training.rollout.max": 11}),
    # the power weight was inert when frozen; it now sets how the TRUNK splits power vs wind
    ("unfreeze_cf300",    SAME, {"training.submodules_to_freeze": ["encoder"],
                                 "training.scalers.power_variable.weights.capacityfactor": 300}),
    # Huber d1 on wind is ~MSE -> conditional mean -> strong winds pulled low, which is the bias
    # the unfrozen runs show above 8 m/s. d0.5 was ruled out when frozen; the trade can differ now.
    ("unfreeze_huber05",  SAME, {"training.submodules_to_freeze": ["encoder"],
                                 "training.training_loss.losses.0.delta": 0.5}),
    # ROUND 4. Round 3 converged: nothing beat unfreeze_proc by more than seed noise. On the test
    # year it ties the MAE-trained CERRA transformer (6.40 vs 6.46) with a crossing lead profile:
    # the converter wins +3h/+6h, the head wins beyond ~18h. Both runs below attack the short
    # leads, one mechanism each -- see training/rollout_tasks.py.
    # Step 1 is 1/6 of the rollout loss and the only step on analysis inputs: weight it up.
    ("unfreeze_earlystep", SAME, {"training.submodules_to_freeze": ["encoder"],
                                  "training.model_task": "rollout_tasks.EarlyStepForecaster"}),
    # The head extrapolates [T, T+3h) from the state at T; target [T-3h, T) instead. Its power
    # output means a different window, so it is SCORED one step later (rank_checkpoints
    # BACKWARD_WINDOW, set automatically below) -- and needs BACKWARD_WINDOW_RUNS in verify_power.
    ("unfreeze_backwin",   SAME, {"training.submodules_to_freeze": ["encoder"],
                                  "training.model_task": "rollout_tasks.BackwardWindowForecaster"}),
]

SCORE_POINTS = 5         # epoch checkpoints scored per run, evenly spread, always incl. the last
# Validation inits per checkpoint. Rounds 1-3 used 16; round 3's control band was 0.23 wide and the
# effects left are ~0.1-0.2, so it could no longer separate them. 48 = ~3x the scoring time. A
# different N_DATES scores a DIFFERENT set of dates, so scores are only comparable at equal
# N_DATES: each run records its own, and the report shows only runs scored at the current value.
N_DATES      = 48
TRAIN_TIMEOUT_H = 12     # a hung DDP job must not eat the weekend
RETRY_FAILED = False     # True: re-attempt runs that failed on an earlier start
# ======================================================================

sys.path.insert(0, str(VERIF_DIR))
import rank_checkpoints as rc      # noqa: E402

STATE = SWEEP_ROOT / "logs" / "sweep_state.json"
FIG = SWEEP_ROOT / "sweep_validation.png"


class _NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):  # write the reorder map out twice rather than as &id001
        return True


def _step(d, k):
    # a numeric segment indexes a list: "training.training_loss.losses.0.delta"
    return d[int(k)] if isinstance(d, list) else d[k]


def dget(d, key):
    for k in key.split("."):
        d = _step(d, k)
    return d


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
    """(stored config, seed) of MixedRollout's final checkpoint."""
    cks = [p for p in ANCHOR_DIR.glob("inference-*.ckpt") if rc.is_runnable(p)]
    last = max(cks, key=lambda p: rc.epoch_step(p)[1])
    with zipfile.ZipFile(last) as z:
        name = next(n for n in z.namelist()
                    if n.endswith(("anemoi-metadata/anemoi.json", "anemoi-metadata/ai-models.json")))
        meta = json.loads(z.read(name))
    return meta["config"], int(meta["seed"]), last


def build_config(base, name, overrides):
    cfg = copy.deepcopy(base)
    for k, v in {**MIXEDROLLOUT, **overrides}.items():
        dset(cfg, k, v)
    root = SWEEP_ROOT / name
    dset(cfg, "system.output.root", str(root))
    dset(cfg, "dataloader.limit_batches.validation", VAL_BATCHES)
    dset(cfg, "diagnostics.log.mlflow.tracking_uri", f"file:{SWEEP_ROOT}/logs/mlflow")
    dset(cfg, "diagnostics.log.mlflow.experiment_name", "Sweep")
    dset(cfg, "diagnostics.log.mlflow.run_name", name)
    return cfg


def rel_cost(cfg):
    """Training cost in rollout-steps, relative to MixedRollout. Rollout grows at EPOCH end
    (rollout.py on_train_epoch_end), so epoch e runs at min(start + ceil(e/inc), max)."""
    def units(c):
        t, r = c["training"], c["training"]["rollout"]
        spe, n = c["dataloader"]["limit_batches"]["training"], t["max_steps"]
        tot = 0
        for e in range(math.ceil(n / spe)):
            R = min(r["start"] + (math.ceil(e / r["epoch_increment"]) if r["epoch_increment"] else 0),
                    r["max"])
            tot += min(spe, n - e * spe) * R
        return tot
    ref = copy.deepcopy(cfg)
    for k, v in MIXEDROLLOUT.items():
        dset(ref, k, v)
    return units(cfg) / units(ref)


def check_control(ctrl, stored):
    """The control config against the anchor's stored one, in BOTH directions: a setting only
    MixedRollout had is as much a difference as one only the sweep has."""
    absent = "(absent)"

    def get(d, key):
        # The YAML writes `layer_kernels: ${model.layer_kernels}`; the checkpoint stores it
        # EXPANDED. A key below an interpolation exists on one side only, so stop at the
        # `${...}` string and let same() treat it as uncomparable rather than as absent.
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
    for p in sorted(ckpt_dir.glob("**/inference-*.ckpt")):
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


def patch_checkpoints(ckpt_dir):
    """Clear dataset.variables_metadata, as inference/patch_all.py does by hand.

    anemoi-inference cannot run a freshly written checkpoint until this is cleared: it dies with
    `KeyError: 'latitudes'` inside the output post-processor, because the initial state never gets
    built. Every run of the first sweep trained fine and then failed to score on exactly this.
    Supporting arrays are read and written straight back, so latitudes/longitudes/cutout_mask/
    grid_indices are untouched -- pick() checks for those before it will score a checkpoint.
    """
    paths = sorted(ckpt_dir.glob("**/inference-*.ckpt"))
    if not paths:
        return
    from anemoi.utils.checkpoints import load_metadata, replace_metadata
    n = 0
    for p in paths:
        meta, arrays = load_metadata(p, supporting_arrays=True)
        if not meta.get("dataset", {}).get("variables_metadata"):
            continue                      # already patched, or written by a version that is clean
        meta["dataset"]["variables_metadata"] = {}
        # replace_metadata prints a tqdm bar and a zipfile "Duplicate name" warning per supporting
        # array (it rewrites them under the same name; the last copy is the one read, so harmless).
        # ~12 lines x 12 checkpoints per run buried the scores. Exceptions still propagate.
        with contextlib.redirect_stderr(io.StringIO()):
            replace_metadata(p, meta, arrays)
        n += 1
    if n:
        print(f"    patched {n} checkpoint(s): cleared dataset.variables_metadata", flush=True)


def score(name, ckpt_dir, work, backward_window=False):
    patch_checkpoints(ckpt_dir)
    ckpts = pick(ckpt_dir)
    if not ckpts:
        return [], "no runnable checkpoint"
    rc.CHECKPOINTS = [str(p) for p in ckpts]
    rc.CKPT_ROOT, rc.OUT_DIR, rc.WORK_DIR = ckpt_dir, work, work / "_rank_work"
    rc.N_DATES = N_DATES
    rc.BACKWARD_WINDOW = backward_window     # module global: reset for every run, never inherited
    try:
        df = rc.main()
    except SystemExit as e:
        return [], f"scoring failed: {e}"
    if df is None:
        # a problem with the code, not the checkpoint -- stop, rather than mark every run unscored
        raise SystemExit(f"{VERIF_DIR / 'rank_checkpoints.py'}: main() returned None. It must end with `return df`; "
                         f"that line is missing from the copy on this machine.")
    cols = ["epoch", "step", "power", "long", "wind", "n"]
    return [{c: (float(r[c]) if c in ("power", "long", "wind") else int(r[c])) for c in cols}
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
    # this mlflow refuses to create a NEW file store ("maintenance mode") unless told otherwise,
    # and logs/mlflow under SWEEP_ROOT is new -- every run died at startup without this
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


LABELS = {"training.rollout.max": "rollout max", "training.rollout.start": "rollout start",
          "training.lr.rate": "lr", "training.max_steps": "steps",
          "training.scalers.power_variable.weights.capacityfactor": "cf weight",
          "training.submodules_to_freeze": "freeze",
          "training.scalers.weather_variable.weights.default": "weather weight",
          "training.training_loss.losses.0.delta": "wind huber delta",
          "training.model_task": "task"}


def describe(overrides, seed, anchor_seed):
    parts = [f"{LABELS.get(k, k.split('.')[-1])} {v:g}" if isinstance(v, float)
             else f"{LABELS.get(k, k.split('.')[-1])} {v.split('.')[-1]}" if k == "training.model_task"
             else f"{LABELS.get(k, k.split('.')[-1])} {v}"
             for k, v in overrides.items() if k != "training.lr.iterations"]
    if seed != anchor_seed:
        parts.append(f"seed {seed}")
    return ", ".join(parts) or "MixedRollout recipe"


def _raise_interrupt(*_):
    raise KeyboardInterrupt      # SIGTERM -> the same cleanup as Ctrl-C: kill the training job


def report(state):
    order = [ANCHOR_NAME] + [n for n, _, _ in RUNS]
    # runs scored before n_dates was recorded used 16
    runs = {n: state[n] for n in order
            if state.get(n, {}).get("scores") and state[n].get("n_dates", 16) == N_DATES}
    if not runs:
        print("nothing scored yet")
        return
    final = {n: r["scores"][-1] for n, r in runs.items()}
    ctrls = [n for n in CONTROLS if n in runs]
    if not ctrls:
        # print, don't raise: report() also runs inside the sweep loop, and a missing band must
        # not kill the sweep between two runs
        print(f"none of CONTROLS {CONTROLS} is scored at N_DATES={N_DATES} yet -- no band to "
              f"compare against. Re-queue them for scoring (see the round-4 note).")
        return
    cp = np.array([final[n]["power"] for n in ctrls])
    cw = np.array([final[n]["wind"] for n in ctrls])
    c_mean, c_lo, c_hi = cp.mean(), cp.min(), cp.max()

    print(f"\n{'=' * 104}\nSWEEP -- FINAL checkpoint of each run, validation window, {N_DATES} inits, "
          f"+3..+33h, BE total\n{'=' * 104}")
    print(f"{'run':15s} {'change':38s} {'POWER %cap':>10s} {'vs ctrl':>8s} {'+21h':>6s} "
          f"{'WIND m/s':>9s} {'vs ctrl':>8s} {'step':>6s} {'train h':>7s}")
    for n in sorted(runs, key=lambda n: final[n]["power"]):
        f, r = final[n], runs[n]
        band = ("" if n in ctrls else
                "  BELOW ctrl band -> candidate" if f["power"] < c_lo else
                "  above ctrl band -> worse" if f["power"] > c_hi else
                "  inside ctrl band")
        print(f"{n:15s} {r.get('change', ''):38.38s} {f['power']:10.2f} {f['power'] - c_mean:+8.2f} "
              f"{f['long']:6.2f} {f['wind']:9.3f} {f['wind'] - cw.mean():+8.3f} {f['step']:6d} "
              f"{r.get('hours_train', float('nan')):7.1f}{band}")
    pending = [(n, state.get(n, {}).get("status", "pending")) for n, _, _ in RUNS
               if state.get(n, {}).get("status") != "scored"]
    if pending:
        print("\nnot finished: " + ", ".join(f"{n} ({s})" for n, s in pending))
    print(f"\n  ctrl band: {len(ctrls)} controls ({', '.join(ctrls)}), power {c_lo:.2f}..{c_hi:.2f}, "
          f"mean {c_mean:.2f}. That spread is seed + GPU noise on these {N_DATES} inits. A run")
    print("  inside it is indistinguishable from the control. A run below it is a CANDIDATE, not a")
    print("  result: take the single best to the test year with verify_power.py, and nothing else.")
    print("  'vs ctrl' is against the mean of the controls. WIND is the trunk's side: a run that")
    print("  buys power with wind is trading the forecast away.")

    # ---- figure ----
    # grey is reserved for the controls -- a treatment drawn grey reads as noise
    palette = [c for i, c in enumerate(plt.cm.tab10.colors) if i != 7]
    style, k = {}, 0
    for n in runs:
        if n == ANCHOR_NAME:
            style[n] = dict(color=to_rgba("k"), ls="--", lw=2.0, label=f"{n} (original)")
        elif n in ctrls:
            style[n] = dict(color=to_rgba("0.55"), ls="-", lw=1.6, label=n)
        else:
            style[n] = dict(color=to_rgba(palette[k % len(palette)]),
                            ls="-" if k < len(palette) else ":", lw=1.3, label=n)
            k += 1
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for n, r in runs.items():
        s = r["scores"]
        x = [q["step"] for q in s]
        axs[0, 0].plot(x, [q["power"] for q in s], marker="o", ms=3, **style[n])
        axs[0, 1].plot(x, [q["long"] for q in s], marker="o", ms=3, **style[n])
        axs[1, 0].plot(x, [q["wind"] for q in s], marker="o", ms=3, **style[n])
    axs[0, 0].axhspan(c_lo, c_hi, color="0.85", zorder=0)     # control band at the final step
    for ax, t, yl in ((axs[0, 0], "Power, all leads +3..33h", "farm power MAE [% cap]"),
                      (axs[0, 1], "Power, leads +21h and later -- where rollout pays", "[% cap]"),
                      (axs[1, 0], "Wind at the farms -- the trunk's side of the trade", "ws100 MAE [m/s]")):
        ax.set_title(t, fontsize=10); ax.set_xlabel("training step"); ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7, ncol=2)
    names = sorted(runs, key=lambda n: final[n]["power"])
    ax = axs[1, 1]
    ax.axvspan(c_lo, c_hi, color="0.85", zorder=0, label="control band (seed noise)")
    ax.axvline(c_mean, color="0.4", lw=1)
    ax.scatter([final[n]["power"] for n in names], range(len(names)),
               c=[style[n]["color"] for n in names], s=40, zorder=3)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
    ax.set_xlabel("farm power MAE at the FINAL checkpoint [% cap]")
    ax.set_title("Selection view: left of the grey band = candidate", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"MixedRollout sweep -- forecast-scored on the validation window "
                 f"({N_DATES} inits, paired)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG, dpi=140); plt.close(fig)
    print(f"\nSaved: {FIG}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    state = load_state()
    if mode == "report":
        report(state)
        return

    base = yaml.safe_load(BASE_YAML.read_text())
    stored, anchor_seed, anchor_ckpt = anchor_metadata()
    print(f"anchor  {anchor_ckpt.name}  (seed {anchor_seed})")

    diffs = check_control(build_config(base, "ctrl_s1", {}), stored)
    if diffs:
        print(f"\nTHE CONTROL DOES NOT REPRODUCE {ANCHOR_NAME}. {len(diffs)} setting(s) differ from the "
              f"config stored in its checkpoint:\n")
        for k, mine, theirs in diffs:
            print(f"  {k:55s} sweep {mine!r:>22}   {ANCHOR_NAME} {theirs!r}")
        print("\nPart of the recipe -> put MixedRollout's value in MIXEDROLLOUT. Pure plumbing -> add "
              "the key to IGNORE_DIFF.")
        raise SystemExit(1)
    print(f"control config matches {ANCHOR_NAME}'s stored config on every shared setting\n")

    plan = []
    for name, seed, ov in RUNS:
        cfg = build_config(base, name, ov)
        plan.append((name, anchor_seed if seed == SAME else seed, ov, cfg, rel_cost(cfg)))
    print(f"{'run':15s} {'seed':>6s} {'cost':>5s}  change")
    for name, seed, ov, cfg, cost in plan:
        print(f"{name:15s} {seed:6d} {cost:5.2f}  {describe(ov, seed, anchor_seed)}")
    total = sum(c for *_, c in plan)
    print(f"\ntotal {total:.1f} MixedRollout-equivalents of training, plus scoring "
          f"({SCORE_POINTS} checkpoints x {N_DATES} inference calls per run).")
    done_h = [(state[n]["hours_train"], state[n]["cost"]) for n in state     # finished runs only:
              if state[n].get("status") in ("trained", "scored")             # a crash at startup
              and state[n].get("hours_train") and state[n].get("cost")]      # is not a timing
    if done_h:
        per = sum(h for h, _ in done_h) / sum(c for _, c in done_h)
        left = sum(c for n, _, _, _, c in plan if state.get(n, {}).get("status") != "scored")
        print(f"measured {per:.2f} h per unit -> ~{per * left:.0f} h of training left")

    if mode == "dry":
        for name, seed, ov, cfg, _ in plan:
            d = SWEEP_ROOT / name / "logs"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.yaml").write_text(yaml.dump(cfg, Dumper=_NoAlias, sort_keys=False))
        print(f"\nYAMLs written under {SWEEP_ROOT}/<run>/logs/. Nothing trained.")
        return

    signal.signal(signal.SIGTERM, _raise_interrupt)

    if state.get(ANCHOR_NAME, {}).get("status") != "scored":
        print(f"\n### scoring {ANCHOR_NAME} (the reference curve; also times the inference)")
        t0 = time.time()
        scores, note = score(ANCHOR_NAME, ANCHOR_DIR, SWEEP_ROOT / "logs" / "anchor")
        state[ANCHOR_NAME] = {"status": "scored" if scores else "failed", "scores": scores,
                              "n_dates": N_DATES,
                              "note": note, "hours_score": (time.time() - t0) / 3600,
                              "change": "the original run"}
        save_state(state)
        print(f"    {ANCHOR_NAME}: {note or 'scored'} in {state[ANCHOR_NAME]['hours_score']:.2f} h")
        report(state)

    for name, seed, ov, cfg, cost in plan:
        st = state.get(name, {})
        if st.get("status") == "scored":
            continue
        if st.get("status") == "failed" and not RETRY_FAILED:
            print(f"\n### {name}: failed on an earlier start ({st.get('note')}) -- skipped")
            continue
        if st.get("status") in ("training", "failed"):
            # interrupted or failed mid-run: its partial output is ours and worthless, start clean
            shutil.rmtree(SWEEP_ROOT / name / "checkpoint", ignore_errors=True)

        if st.get("status") != "trained":
            print(f"\n### {name}: training  [{describe(ov, seed, anchor_seed)}]  "
                  f"seed {seed}, cost {cost:.2f}  -- log: {SWEEP_ROOT / name / 'logs' / 'train.log'}",
                  flush=True)
            state[name] = {"status": "training", "cost": cost, "seed": seed,
                           "change": describe(ov, seed, anchor_seed),
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
        scores, note = score(name, SWEEP_ROOT / name / "checkpoint", SWEEP_ROOT / name / "logs",
                             backward_window=ov.get("training.model_task") == "rollout_tasks.BackwardWindowForecaster")
        state[name].update(scores=scores, hours_score=(time.time() - t0) / 3600, n_dates=N_DATES,
                           status="scored" if scores else "trained",
                           note=note or state[name].get("note", ""))
        save_state(state)
        print(f"    {name}: {note or 'scored'} in {state[name]['hours_score']:.2f} h", flush=True)
        report(state)

    print("\nSweep finished.")


if __name__ == "__main__":
    main()
