import os
import random
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# ============================== SETTINGS ==============================
# N_PER_DAY random inits per day instead of all 8. The SAME sample is used for every checkpoint --
# verify_power.py intersects on common inits, so drawing a fresh sample per run would leave almost
# nothing in common. SEED makes it reproducible; change it only to draw a genuinely new sample.
#
# Cost of subsampling is small: consecutive 3-hourly inits share a synoptic situation, so the
# effective sample size is set by the ~3-5 day decorrelation time, not by the init count. 2/day
# gives ~730 inits and ~8000 scored cases per lead -- ~1800 of them above 12 m/s.
START      = datetime(2024, 8, 1, 0)
END        = datetime(2025, 7, 31, 9)
STEP       = timedelta(hours=3)
N_PER_DAY  = 8
SEED       = 0
LEAD_TIME  = 37
INIT_LIST  = "sampled_inits.txt"     # written once, so the sample is on the record

# Parallelism. Every task is a separate `anemoi-inference` process pinned to one GPU by
# CUDA_VISIBLE_DEVICES, so WORKERS_PER_GPU processes share each card. The model is 23.8M
# parameters, so compute -- not memory -- is normally the limit; drop WORKERS_PER_GPU to 1 if a
# card OOMs. THREADS is set per process so the pool does not oversubscribe the CPU: torch
# otherwise grabs every core in every process and they thrash.
N_GPUS          = 2
WORKERS_PER_GPU = 3
N_CORES         = 24
WORKERS = N_GPUS * WORKERS_PER_GPU
THREADS = max(1, N_CORES // WORKERS)


INNER = "/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr"
OUTER = "/mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_era5_A.zarr"
CHECKPOINTS = {
    # "WPDistr/HC_Finetune": (
    #     "/mnt/weatherloss/WindPower/training/WPDistr/HighCapacityGTFinetune/checkpoint/a1c74e76ef364f2daca5c101683ed083/",
    #     "inference-last.ckpt"),
    # "WPDistr/SHC_Finetune": (
    #     "/mnt/weatherloss/WindPower/training/WPDistr/SemiHighCapacityGTFinetune/checkpoint/98e167eb8e1c43f8a00c251844f10ea9/",
    #     "inference-last.ckpt"),
    # "WPDistr/VHC_10k_Finetune": (
    #     "/mnt/weatherloss/WindPower/training/WPDistr/VeryHighCapacityGTFinetune/checkpoint/81daa05665cb4f4daf1452e60657465d/",
    #     "inference-anemoi-by_time-epoch_019-step_010000.ckpt"),
    "WPDistr/VHC_5k_Finetune": (
        "/mnt/weatherloss/WindPower/training/WPDistr/VeryHighCapacityGTFinetune/checkpoint/81daa05665cb4f4daf1452e60657465d/",
        "inference-anemoi-by_epoch-epoch_009-step_005000.ckpt"),
    # "WPDistr/VHC_Half_Finetune": (
    #     "/mnt/weatherloss/WindPower/training/WPDistr/VeryHighCapacityGTFinetune/checkpoint/81daa05665cb4f4daf1452e60657465d/",
    #     "inference-anemoi-by_epoch-epoch_002-step_001500.ckpt"),
    # "WPDistr/VHC_Finetune_7var": (
    #     "/mnt/weatherloss/WindPower/training/WPDistr/VeryHighCapacityGTFinetune7var/checkpoint/c9816c65a50242ac82f2beb917bfef5f/",
    #     "inference-last.ckpt"),
    # "WPDistr/VHC_Finetune": (
    #     "/mnt/weatherloss/WindPower/training/WPDistr/VeryHighCapacityGTFinetune/checkpoint/81daa05665cb4f4daf1452e60657465d/",
    #     "inference-last.ckpt"),
    # "WPDistr/Vanilla_Finetune": (
    #     "/mnt/weatherloss/WindPower/training/WPDistr/VanillaPowerGTFinetune/checkpoint/b5b86e46dc9b433fa6e3f7383a9f6c43/",
    #     "inference-last.ckpt"),
}
# ======================================================================

_lock = threading.Lock()
_done = [0]


def sample_inits():
    """N_PER_DAY inits per calendar day, drawn once and shared by every checkpoint."""
    by_day = {}
    t = START
    while t <= END:
        by_day.setdefault(t.date(), []).append(t)
        t += STEP
    rng = random.Random(SEED)
    picked = sorted(i for day in by_day.values()
                    for i in rng.sample(day, min(N_PER_DAY, len(day))))
    hours = {}
    for i in picked:
        hours[i.hour] = hours.get(i.hour, 0) + 1
    print(f"{len(picked)} inits over {len(by_day)} days (seed {SEED})")
    print("  init hour spread: " + " ".join(f"{h:02d}h:{n}" for h, n in sorted(hours.items())))
    with open(INIT_LIST, "w") as f:
        f.write("\n".join(i.strftime("%Y-%m-%dT%H:%M:%S") for i in picked) + "\n")
    return picked


def run_one(job, tmpdir, total):
    """One forecast in its own process on one GPU. Returns (tag, ok)."""
    idx, tag, ckpt, init, out = job
    gpu = idx % N_GPUS
    date_str = init.strftime("%Y-%m-%dT%H:%M:%S")
    cfg = os.path.join(tmpdir, f"cfg_{idx}.yaml")     # per TASK: concurrent tasks must not share
    with open(cfg, "w") as f:
        f.write(f"""\
checkpoint: {ckpt}
lead_time: {LEAD_TIME}
date: "{date_str}"
device: cuda
input:
  dataset:
    dataset:
      cutout:
        - dataset: {INNER}
        - dataset: {OUTER}
      min_distance_km: 0
      adjust: all
output:
  extract_lam:
    output:
      netcdf: {out}
""")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS))
    # output is captured so six concurrent runs do not interleave into unreadable noise; it is
    # printed only when the run fails, which is when you actually need it
    p = subprocess.run(["anemoi-inference", "run", cfg], env=env,
                       capture_output=True, text=True)
    with _lock:
        _done[0] += 1
        n = _done[0]
    if p.returncode:
        print(f"[{n}/{total}] FAILED gpu{gpu} {tag} {date_str}\n{p.stderr[-800:]}", flush=True)
    elif n % 25 == 0 or n == total:
        print(f"[{n}/{total}] gpu{gpu} {tag} {date_str}", flush=True)
    return tag, p.returncode == 0


def main():
    inits = sample_inits()

    # one flat job list across every checkpoint, so the pool stays saturated to the last task
    jobs, skipped = [], {}
    for tag, (ckpt_dir, ckpt_name) in CHECKPOINTS.items():
        os.makedirs(tag, exist_ok=True)
        ckpt = os.path.join(ckpt_dir, ckpt_name)
        skipped[tag] = 0
        for init in inits:
            stamp = init.strftime("%Y%m%d%H%M%S")
            out = f"{tag}/forecast_{stamp}.nc"
            if os.path.exists(out):
                skipped[tag] += 1
            else:
                jobs.append((len(jobs), tag, ckpt, init, out))

    print(f"\n{len(jobs)} forecasts to run, {sum(skipped.values())} already present")
    print(f"{WORKERS} workers ({N_GPUS} GPUs x {WORKERS_PER_GPU}), {THREADS} threads each\n")
    if not jobs:
        return

    tmpdir = tempfile.mkdtemp(prefix="anemoi_cfg_")
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(lambda j: run_one(j, tmpdir, len(jobs)), jobs))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    for tag in CHECKPOINTS:
        ok = sum(1 for t, good in results if t == tag and good)
        bad = sum(1 for t, good in results if t == tag and not good)
        print(f"[{tag}] {ok + skipped[tag]}/{len(inits)} present"
              + (f" -- {bad} FAILED" if bad else ""))
    # any missing file drops that init from the intersection for EVERY run, so a partial failure
    # costs more than the forecasts it lost
    if any(not good for _, good in results):
        print("\nRerun to retry the failures: existing files are skipped.")


if __name__ == "__main__":
    main()
