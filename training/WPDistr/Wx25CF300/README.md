# Wx25CF300 — the baseline power model

Frozen recipe. **One file per stage, never overwritten** — that is what made the earlier lineages
unreproducible. Its weather counterpart is `../RegularWeather/` (same structure, no power).

| file | trains | output |
|---|---|---|
| `pretrain_150k.yaml` | 150k steps, `capacityfactor` weight **0** -> a pure weather model | `Wx25CF300/pretrain/` |
| `finetune_25k_power.yaml` | 25k fine-tune, `capacityfactor` weight **300** | `Wx25CF300/finetune/` |

## Recipe

**Pre-training** — from scratch (`warm_start: null`), 150 000 steps, `limit_batches.training: null`
(full dataset, ~13 000 steps/epoch, so ~12 epochs), rollout **1**, lr 1.25e-4 / warmup 1000 /
cosine to 3e-7, AdamW beta = (0.9, **0.999**), MSE loss with the full pre-training weights
(z 12, t 6, u 0.8, v 0.5, q 0.6, w 0.001, mcc 0.1, ws10/ws100 1) and **capacityfactor 0**,
training window **2015-01-01 -> 2024-01-31**.

The `cf_head` decoder is present but inert here: its output layer is zero-initialised
(`ZERO_INIT_OUTPUT`) and, at loss weight 0, never receives a gradient — so the branch contributes
exactly 0 and every weather channel is computed as the standard decoder would. It is kept so both
stages are architecturally identical and so the `EXPECT_N_OUT == 55` assertion guards this stage too.

**Fine-tune** — warm start from the 150k (`load_weights_only: True`, so the step counter and the LR
cycle restart; ECMWF report that restarting the optimiser for rollout improves things, AIFS Single
v1.1 model card), 25 000 steps, `limit_batches` 1000/10 (-> 25 epochs), rollout 1 -> **11**,
**nothing frozen**, lr 3e-5 / warmup 1000 / cosine over 25 000, AdamW beta = (0.9, **0.95**),
`ScalerAwareCombinedLoss` [1, 1] = Huber(delta=1) on weather + MAE on power, weather weights
x0.25 of pre-training (**ws10/ws100 held at 0.5, i.e. x0.5 — the wind is up-weighted 2x relative to
the rest; inherited from FS2, never tuned**), power weight **300**, window **2020-01-01 ->
2024-01-31** (the capacityfactor observations start in 2020).

Rollout 11 = the 11 scored leads (+3..+33 h): the rollout loss is a mean over steps 1..N
(`anemoi/training/train/tasks/rollout.py:163-177`) and the reported metric is a mean over the same
11 leads, so N = 11 makes them the same functional.

`config_validation: false` is **required**: validation strips `cf_index`/`cf_hidden` before hydra
instantiates the decoder (see `cf_head.py:90`) and silently drops `training.optimizer.betas`.

## Running

```bash
export PYTHONPATH=/mnt/weatherloss/WindPower/training:$PYTHONPATH   # rollout_tasks, cf_head, combined_loss
cd /mnt/weatherloss/WindPower/training/WPDistr/Wx25CF300
setsid nohup anemoi-training train --config-path=$PWD --config-name=pretrain_150k \
    > pretrain.out 2>&1 < /dev/null & disown
sleep 120 && pgrep -af anemoi-training | wc -l     # expect 2 (one per GPU)
```

**Check in the first minutes:** the epoch length must be ~13 000 steps. If it reports 1000 or 500,
`limit_batches: null` did not pass through as "no limit" and the run will stop at the wrong place.

When pre-training finishes, put its checkpoint path into `finetune_25k_power.yaml`'s
`warm_start: FILL_IN_AFTER_PRETRAINING` and launch the same way with `--config-name=finetune_25k_power`.

Do not run two trainings on the same pair of GPUs. `graph.overwrite: false` in all four configs so
they share the existing `WPDistrgraph.pt` instead of racing to regenerate it.

## Checkpoint selection

Checkpoints are written every epoch. Select on the **validation window** (2024-02-01..2024-07-31),
never on the test year:

```bash
cd /mnt/weatherloss/WindPower/verification
# rank_checkpoints.py: _CKPT_DIR -> Wx25CF300/finetune/checkpoint/<uuid>, BACKWARD_WINDOW = True
python rank_checkpoints.py
```

Rank by **`+21h`** (leads >= 21 h) rather than the all-lead mean — the all-lead average is diluted by
the short leads and has hidden a real long-lead gain before. Check the farm WIND column has not
degraded.

## Notes

- 25k is not `s1_wx25_cf300_17k` extended: the cosine now decays over 25 000, so every checkpoint
  sits on a different schedule. It will not reproduce 5.77 at step 17 500.
- `verify_power.py`: add the inference tag to `BACKWARD_WINDOW_RUNS` (exact string match) or the
  power reads ~8-9 % instead of ~5-6 %. The tell is a bad direct row beside a normal curve row.
- The causal "what does the power objective cost the weather" number does NOT come from comparing
  this to `../RegularWeather` — they differ in loss, weights, task, decoder and data window. It comes
  from the matched control pair already measured (`wx25_cf300_R11` vs `nw_wx0_10k_r11`, <=0.7 %
  domain RMSE). See `../MODELS.md`.
