# Model registry — direct power head (WPDistr)

## 2026-09-25: the frozen recipe (supersedes the ad-hoc lineages below)

Two directories, four YAMLs, **one file per stage — never overwrite them again**:

| dir | files | what it is |
|---|---|---|
| `RegularWeather/` | `pretrain_150k.yaml`, `finetune_25k.yaml` | what a **conventional weather model** looks like in this setup: capacityfactor DROPPED from the dataset, standard decoder, `GraphForecaster`, MSE with full pre-training weights, fine-tune on 2015+. A BASELINE, deliberately not a control. |
| `Wx25CF300/` | `pretrain_150k.yaml`, `finetune_25k_power.yaml` | the power model: `cf_head`, `BackwardWindowForecaster`, capacityfactor weight 0 in pre-training -> 300 in the fine-tune, weather x0.25, fine-tune on 2020+ (power obs start then). |

Both: 150k pre-train (no batch limit, rollout 1, lr 1.25e-4, betas 0.9/0.999) then 25k fine-tune
(limit_batches 1000/10, rollout ->11, lr 3e-5 warmup 1000, betas 0.9/0.95, nothing frozen,
max_epochs 50). `config_validation: false` in all four — `true` silently drops the optimizer betas.

**The causal power-cost number does NOT come from this pair** (they differ in loss, weights, task,
decoder and data window). It comes from the matched control already measured on the previous
lineage: `wx25_cf300_R11` vs `nw_wx0_10k_r11`, <=0.7 % domain RMSE (finding 3). This pair answers
the different question: is the power model's weather comparable to a normal weather model?


Source of truth for which model is which. Updated 2026-09-25.
Numbers are MEASURED; anything unmeasured says so. Append, don't rewrite history.

**Current best power model: `s1_wx25_cf300_17k`** (test year 5.68 % MAE of 2261 MW, leads 3–33 h).
**Current best weather model: `nw_wx0_10k_r11`** (−6.4 % domain RMSE vs `base_7500`), pending `s1_wx25_17k`.

---

## Shared lineage

All WPDistr models below descend from one stage-1 checkpoint, trained with `capacityfactor`
weight **0** (so it is a pure weather model, and its power head is untrained):

```
NoWeightPower/checkpoint/6d6611e1a6e44c93a37c4e7a34f6ce92/anemoi-by_time-epoch_011-step_150000.ckpt
```

Stage 2 of that lineage lives in the **same directory** (the step counter restarts when
`load_weights_only: true`):

```
.../6d6611e1a6e44c93a37c4e7a34f6ce92/anemoi-by_time-epoch_007-step_007500.ckpt     # = base_nw7500
```

`RegularWeather` is NOT in this lineage: separate model, no `capacityfactor` variable at all,
different graph (`WindAIgraph.pt`), 150k + 7.5k with rollout reaching ~8.

---

## Bases (power head untrained or co-trained; no stage-3 fine-tune)

| name | recipe | power % | weather vs `base_7500` |
|---|---|---|---|
| `base_7500` | co-trained stage 2, `capacityfactor` weight 100 | 11.12 | 0 (the reference) |
| `base_nw7500` | weather-0 stage 2, 7.5k, rollout→8 | 27.29 | **−1.7 %** |

`base_nw7500`'s −1.7 % is the **pre-training cost of the power co-target** (z500 −2.4, z850 −3.0,
msl −4.6, ws100 −0.6). `base_7500`: `VHCapacityBackWin/checkpoint/538e92ed028f46bf9a0826ef7bb3dee3/`.

---

## Power models (stage 3 = weather ×0.25 of pre-training weights, power weight 300 unless noted,
## backward window, `cf_head` decoder, lr 3e-5, AdamW β=(0.9,0.95), train 2020-01-01→2024-01-31)

**Weight detail** (source: the commented `general_variable` block in `NoWeightPower/NoWeightPower.yaml`
— this lineage's own record; `VHCapacityBackWin.yaml` has the same weather weights but
`capacityfactor: 100` instead of `0`, which is the only difference between the two pre-trainings):
pre-training used z 12, t 6, u 0.8, v 0.5, q 0.6, w 0.001, mcc 0.1, default 1,
**ws10 1, ws100 1**, capacityfactor 0. The fine-tune scales the ordinary variables by ×0.25 but
holds ws10/ws100 at **0.5**, i.e. ×0.5 — so the wind is up-weighted **2× relative to every other
weather variable** compared to pre-training (ws:z goes from 1:12 to 1:6). That is inherited from
FS2's recipe, never tuned, and is the likely reason ws100 is the field the power objective spares
domain-wide (finding 3). `wx()` in `SweepFS2/sweep.py` builds this: its `PRETRAIN` dict omits the
two wind speeds and sets them as a constant.

| name | stage 3 | rollout | frozen | val power / +21h | test yr | weather | farm wind |
|---|---|---|---|---|---|---|---|
| `FS2` | co-trained base + 5k, power only (no weather anchor) | →6 | encoder | 6.65 / 7.77 | ~6.0 | **+14.2 %** | — |
| `wx25_300` = `nw_wx25_cf300_10k` | nw base + 10k | →6 | encoder | 6.09 / 6.65 | 5.81 | −5.0 % | 1.092 |
| `wx25_cf300_R11` = `nw_wx25_cf300_10k_r11` | nw base + 10k | →11 | encoder | 6.03 / 6.30 | 5.78 | −6.2 % | **1.077** |
| **`s1_wx25_cf300_17k`** | **150k → ONE 17.5k fine-tune** | →11 | **none** | **5.77 / 6.20** | **5.68** | **−8.3 %** | 1.106 |

Checkpoints (final, `inference-*` prefix, epoch/step as named):

```
FS2                  VHCapacityBackWinFinetune/checkpoint/f9ff915ed31f4356b1da9c48217377fc/  e009s005000
wx25_300             SweepFS2/nw_wx25_cf300_10k/checkpoint/47fd110a1be8493990fa43406c15db90/ e019s010000
wx25_cf300_R11       SweepR11/nw_wx25_cf300_10k_r11/checkpoint/011b4ead25f5425685cf0e00f8e7b4b3/ e019s010000
s1_wx25_cf300_17k    SweepS1/s1_wx25_cf300_17k/checkpoint/c0610d42a795424894d2a54c26694435/ e034s017500
```

`s1_wx25_cf300_17k` plateaued: 14 000 steps gives 5.76 / 6.13, 17 500 gives 5.77 / 6.20 — tied.

---

## Controls (identical to the row above them except `capacityfactor` weight 300 → **0**)

These exist so the power objective's cost can be measured causally. `capacityfactor` is diagnostic
(output-only), so inputs, architecture and data are untouched; only that one output's gradient differs.

| name | control for | weather | checkpoint |
|---|---|---|---|
| `nw_wx0_10k` (inference tag `RegularWeather10k`) | `wx25_300` | see sweep report | `SweepFS2/nw_wx0_10k/checkpoint/4b66cd6b48ca49f68c49ca570fb25a55/` |
| `nw_wx0_10k_r11` | `wx25_cf300_R11` | **−6.4 %** | `SweepR11/nw_wx0_10k_r11/checkpoint/e548b7ed31e6473aac274b1a306957b8/` |
| `s1_wx25_17k` | `s1_wx25_cf300_17k` | **PENDING** | `SweepS1/s1_wx25_17k/...` |

A control's POWER column is meaningless (~27 %, head never trained) — that is the correctness check.

---

## Baselines (test year, mean over leads 3–33 h, BE regional total, MAE % of 2261 MW)

| system | |
|---|---|
| `s1_wx25_cf300_17k` direct head | **5.68** |
| `wx25_cf300_R11` direct head | 5.78 |
| `wx25_300` direct head | 5.81 |
| `Noweight10k` direct head (no weather anchor) | 5.85 |
| WindPowerTransformer post-processor | 6.36 |
| RegularWeather + measured curve (fit 2021-01-01→2024-01-31) | 7.13 |
| RegularWeather + specs curve | 10.37 |

Curve baselines must use **RegularWeather's** wind. Applying the curve to a power model's own wind
handicaps it — those models have a growing low wind bias at the farm cells (see caveats).

---

## Established findings

1. **Weather anchoring works.** Giving the other weather variables ×0.25 of their pre-training
   weight turns FS2's +14.2 % weather drift into an −8.3 % improvement, at no power cost.
2. **The ×0.25 "cost" was loss rescaling**, not a power/weather conflict: scale the power weight
   with it (100 → 300) and it inverts. Compensation ≈ `cf = 100 + 1000·frac`.
3. **The power objective's domain cost depends strongly on rollout.** Measured against the matched
   control (730 inits, whole domain, scorecard):
   - at **rollout 6** (`wx25_300` vs `nw_wx0_10k`): 2–3 % on z/msl at long leads, ≤1 % on winds.
   - at **rollout 11** (`wx25_cf300_R11` vs `nw_wx0_10k_r11`): **≤0.7 % on everything**, mostly
     0.2–0.4 % (ws100 +0.2..+0.6, z500 +0.1..+0.4, msl ~0.0, t2m +0.2..+0.5).
   So matching the training horizon to the evaluation range both improved power AND nearly
   removed the weather cost of carrying the power objective.
3b. **Locally (15 farm cells) the cost is real and large, and it is NOT the same fields.**
   `wx25_cf300_R11` vs its control at BE: u500 +1.2→+27.3 %, u600 +3.6→+28.8 %, v500 →+13.6 %,
   t2m →+9.2 %, **ws100 +2.5..+4.6 %**, while q1000 is BETTER (−2.5..−4.9 %). Domain-wide u500 is
   +0.2 %, so this is confined to the cells the power loss is applied on: the model sacrifices
   fields power does not need, exactly there.
   **Consequence for the paper: the model does not produce a better wind at the farm cells; it
   produces a better power forecast.** The head beats the curve while feeding on a slightly worse
   wind, by compensating internally — see caveat 1.
4. **Beating RegularWeather domain-wide was training length, not the power head** (the 10k
   continuation alone is worth up to +13 % at 36 h). Do not claim otherwise.
5. **The direct head beats the measured curve** by ~1.3 pp and the post-processor by ~0.5 pp.
6. **Freezing more does not protect the weather**: encoder+processor frozen gave 6.92 power AND
   +7.4 % weather drift. Loss weighting is the lever, not architecture freezing.
7. **Validation and test inference both unroll to 36 h** (`LEAD_HOURS = 37`), scored on 11 leads
   3–33 h — so rollout 11 matches the training horizon to the evaluation range.

## Open caveats

1. **Farm-cell wind bias grows with stage-3 training.** Measured-curve bias at 3 h:
   `wx25_300` −10.5 MW → `R11` −29.9 → `s1` −53.7. The head absorbs it (its own bias stays
   −10..−29). `s1_wx25_17k` will say whether the power head causes this or the fine-tune does.
2. **`s1`'s gain is confounded**: it differs from `R11` by three things — no stage 2, encoder
   unfrozen, warmup 1000 vs 200. The disambiguating run (`R11` with `submodules_to_freeze: []`,
   ~4.4 h) has NOT been done.
3. **Single seed everywhere.** Seed spread: 0.01 pp on power for a repeated recipe, 0.3 pp on the
   WEATHER column. Differences smaller than that are not differences.
4. **Test year is 730 of 2916 inits** for `R11` and `s1` — roughly one season. Only `wx25_300`,
   `RegularWeather`, `Noweight10k` and `Transformer` have the full year.
5. **No scorecards for `s1`** yet. `R11` has both (domain + BE) against `nw_wx0_10k_r11`, 730 inits,
   inference tag ` RegularWeather10k_wx0_r11` (note the leading space in the tag as used).
   `wx25_300` has four (vs RegularWeather and vs `RegularWeather10k`, each domain + BE).
6. **17.5k steps is not a tuned choice** — it was picked to step-match 7 500 + 10 000. A 25k run
   has not been tried.
7. **Selection pressure**: many decisions have now been made on the same 48-init validation window.
   Keep the test year as the final arbiter; do not select on it.

## Gotchas that have cost time

- **`BACKWARD_WINDOW_RUNS` in `verify_power.py`** is an exact-string list. A run missing from it is
  graded one step off and reads ~8–9 % instead of ~5–6 %. The tell: the direct row is bad while the
  run's measured-curve row is normal.
- `rank_checkpoints.py` has the same switch (`BACKWARD_WINDOW = True`).
- The sweep's WEATHER column is always vs `base_7500`. Runs from the nw lineage (`nw_*`, `s1_*`)
  must be read against `base_nw7500` (−1.7 %), not against 0.
