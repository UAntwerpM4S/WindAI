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

## FULL TEST YEAR, 2909 common inits (2026-09-25) — the definitive table

| system | 3–33 h | 3–12 h | 21–33 h |
|---|---|---|---|
| **`s1_wx25_cf300_17k` / direct** | **5.63** | **4.92** | **6.19** |
| `wx25_cf300_R11` / direct | 5.73 | 5.00 | 6.34 |
| `wx25_300` / direct | 5.81 | 5.02 | 6.46 |
| WindPowerTransformer post-processor | 6.43 | 5.53 | 7.19 |
| RegularWeather + measured curve (2021 fit) | 7.25 | 6.39 | 7.96 |

Same ordering as the earlier 730-init subset, so the result is not a seasonal artefact.

**The curve applied to each model's OWN wind**: R11 6.94, s1 6.99, wx25_300 7.00,
RegularWeather 7.25.

**DO NOT read that as "the power models' wind is better than RegularWeather's."** The
`RegularWeather` in this table is the OLD one (150k + 7.5k); the power models have 10 000-17 500
MORE steps on top of the same base. It is the training-length confound (finding 4) restated as a
wind claim. The comparison is only meaningful once the retrained RegularWeather (150k + 25k,
`../RegularWeather/`) exists.

What IS valid here, because it is internally controlled — same model, same wind, same sample:
**head vs curve on `s1` = 6.99 -> 5.63, a 1.36 pp gain.** That one owes nothing to a wind-quality
difference between models.

Still open, not settled by this table: whether the power models' growing negative farm-wind bias
(curve bias at 3 h: wx25_300 -14.3, R11 -32.2, s1 -54.8 MW) costs anything against a properly
trained weather model. The retrained RegularWeather answers it.

## 2026-09-27 — THE FAIR COMPARISON (frozen recipe, matched budgets, full test year)

All four systems below sit on models with the SAME training budget (150k + 25k). Previously both
baselines were fed by the OLD RegularWeather (150k + 7.5k), i.e. 17 500 fewer fine-tune steps than
the head — the training-length confound (finding 4) hiding inside the baselines. Fixed by:
`extract_cells.py --runs RegularWeather_20K=...` -> `build_targets.py --cells cells_BE_RegularWeather_20K.nc`
-> `cerra_norated/write_inference_norated.py` with DATASET/SRC_DIR/OUT_DIR repointed. No retraining:
the post-processor learned from CERRA and is indifferent to which forecast wind it is handed.

**POWER, 2909 inits, mean over leads 3–33 h, BE total, MAE % of 2261 MW:**

| system | 3–33 h | was (old wind) |
|---|---|---|
| `s1_wx25_cf300_17k` / direct | **5.63** | — |
| **`Wx25CF300_20` / direct** | **5.70** | — |
| CERRA Transformer (no-rated) / direct | 6.12 | 6.43 |
| RegularWeather_20K + measured curve | 6.93 | 7.25 |

Head's margin: **+0.42 over the post-processor** (was 0.73), **+1.23 over the measured curve**
(was 1.55). Both baselines gained ~0.3 from the better wind; the conclusion survives the fix.

**WEATHER vs `RegularWeather_20K` (scorecard, 2916 inits, + = the power model is WORSE), at 36 h:**

| | whole domain | farm cells (15) |
|---|---|---|
| ws100 | **+1.8 %** | +2.8 % |
| ws10  | +1.8 % | +3.7 % |
| t2m   | +2.2 % | +10.7 % |
| msl   | +5.5 % | **+59.0 %** |
| z850  | +7.7 % | +42.4 % |
| z500  | +10.2 % | — |
| q1000 | +1.6 % | +37.0 % |

**Defensible claim: domain-wide the power model matches a matched-budget weather model on wind and
near-surface fields, at a cost of 5–10 % on geopotential and MSL at long leads; at the trained cells
the local weather is substantially degraded.** NOT "one model does both" unqualified.
Consistent with the loss weights: ws is held at 0.5 while z is 3.0 x 0.25, so the wind is what the
objective protects and geopotential is what it spends. At 3 h several near-surface fields are
BETTER (q1000 -1.3, t2m -0.8), so the cost grows with lead — a compounding effect through the
rollout, not a fixed offset.

**Caveat:** these are two separately trained models, so this mixes the power objective with
different pre-trainings, losses, weights and fine-tune windows. The CONTROLLED figure is still the
R11 pair (<=0.7 % domain, finding 3). The gap between 0.7 % and these numbers is itself a measure
of how much is lineage rather than objective.

**CORRECTED 2026-09-27:** the curve applied to each model's own wind (head 6.78 vs RegularWeather_20K
6.93) does NOT mean the head's wind is better. Its ws100 RMSE is worse at every lead
(0.978->1.71 vs 0.906->1.66). The measured curve is nonlinear and fitted to CERRA truth, so a
differently-biased wind can give better power MAE while being a worse wind. Do not use the curve row
as a wind-quality measure.

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
4. **Test year is 730 of 2916 inits** for `R11` and `s1` — every 4th init, **spread across the
   whole year**, so it is seasonally representative: smaller n, not a seasonal slice. (Earlier
   notes in this file called it "one season"; that was wrong.) Only `wx25_300`,
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

---

## 2026-09-29 — why the two scorecards differ, and what the SweepWeather runs can/cannot see

### The domain card and the farm card are TWO DIFFERENT MECHANISMS

**Domain (z +10.2 %, msl +5.5 %, ws100 +1.8 % vs `RegularWeather_20K`) = the loss weights.**
The ordering ranks by how much weight each field lost: z 12→3.0 (÷4), t 6→1.5 (÷4),
ws10/ws100 1→0.5 (**÷2**). Plus the loss SHAPE: RegularWeather uses MSE, the fine-tune uses
Huber(δ=1), which is linear beyond δ and so stops chasing large residuals — exactly where
long-lead z error lives. That is why z degrades 5.7× more than ws off a weight ratio of only 2.
The matched control puts the POWER OBJECTIVE's share of this at ≤0.7 % (finding 3), so ~90 % of
the domain gap is weighting, not power.

**Farm cells (msl +59 %, z850 +42 %, q1000 +37 %) = the loss is a MEAN OVER CELLS.**
Per farm cell, with 9 levels: z 27 + t 13.5 + u/v/q 4.3 + surface 1.5 ≈ **46** of weather weight
against **300** for power → the power term is ~6.5× everything else combined, i.e. **87 % of the
loss arriving at that node**. Domain-wide the same term is 300 × 172/72 668 = 0.71 against 46,
i.e. **1.5 % of the total loss**. Same number, two views.
Wrecking msl at the 15 BE cells by 59 % raises the domain msl term by 15/72 668 × 1.5 ≈ **0.03 %**.
That is the entire penalty the loss charges for the saturated bottom card.
(Weights-per-cell are approximate: they ignore the `pressure_level` scaler, which does not change
the order of magnitude.)

Two ingredients are both required:
1. **Shared readout.** `node_data_extractor = LayerNorm(512) → Linear(512, 55)`; `cf_head` reads
   the SAME 512-dim `x_dst`. Serving power must move that vector, and every weather channel at
   that node is a linear readout of it. → why there is any coupling.
2. **Mean-over-cells loss.** The pushback is divided by 72 668. → why the coupling is not resisted.

Supporting evidence it is SLACK rather than a real capacity conflict: the sacrificed fields MOVE
between runs (R11 control: u500 +27 %, u600 +29 %; `Wx25CF300_20`: msl/z850/q1000). A physical
mechanism would hit the same fields each time. Against: farm ws100 is +2.8 %, so the model degrades
even what it needs — small, but the trade is not purely in unused channels.

### Why the damage does NOT spread, and the test that would prove it

The gradient is NOT spatially local: processor = `GraphTransformerProcessor`, **8 layers**,
`MultiScaleEdges x_hops: 1` on a `LimitedAreaTriNodes` resolution-9 mesh (~14 km spacing —
ESTIMATED from 10·4⁹+2, confirm by measuring `edge_length` in the graph .pt). Encoder KNN 12,
decoder KNN 3. So **one step reaches ~110 km**, and 12 rollout steps ≈ **1300–1400 km** — the whole
inner domain at 36 h. The power gradient therefore does touch hidden nodes serving other cells.

It stays local because the model can CONDITION ON FARM IDENTITY: `turbmask`, `capacity`,
`turbinecount` are **forcings** (per-node static inputs, `finetune_25k_power.yaml:64-66`), so the
cheap thing to learn is "where turbmask = 1, do this" rather than a diffuse spatial change.

**UNTESTED, cheap, and worth doing:** score a RING of cells at ~20–60 km and ~60–150 km from the
farms (`select_cells` subset + one scorecard run). A halo is the default expectation given the
110 km/step reach. If damage drops to ~0 immediately outside the farm cells, that proves the
correction is genuinely turbmask-keyed — a stronger claim than "domain-wide it's fine".

### CONSEQUENCE: domain weather feeds LONG-LEAD POWER

A farm cell's +33 h forecast is downstream of ~1300 km of domain weather, so domain degradation is
not cosmetic. Already visible: the weather-clean base held **+21h 6.65 vs 7.10, a 0.45 pp gap that
did not shrink** with more fine-tuning, while the all-lead gap collapsed to noise. So read the
sweeps' **`+21h` column**, not the all-lead `POWER` mean, for the weather-anchoring effect.

### What SweepWeatherA / B can and cannot answer

`rank_checkpoints` scores farm POWER, farm ws100 (WIND), and domain weather on every 7th cell.
**Farm-cell msl/z850/q1000 are scored by NEITHER sweep.** The bottom card will look unchanged after
both finish. That needs `verify_scorecard` with `DOMAIN = "BE"` on the winners afterwards.

Local power share (power / (power + ~46·frac·4) at a farm cell) per run:

| 62 % | 76 % | 87 % (= shipped) | 93 % |
|---|---|---|---|
| A `wx100_cf300` | A `wx50_cf300`, B `cf150`, B `lw_1_05` | A: the 6 compensated runs, B: base/huber/lr/rollout | A `wx100_cf2400`, B `cf600`, B `lw_1_2` |

**Compensating the power weight holds the local ratio EXACTLY fixed** — that is what "compensated"
means — so 7 of Sweep A's 10 runs cannot move the farm card by construction. Sweep A is a DOMAIN
experiment; Sweep B is the local one.

### VERIFIED FROM SOURCE: `lw_*` and `cf*` are the same knob

`CombinedLoss.forward` is `Σ loss_weights[i] × loss_fn_i(...)` (`losses/combined.py:164-170`), and
`BaseLoss.scale` multiplies by the scaler then `reduce` averages — **no normalisation by the scaler
sum** (`losses/base.py:76-182`). The loss is exactly linear in both knobs, so:

    lw_1_05 (0.5 × scaler 300) ≡ cf150      lw_1_2 (2 × scaler 300) ≡ cf600

Same seed, same objective. **Do not read them as different settings.** Keep them as a run-to-run
NOISE ESTIMATE (DDP + non-deterministic kernels make same-seed reruns non-bitwise) — that is the
error bar for every other row in the table.

### `Wx25CF300` vs `RegularWeather_20K`: what actually differs (corrects an overstatement)

IDENTICAL: both pre-trainings (150k, 2015+, rollout 1, lr 1.25e-4, betas 0.9/0.999, MSE, full
weights) and the whole fine-tune schedule — `limit_batches` 1000/10, **rollout start 1 / increment 1
/ max 11 in BOTH**, lr 3e-5 / warmup 1000, betas 0.9/0.95, nothing frozen, same validation window.

DIFFERS, only three things (+ task/decoder/dataset): fine-tune **loss** (MSE vs Huber δ=1 + MAE),
**weather weights** (full vs ×0.25), **window** (2015+ vs **2020+**, forced — power obs start 2020).
The window is the one genuinely awkward item: RegularWeather fine-tunes on 9 years, the power model
on 4, same 25k steps. It favours RegularWeather. Volunteer it.

NOT differences (I claimed these and was wrong): the rollout schedule, and the pre-training recipe.
Rollout →8 was the OLD RegularWeather (150k + 7.5k), not the retrained twin.

### OPERATIONAL: `base12k` HUNG DURING A CHECKPOINT SAVE (not slow)

2026-09-28, SweepWeatherA. Training started ~08:51; `base12k` reached **step 9000 at 14:05:56**
and its last log lines are `anemoi.utils.checkpoints` writing the inference checkpoint's supporting
arrays. Then **nothing for 6.75 h** until `TRAIN_TIMEOUT_H = 12` fired at ~20:51 and SIGTERM'd it.
The next run died instantly with `CUDA driver initialization failed` — the SIGKILL'd DDP ranks had
not released the GPUs 10 s later — and with `RETRY_FAILED = False` every following run is marked
failed and skipped, so the sweep empties its own run list in minutes.

**RATE (corrects the panic estimate written earlier the same day): 9000 steps in 5.24 h = ~1720
steps/h, so a 12k run is ~7 h and ONE FS2-EQUIVALENT IS ~21 h.** Sweep A's 3.7 units ≈ 78 h training
+ ~7 h scoring ≈ **3.5 days**; Sweep B's 3.2 ≈ 3 days. NOT six days — the run was on pace, it stalled.

Suspected cause: checkpoint save to the shared `/mnt/weatherloss` mount stalling, aggravated by the
second container checkpointing the same storage concurrently (the same deterministic anchor scoring
took 0.88 h on A vs 0.68 h on B, a 29 % throughput spread). Anemoi writes a training AND an
inference checkpoint every epoch = 12 saves per 12k run, of which `SCORE_POINTS = 5` are ever read.

Fixes: `TRAIN_TIMEOUT_H = 24` (the 20k run at cost 0.74 needs ~16 h — do NOT set 40, that just lets
a hang burn longer), `RETRY_FAILED = True`, and a `time.sleep(120)` between runs so CUDA contexts
drain. If the hang recurs, cut checkpoint frequency or run the two sweeps SEQUENTIALLY.
