# MixedRollout — the run that beat the measured power curve

The first configuration whose **direct** `capacityfactor` output beats a farm-fitted
method-of-bins power curve, on the full out-of-sample test year, with the weather forecast no
worse than the weather-only model's.

Written 2026-09-09. Numbers are TOTAL BE, MAE as % of 2261 MW, bin-weighted,
**2916 inits = every init in 2024-08-01 .. 2025-07-31**, scored by `verification/verify_power.py`.

---

## Result

| system | direct | own measured curve | conversion | wind | vs benchmark |
|---|---|---|---|---|---|
| RegularWeather + measured curve | — | **7.25** | — | — | benchmark |
| VH base (no fine-tune) | 11.67 | 7.37 | +4.30 | +0.12 | +4.42 |
| **MixedRollout** | **6.94** | 7.19 | **−0.26** | **−0.05** | **−0.31** |

- **conversion** = direct − this model's own curve. Same forecast both ways, so it isolates the
  conversion step. Matched on training budget by construction (it is one checkpoint).
- **wind** = this model's own curve − RegularWeather's curve. Same conversion both ways, so it
  isolates forecast quality.

Stable under resampling: the 730-init subset gave −0.28 / −0.04 / −0.34 for the same three
columns. 4x the data moved every margin by ≤0.03 pp.

**Where the win is:**

| wind bin | share | RW curve | head | diff |
|---|---|---|---|---|
| 0–4.5 | 15.5% | 1.72 | 1.80 | +0.09 |
| 4.5–8 | 29.7% | 4.71 | 4.82 | +0.11 |
| **8–12** | **32.1%** | **11.04** | **10.08** | **−0.96** |
| 12+ | 22.7% | 8.99 | 8.77 | −0.22 |

Almost all of it is the 8–12 m/s band — the steep part of the power curve, where conversion is
most sensitive to anything beyond a single wind speed. The head is marginally worse in the two low
bins, where the curve is nearly flat and there is nothing extra to exploit.

**Lead structure** (head − RW curve): +0.46, +0.09, −0.23, −0.39, −0.39, −0.40, −0.46, −0.46,
−0.50, −0.55, −0.60 at 3…33h. Wins 9/11 leads, monotone with lead. The short-lead loss is
mechanical: `train/tasks/rollout.py:176` averages the loss as `1/R` over rollout steps, so at R=6
the 3h step carries a sixth of the gradient weight it had at R=1.

**Controls.**
`RegularWeather5k` — RegularWeather given the same +5k steps, wind-specialised, same rollout
curriculum and LR schedule — scored 7.20 against its own 7.18 on the 730-init subset. The extra
budget is worth +0.015 pp, i.e. nothing. The budget objection is closed by measurement.

---

## Lineage

```
150k  regular training          (weather + capacityfactor co-target)
 7.5k rollout stage, max 8      -> VeryHighCapacityGT/checkpoint/
                                   f2baebf0b95b4d8dafb493d3fb36d29a/
                                   anemoi-by_time-epoch_007-step_007500.ckpt
 5k   MixedRollout fine-tune    <- this document
```

RegularWeather shares the first two stages with no power target at any point, and is the
System A comparator.

---

## Architecture

`anemoi.models.models.AnemoiModelEncProcDec`, 512 channels, anemoi-models 0.11.2 /
anemoi-training 0.8.1 (conda env `anemoi_old`).

| | |
|---|---|
| encoder | `GraphTransformerForwardMapper`, 1 chunk, mlp_hidden_ratio 2, 8 heads |
| processor | `GraphTransformerProcessor`, 8 layers, 2 chunks, mlp_hidden_ratio 2, 8 heads |
| decoder | `cf_head.CFHeadGraphTransformerBackwardMapper`, `cf_index: 54`, `cf_hidden: 128` |
| residual | `SkipConnection`, step −1 |
| output_mask | `Boolean1DMask` on `cutout` |
| bounding | `ReluBounding` on q_500…q_1000; `HardtanhBounding` on `capacityfactor` [0, 1] |
| trainable_parameters | all zero — no node or edge embeddings |

```
Total params      23,915,832
Trainable          2,534,712     decoder body (2,467,895) + cf_head (66,817)
Non-trainable     21,381,120     encoder + processor, frozen
```

**The private head** (`training/cf_head.py`):

```
LayerNorm(512) -> Linear(512, 128) -> GELU -> Linear(128, 1)     66,817 params
```

Zero-initialised final layer, added as a residual into output channel 54 only, so at step 0 the
model is bit-identical to the warm start. Pointwise, therefore safe under grid sharding. It lives
inside `decoder`, which is why `submodules_to_freeze: [encoder, processor]` leaves it trainable.
That is the whole head — `cf_head.py` now contains nothing else, every switch it used to carry
having been tested and removed (see *Ruled out* below).

`cf_index: 54` is the **model output** index, not the zarr index (64). 67 data variables − 12
forcings = 55 output channels, `capacityfactor` last.

---

## Data

CERRA ~5.5 km inner (72,668 cells) cut out of ERA5 ~31 km outer (13,965), 82,068 nodes total,
77,419 after `RemoveUnconnectedNodes`. `capacityfactor` exists on 172 cells, 15 of them BE farm
cells; NaN everywhere else and imputed to 0.

| | |
|---|---|
| frequency / timestep | 3h / 3h, `multistep_input: 2` |
| forcings (12) | lsm, insolation, sin/cos julian day, sin/cos lat, sin/cos lon, z_sfc, capacity, turbinecount, turbmask |
| diagnostic (1) | capacityfactor |
| imputer | `ConstantImputer` 0.0 for capacity, capacityfactor, turbinecount, turbmask |
| normalizer | mean-std default; std for q_*; max for z_sfc, turbinecount; **min-max for capacityfactor**; none for lsm, mcc, lat/lon, turbmask |
| train | 2020-01-01 .. 2024-01-31 |
| validation | 2024-02-01 .. 2024-07-31 |
| test | 2024-08-01 .. 2025-07-31 — **the evaluation window, fully out of sample** |
| limit_batches | training 500, validation 100 |

`capacityfactor` is min-max normalised to ~[0,1], which is why a Huber delta of 0.1 behaves like
MAE and 1.0 like MSE — deltas are in normalised units.

**Graph**: `LimitedAreaTriNodes` resolution 9, margin 300 km. data→hidden KNN 12 (cutout) +
CutOff 15 km (boundary), then `RestrictEdgeLength` 20 km. hidden→hidden `MultiScaleEdges`
x_hops 1. hidden→data KNN 3.

---

## The 5k fine-tune

```yaml
warm_start:            VeryHighCapacityGT .../anemoi-by_time-epoch_007-step_007500.ckpt
load_weights_only:     true
transfer_learning:     false
submodules_to_freeze:  [encoder, processor]

max_steps:             5000          # 500 batches/epoch -> 10 epochs
precision:             16-mixed
batch_size:            1
accum_grad_batches:    1
gradient_clip:         {val: 32.0, algorithm: value}
hardware:              2 GPUs, num_gpus_per_model 1, DDPGroupStrategy

optimizer:             AdamW, betas [0.9, 0.95]     # weight_decay = torch default 0.01
lr:                    warmup 200, rate 3.0e-5, cosine to 3.0e-7 over 5000
                       # effective 6.0e-5 — anemoi scales by total GPU count / model group size

rollout:               {start: 1, epoch_increment: 1, max: 6}
```

Rollout schedule as anemoi actually applies it (`rollout.py:189-191` increments at *epoch end*):

```
epoch    0    1    2    3    4    5-9
R        1    2    3    4    5     6      -> 2500 of 5000 steps at full rollout
```

The loss is an **unweighted mean over rollout steps** (`rollout.py:176`, `loss *= 1.0/self.rollout`).

### Loss

```yaml
training_loss:
  _target_: combined_loss.ScalerAwareCombinedLoss     # local subclass, see below
  loss_weights: [1, 1]
  scalers: [pressure_level, weather_variable, power_variable, node_weights, nan_mask_weights]
  losses:
    - _target_: anemoi.training.losses.HuberLoss      # WEATHER
      delta: 1.0                                      # ~MSE in normalised units
      ignore_nans: true
      scalers: [pressure_level, weather_variable, node_weights, nan_mask_weights]
    - _target_: anemoi.training.losses.MAELoss        # POWER
      ignore_nans: true
      scalers: [power_variable, node_weights, nan_mask_weights]

scalers:
  weather_variable:   {default: 0, ws10: 0.5, ws100: 0.5, capacityfactor: 0}
  power_variable:     {default: 0, capacityfactor: 100}
  pressure_level:     ReluVariableLevelScaler, group pl, y_intercept 0.2, slope 0.001
  nan_mask_weights:   NaNMaskScaler
  node_weights:       GraphNodeAttributeScaler, area_weight, norm unit-sum
```

**Every weather variable except ws10/ws100 has weight zero.** The trunk is supervised on wind and
power only; t, q, z, u, v, msl, t2m, mcc, wdir are unconstrained and free to drift. This has not
been measured — see open items.

`config_validation: false` is required: pydantic validates `model.decoder` as a discriminated
union on `_target_` and a custom class is not an accepted tag; it also strips unknown keys
(`cf_index`, `cf_hidden`) because the model schemas set no `extra="allow"`.

`PYTHONPATH` must include the directory holding `cf_head.py` and `combined_loss.py`, for
**inference as well as training** — the checkpoint is a pickled model object and needs the class
importable at the same module path forever.

---

## Why each non-default choice is there, and what it bought

| choice | reason | measured effect |
|---|---|---|
| private `cf_head` | every output is a *linear* readout of one 512-d latent; `capacityfactor` owned 513 params and, being diagnostic, has no residual skip | 7.80 → 7.66; 12+ MAE 11.29 → 10.80, ceiling −158 → −138 MW; wind untouched |
| split loss functionals | one Huber delta cannot serve both: power is censored/left-skewed above rated and needs a **median**, the wind needs a **mean** | 7.66 → 7.45; fleet ceiling −138 → −70 MW; 12+ MAE −0.96 |
| power = MAE | above rated the target is left-skewed (curtailment, outages), so squared error returns a central tendency below the plateau | this is the ceiling fix |
| weather = Huber δ1.0 | ≈MSE, so the shared trunk is not pulled toward a median | |
| `rollout` matched to the base | fine-tuning at R=1 **destroys** the multi-step skill the 7.5k stage built; the wind cost was that damage, not an inherent price of power | 7.45 → 6.94; wind +0.54 → −0.05 |
| `cf_hidden: 128` | 66,817 private params vs 513 | 256 was tried only in the frozen runs |
| `capacityfactor: 100` | targets exist on 172 of 77,419 cells; `node_weights` is unit-sum over *all* nodes, so a sparse variable gets N/M of a full field's grid weight | 100 × 172/72,668 ≈ 0.24, comparable to ws100's 0.5 |
| `nan_mask_weights` | without it the ~72k imputed zeros are trained as real observations | required |
| `combined_loss.ScalerAwareCombinedLoss` | anemoi 0.8.1 bug: `CombinedLoss.__init__` does `del self.scaler` but `GraphForecaster.update_scalers` reads `if name in self.loss.scaler` unconditionally (`train/tasks/base.py:319`), so any updating scaler crashes on batch 1 | subclass restores `.scaler` as a membership test over the sub-losses |

**Ruled out along the way**, each with a mechanism: Huber δ0.5 and δ0.175; `ws100: 1.0`;
freezing the whole decoder (wind +0.05, conversion +1.15); `LeakyHardtanhBounding` (upper clamp
inert, upside ≤0.06 pp); `trainable_parameters.data` node embeddings (`.data` is dead config,
only `.hidden` is read and changing it breaks the warm start).

### The per-farm plateau: four nulls, and what they close off

The head emits ~one plateau (~93% of nameplate) for farms whose real plateaus span 89.7%
(Northwester2) to 97.4% (Nobelwind); that plateau correlates −0.88 with each farm's 12+ bias. Four
attempts to fix it were trained and scored on the full test year. **All four landed inside the
~0.07 pp run-to-run seed noise**, and the code for each has been deleted rather than left as a dead
switch — this table is the record.

| attempt | what it supplied | result |
|---|---|---|
| `STATIC_INDEX = [129,130,131]` | `capacity`/`turbinecount`/`turbmask` handed straight to the branch, bypassing the frozen trunk | +0.07 — **information** is not the limit |
| `cf_layers = 2` | a second hidden layer, +16k params, so farm identity and wind level can interact | +0.07 — **capacity** is not the limit |
| `wdir100_cos/sin` in the loss | direction preserved in the shared latent | +0.06 — **representation** is not the limit |
| `CF_CELL_AFFINE` | a **free** scale and offset per data node on capacityfactor, init at identity | +0.00 fleet, plateau 0.27 pp **worse** |

The first two agreed with each other to 0.001 pp at every lead in every bin — identical training
trajectories, i.e. the additions changed nothing the head computes. The fourth is the decisive one:
that parameter needed no learning at all, and the bin table says where it went — the output was
pulled **down** everywhere, erasing the 8–12 over-prediction (+21.8 → −0.1 MW) and deepening the
12+ under-prediction (−47.7 → −74.3 MW).

So the plateau is not something the model cannot express. It is something the **objective does not
want**: above rated is a fraction of the 22% of hours in the 12+ bin, each farm is 1–2 of the 172
target cells, and a fleet MAE will always rather spend a parameter on the 8–12 band that carries
32% of the samples.

A `PlateauWeightedMAE` confirmed that directly, by up-weighting the power term where the *target*
was near rated: the hard version cost **+0.19 pp** on fleet MAE; the soft version was neutral
(−0.02) and moved the 12+ bias by −15.2 MW. The plateau is therefore *fixable and worth ~0 on the
headline* — a trade to make deliberately if per-farm error is the claim, not a gain.

### Also removed

`farm_scalers.FarmBoostScaler` (up-weight the wind loss at the 172 farm cells — it would have moved
the claim from "converts better" to "knows where the farms are", and was never run to a scored
result), `trunk_lr.SplitLRAdamW` (a continuous knob between the frozen-decoder and full-LR
endpoints — superseded, because matching the rollout recovered the wind outright: +0.54 → −0.05),
and `farm_metrics.FarmMaskedMAE` (a validation metric masked to cells with a real target — the
right fix for the ~99.8%-fabricated-zeros problem, but no config ever used it; `rank_checkpoints.py`
does the selection from forecasts instead). All recoverable from git if a future run needs them.

---

## Open / to verify before publication

1. **Which checkpoint was scored, and by what rule.** Must be the same rule for every run in the
   table. Do not select by `verify_power` output — that is selection on the test window.
2. **Did `nan_mask_weights` reach `validation_metrics`?** Without it the metric is ~99.8%
   fabricated zeros and best-validation selection is actively harmful (measured: the
   best-validation checkpoint of the pre-rollout Mixed run was 0.71 pp *worse* on power).
   Check the run's `anemoi-inference metadata --dump`.
3. **Paired bootstrap** over the 2916 inits for the −0.31 margin. Nothing else outstanding
   depends on compute.
4. **Seed / reproducibility** — one training run per configuration. A second seed converts the
   headline from "measured" to "reproduced".
5. **Non-wind weather variables unmeasured.** The weather term zeroed everything but wind;
   domain-wide RMSE for z_500, t_850, q_850, msl vs RegularWeather is needed before claiming the
   LAM remains a general weather model.
6. **Per-farm plateau term.** 12+ bias correlates −0.88 (R² 0.77) with each farm's true plateau:
   the head emits ~one plateau (~93%) for farms whose real plateaus span 89.7–97.0%. It largely
   *cancels* at fleet level (capacity-weighted mean bias −1.75 %cap vs per-farm RMS 5.48 %cap),
   so it costs the headline little but dominates per-farm error. Two attempts failed; the
   static-passthrough one is confounded and worth one retest.
7. **Intermediate ladder runs are scored on the 730-init subset**, only the headline is on 2916.
   Footnote it.
