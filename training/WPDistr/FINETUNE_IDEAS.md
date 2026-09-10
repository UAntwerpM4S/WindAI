# Improving the power head — options grounded in the AnemoiOLD source

Written against `VeryHighCapacityGT.yaml` and
`AnemoiOLD/lib/python3.11/site-packages/anemoi/` (the env you actually train in —
the newer `WindAI/Anemoi/` tree has different features, don't copy from it).

Current pipeline: 150k base steps → 7.5k rollout → 5k fine-tune (encoder+processor frozen).

---

## STATUS as of 2026-09-04 — what has actually been run

| item | status |
|---|---|
| **A1 Huber loss** | **DONE.** `delta: 0.25` is the keeper. `delta: 0.1` fixes the ceiling harder but wrecks the wind (see below). |
| **B2 `ws100` nonzero** | **DONE.** `ws100: 0.5`. This is what protects the shared decoder — with it the wind costs +0.41 pp vs `RegularWeather`; without it, +2.34 pp. |
| **B1 rollout fine-tuning** | **DO NOT DO.** See the reversal note below. |
| A2 LeakyHardtanhBounding | untried |
| A3 `accum_grad_batches` | untried |
| A4 best-checkpoint selection | untried — still the cheapest thing on the list |
| A5 LR sweep | untried |
| B3 CombinedLoss split | **now newly relevant** — with `ws100: 0.5` the Huber loss is being applied to the WIND too. Splitting (Huber on power, MSE on weather) is no longer cosmetic. |
| B4 freeze granularity | untried |
| C1-C5 | untried |

Best configuration so far — **`Huber_Finetune2`**: `HuberLoss delta 0.25`, `ws100: 0.5`,
`capacityfactor: 100`, freeze `[encoder, processor]`, `rollout max 1`, 5500 steps.
729 common inits, sample-weighted MAE % of 2261 MW:

    direct head 7.80 | wind proxy 7.59 (RegularWeather 7.18) | head minus own curve +0.21
    Huber_Finetune (delta 0.1, no ws100 anchor): 8.21 | 9.52 | -1.31

CAUTION: `Huber_Finetune` appears to "beat its own measured curve" by 1.31 pp. That is bias
cancellation, not skill -- its wind is so degraded that the curve on that wind is bad too. When
the wind moves between runs the measured curve is NOT a fixed yardstick; compare ABSOLUTE scores.

### TWO REVERSALS -- read before using anything below

1. **B1 (rollout fine-tuning): the recommendation is now AGAINST.** The head's deficit against
   its own measured curve CLOSES with lead (+0.73 pp at 3h -> +0.10 at 33h, at parity from 21h),
   so there is no compounding error for rollout to fix. The base model was already rollout-trained
   (7.5k steps, 7 steps) and its 12+ MAE was 26.26% -- worse than the datasheet curve. Rollout
   training demonstrably does not fix the ceiling. It also costs memory (`_advance_input` does not
   detach) and reopens the wind risk that `ws100: 0.5` just closed.

2. **The "59% / 41%" diagnostic below is superseded.** It was computed without an un-fine-tuned
   control. With the base VH scored (267 inits): base 12.31 overall / 26.26 at 12+ / bias -545 MW,
   vs fine-tunes at 7.8-8.5 overall. FINE-TUNING IS WORTH -3.4 to -4.0 pp and is the single
   biggest lever found. An earlier hypothesis that the rollout=1 fine-tune UNDOES the base's
   rollout training is FALSIFIED -- it rescues a head that is unusable above rated.

---

## 0. Facts I verified first (so you can trust the rest)

| Thing | Value | Where |
|---|---|---|
| Effective LR | **6e-5, not 3e-5** | `base.py:230` multiplies `lr.rate` by `num_nodes × num_gpus_per_node / num_gpus_per_model` = ×2 |
| Effective batch | **2** | `batch_size.training: 1` × 2 GPUs, `accum_grad_batches: 1` |
| Fine-tune rollout | **`max: 1`** — one 3h step | `training.rollout` |
| `capacityfactor` | **diagnostic** — never fed back | only `output.prognostic` recycles, `rollout.py:95` |
| Weather output heads during fine-tune | **NOT protected -- the decoder trunk is SHARED** | `node_data_extractor` is one `Linear(512->55)` for every variable; only its 55 output rows are per-variable. Everything upstream (LayerNorm, `emb_nodes_dst`, the mapper block -- all 2,467,895 trainable params) is shared, so gradient from `capacityfactor` moves the weather too. MEASURED: the Huber fine-tune's specs-curve score is 3.78 pp worse than the MSE one, and the specs curve sees only `ws100`. |
| `capacityfactor` normaliser | **min-max** → values live in ~[0,1] | `data.normalizer.min-max` |
| Bounding | `HardtanhBounding[0,1]` — **gradient is identically zero outside the interval** | `bounding.py:202` |
| NaN masking | **correct**, per-batch, from that batch's input t=0 | `imputer.py:213-220` |
| `turbine_mask` graph attribute | **commented out**, and with two known-bad values | lines 441-443 |
| Losses the schema will accept | MSE, RMSE, MAE, Huber, LogCosh, WeightedMSE, CombinedLoss | `ImplementedLossesUsingBaseLossSchema` |
| `FilteringLossWrapper` | **NOT schema-allowed** — will fail validation | not in that enum |
| `LeadTimeDecayScaler` | **does not exist in this version** | newer anemoi only |
| SWA | available, currently `enabled: false` | `training.swa` |
| Checkpoint selection | **none — `monitor="step"`** | `callbacks/__init__.py:111`; no `save_top_k` on a metric, no EarlyStopping (`diagnostics.callbacks: []`) |
| `trainable_parameters.data` | **dead key — referenced nowhere in the codebase** | only `.hidden` is read, `base.py:66` |
| `self.node_attributes` | direct child of the model → **NOT frozen** by `['encoder','processor']` | `models/base.py:66` |

### The diagnostic that orders everything below

`VH_Finetune_5k` bias in the 12+ m/s bin:

```
 3h     6h     9h    12h    15h    18h    21h    24h    27h    30h    33h
-180   -146   -182   -235   -245   -272   -246   -262   -249   -303   -252
```

−180 MW is already there at **3h — one step, exactly the horizon it was trained on.**
So the ceiling splits: **~59% is a loss/target problem** (present at the trained horizon),
**~41% is a rollout problem** (accumulates with lead). Different fixes for each.

---

## Tier A — cheap, high expected value, run these first

### A1. Huber loss with a *small* delta ← attacks the 59%

MSE's minimiser is the conditional **mean**. Above rated, observed CF can only fall below the
plateau (curtailment, outages, availability) and never exceed it. So the conditional mean sits
below the plateau and a well-trained MSE model correctly predicts a number that is too low.
**That is the ceiling.** It's the same asymmetry that made your measured curve (a bin median) work.

The trap: `delta` is in **normalised** units and `capacityfactor` is min-max normalised to ~[0,1],
so **`delta: 1.0` makes Huber numerically identical to MSE.** You need it well below the typical
error magnitude.

```yaml
training_loss:
  _target_: anemoi.training.losses.HuberLoss
  delta: 0.1                    # try 0.05 / 0.1 / 0.2 — NOT 1.0
  scalers: [pressure_level, general_variable, node_weights, nan_mask_weights]
  ignore_nans: true
```

Also worth one run: `anemoi.training.losses.MAELoss` (pure median) and
`anemoi.training.losses.LogCoshLoss` (smooth, MAE-like in the tails, no delta to tune).

### A2. `LeakyHardtanhBounding` — one word

`F.hardtanh` is flat outside [0,1]; gradient is *identically zero* there. Any cell whose
pre-bounding activation leaves the interval is frozen out of learning for that sample. Your
failure mode is specifically at the top of the range, so rule this out:

```yaml
bounding:
  - _target_: anemoi.models.layers.bounding.LeakyHardtanhBounding
    variables: [capacityfactor]
    min_val: 0.0
    max_val: 1.0
```

Cheapest experiment on the list.

### A3. `accum_grad_batches` — your effective batch is 2

Batch size 1 on 2 GPUs, with a loss supported by 172 of 77,419 cells. Gradient variance must be
enormous. Accumulation costs no extra memory:

```yaml
accum_grad_batches: 8         # effective batch 16
```

Pair with a higher LR (below) — larger batch tolerates it.

### A4. You have no best-checkpoint selection — fix this before anything else

`monitor="step"` (`callbacks/__init__.py:111`). Checkpoints are saved every epoch and every 5
minutes, and **nothing ever selects the best one on a validation metric.** `diagnostics.callbacks`
is empty, so there is no EarlyStopping either. Whichever checkpoint you ran inference from was
picked by hand — almost certainly the last epoch.

This matters a lot here: a 3000-step fine-tune on a noisy 172-cell objective will bounce around,
and the last epoch is not the best epoch. **Some of the run-to-run variation you've been
attributing to hyperparameters may just be which epoch you happened to grab.**

You already log per-variable validation metrics — `training.metrics` includes `capacityfactor`, so
`_get_metric_ranges` (`loss.py:113`) gives it its own entry. Two options:

1. **Free, do this now:** open the mlflow run, plot the `capacityfactor` validation metric across
   epochs, and re-run inference from the best epoch rather than the last. This costs nothing and
   may move your numbers more than any config change below.
2. **Add EarlyStopping** (`anemoi.training.diagnostics.callbacks.stopping.EarlyStopping`) to
   `diagnostics.callbacks` with `monitor` set to the capacityfactor metric. The name is built as
   `f"{metric_name}_metric/{mkey}{suffix}"` → something like `val_mse_metric/capacityfactor/1`,
   but the docstring in `stopping.py` gives a slightly different pattern, so **read the exact
   string off mlflow rather than trusting either of us.**

### A5. Sweep the LR — it has never been swept

Only 3e-5 (→ 6e-5 effective) has ever run. You are training a small fraction of parameters, so
the optimum is likely **higher** than a full-model value. Try `rate:` 1e-5, 3e-5 (current),
1e-4, 3e-4. This is the single most likely source of free improvement and it costs four short runs.

Note `warmup: 200` out of `max_steps: 3000` — if you raise the LR, raise warmup to ~500.

---

## Tier B — worth running

### B1. Rollout fine-tuning ← attacks the other 41%

The head is trained on **one** 3h step and scored to 33h.

```yaml
rollout: {start: 1, epoch_increment: 1, max: 11}    # 11 steps = 33h
```


**Two things you must handle:**

1. **Anchor the weather heads.** They are ALREADY drifting at rollout=1 — the decoder trunk is
   shared, so the power gradient moves the weather whether you want it to or not (see the table
   above; it costs ~1 pp of measured-curve score between two fine-tunes). At rollout>1 they also
   gain an indirect path through the next step's input, so it gets worse. Give them a floor:
   ```yaml
   general_variable:
     weights: {default: 0, ..., ws100: 0.1, ws10: 0.05, capacityfactor: 100}
   ```
2. **Memory.** `_advance_input` does **not** detach, so activations scale ~linearly with rollout.
   Freezing stops the *updates*, not the backward pass through the frozen modules. Ramp with
   `epoch_increment` rather than starting at 11, and expect to reduce something else.

Note `loss *= 1.0/self.rollout` (`rollout.py:176`) — all steps are weighted **equally**. There is
no lead-time weighting scaler in this version, so you cannot preferentially weight long leads
without patching.

### B2. Let `ws100` have nonzero weight even at rollout=1

At rollout=1 a nonzero `ws100` weight does not change the power path directly, but it is NOT idle:
it pushes back on the shared decoder trunk, which the power loss is otherwise free to reshape.
This is now the MAIN reason to set it, not a nicety:

- it stops the shared decoder trunk drifting, which is MEASURED to cost ~1 pp of wind quality
- it lets you report the fine-tuned run's **wind** honestly. Right now you cannot: the fine-tune
  silently degrades it, so a fine-tuned run's ws100 is not the base model's ws100
- it gives you a co-training knob *inside* the fine-tune, which is a clean ablation axis for the
  paper: `ws100 ∈ {0, 0.1, 1, 10}` against `capacityfactor: 100`

I'd run `ws100: 0` (current) and `ws100: 1` as a pair. My expectation is no power difference at
rollout=1 and a real difference at rollout>1 — which is itself a publishable observation about
where the coupling actually lives.

### B3. Split the loss: Huber on power, MSE on weather

`CombinedLoss` is schema-allowed and supports **per-loss scalers**. This is the clean way to use
a median-ish loss on power without changing how weather is trained:

```yaml
scalers:
  power_only:
    _target_: anemoi.training.losses.scalers.VariableMaskingLossScaler
    variables: [capacityfactor]
    invert: true            # capacityfactor -> 1, everything else -> 0
  weather_only:
    _target_: anemoi.training.losses.scalers.VariableMaskingLossScaler
    variables: [capacityfactor]
    invert: false           # capacityfactor -> 0, everything else -> 1
  # keep general_variable, pressure_level, node_weights, nan_mask_weights as they are

training_loss:
  _target_: anemoi.training.losses.combined.CombinedLoss
  losses:
    - _target_: anemoi.training.losses.HuberLoss
      delta: 0.1
      scalers: [power_only, node_weights, nan_mask_weights]
    - _target_: anemoi.training.losses.MSELoss
      scalers: [weather_only, general_variable, pressure_level, node_weights, nan_mask_weights]
  loss_weights: [1.0, 0.1]
  scalers: []
  ignore_nans: true
```

`VariableMaskingLossScaler` is also just a tidier way to say "power only" than listing ten weather
variables at 0 — it won't silently break when an eleventh variable appears.

### B4. Freeze granularity

`freeze_submodule_by_name` (`checkpoint.py:131`) recurses through `named_children()` and freezes
**any** submodule matching that name at **any depth**. You're using the blunt setting. Options:

- `['encoder']` — let the processor adapt. More capacity for power, some cost to the shared
  representation. This is a direct advantages/disadvantages experiment for the paper.
- `['encoder', 'processor']` — current.
- `[]` with a very low LR — full-model fine-tune, closer to your rollout stage.

You can also freeze deeper names if the processor's children are individually named (check
`for n,_ in model.named_children()` on a loaded checkpoint to see the actual tree).

---

## Tier C — bigger changes, speculative

### C1. Trainable node parameters — currently all zero, and one key is dead

```yaml
trainable_parameters: {data: 0, hidden: 0, data2hidden: 0, hidden2data: 0, hidden2hidden: 0}
```

I checked this properly and it does not work the way the key names suggest:

- **`trainable_parameters.data` is never read.** `grep` finds no reference anywhere in
  `anemoi/models` or `anemoi/training`. Setting it does nothing.
- `models/base.py:66` passes only `trainable_parameters.**hidden**` into `NamedNodesAttributes`,
  which then loops over **every** node group and gives them all the same trainable width
  (`graph.py`, `register_tensor` inside `for nodes_name, nodes in graph_data.node_items()`).
  So **`hidden: 8` gives 8 learnable channels to the data nodes *and* the hidden nodes.**
- Those tensors live in `self.node_attributes`, a **direct child of the model**, so they are
  **not** frozen by `submodules_to_freeze: ['encoder','processor']`.

That combination is genuinely interesting: per-data-node embeddings would let the model learn a
per-cell correction — the closest thing in this architecture to "learn each farm's own curve" —
and they'd stay trainable through your fine-tune.

**But you cannot bolt it on at fine-tune time.** Adding trainable width changes `attr_ndims`,
which changes the encoder's `in_channels_src`, so the checkpoint's encoder input layer no longer
matches. `transfer_learning: true` handles that by **skipping mismatched parameters**
(`checkpoint.py:96-99`) — i.e. randomly re-initialising that layer. If you then freeze the
encoder you would be freezing random weights. So this has to be introduced at the **150k base
stage**, not as a fine-tune tweak.

Same reasoning applies to `hidden2data`, which feeds the **decoder's** `trainable_size` — the
decoder *is* trainable in your fine-tune, but the shape change still breaks checkpoint loading.

### C2. Feed `turbmask` / `capacity` in as node attributes

```yaml
model:
  attributes:
    nodes: []          # currently empty
```

`turbmask`, `capacity` and `turbinecount` are already **forcing** variables, so the model does see
them per-cell. Adding them as graph node attributes as well is probably redundant — listed for
completeness, low priority.

### C3. Fix and use the node reweighting

`turbine_mask` is commented out in this config **and has two bugs in the commented text**:
`anemoi.graphs.nodes.attributes.masks.NonmissingAnemoiDatasetVariable` (drop `masks.`) and
`variable: turbinemask` (should be `turbmask`). Working version, from `VHCapacity_025`:

```yaml
turbine_mask:
  _target_: anemoi.graphs.nodes.attributes.NonmissingAnemoiDatasetVariable
  variable: turbmask
```

**But I don't expect this to help the fine-tune much.** With a power-*only* loss the grid dilution
is a uniform constant on the whole loss, and AdamW normalises per-parameter by √v, so a global
loss scale is largely absorbed. It matters for the **co-trained** runs, where it changes power's
weight *relative* to weather. Low priority here, high priority if you go back to co-training.

### C4. SWA

`swa: {enabled: false, lr: 1e-4}`. For a short fine-tune on a noisy objective, weight averaging
over the last epochs is a cheap variance reducer. One-line experiment, unknown payoff.

### C5. `bf16-mixed` instead of `16-mixed`

Your loss is tiny (172/77419 dilution). fp16 gradients can underflow; Lightning's GradScaler
usually handles it, but if your GPUs are Ampere or newer, `precision: bf16-mixed` removes the
question entirely and costs nothing.

---

## Tier D — do NOT bother (and why), so you don't spend time on it

| Idea | Why not |
|---|---|
| `NaNMaskScaler(norm: ...)` to fix dilution | `norm` normalises over the **whole** mask tensor, not per variable. It's a global scale, not a differential boost for `capacityfactor`. |
| `FilteringLossWrapper` | Exists in the code but is **not** in `ImplementedLossesUsingBaseLossSchema`. With `config_validation: True` it will fail. Use `CombinedLoss` + `VariableMaskingLossScaler` instead. |
| `LeadTimeDecayScaler` | Newer-anemoi only. Not in your env. |
| `WeightedMSELoss` | It's for diffusion models — takes an explicit `weights` tensor from the diffusion machinery. |
| `FractionBounding` | Bounds a variable as a fraction of another. Nothing sensible to make CF a fraction of. |
| `loss_gradient_scaling` | Schema comment says "Not yet tested." |
| IEC 61400-12-3 as a citation | It's *site calibration for terrain*, not farm power performance. Wrong standard. |
| `trainable_parameters.data: N` | Dead key. Nothing reads it. Use `.hidden`, which sets the width for **all** node groups. |
| Adding trainable params during the fine-tune | Changes `attr_ndims` → encoder input shape → checkpoint layer skipped and randomly re-initialised, then frozen. Do it at the base stage or not at all. |

---

## Suggested run order

Independent axes, so a fractional factorial gets you a long way:

**Before any new run — costs nothing:** pull the `capacityfactor` validation curve for the
existing `VeryHighCapacityGT` run out of mlflow and check whether the checkpoint you scored was
the best epoch or just the last one (A4). If it wasn't the best, re-run inference from the best
one first — your baseline may move before you change a single hyperparameter, and every
comparison below is measured against it.

**Night 1 (4 runs, all cheap fine-tunes):**
1. baseline (current config, best-epoch checkpoint)
2. `LeakyHardtanhBounding`
3. `HuberLoss delta=0.1`
4. `accum_grad_batches: 8` + `lr.rate: 1e-4`, `warmup: 500`

**Night 2:** take whatever wins, then LR sweep around it (1e-5 / 1e-4 / 3e-4).

**Night 3:** `rollout max: 11` with `ws100: 0.1` anchoring, and the `ws100 ∈ {0, 1}` pair.

**Then:** `trainable_parameters.data: 8`, and `submodules_to_freeze: ['encoder']`.

Score everything with `verify_power.py`, `PER_FARM=False`, `BINNING=regimes`, and watch the
**12+ bias at 3h vs 30h separately** — that's the number that tells you which of the two
mechanisms you actually moved.
