#!/usr/bin/env python3
"""Two GraphForecaster variants aimed at the short leads, where the head still loses.

MEASURED (test year, 2909 inits, verify_power): unfreeze_proc 6.40 vs a CERRA transformer trained
on the same MAE loss 6.46 -- a tie on average, but the lead profile crosses. The converter wins at
+3h/+6h (5.08 vs 5.68, 5.45 vs 5.56), the head wins beyond ~18h (+33h: 7.24 vs 7.53). Both classes
below go after the short leads, one mechanism each.

Wire either in with one config line; nothing else changes:

    training:
      model_task: rollout_tasks.EarlyStepForecaster        # or rollout_tasks.BackwardWindowForecaster

The task class is NOT part of an inference checkpoint (checkpoint.py saves pl_module.model only),
so forecasts are produced exactly as before. BackwardWindowForecaster does change what the power
output MEANS, and scoring has to be told -- see its docstring.

Written against anemoi-training 0.8.1 (AnemoiOLD): rollout.py `_step` and forecaster.py
`_rollout_step`. Neither class adds parameters, so the weights-only warm start
(`load_from_checkpoint(..., strict=False)`) loads them exactly as it loads GraphForecaster.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks import GraphForecaster

LOGGER = logging.getLogger(__name__)

# ---- EarlyStepForecaster ---------------------------------------------------------------------
# Loss weight per rollout step, 0-based; steps beyond the tuple get 1.0. Normalised by the sum of
# the weights actually used, so the loss keeps its scale and at rollout 1 it is identical to the
# stock mean. (2.0, 1.5) at R=6 gives step 1 2/7.5 = 27% of the gradient instead of 1/6 = 17%.
STEP_WEIGHTS = (2.0, 1.5)

# ---- BackwardWindowForecaster ----------------------------------------------------------------
CF_NAME = "capacityfactor"
OBS_STEP_H = 3          # the observation window, which is also the model step


class EarlyStepForecaster(GraphForecaster):
    """GraphForecaster whose rollout loss up-weights the first steps.

    Stock anemoi averages the rollout steps with EQUAL weight (rollout.py: `loss *= 1/rollout`).
    Step 1 is then 1/6 of the gradient at R=6, and it is the only step whose inputs are analysis
    states rather than the model's own forecast -- so it is both under-weighted and out of
    distribution for a head trained mostly on rolled-out latents. That is the lead where the head
    loses to the converter.

    The whole step loss is re-weighted (wind and power together), not just capacityfactor:
    anemoi computes one scalar per step, and splitting it would mean re-implementing the combined
    loss here.
    """

    def _step(self, batch: torch.Tensor, validation_mode: bool = False) -> tuple[torch.Tensor, dict, list]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics, y_preds, wsum = {}, [], 0.0
        for k, (loss_next, metrics_next, y_preds_next) in enumerate(
            self._rollout_step(batch, rollout=self.rollout, validation_mode=validation_mode)
        ):
            w = STEP_WEIGHTS[k] if k < len(STEP_WEIGHTS) else 1.0
            loss = loss + w * loss_next
            wsum += w
            metrics.update(metrics_next)
            y_preds.append(y_preds_next)
        return loss / wsum, metrics, y_preds


class BackwardWindowForecaster(GraphForecaster):
    """GraphForecaster whose capacityfactor target is the window that ENDS at the valid time.

    In the data, capacityfactor at T is the mean observed power over [T, T+3h). The stock target
    therefore asks the head, reading the state at T, to predict the next 3 hours -- beyond anything
    the model has computed. The measured curve reads the wind at T AND T+3h, and the converter
    attends to every lead, so both see the end of that window and the head does not.

    Here the target at output step T is capacityfactor at T-3h, i.e. the mean over [T-3h, T): a
    window whose start the model had as input and whose end it has just predicted. Interpolation
    instead of extrapolation. The previous time step is always in the batch -- for the first output
    it is the last input step -- so no data changes. Every other variable keeps its target.

    SCORING MUST FOLLOW. A checkpoint trained this way emits, at valid time T, power for [T-3h, T).
    The observation window [vt, vt+3h) is therefore read from the output at vt+3h:
    `BACKWARD_WINDOW = True` in rank_checkpoints.py, and the run's label in `BACKWARD_WINDOW_RUNS`
    in verify_power.py. Scored without that, it is compared against the wrong window.
    """

    def _cf_positions(self) -> tuple[int, int]:
        """(position of capacityfactor in the output tensor y, its index in the batch)."""
        pos = getattr(self, "_bw_pos", None)
        if pos is None:
            data_idx = int(self.data_indices.data.output.name_to_index[CF_NAME])
            hits = (self.data_indices.data.output.full == data_idx).nonzero()
            assert hits.numel() == 1, f"{CF_NAME} (data index {data_idx}) not in the output variables"
            pos = self._bw_pos = (int(hits.item()), data_idx)
        return pos

    def _rollout_step(self, batch: torch.Tensor, rollout: int | None = None, validation_mode: bool = False):
        """forecaster.py `_rollout_step`, with capacityfactor's target moved back one step."""
        y_pos, cf_idx = self._cf_positions()
        x = batch[:, 0 : self.multi_step, ..., self.data_indices.data.input.full]
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            y_pred = self(x)

            # advanced indexing returns a copy, so writing into y never touches the batch
            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.data.output.full]
            y[..., y_pos] = batch[:, self.multi_step + rollout_step - 1, ..., cf_idx]

            loss, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=rollout_step,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            x = self._advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred
