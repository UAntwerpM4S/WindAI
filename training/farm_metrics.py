#!/usr/bin/env python3
"""A validation metric for `capacityfactor` that is actually computed on the farm cells.

WHY THE STOCK METRIC IS UNUSABLE FOR THIS TARGET
------------------------------------------------
`capacityfactor` is NaN at every cell without turbines (~72,496 of 77,419) and the ConstantImputer
fills those with 0.0. Training handles it: the imputer builds `loss_mask_training` by pairing
`index_training_input` with `index_inference_output` (imputer.py:220), capacityfactor has both, so
`nan_mask_weights` zeroes the fabricated cells and the training loss is clean.

VALIDATION IS NOT. `calculate_val_metrics` scores in POST-PROCESSED space, and the inverse imputer
pairs `index_inference_input` with the output index instead (imputer.py:260). capacityfactor is a
DIAGNOSTIC, so it has no inference-input index, the pair is skipped, and its NaNs are never put
back -- the file even says so at imputer.py:93-98: "if the variable is not in inference input
(diagnostic variable), we cannot place NaNs in its inference output ... this needs to be handled
by postprocessors". So the stock metric averages ~99.8% fabricated zeros.

Measured symptom in MixedRollout's own logs: `val_huber_metric/capacityfactor` moves from 0.06420
at +3h to 0.06399 at +18h -- a 0.3% spread across six lead times, where a real wind-power forecast
degrades by tens of percent. It is measuring a constant. It also ranked step 499 as the best
checkpoint while step 4999 was the one that scored 6.94 on the test year.

Adding `nan_mask_weights` does NOT fix it: NaNMaskScaler declares TensorDim.VARIABLE among its
scale_dims, and base.py:638 rejects any variable-dimension scaler on a validation metric outright
("Validation metrics cannot be scaled over the variable dimension in the post processed space").

WHAT THIS DOES INSTEAD
----------------------
Mask inside `calculate_difference`, where no scaler machinery is involved. BaseLoss reduces the
GRID dimension with a SUM, not a mean -- normalisation is assumed to come from node_weights
(base.py:174) -- so each channel is divided by its own count here and the sum then IS the mean:

  capacityfactor  NaN outside the farm cells, divided by the number of farm cells
                  -> nansum over grid = MEAN ABSOLUTE ERROR OVER FARM CELLS, in capacity-factor
                     units. Multiply by 100 to compare with the %-of-capacity numbers in
                     verify_power.py.
  every other     divided by the full grid count -> plain MAE over all cells, physical units.
  channel

Use it with `scalers: []`. With no scalers, BaseLoss.scale returns `x[subset_indices]`
(base.py:106) -- variable subsetting still happens, so the per-group logging is unaffected -- and
the variable-dimension check cannot trip. It also means node_weights is NOT applied, which is
what makes the number above a plain mean rather than an area-weighted sum.

    training:
      validation_metrics:
        farm:
          _target_: farm_metrics.FarmMaskedMAE
          cf_index: 54
          scalers: []
          ignore_nans: true

and read `val_farmmaskedmae_metric/capacityfactor/{1..R}` for the power and
`val_farmmaskedmae_metric/ws100/{1..R}` for the wind. Those two are the validation-time version of
the conversion/wind split that verify_power.py measures on the test year.

THE MASK IS `target > TOL`, AND WHAT THAT COSTS
-----------------------------------------------
The imputed value is exactly 0.0 and survives the min-max inverse as 0.0, so a threshold separates
fabricated cells from real ones. It also drops farm cells whose true output is genuinely zero --
below cut-in, curtailed, or out. That biases the level slightly low, but the mask depends ONLY on
the target, so it is identical for every checkpoint and every run: the RANKING, which is the whole
purpose here, is unaffected. Do not quote this number as a headline; quote verify_power.py.

`cf_index` is the MODEL OUTPUT index, the same convention as cf_head.py -- 55 output channels for
this dataset with capacityfactor last, so 54. It moves if the variable set changes.

NOT VALID UNDER GRID SHARDING. The farm-cell count is taken over the grid dimension of the tensor
this rank was handed. With `num_gpus_per_model: 1` that is the whole grid and the count is right.
Split the grid across GPUs and each shard would normalise by its own share of the farm cells,
which is wrong and would not fail loudly -- so `supports_sharding` is left False.
"""

from __future__ import annotations

import torch

from anemoi.training.losses.base import FunctionalLoss

CF_TOL = 1.0e-6          # target at or below this is an imputed cell, not a farm


class FarmMaskedMAE(FunctionalLoss):
    """MAE, with the capacityfactor channel restricted to cells that carry a real target."""

    name: str = "farmmaskedmae"
    supports_sharding: bool = False    # see the grid-sharding note in the module docstring

    def __init__(self, cf_index: int = 54, ignore_nans: bool = True, **kwargs) -> None:
        # ignore_nans must be on: the mask below is expressed as NaN, and it is the nan-aware
        # sum in BaseLoss.reduce that drops those cells rather than propagating them.
        super().__init__(ignore_nans=True)
        del ignore_nans, kwargs
        self.cf_index = int(cf_index)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Absolute error, per-channel normalised so the grid SUM downstream is a MEAN.

        pred/target are full width (bs, ensemble, grid, n_out) and post-processed; the variable
        subsetting happens afterwards in `scale`, so both the farm mask and the counts have to be
        formed here, before this method knows which group it will be reported under.
        """
        err = torch.abs(pred - target)
        n_grid = err.shape[-2]

        valid = target[..., self.cf_index] > CF_TOL                 # (bs, ensemble, grid)
        n_valid = valid.sum(-1, keepdim=True).clamp(min=1)          # (bs, ensemble, 1)
        cf = torch.where(valid, err[..., self.cf_index],
                         torch.full_like(err[..., self.cf_index], float("nan")))

        out = err.clone()
        out[..., self.cf_index] = cf * (n_grid / n_valid)           # /n_valid after the /n_grid
        return out / n_grid
