#!/usr/bin/env python3
"""The two loss pieces the power fine-tune needs that anemoi 0.8.1 does not provide.

  ScalerAwareCombinedLoss   a CombinedLoss that survives the per-batch scaler update
  PlateauWeightedMAE        an MAE that pays attention above rated wind

They live together because they are used together -- the second is a sub-loss of the first -- and
because one PYTHONPATH module is one thing to keep in sync instead of two. ScalerAwareCombinedLoss
keeps this import path: every config written so far names `combined_loss.ScalerAwareCombinedLoss`,
and moving it would break reproducing any of them.


ScalerAwareCombinedLoss
=======================
`CombinedLoss.__init__` ends with `del self.scaler` ("not used here" -- it routes scalers to the
sub-losses instead). But `GraphForecaster.update_scalers` reads `if name in self.loss.scaler:`
unconditionally (train/tasks/base.py:319), so ANY updating scaler crashes the run on the first
batch with `AttributeError: 'CombinedLoss' object has no attribute 'scaler'`.

`nan_mask_weights` is exactly such a scaler -- NaNMaskScaler refreshes every batch -- and it
cannot be dropped: without it the imputed zeros at the ~72k cells that carry no capacityfactor
target are trained as if they were real observations.

The fix is to give the attribute back as a read-only membership test. That is all line 319 asks
of it, and the routing it guards (`CombinedLoss.update_scaler`) already works: it forwards to
whichever sub-losses named that scaler. Nothing else in the training code touches `loss.scaler`.


PlateauWeightedMAE
==================
Four experiments have failed to make the head produce per-farm plateaus, each inside the ~0.07 pp
run-to-run noise:
    STATIC_INDEX = [129,130,131]   capacity handed straight to the branch       information
    cf_layers = 2                  a second hidden layer                        capacity
    wdir100_cos/sin in the loss    direction preserved in the latent            representation
    CF_CELL_AFFINE                 a FREE scale and offset per cell             none needed
The fourth settles it. That parameter needed no learning -- a free number per cell, wired straight
to the output, initialised at the identity. It moved, and the bin table says where it went: the
output was pulled DOWN everywhere, erasing the 8-12 over-prediction (+21.8 -> -0.1 MW) and
DEEPENING the 12+ under-prediction (-47.7 -> -74.3). The plateau got 0.27 pp WORSE while the fleet
number stayed flat.

So the plateau is not something the model cannot express. It is something the OBJECTIVE does not
want. Above rated is a fraction of the 22% of hours in the 12+ bin, each farm is 1-2 of the 172
target cells, and a fleet MAE will always rather spend a parameter on the 8-12 band where 32% of
the samples are.

This reweights the per-element MAE on capacityfactor by the TARGET's own level:

    w = 1 + (PLATEAU_BOOST - 1) * sigmoid((target - PLATEAU_FROM) / PLATEAU_SOFT)

so hours near a farm's ceiling count more and everything below rated is untouched (w ~ 1.0 below
0.7). It is applied inside `calculate_difference`, the only place a loss sees the target.

A SCALER CANNOT DO THIS. `BaseUpdatingScaler.on_batch_start(model)` is handed the model and
nothing else; NaNMaskScaler works only because the imputer leaves `loss_mask_training` on itself
as a side effect of pre-processing. No hook carries the target, so a regime-conditional weight has
to live in the loss.

PLATEAU_RENORM divides by the mean weight over cells that carry a target, so the power term keeps
its total magnitude and only its DISTRIBUTION across wind regimes changes. Without it, boosting
would also raise power relative to weather in the CombinedLoss and two things would move at once.

WEIGHTING BY THE TARGET IS A TRAINING CHOICE, NOT LEAKAGE. The weight computes gradients; it never
enters a prediction and never touches the scored year -- the same kind of decision as choosing MAE
over MSE. It does need stating in the methods.

AND IT IS A TRADE. Fleet MAE is the headline number and this deliberately spends some of it to buy
plateau accuracy. Worth it only if 12+ improves by more than 8-12 degrades; with 12+ at 22% of
hours and a -47 MW bias there the room exists, but measure both bins before believing it.

    training:
      training_loss:
        _target_: combined_loss.ScalerAwareCombinedLoss
        loss_weights: [1, 1]
        losses:
          - _target_: anemoi.training.losses.HuberLoss
            delta: 1.0
            scalers: ['pressure_level', 'weather_variable', 'node_weights', 'nan_mask_weights']
            ignore_nans: true
          - _target_: combined_loss.PlateauWeightedMAE        # was anemoi...MAELoss
            cf_index: 54
            scalers: ['power_variable', 'node_weights', 'nan_mask_weights']
            ignore_nans: true

`cf_index` is the MODEL OUTPUT index, the same convention as cf_head.py -- 54 for this dataset.
"""

from __future__ import annotations

import torch

from anemoi.training.losses import CombinedLoss
from anemoi.training.losses.base import FunctionalLoss

# ---- PlateauWeightedMAE settings -------------------------------------
# capacityfactor is min-max normalised, so the target arrives on the same [0, 1] scale as a
# capacity factor. Real plateaus run 0.897 (Northwester2) to 0.974 (Nobelwind), so a threshold of
# 0.80 catches the approach to rated as well as the ceiling itself.
PLATEAU_FROM   = 0.90
PLATEAU_SOFT   = 0.05    # transition width; a sigmoid, not a step, so no gradient discontinuity
PLATEAU_BOOST  = 2.5     # weight at the ceiling relative to 1.0 well below it
PLATEAU_RENORM = True    # hold the power term's total magnitude fixed; only redistribute it
# ----------------------------------------------------------------------


class _SubLossScalerNames:
    """Membership over the union of the sub-losses' scaler names."""

    def __init__(self, owner: "ScalerAwareCombinedLoss") -> None:
        self._owner = owner

    def __contains__(self, name: str) -> bool:
        return any(name in loss.scaler for loss in self._owner.losses)


class ScalerAwareCombinedLoss(CombinedLoss):
    """CombinedLoss with `self.scaler` restored as a membership test."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)      # deletes self.scaler
        self.scaler = _SubLossScalerNames(self)


class PlateauWeightedMAE(FunctionalLoss):
    """MAE with the capacityfactor channel up-weighted where the TARGET is near a plateau."""

    name: str = "plateauweightedmae"

    def __init__(self, cf_index: int = 54, ignore_nans: bool = True, **kwargs) -> None:
        super().__init__(ignore_nans=ignore_nans)
        del kwargs
        self.cf_index = int(cf_index)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Absolute error, with capacityfactor reweighted by how close the target is to rated.

        pred/target are full width (bs, ensemble, grid, n_out) in NORMALISED space -- this runs
        inside the training loss, before any post-processing, so the [0, 1] range is the min-max
        normalised capacity factor.
        """
        err = torch.abs(pred - target)
        cf = target[..., self.cf_index]

        w = 1.0 + (PLATEAU_BOOST - 1.0) * torch.sigmoid((cf - PLATEAU_FROM) / PLATEAU_SOFT)

        if PLATEAU_RENORM:
            # mean over cells that carry a real target. The imputed cells sit at exactly 0.0 and
            # would drag the mean toward 1, inflating the boost by a factor of ~450.
            real = cf > 1.0e-6
            denom = torch.where(real, w, torch.zeros_like(w)).sum(-1, keepdim=True)
            count = real.sum(-1, keepdim=True).clamp(min=1)
            w = w * (count / denom.clamp(min=1.0e-6))

        out = err.clone()
        out[..., self.cf_index] = err[..., self.cf_index] * w
        return out