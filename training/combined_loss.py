#!/usr/bin/env python3
"""A CombinedLoss that survives the per-batch scaler update.

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

This keeps its import path: every config written so far names `combined_loss.ScalerAwareCombinedLoss`,
and moving it would break reproducing any of them.

    training:
      training_loss:
        _target_: combined_loss.ScalerAwareCombinedLoss
        loss_weights: [1, 1]
        losses:
          - _target_: anemoi.training.losses.HuberLoss
            delta: 1.0
            scalers: ['pressure_level', 'weather_variable', 'node_weights', 'nan_mask_weights']
            ignore_nans: true
          - _target_: anemoi.training.losses.MAELoss
            scalers: ['power_variable', 'node_weights', 'nan_mask_weights']
            ignore_nans: true

A plateau-weighted MAE also lived here, up-weighting the power term where the TARGET was near
rated. It was trained and scored: the hard version cost 0.19 pp on fleet MAE, the soft version was
neutral (-0.02) and moved the 12+ bias by -15.2 MW. So the plateau is reachable by reweighting and
worth ~0 on the headline -- a trade, not a gain. Removed; see RUN_MixedRollout.md.
"""

from __future__ import annotations

from anemoi.training.losses import CombinedLoss


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
