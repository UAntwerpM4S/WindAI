#!/usr/bin/env python3
"""CombinedLoss that survives anemoi 0.8.1's per-batch scaler update.

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
