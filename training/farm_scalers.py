#!/usr/bin/env python3
"""A grid scaler that makes the loss care about the wind WHERE THE FARMS ARE.

WHY
---
The error budget says the remaining error is wind, not conversion:

    bin      share   head MAE   contributes   vs the fitted curve
    0-4.5    15.5%      1.83       4.0%            +0.06
    4.5-8    29.8%      4.91      20.7%            +0.11
    8-12     32.0%     10.18      46.0%            -1.05
    12+      22.8%      9.12      29.4%            +0.14

8-12 m/s alone is 46% of the total, and it is the band the head already WINS -- so what is
left there is not conversion. And the head at 6.94 already beats a per-farm method-of-bins
curve fitted on its own wind (7.18), so the conversion headroom is largely spent.

Meanwhile the loss barely looks at the wind that produces the metric. `weather_variable` gives
ws100 a weight of 0.5, and `node_weights` is unit-sum over all 77,419 data nodes, so the 172
cells that carry a capacityfactor target receive 0.22% of the wind loss. capacityfactor gets a
weight of 100 at those cells; ws100 gets 0.5, diluted across a domain that the metric never
looks at.

This multiplies the grid weighting by

    1 + (BOOST - 1) * turbine_mask

so the farm cells count BOOST times more and everywhere else is unchanged. Scalers multiply, so
combined with `weather_variable` (0 for everything except ws10/ws100) the net effect is exactly
"ws10 and ws100, up-weighted at farm cells, nothing else touched".

WHY NOT GraphNodeAttributeScaler DIRECTLY
-----------------------------------------
It returns the attribute as-is. `turbine_mask` is 0/1, so it would ZERO the wind loss at every
non-farm cell -- and with rollout that is destructive, not merely narrow: the model feeds its own
output back as the next input, so letting the domain wind rot would poison every step after the
first. The offset of 1 is the whole point.

WHAT BOOST MEANS
----------------
Share of the wind loss landing on the 172 target cells, against 0.22% at BOOST = 1:

    BOOST     share
        1     0.2%
       10     2.2%
       50    10.0%
      100    18.2%
      500    52.6%

CAVEATS, BOTH REAL
------------------
It trades domain wind for farm wind. Report a domain-wide ws100 RMSE alongside, or the run cannot
be defended -- and if the rollout degrades, this is the first thing to suspect.

And it moves the claim. The gain will appear in the WIND term of the conversion/wind split, not
the conversion term, so the paper stops saying "the head converts better" and starts saying
"an integrated model knows where the farms are and can spend capacity there". That is a different
argument, arguably a stronger one for the approach, but it has to be made deliberately.

    graph:
      nodes:
        data:
          attributes:
            turbine_mask:
              _target_: anemoi.graphs.nodes.attributes.masks.NonmissingAnemoiDatasetVariable
              variable: turbmask          # the zarr's name -- NOT turbinemask

    training:
      scalers:
        farm_boost:
          _target_: farm_scalers.FarmBoostScaler
          nodes_name: ${graph.data}
          nodes_attribute_name: turbine_mask
          boost: 50
          norm: null
      training_loss:
        losses:
          - _target_: anemoi.training.losses.HuberLoss     # the WEATHER sub-loss only
            scalers: ['pressure_level', 'weather_variable', 'node_weights',
                      'nan_mask_weights', 'farm_boost']

Not in combined_loss.py because a scaler is not a loss and is instantiated from a different part
of the config; putting it there would make that module's name a lie.
"""

from __future__ import annotations

import torch

from anemoi.training.losses.scalers.node_attributes import GraphNodeAttributeScaler


class FarmBoostScaler(GraphNodeAttributeScaler):
    """Grid scaler of 1 everywhere and `boost` on the cells the mask selects."""

    def __init__(self, *args, boost: float = 50.0, **kwargs) -> None:
        # `boost` is named here rather than left in kwargs: the parent ends with `del kwargs`
        # and would swallow it silently, leaving a run that looks configured and is not.
        super().__init__(*args, **kwargs)
        self.boost = float(boost)

    def get_scaling_values(self) -> torch.Tensor:
        mask = super().get_scaling_values().to(torch.float32)
        n = int(mask.sum())
        share = self.boost * n / (mask.numel() - n + self.boost * n)
        print(f"[farm_scalers] boost {self.boost:g} on {n:,} of {mask.numel():,} nodes "
              f"-> {100*share:.1f}% of the grid-weighted loss (was "
              f"{100*n/mask.numel():.2f}%)", flush=True)
        return 1.0 + (self.boost - 1.0) * mask
