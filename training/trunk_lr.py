#!/usr/bin/env python3
"""AdamW that gives the private cf head and the shared trunk different learning rates.

WHY. Freezing the decoder leaves the wind untouched (+0.05) but destroys the conversion (+1.15);
letting it train at full rate wins the conversion (-0.27) but costs 0.53 of wind. Those are the
two endpoints of one knob and the interior has never been looked at, even though the entire
remaining deficit to the 7.19 benchmark is the wind term. This makes the knob continuous:
`trunk_lr_scale` 0.0 approximates the frozen run, 1.0 reproduces Mixed.

WHY NOT A GRADIENT HOOK. Adam is scale-invariant -- scaling g by s scales m by s and sqrt(v) by s,
so the step m/sqrt(v) is unchanged. Only a separate param group with its own lr actually slows a
subset of the model down.

HOW THE SPLIT IS MADE. anemoi builds the optimiser as
`instantiate(opt_cfg, params=filter(requires_grad, self.parameters()), lr=self.lr)`
(train/tasks/base.py:736-744) -- one flat iterator of tensors, no names. So cf_head marks its own
parameters with `_cf_head_param = True` at construction (an attribute on the Parameter object,
which survives DDP wrapping since the objects are the same), and this class splits on that tag.

THE SCHEDULER STILL WORKS. timm's CosineLRScheduler stores `base_values` per param group from each
group's own initial lr and writes one value back per group, so the ratio between the two groups is
preserved across the whole cosine schedule rather than being flattened at the first step. Verify it
on the startup print below the first time you run this.

Wire it in with:

    training:
      optimizer:
        _target_: trunk_lr.SplitLRAdamW
        betas: [0.9, 0.95]
        trunk_lr_scale: 0.1
"""

from __future__ import annotations

import torch


class SplitLRAdamW(torch.optim.AdamW):
    """AdamW with a reduced learning rate for every parameter outside the private cf head."""

    def __init__(self, params, lr: float, trunk_lr_scale: float = 1.0, **kwargs) -> None:
        params = list(params)
        head = [p for p in params if getattr(p, "_cf_head_param", False)]
        trunk = [p for p in params if not getattr(p, "_cf_head_param", False)]

        assert head, (
            "no parameter carries _cf_head_param -- the decoder is not cf_head."
            "CFHeadGraphTransformerBackwardMapper, or FREEZE_ALL_BUT_HEAD dropped the head from "
            "the optimiser. SplitLRAdamW would silently become a plain AdamW."
        )

        groups = [{"params": head, "lr": lr}]
        if trunk:                       # empty under FREEZE_ALL_BUT_HEAD
            groups.append({"params": trunk, "lr": lr * trunk_lr_scale})

        print(f"[trunk_lr] head {sum(p.numel() for p in head):,} params @ lr {lr:.3g} | "
              f"trunk {sum(p.numel() for p in trunk):,} params @ lr {lr * trunk_lr_scale:.3g} "
              f"(scale {trunk_lr_scale})", flush=True)

        super().__init__(groups, lr=lr, **kwargs)
