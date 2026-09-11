#!/usr/bin/env python3
"""A private non-linear head for `capacityfactor`, bolted onto the graph-transformer decoder.

Anemoi's GraphTransformerBackwardMapper ends in

    node_data_extractor = LayerNorm(512) -> Linear(512, n_out)

so every output variable is a LINEAR readout of the same 512-dim latent. `capacityfactor` owns
one row of that Linear -- 513 parameters -- and, being a diagnostic, gets no residual skip, so its
whole value must come out of that projection. This adds a small MLP that reads the same latent and
writes ONLY into the capacityfactor channel, as a residual branch whose last layer is
zero-initialised: at step 0 the branch outputs exactly zero, so the model is bit-identical to the
checkpoint it warm-starts from and fine-tuning can only move away from it deliberately. The branch
is pointwise, so it is safe under grid sharding, and it lives inside `decoder`, so
`submodules_to_freeze: [encoder, processor]` leaves it trainable.

MEASURED (Huber_Head1 vs Huber_Finetune2, 730 inits, BE total): fleet MAE 7.80 -> 7.67, and the
gain is where the design predicted capacity was missing -- above rated, 12+ MAE 11.29 -> 10.80 and
the ceiling -158 -> -138 MW. The wind is untouched (own measured curve 7.59 both), confirming that
66k private parameters do not reshape the shared trunk.

Wire it in by pointing the decoder at this class:

    model:
      decoder:
        _target_: cf_head.CFHeadGraphTransformerBackwardMapper
        cf_index: 54
        cf_hidden: 128
        # ... every existing key unchanged

`cf_index` is an integer rather than a name because the decoder is not handed data_indices. It is
the MODEL OUTPUT ordering, not the zarr ordering -- the two differ, and only the first is what the
decoder writes. Anemoi builds it as the dataset order filtered to prognostic+diagnostic and
reindexed (data_indices/collection.py). For this dataset: 67 variables, 12 forcings, 54 prognostic
and `capacityfactor` as the single diagnostic, so the output has 55 channels and capacityfactor is
the last of them -> 54. Adding or dropping any non-forcing variable MOVES THIS NUMBER. It is
asserted at construction against the channel count it implies, so a stale index fails loudly
instead of training into a weather channel.

The whole scheme assumes multistep_output == 1; with more output steps the decoder's channel axis
becomes (time vars) flattened and cf_index would have to be strided.

Loading is non-strict (anemoi's transfer_learning_loading and the weights-only path both use
strict=False), so the new parameters are simply absent from the checkpoint and keep their init.

Written against anemoi-models 0.11.2 / anemoi-training 0.8.1 (the `anemoi_old` env). There,
`post_process` is called ONCE PER CHUNK inside run_processor_chunk_edge_sharding, which always
passes keep_x_dst_sharded=True and gathers outside -- so the gather branch below is dead on that
path, and is kept only so the override stays faithful to the mixin it replaces. NEWER anemoi drops
the extra arguments and the gather entirely; re-check the override against the mixin if the env is
upgraded.

WHAT IS NOT HERE, AND WHY
-------------------------
Four additions to this head were trained and scored on the full test year, and all four came back
inside the ~0.07 pp run-to-run seed noise. They are gone from the code and recorded in
RUN_MixedRollout.md instead: the raw-forcing static passthrough, a second hidden layer, a
per-cell affine calibration on capacityfactor, and wind direction in the loss. Together they rule
out information, capacity and representation as the reason the head emits ~one plateau for farms
whose real plateaus span 89.7-97.4% of nameplate -- the fourth is decisive, because a free scale
and offset per cell needs no learning at all and still did not help. What is left is gradient
weight: above rated is ~5% of hours on 1-2 cells per farm, and a fleet MAE has almost no reason to
chase it. Do not re-add any of them to this file without a new reason.

BACKWARD COMPATIBILITY -- READ BEFORE EDITING THIS FILE
--------------------------------------------------------
An anemoi INFERENCE checkpoint is a pickled MODEL OBJECT, not a state dict. Unpickling restores
the instance's __dict__ and submodules and NEVER calls __init__. So every attribute this class
creates in __init__ exists on an old checkpoint only if it existed when that checkpoint was
written -- and every method body here runs against objects pickled by OLDER versions of this file.

Editing a method to use a new attribute therefore breaks inference for every checkpoint already
trained, with an AttributeError deep inside run_processor_chunk_edge_sharding. That happened once,
when the static passthrough landed and HuberCFHead1's checkpoints stopped loading.

The rule: reach for anything added after the first release through `getattr(self, name, default)`
or `hasattr`, and keep the old arithmetic reachable. `_cf_branch` below is where that lives.

The converse also holds, and is why the four dead experiments could be deleted safely: removing a
branch only affects checkpoints that HAVE the attribute it tested for. MixedRollout has neither
`static_index` nor `cf_cell_scale`, so this file computes exactly what it did when that checkpoint
was trained. The checkpoints from those four runs still load -- their extra parameters sit unused
in the pickle -- but they no longer reproduce their own forecasts. Re-score them from saved output,
not by re-running inference.

Pydantic validates `model.decoder` as a discriminated union tagged on `_target_`, and a custom
class is not one of the accepted tags, so this needs `config_validation: false` at the top of the
training config. That switch is the supported escape hatch (train.py picks UnvalidatedBaseSchema),
and it is also REQUIRED for a second reason: the model schemas set no `extra="allow"`, so pydantic
would silently strip `cf_index`/`cf_hidden` before hydra ever instantiated this class.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper

CF_HIDDEN = 128          # width of the private head; ~66k params vs the 513 capacityfactor owns
ZERO_INIT_OUTPUT = True  # start as an exact no-op so a warm start is unchanged at step 0
EXPECT_N_OUT = 55        # decoder output channels for this dataset; asserted against config drift


class CFHeadGraphTransformerBackwardMapper(GraphTransformerBackwardMapper):
    """GraphTransformerBackwardMapper with an extra private MLP on one output channel."""

    def __init__(self, *args, cf_index: int, cf_hidden: int = CF_HIDDEN, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.out_channels_dst == EXPECT_N_OUT, (
            f"decoder writes {self.out_channels_dst} channels, expected {EXPECT_N_OUT}. The "
            f"variable set changed, so cf_index={cf_index} is stale -- recompute it as the "
            f"position of capacityfactor in data_indices.model.output.name_to_index."
        )
        assert 0 <= cf_index < self.out_channels_dst, f"cf_index {cf_index} out of range"
        self.cf_index = int(cf_index)

        # The latent is normalised on its own rather than inside cf_head, so the branch starts
        # from the same scale the node_data_extractor sees.
        self.cf_norm = nn.LayerNorm(self.hidden_dim)
        self.cf_head = nn.Sequential(
            nn.Linear(self.hidden_dim, cf_hidden),
            nn.GELU(),
            nn.Linear(cf_hidden, 1),
        )
        if ZERO_INIT_OUTPUT:
            nn.init.zeros_(self.cf_head[-1].weight)
            nn.init.zeros_(self.cf_head[-1].bias)

    def _cf_branch(self, x_dst: torch.Tensor) -> torch.Tensor:
        """The private branch's contribution to the capacityfactor channel.

        Two shapes of this module exist in pickled checkpoints and both must work:
          NO cf_norm   pickled before the static passthrough; cf_head carries its own LayerNorm
                       and reads the latent alone. Reproduce the old arithmetic exactly.
          cf_norm      current; the latent is normalised before the branch reads it.
        """
        if not hasattr(self, "cf_norm"):
            return self.cf_head(x_dst).squeeze(-1)
        return self.cf_head(self.cf_norm(x_dst)).squeeze(-1)

    def post_process(
        self,
        x_dst: torch.Tensor,
        shapes_dst: list,
        model_comm_group: ProcessGroup | None = None,
        keep_x_dst_sharded: bool = False,
    ) -> torch.Tensor:
        """Mirror BackwardMapperPostProcessMixin.post_process, plus the private cf branch."""
        out = self.node_data_extractor(x_dst).clone()
        out[..., self.cf_index] += self._cf_branch(x_dst)
        if not keep_x_dst_sharded:
            out = gather_tensor(
                out, 0, change_channels_in_shape(shapes_dst, self.out_channels_dst), model_comm_group
            )
        return out
