#!/usr/bin/env python3
"""A private non-linear head for `capacityfactor`, bolted onto the graph-transformer decoder.

Anemoi's GraphTransformerBackwardMapper ends in

    node_data_extractor = LayerNorm(512) -> Linear(512, n_out)

so every output variable is a LINEAR readout of the same 512-dim latent. `capacityfactor` owns
one row of that Linear -- 513 parameters -- and, being a diagnostic, gets no residual skip, so its
whole value must come out of that projection. The clamp to [0,1] supplies the two flat segments
of a power curve for free; everything between cut-in and rated has to be linear in a latent that
is shared with 54 weather fields and built by a frozen encoder/processor. Measured consequence: at
short lead, where the wind forecast is nearly exact and the mapping error is therefore the whole
residual, the head loses ~1 pp of capacity to a binned measured curve fitted on its own wind.

This adds a small MLP that reads the same latent and writes ONLY into the capacityfactor channel,
as a residual branch whose last layer is zero-initialised. At step 0 the branch outputs exactly
zero, so the model is bit-identical to the checkpoint it warm-starts from and fine-tuning can only
move away from it deliberately. The branch is pointwise, so it is safe under grid sharding, and it
lives inside `decoder`, so `submodules_to_freeze: [encoder, processor]` leaves it trainable.

Wire it in by pointing the decoder at this class and giving it the output index of capacityfactor:

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
the last of them -> 54. Adding or dropping any non-forcing variable MOVES THIS NUMBER.

Two guards below: out_channels_dst must be 55 exactly as the index arithmetic assumes, and the
index must be in range. Both are asserted at construction so a stale index fails loudly instead of
training into a weather channel. Note the whole scheme assumes multistep_output == 1; with more
output steps the decoder's channel axis becomes (time vars) flattened and the index would have to
be strided.

Loading is non-strict (anemoi's transfer_learning_loading and the weights-only path both use
strict=False), so the new parameters are simply absent from the checkpoint and keep their init.

Written against anemoi-models 0.11.2 / anemoi-training 0.8.1 (the `anemoi_old` env). `post_process`
there comes from BackwardMapperPostProcessMixin and takes (x_dst, shapes_dst, model_comm_group,
keep_x_dst_sharded) and performs a shard gather; NEWER anemoi drops the extra arguments and the
gather, so the override below must be re-checked against the mixin if the env is upgraded.

Pydantic validates `model.decoder` as a discriminated union tagged on `_target_`, and a custom
class is not one of the accepted tags, so this needs `config_validation: false` at the top of the
training config. That switch is the supported escape hatch (train.py picks UnvalidatedBaseSchema).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper

CF_HIDDEN = 128          # width of the private head; 512->128->1 is ~66k params vs the 513 it has
ZERO_INIT_OUTPUT = True  # start as an exact no-op so a warm start is unchanged at step 0
EXPECT_N_OUT = 55        # output channels for this dataset; asserted so a config drift is caught


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
        self.cf_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, cf_hidden),
            nn.GELU(),
            nn.Linear(cf_hidden, 1),
        )
        if ZERO_INIT_OUTPUT:
            nn.init.zeros_(self.cf_head[-1].weight)
            nn.init.zeros_(self.cf_head[-1].bias)

    def post_process(
        self,
        x_dst: torch.Tensor,
        shapes_dst: list,
        model_comm_group: ProcessGroup | None = None,
        keep_x_dst_sharded: bool = False,
    ) -> torch.Tensor:
        """Mirror BackwardMapperPostProcessMixin.post_process, plus the private cf branch.

        The branch is applied to the still-SHARDED latent, before the gather: it is pointwise, so
        each rank can compute its own slice, and the result then rides the existing gather. The
        signature must match the mixin's exactly -- it is called positionally by the base mapper.
        """
        out = self.node_data_extractor(x_dst).clone()
        out[..., self.cf_index] += self.cf_head(x_dst).squeeze(-1)
        if not keep_x_dst_sharded:
            out = gather_tensor(
                out, 0, change_channels_in_shape(shapes_dst, self.out_channels_dst), model_comm_group
            )
        return out
