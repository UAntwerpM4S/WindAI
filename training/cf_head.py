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

STATIC PASSTHROUGH -- why the head also reads raw input channels
----------------------------------------------------------------
What is left above rated is partly an IDENTIFIABILITY problem, measured per farm: regressing each
farm's 12+ bias on its real plateau (the measured curve's top, 89.7% of nameplate at Northwester2
to 97.0% at Nobelwind) gives corr -0.69. The head over-predicts the low-plateau farms and
under-predicts the high-plateau ones -- it is emitting something close to one average plateau for
everybody, because it cannot tell the farms apart.

The information to tell them apart IS in the model input -- `capacity`, `turbinecount` and
`turbmask` are per-cell forcings and every farm has a distinct capacity -- but it need not survive
into the 512-dim latent, because the encoder and processor are FROZEN and were trained when power
carried almost no loss weight. They had no reason to preserve it.

So the branch is given those channels directly, bypassing the frozen trunk. `pre_process` receives
x_dst as the RAW assembled data-node features (see the layout note on STATIC_INDEX) before
`emb_nodes_dst` embeds them, so the slice is stashed there and consumed by `post_process` for the
same chunk. They are detached: these are inputs, and nothing should learn to change them.

The latent is LayerNorm-ed on its own BEFORE the static channels are concatenated. Normalising the
concatenation instead would spread one LayerNorm over 512 latent dimensions plus 3 static ones and
bury the static signal, which is the entire point of adding it.

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
the last of them -> 54. Adding or dropping any non-forcing variable MOVES THIS NUMBER, as it moves
STATIC_INDEX below. Both are asserted at construction against the channel counts they imply, so a
stale index fails loudly instead of training into a weather channel.

The whole scheme assumes multistep_output == 1; with more output steps the decoder's channel axis
becomes (time vars) flattened and cf_index would have to be strided.

Loading is non-strict (anemoi's transfer_learning_loading and the weights-only path both use
strict=False), so the new parameters are simply absent from the checkpoint and keep their init.

Written against anemoi-models 0.11.2 / anemoi-training 0.8.1 (the `anemoi_old` env). There,
`pre_process`/`post_process` are called ONCE PER CHUNK inside run_processor_chunk_edge_sharding,
which always passes keep_x_dst_sharded=True and gathers outside -- so the gather branch below is
dead on that path, and is kept only so the override stays faithful to the mixin it replaces.
NEWER anemoi drops the extra arguments and the gather entirely; re-check both overrides against
the mixin if the env is upgraded.

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

# THE TRADE-OFF THIS FILE EXISTS TO NAVIGATE
# ------------------------------------------
# Split the fleet MAE into CONVERSION (direct - this model's own measured curve, same wind both
# ways) and WIND (that curve - RegularWeather's curve, same conversion both ways). Measured:
#     run                    wind   conversion   direct
#     decoder frozen        +0.05      +1.15      8.29
#     Mixed  (full LR)      +0.53      -0.27      7.45     vs the 7.19 benchmark
# Freeze the decoder and the wind is untouched but the conversion collapses; let it train freely
# and the conversion wins but the wind pays 0.53. Both endpoints are measured and nothing in
# between is. The whole remaining deficit is the wind term, so the interior is where to look.
#
# Two knobs for that interior:
#   FREEZE_ALL_BUT_HEAD   the hard endpoint -- decoder parameters get requires_grad=False, the
#                         optimiser drops them (train/tasks/base.py:738) and DDP excludes them
#                         from the reducer, so the weather output is bit-identical to the warm
#                         start. Reproduces the CFHeadFrozen runs.
#   the _cf_head_param    the soft version -- every private parameter is tagged below so
#   tag                   trunk_lr.SplitLRAdamW can put it in its own param group and give the
#                         trunk a fraction of the head's learning rate.
#
# A GRADIENT HOOK CANNOT DO THIS. Adam is scale-invariant: rescaling g uniformly rescales m and
# sqrt(v) together and leaves m/sqrt(v) -- the step -- unchanged. Slowing the trunk down has to be
# a separate param group with its own lr, which is why the tag exists instead of a hook.
FREEZE_ALL_BUT_HEAD = False

# Raw data-node channels handed to the branch alongside the latent. in_channels_dst is
# multistep_input * len(model.input) + node_attrs = 2 * 66 + 4 = 136, laid out as
#   [ t-1: 0..65 ][ t: 66..131 ][ lat/lon sin/cos: 132..135 ]
# so a model-input index i sits at 66+i for the most recent step. capacity, turbinecount and
# turbmask are model-input 63, 64, 65 -> 129, 130, 131. Set to [] to disable the passthrough.
#
# DISABLED after HuberCFHead4Static: handing the branch these channels changed the forecast by
# 0.01-0.03 pp, against the 0.13 pp that adding the branch at all bought. The head can be given
# farm identity and still does not produce per-farm plateaus, so the -0.69 plateau/bias
# correlation is not an information problem. Left in place, and re-enabled by restoring
# [129, 130, 131], because the negative result is worth being able to reproduce.
#
# Do NOT "revert" this file to the pre-passthrough version to turn the feature off: the
# HuberCFHead4Static checkpoints are pickled against the 515-wide layer and would fail to load.
# Empty here is the off switch, and it reproduces the v1 arithmetic exactly.
STATIC_INDEX = []
EXPECT_IN_DST = 136      # asserted, because STATIC_INDEX is meaningless if the layout changed


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

        self.static_index = list(STATIC_INDEX)
        if self.static_index:
            assert self.in_channels_dst == EXPECT_IN_DST, (
                f"decoder reads {self.in_channels_dst} data-node channels, expected "
                f"{EXPECT_IN_DST}. The input layout changed, so STATIC_INDEX is stale -- "
                f"recompute it as multistep_input*len(model.input) + model-input position."
            )
            assert max(self.static_index) < self.in_channels_dst, "STATIC_INDEX out of range"
        self._cf_static = None

        # the latent is normalised alone; the static channels are appended afterwards so one
        # LayerNorm over 512+3 dimensions cannot bury them
        self.cf_norm = nn.LayerNorm(self.hidden_dim)
        self.cf_head = nn.Sequential(
            nn.Linear(self.hidden_dim + len(self.static_index), cf_hidden),
            nn.GELU(),
            nn.Linear(cf_hidden, 1),
        )
        if ZERO_INIT_OUTPUT:
            nn.init.zeros_(self.cf_head[-1].weight)
            nn.init.zeros_(self.cf_head[-1].bias)

        # Tag the private parameters. trunk_lr.SplitLRAdamW reads this to build two param groups;
        # the tag is inert under a plain AdamW, so it is always safe to set.
        for p in list(self.cf_norm.parameters()) + list(self.cf_head.parameters()):
            p._cf_head_param = True

        if FREEZE_ALL_BUT_HEAD:
            # self is the decoder, so this covers the decoder only; encoder and processor are
            # frozen by submodules_to_freeze as before.
            kept = frozen = 0
            for name, param in self.named_parameters():
                if name.startswith(("cf_head.", "cf_norm.")):
                    kept += param.numel()
                else:
                    param.requires_grad_(False)
                    frozen += param.numel()
            print(f"[cf_head] trainable {kept:,} | frozen in decoder {frozen:,} "
                  f"-> the wind is bit-identical to the warm-start checkpoint", flush=True)

    def pre_process(self, x, shard_shapes, model_comm_group=None,
                    x_src_is_sharded=False, x_dst_is_sharded=False):
        """Stash this chunk's raw static channels, then embed as usual.

        x[1] here is the assembled data-node input, BEFORE emb_nodes_dst -- the only point at
        which the raw forcings are visible to the decoder. post_process runs on the same chunk
        immediately after, so a plain attribute is enough; under activation checkpointing the
        pair is recomputed together, so it stays consistent.

        getattr, not self.static_index: see the BACKWARD COMPATIBILITY note in the module
        docstring. An inference checkpoint pickled before the passthrough existed has no such
        attribute, and must keep running.
        """
        if getattr(self, "static_index", None):
            self._cf_static = x[1][..., self.static_index].detach()
        return super().pre_process(x, shard_shapes, model_comm_group,
                                   x_src_is_sharded, x_dst_is_sharded)

    def _cf_branch(self, x_dst: torch.Tensor) -> torch.Tensor:
        """The private branch's contribution to the capacityfactor channel.

        Two shapes of this module exist in pickled checkpoints and both must work:
          NO cf_norm   pickled before the static passthrough; cf_head carries its own LayerNorm
                       and reads the latent alone. Reproduce the old arithmetic exactly.
          cf_norm      current; the latent is normalised on its own and the raw static channels
                       are appended afterwards.
        """
        if not hasattr(self, "cf_norm"):
            return self.cf_head(x_dst).squeeze(-1)
        feat = self.cf_norm(x_dst)
        if getattr(self, "static_index", None):
            feat = torch.cat([feat, self._cf_static.to(feat.dtype)], dim=-1)
        return self.cf_head(feat).squeeze(-1)

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
