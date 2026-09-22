#!/usr/bin/env python3
"""What would config_validation: true have done to this YAML?

Our configs run with config_validation: false (custom model_task / decoder / loss can't pass the
pydantic unions). The original VeryHighCapacity 150k and 7.5k stages ran with it TRUE, and
validation silently drops keys its schemas don't declare -- training.optimizer.betas among them,
so those runs used AdamW's default (0.9, 0.999). This runs the same BaseSchema validation on a
YAML and prints every key it would DROP or CHANGE, i.e. every place the unvalidated run behaves
differently from a validated one.

Not reported, because they change nothing: keys that only feed ${...} interpolations (validation
drops the source, the resolved copy survives), defaults validation fills in (the schemas mirror
the class defaults), str -> Path. Custom targets are swapped for stock ones before validating
(they would fail the unions) and listed as swapped.

    python check_validation.py path/to/config.yaml
"""

from __future__ import annotations

import re
import sys

from omegaconf import OmegaConf

from anemoi.training.schemas.base_schema import BaseSchema

STOCK = {
    "rollout_tasks.BackwardWindowForecaster": "anemoi.training.train.tasks.GraphForecaster",
    "rollout_tasks.EarlyStepForecaster": "anemoi.training.train.tasks.GraphForecaster",
    "cf_head.CFHeadGraphTransformerBackwardMapper": "anemoi.models.layers.mapper.GraphTransformerBackwardMapper",
    "combined_loss.ScalerAwareCombinedLoss": "anemoi.training.losses.CombinedLoss",
}
CUSTOM_ONLY = ("model.decoder.cf_index", "model.decoder.cf_hidden")   # args of the custom decoder
IGNORE = ("system.output.", "diagnostics.log.mlflow.save_dir")       # rewritten by expand_paths


def flat(d, prefix=""):
    if isinstance(d, (dict, list, tuple)):
        items = d.items() if isinstance(d, dict) else enumerate(d)
        out = {}
        for k, v in items:
            out.update(flat(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


text = open(sys.argv[1]).read()
sources = tuple(f"{s}." for s in re.findall(r"\$\{([\w.]+)\}", text)) + \
          tuple(re.findall(r"\$\{([\w.]+)\}", text))
raw = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)
raw.pop("defaults", None)                                             # consumed by hydra

swapped = []
for path, v in flat(raw).items():
    if isinstance(v, str) and v in STOCK:
        node = raw
        *parents, leaf = path.split(".")
        for p in parents:
            node = node[int(p)] if isinstance(node, list) else node[p]
        node[leaf] = STOCK[v]
        swapped.append(f"{path}: {v}")
for path in CUSTOM_ONLY:
    sec, sub, key = path.split(".")
    raw.get(sec, {}).get(sub, {}).pop(key, None)

val = flat(OmegaConf.to_container(OmegaConf.create(                  # as train.py: from a DictConfig
    BaseSchema(**OmegaConf.create(raw)).model_dump(by_alias=True))))
src = flat(raw)


def same(a, b):
    return a == b or str(a) == str(b)


dropped = [k for k in src if k not in val and not k.startswith(IGNORE)
           and not (k.startswith(sources) or k in sources)]
changed = [k for k in src if k in val and not same(src[k], val[k]) and not k.startswith(IGNORE)]

print(f"swapped for validation ({len(swapped)}):")
for s in swapped:
    print(f"   {s}")
print(f"\nDROPPED by validation ({len(dropped)}) -- the unvalidated run USES these, a validated one did not:")
for k in dropped:
    print(f"   {k} = {src[k]}")
print(f"\nCHANGED by validation ({len(changed)}):")
for k in changed:
    print(f"   {k}: yaml {src[k]!r} -> validated {val[k]!r}")
