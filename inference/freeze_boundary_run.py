"""Run anemoi-inference with the LAM lateral boundary FROZEN at its initial value.

Normal anemoi LAM inference streams fresh ERA5 into the boundary nodes at every
rollout step (runner.add_boundary_forcings_to_input_tensor -> BoundaryForcings.
load_forcings_array at `next_date`). This wrapper monkeypatches that call so the
boundary is loaded once (first rollout step) and then held constant for the whole
forecast. Everything else (inner-region autoregression, forcings, output) is
unchanged.

Usage (same args as `anemoi-inference run`):
    python freeze_boundary_run.py temp_config.yaml

Diagnostic test:
    - If a model's RMSE-vs-lead-time curve stays flat with fresh boundary but
      climbs steeply when frozen here, its skill was coming from the boundary
      stream, not from evolving the inner state.
"""

import sys

import anemoi.inference.forcings as forcings_mod

_orig_load = forcings_mod.BoundaryForcings.load_forcings_array
_frozen_cache = {}


def _frozen_load(self, dates, current_state):
    # Cache per BoundaryForcings instance: load the boundary once, reuse forever.
    key = id(self)
    if key not in _frozen_cache:
        _frozen_cache[key] = _orig_load(self, dates, current_state)
        print(f"[freeze_boundary] froze boundary forcings for {self!r} at {dates}", flush=True)
    return _frozen_cache[key]


forcings_mod.BoundaryForcings.load_forcings_array = _frozen_load

from anemoi.inference.__main__ import main  # noqa: E402

sys.argv = ["anemoi-inference", "run", *sys.argv[1:]]
main()
