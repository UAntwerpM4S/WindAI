#!/usr/bin/env python3
"""Count forecast dates per model dir and show the intersection across all."""
import glob
import os
import re

BASE = "/mnt/weatherloss/WindPower/inference/WindAI"
TS = re.compile(r"forecast_(\d{14})\.nc$")


def dates_in(folder):
    return {TS.search(os.path.basename(f)).group(1)
            for f in glob.glob(os.path.join(folder, "forecast_*.nc"))
            if TS.search(os.path.basename(f))}


models = sorted(d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)))
per_model = {m: dates_in(os.path.join(BASE, m)) for m in models}

for m in models:
    print(f"{m:<28} {len(per_model[m])}")

inter = set.intersection(*per_model.values()) if per_model else set()
union = set().union(*per_model.values()) if per_model else set()
print("-" * 40)
print(f"{'union (any model)':<28} {len(union)}")
print(f"{'intersection (all models)':<28} {len(inter)}")
