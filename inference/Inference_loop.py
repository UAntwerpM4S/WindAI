import os
import subprocess
from datetime import datetime, timedelta

start_date = datetime(2024, 8, 1, 0)
end_date = datetime(2024, 9, 15, 21)
interval = timedelta(hours=3)

checkpoints = {
    "CI/GT156": ("/mnt/weatherloss/WindPower/training/CI/GraphTransformer/checkpoint/GTNEW", "inference-anemoi-by_time-epoch_011-step_156000.ckpt"),
     "CI/GT150": ("/mnt/weatherloss/WindPower/training/CI/GraphTransformer/checkpoint/GTNEW", "inference-anemoi-by_time-epoch_005-step_150000.ckpt"),

}

for tag, (ckpt_dir, ckpt_name) in checkpoints.items():
    checkpoint_path = os.path.join(ckpt_dir, ckpt_name)
    output_dir = tag
    os.makedirs(output_dir, exist_ok=True)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%dT%H:%M:%S")
        output_file = f"{output_dir}/forecast_{date_str.replace(':', '').replace('-', '').replace('T', '')}.nc"

        if os.path.exists(output_file):
            print(f"[{tag}] Skipping {date_str} (already exists)")
            current += interval
            continue

        temp_yaml = "temp_config.yaml"
        with open(temp_yaml, "w") as f:
            f.write(f"""\
checkpoint: {checkpoint_path}
lead_time: 72
date: "{date_str}"
input:
  dataset:
    dataset:
      cutout:
        - dataset: /mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/Cerra_A.zarr
        - dataset: /mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/era5_A2.zarr
      min_distance_km: 0
      adjust: all
output:
  netcdf: {output_file}
""")

        print(f"[{tag}] Running forecast for {date_str}")
        subprocess.run(["anemoi-inference", "run", temp_yaml])
        current += interval

print("All forecasts complete.")