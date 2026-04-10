import os
import subprocess
from datetime import datetime, timedelta

start_date = datetime(2024, 8, 1, 0)
end_date = datetime(2024, 8, 31, 21)
interval = timedelta(hours=3)

checkpoints = {
    #"EGU/NoPowerLarge": ("/mnt/weatherloss/WindPower/training/EGU26/NoPower/checkpoint/NoPowerLarge", "inference-last.ckpt"),
     #"EGU/NoPowerLarge2": ("/mnt/weatherloss/WindPower/training/EGU26/NoPower/checkpoint/NoPowerLarge", "inference-anemoi-by_time-epoch_025-step_210000.ckpt"),
        "EGU/NoPowerLargeNoRollout": ("/mnt/weatherloss/WindPower/training/EGU26/NoPower/checkpoint/NoPowerLarge", "inference-anemoi-by_epoch-epoch_015-step_200000.ckpt"),
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
lead_time: 30
date: "{date_str}"
device: cuda
input:
  dataset:
    dataset:
      cutout:
        - dataset: /mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/Cerra_A_large.zarr
        - dataset: /mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/era5_A_large.zarr
      min_distance_km: 0
      adjust: all
output:
  extract_lam:      
    output:
      netcdf: {output_file}
""")

        print(f"[{tag}] Running forecast for {date_str}")
        subprocess.run(["anemoi-inference", "run", temp_yaml])
        current += interval

print("All forecasts complete.")