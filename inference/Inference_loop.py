import os
import subprocess
from datetime import datetime, timedelta

# Common forecasting window
start_date = datetime(2020, 10, 1, 0)
end_date = datetime(2025, 7, 31, 21)
interval = timedelta(hours=3)

ckpt_dir = "/mnt/weatherloss/WindPower/training/CI/GraphTransformer/checkpoint/8276a997-e9e2-4382-89ef-ae78f97ac47c"
checkpoints = { 
"CI/GraphTransformer": "inference-last.ckpt"
}

for tag, ckpt_name in checkpoints.items():
    checkpoint_path = os.path.join(ckpt_dir, ckpt_name)
    output_dir = tag
    os.makedirs(output_dir, exist_ok=True)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%dT%H:%M:%S")
        output_file = f"{output_dir}/forecast_{date_str.replace(':', '').replace('-', '').replace('T', '')}.nc"

        if os.path.exists(output_file):
            print(f"[{tag}] Skipping {date_str} (NetCDF already exists)")
            current += interval
            continue


        temp_yaml = "temp_config.yaml"
        with open(temp_yaml, "w") as f:
            f.write(
                f"""\
checkpoint: {checkpoint_path}
lead_time: 72
date: "{date_str}"
input:
  dataset:
    dataset:
      cutout:
        - dataset: /mnt/weatherloss/WindPower/data/NorthSea/Anemoidatasets/Cerra_CI_HR_A.zarr
        - dataset: /mnt/weatherloss/WindPower/data/NorthSea/Anemoidatasets/Cerra_CI_LR_A.zarr
      min_distance_km: 0
      adjust: all
output:
  netcdf: {output_file}
"""
            )

        print(f"[{tag}] Running forecast for {date_str}")
        subprocess.run(["anemoi-inference", "run", temp_yaml])

        current += interval



