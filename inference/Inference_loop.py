import os
import subprocess
from datetime import datetime, timedelta

start_date = datetime(2024, 8, 1, 0)
end_date = datetime(2025, 7, 31, 9)
interval = timedelta(hours=3)

checkpoints = {
    #     "WPDistr/SemiHighCapacity": (
    #      "/mnt/weatherloss/WindPower/training/WPDistr/SemiHighCapacityGT/checkpoint/79c61ab7f58845249c7619a5d2ae3adc/",
    #     "inference-last.ckpt"
    # ),
        "WPDistr/HighCapacityGT": (
        "/mnt/weatherloss/WindPower/training/WPDistr/HighCapacityGT/checkpoint/c9c70adb05ec487da3dadb5e71ba6e8f/",
        "inference-last.ckpt"
    ),

        #   "EGU/TF00":  ("/mnt/weatherloss/WindPower/training/EGU26/BigTransformer/checkpoint/606532fc724149bcb7eb378f22d29d61","inference-anemoi-by_epoch-epoch_015-step_100000.ckpt"),
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
lead_time: 37
date: "{date_str}"
device: cuda
input:
  dataset:
    dataset:
      cutout:
        - dataset: /mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_cerra_A.zarr
        - dataset: /mnt/weatherloss/WindPower/data/WPDistr/Anemoidatasets/power_era5_A.zarr
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