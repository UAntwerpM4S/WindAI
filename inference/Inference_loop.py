import os
import subprocess
from datetime import datetime, timedelta

start_date = datetime(2024, 8, 1, 0)
end_date = datetime(2025, 7, 31, 9)
interval = timedelta(hours=3)

checkpoints = {

        "WPDistr/Vanilla_Finetune": (
        "/mnt/weatherloss/WindPower/training/WPDistr/VanillaPowerGTFinetune/checkpoint/b5b86e46dc9b433fa6e3f7383a9f6c43/",
        "inference-last.ckpt"
    ),

  
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