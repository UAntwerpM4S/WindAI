import os
import subprocess
from datetime import datetime, timedelta

start_date = datetime(2024, 8, 1, 0)
end_date = datetime(2025, 7, 31, 21)
interval = timedelta(hours=3)

checkpoints = {
    #     "WindAI/WindWeather": (
    #     "/mnt/weatherloss/WindPower/training/WindAI/WindWeather/checkpoint/5aa1c5854b234c97a103ff630801e779/",
    #     "inference-last.ckpt"
    # ),
            "WindAI/RegularWeather": (
        "/mnt/weatherloss/WindPower/training/WindAI/RegularWeather/checkpoint/80b001d0e79942d086812e255a19b0e1/",
        "inference-last.ckpt"
    ),

    #     "WindAI/VanillaPowerGT": (
    #     "/mnt/weatherloss/WindPower/training/WindAI/VanillaPowerGT/checkpoint/45fbdb76e7ce4f568373a524b5897edf/",
    #     "inference-last.ckpt"
    # ),

    #     "WindAI/VanillaPowerTF": (
    #     "/mnt/weatherloss/WindPower/training/WindAI/VanillaPowerTF/checkpoint/3966833d6cd841d1a8d298c5603b2ee0/",
    #     "inference-last.ckpt"
    # ),

    #     "WindAI/WindHeavyTinyPower": (
    #     "/mnt/weatherloss/WindPower/training/WindAI/WindHeavyTinyPower/checkpoint/01bd98167d054c85b4266f24a239dc37/",
    #     "inference-last.ckpt"
    # ),

    #     "WindAI/WindHeavyVanillaPower": (
    #     "/mnt/weatherloss/WindPower/training/WindAI/WindHeavyVanillaPower/checkpoint/6d3b66b2e04140e0abadbf5359cd8d71/",
    #     "inference-last.ckpt"
    # ),


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
        - dataset: /mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/New_Cerra_A_large.zarr
        - dataset: /mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/era5_A_large.zarr
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