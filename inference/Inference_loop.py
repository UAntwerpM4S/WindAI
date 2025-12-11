import os
import subprocess
from datetime import datetime, timedelta

# Common forecasting window
start_date = datetime(2024, 8, 1, 3)
end_date = datetime(2025, 7, 31, 21)
interval = timedelta(hours=3)

ckpt_dir = "/mnt/data/weatherloss/WindPower/training/RegularWeather/checkpoint/Regular"
checkpoints = { "Regular": "inference-last.ckpt"
}

for tag, ckpt_name in checkpoints.items():
    checkpoint_path = os.path.join(ckpt_dir, ckpt_name)
    output_dir = tag
    os.makedirs(output_dir, exist_ok=True)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%dT%H:%M:%S")
        output_file = f"{output_dir}/forecast_{date_str.replace(':', '').replace('-', '').replace('T', '')}.nc"

        temp_yaml = "temp_config.yaml"
        with open(temp_yaml, "w") as f:
            f.write(
                f"""\
checkpoint: {checkpoint_path}
lead_time: 72
date: "{date_str}"
input: test
output:
  netcdf: {output_file}
runner:
  external_graph:
    graph: /mnt/data/weatherloss/WindPower/graphs/tryout.pt
    updated_number_of_grid_points: x
"""
            )

        print(f"[{tag}] Running forecast for {date_str}")
        subprocess.run(["anemoi-inference", "run", temp_yaml])

        current += interval
