import subprocess
from datetime import datetime, timedelta
import os

start_date = datetime(2024, 8, 1, 3)
end_date = datetime(2024, 8, 31, 21)
interval = timedelta(hours=3)

output_dir = "Huber4"
os.makedirs(output_dir, exist_ok=True)

checkpoint_path = "/mnt/data/weatherloss/WindPower/training/GraphTransformer/fullloss/checkpoint/d1563676-4bbf-4dad-b86b-941bb8de4288/inference-last.ckpt"

while start_date <= end_date:
    date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    output_file = f"{output_dir}/forecast_{date_str.replace(':', '').replace('-', '').replace('T', '')}.nc"

    temp_yaml = "temp_config.yaml"
    with open(temp_yaml, "w") as f:
        f.write(f"""\
checkpoint: {checkpoint_path}
lead_time: 24
date: "{date_str}"
input: test
output:
  netcdf: {output_file}
""")

    print(f"Running forecast for {date_str}")
    subprocess.run(["anemoi-inference", "run", temp_yaml])

    start_date += interval