import os
import shutil

bad_dates = [
    "20250206150000", "20250207030000", "20250207090000",
    "20250730120000", "20250730150000", "20250730180000", "20250730210000",
    "20250731000000", "20250731030000", "20250731060000", "20250731090000",
    "20250731120000", "20250731150000", "20250731180000", "20250731210000",
]

infer_dir = "EGU/NoPowerTFNew"  # <-- set this

for date in bad_dates:
    for entry in os.listdir(infer_dir):
        if date in entry:
            full_path = os.path.join(infer_dir, entry)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
            print(f"Deleted: {full_path}")
