import os
import pandas as pd
import pytz
from tqdm import tqdm
from entsoe import EntsoePandasClient

# ---------------- CONFIG ----------------
COUNTRY = "BE"                        
START_YM = "2020-01"
END_YM   = "2022-12"
OUT_FILE = "BE_offshore_per_unit_hourly_2020-01_to_2025-07.csv"

# ---------------- API TOKEN ----------------
with open("token.txt") as f:
    api_key = f.read().strip()

client = EntsoePandasClient(api_key=api_key)

# ---------------- DATE WINDOWS ----------------
month_starts = pd.date_range(
    pd.Timestamp(f"{START_YM}-01", tz="UTC"),
    pd.Timestamp(f"{END_YM}-01", tz="UTC"),
    freq="MS"
)

def month_end(s):
    return (s + pd.offsets.MonthEnd(1)) + pd.Timedelta(hours=23)

# ---------------- DOWNLOAD LOOP ----------------
dfs = []
for s in tqdm(month_starts, total=len(month_starts), unit="month"):
    e = month_end(s)

    df = client.query_generation_per_plant(
        COUNTRY, start=s, end=e, psr_type="B18"
    )

    # Ensure UTC and clean index
    df.index = df.index.tz_convert("UTC")
    df.index.name = "time"

    # Drop multi-index levels we don’t need (unit_eic, psr)
    df = df.droplevel([1, 2], axis=1)

    dfs.append(df)

# ---------------- MERGE MONTHS ----------------
df_all = pd.concat(dfs).sort_index()

# Remove duplicate timestamps if overlap
df_all = df_all[~df_all.index.duplicated(keep="first")]

# ---------------- RESAMPLE TO 3-HOURLY ----------------
df_3h = df_all.resample("3H").mean()

# ---------------- SAVE ----------------
os.makedirs("testdata", exist_ok=True)
df_all.to_csv("testdata/" + OUT_FILE)
df_3h.to_csv("testdata/BE_offshore_per_unit_3H_meanMW_2020-01_to_2025-07.csv")

print("✅ Saved:")
print("  testdata/" + OUT_FILE)
print("  testdata/BE_offshore_per_unit_3H_meanMW_2020-01_to_2025-07.csv")
