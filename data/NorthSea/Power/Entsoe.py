import os
import time
import pandas as pd
from tqdm import tqdm
from entsoe import EntsoePandasClient
from entsoe.exceptions import InvalidPSRTypeError, NoMatchingDataError

# ---------------- CONFIG ----------------
# Bidding zones to pull generation-per-plant (psr_type B18 = offshore wind).
TARGET_ZONES = {
    #"NL": "Netherlands",
    #"GB": "Great Britain",
    #"DE_LU": "Germany/Luxembourg",
   # "DK" : "Denmark"
    #"BZN|DK1": "Denmark West (DK1)",
    #"BZN|DK2": "Denmark East (DK2)",
    "BE": "Belgium",
}
PSR_TYPE = "B18"
START_YM = "2021-01"
END_YM = "2021-02"
OUT_DIR = "entsoe_offshore"

# ---------------- API TOKEN ----------------
with open("token.txt") as f:
    api_key = f.read().strip()

client = EntsoePandasClient(api_key=api_key)

# ---------------- DATE WINDOWS ----------------
START_TS = pd.Timestamp(f"{START_YM}-01", tz="UTC")
END_TS = (pd.Timestamp(f"{END_YM}-01", tz="UTC") + pd.offsets.MonthEnd(1)) + pd.Timedelta(hours=23)
YEARS = range(START_TS.year, END_TS.year + 1)


def month_end(ts):
    return (ts + pd.offsets.MonthEnd(1)) + pd.Timedelta(hours=23)


def month_starts_for_year(year):
    """Return month starts for the requested year, clipped to the configured start/end range."""
    year_start = max(START_TS, pd.Timestamp(f"{year}-01-01", tz="UTC"))
    year_end = min(END_TS, pd.Timestamp(f"{year}-12-01", tz="UTC"))
    if year_end < year_start:
        return []
    return list(pd.date_range(year_start, year_end, freq="MS"))


def filter_psr(df, psr_type):
    """Keep only columns that match the requested psr_type if present; otherwise return as-is."""
    if not psr_type or not isinstance(df.columns, pd.MultiIndex):
        return df

    level_names = df.columns.names
    if "psr_type" in level_names:
        lvl = level_names.index("psr_type")
    elif "psr" in level_names:
        lvl = level_names.index("psr")
    else:
        # Heuristic: second level often holds psr
        lvl = 1 if df.columns.nlevels > 1 else None

    if lvl is None:
        return df

    mask = df.columns.get_level_values(lvl) == psr_type
    return df.loc[:, mask] if mask.any() else df


def download_zone_year(zone_code, zone_name, year):
    months = month_starts_for_year(year)
    if not months:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    hourly_path = os.path.join(OUT_DIR, f"{zone_code}_{year}_offshore_per_unit_hourly.csv")
    resampled_path = os.path.join(OUT_DIR, f"{zone_code}_{year}_offshore_per_unit_3H_meanMW.csv")
    if os.path.exists(hourly_path) and os.path.exists(resampled_path):
        print(f"\n→ {zone_name} ({zone_code}) {year}: outputs exist, skipping")
        return

    print(f"\n→ {zone_name} ({zone_code}) {year}: {len(months)} month window(s)")
    dfs = []
    for s in tqdm(months, total=len(months), unit="month", desc=f"{zone_code} {year}"):
        e = min(month_end(s), END_TS)
        # Try requesting only the PSR; if the zone rejects psrType, fall back to all and filter client-side.
        try:
            df = client.query_generation_per_plant(zone_code, start=s, end=e, psr_type=PSR_TYPE)
        except InvalidPSRTypeError:
            df = client.query_generation_per_plant(zone_code, start=s, end=e)
            df = filter_psr(df, PSR_TYPE)
        except NoMatchingDataError:
            print(f"  ⚠ no data for {zone_code} {s.date()}–{e.date()}, skipping")
            continue
        except Exception as exc:
            print(f"  ⚠ error on {zone_code} {s.date()}–{e.date()}: {exc}. retrying once in 10s…")
            time.sleep(10)
            try:
                df = client.query_generation_per_plant(zone_code, start=s, end=e, psr_type=PSR_TYPE)
            except InvalidPSRTypeError:
                df = client.query_generation_per_plant(zone_code, start=s, end=e)
                df = filter_psr(df, PSR_TYPE)
            except Exception as exc2:
                print(f"  ❌ failed after retry: {exc2}. skipping window.")
                continue

        # Ensure UTC and clean index
        df.index = df.index.tz_convert("UTC")
        df.index.name = "time"

        # Drop multi-index levels we don’t need (unit_eic, psr)
        df = df.droplevel([1, 2], axis=1)
        dfs.append(df)

    if not dfs:
        print(f"  ⚠ no data collected for {zone_code} {year}, skipping saves")
        return

    df_all = pd.concat(dfs).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]

    df_3h = df_all.resample("3h").mean()

    os.makedirs(OUT_DIR, exist_ok=True)
    hourly_path = os.path.join(OUT_DIR, f"{zone_code}_{year}_offshore_per_unit_hourly.csv")
    resampled_path = os.path.join(OUT_DIR, f"{zone_code}_{year}_offshore_per_unit_3H_meanMW.csv")

    df_all.to_csv(hourly_path)
    df_3h.to_csv(resampled_path)
    print(f"  ✔ saved {hourly_path}")
    print(f"  ✔ saved {resampled_path}")


if __name__ == "__main__":
    for zone_code, zone_name in TARGET_ZONES.items():
        for year in YEARS:
            download_zone_year(zone_code, zone_name, year)

    print("\n✅ Done.")
