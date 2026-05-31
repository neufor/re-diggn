"""
EURUSD M1 — CSV cleaning and Parquet conversion
================================================
Source  : MetaTrader 5 export  (View -> Symbols -> Bars -> Request)
Provider: MT5 broker export
Pair    : EURUSD
Timeframe: M1 (1-minute OHLCV bars, bid-based)
Date range: 2000-01-03 to 2025-05-23
Pull date : 2026-05-31
Columns   : date, time, open, high, low, close, tick_volume, volume, spread
Notes     : Prices are 5-decimal (0.00001 pip precision for EURUSD)
            Spread is in points (0.00001 units). volume=0 always (MT5 FX limitation).
"""

import os
import sys
import glob
import time
import logging

import numpy as np
import pandas as pd

# ── Setup ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Projects\My\Re-Diggn"
os.chdir(PROJECT_ROOT)

LOG_PATH = os.path.join(PROJECT_ROOT, "scripts", "clean_eurusd_m1.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RAW_DIR    = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "eurusd_m1.parquet")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

COL_NAMES = ["date", "time", "open", "high", "low", "close",
             "tick_volume", "volume", "spread"]

# ── 1. Read all CSVs ──────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "EURUSD_i_M1_*.csv")))
log.info(f"Found {len(csv_files)} CSV files:")
total_raw_rows = 0
for f in csv_files:
    size_mb = os.path.getsize(f) / 1024**2
    log.info(f"  {os.path.basename(f)}  ({size_mb:.1f} MB)")

t0 = time.time()
frames = []
for f in csv_files:
    df_part = pd.read_csv(
        f,
        sep="\t",
        header=None,
        names=COL_NAMES,
        dtype={
            "date":  str,
            "time":  str,
            "open":  "float32",
            "high":  "float32",
            "low":   "float32",
            "close": "float32",
            "tick_volume": "int32",
            "volume":      "int64",
            "spread":      "int16",
        },
    )
    frames.append(df_part)
    log.info(f"  Loaded {os.path.basename(f)}: {len(df_part):,} rows")
    total_raw_rows += len(df_part)

df = pd.concat(frames, ignore_index=True)
log.info(f"Total rows after concat: {len(df):,}  (elapsed: {time.time()-t0:.1f}s)")
log.info(f"Raw dtypes:\n{df.dtypes}")

# ── 2. Parse timestamps → UTC datetime index ─────────────────────────────────
log.info("\n--- Parsing timestamps ---")
df["timestamp"] = pd.to_datetime(
    df["date"] + " " + df["time"],
    format="%Y.%m.%d %H:%M:%S",
    utc=True,
)
df = df.drop(columns=["date", "time"])
df = df.set_index("timestamp")
log.info(f"Index dtype : {df.index.dtype}")
log.info(f"Index range : {df.index.min()}  →  {df.index.max()}")

# ── 3. Data quality checks and fixes ─────────────────────────────────────────
issues = {}

# 3a. Duplicate timestamps ────────────────────────────────────────────────────
n_dupes = int(df.index.duplicated(keep="last").sum())
issues["duplicate_timestamps"] = n_dupes
log.info(f"\n[Duplicates] {n_dupes:,} duplicate timestamps → keeping last")
if n_dupes > 0:
    df = df[~df.index.duplicated(keep="last")]

# 3b. Out-of-order rows ───────────────────────────────────────────────────────
is_sorted = df.index.is_monotonic_increasing
log.info(f"[Order] Index monotonic increasing: {is_sorted}")
if not is_sorted:
    time_deltas = df.index.to_series().diff().dt.total_seconds()
    n_ooo = int((time_deltas < 0).sum())
    issues["out_of_order_rows"] = n_ooo
    log.info(f"  → {n_ooo:,} out-of-order timestamps — sorting ascending")
    df = df.sort_index()
else:
    issues["out_of_order_rows"] = 0

# 3c. Zero / negative prices ──────────────────────────────────────────────────
mask_nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
n_nonpos = int(mask_nonpos.sum())
issues["zero_or_negative_prices"] = n_nonpos
log.info(f"\n[Zero/Negative prices] {n_nonpos:,} rows → dropping")
if n_nonpos > 0:
    df = df[~mask_nonpos]

# 3d. OHLC sanity ─────────────────────────────────────────────────────────────
# Hard violation: high < low — definitely bad data, drop
high_lt_low = df["high"] < df["low"]
n_high_lt_low = int(high_lt_low.sum())
issues["ohlc_high_lt_low_dropped"] = n_high_lt_low
log.info(f"\n[OHLC hard — high < low] {n_high_lt_low:,} rows → dropping")
if n_high_lt_low > 0:
    df = df[~high_lt_low]

# Soft violations: flag but keep
high_below_oc = df["high"] < df[["open", "close"]].max(axis=1)
low_above_oc  = df["low"]  > df[["open", "close"]].min(axis=1)
soft_mask = high_below_oc | low_above_oc
n_soft = int(soft_mask.sum())
issues["ohlc_soft_violations_flagged_kept"] = n_soft
log.info(f"[OHLC soft — high<max(O,C) or low>min(O,C)] {n_soft:,} flagged (kept)")
if n_soft > 0:
    log.info("  Sample soft violations:")
    log.info(df[soft_mask].head(5).to_string())

# 3e. Weekend bars ────────────────────────────────────────────────────────────
dow = df.index.dayofweek   # Mon=0 … Sun=6
mask_weekend = dow >= 5
n_weekend = int(mask_weekend.sum())
issues["weekend_bars_dropped"] = n_weekend
n_sat = int((dow == 5).sum())
n_sun = int((dow == 6).sum())
log.info(f"\n[Weekend bars] {n_weekend:,} total (Sat: {n_sat:,}, Sun: {n_sun:,}) → dropping")
df = df[~mask_weekend]

# 3f. tick_volume == 0 ────────────────────────────────────────────────────────
n_zero_tv = int((df["tick_volume"] == 0).sum())
issues["zero_tick_volume_flagged_kept"] = n_zero_tv
log.info(f"\n[tick_volume == 0] {n_zero_tv:,} bars (kept — thin market / data artifact)")

# volume column diagnostics
# NOTE: 2000–2014 files have volume=0 (MT5 FX limitation for that era);
#       2015–2025 files have real non-zero volume from broker aggregation.
n_zero_vol  = int((df["volume"] == 0).sum())
n_nonz_vol  = len(df) - n_zero_vol
pct_zero_vol = n_zero_vol / len(df) * 100
log.info(f"[volume == 0] {n_zero_vol:,} bars ({pct_zero_vol:.1f}%)")
log.info(f"[volume > 0]  {n_nonz_vol:,} bars ({100-pct_zero_vol:.1f}%) — 2015+ broker aggregation data")

# 3g. Gap analysis ────────────────────────────────────────────────────────────
log.info("\n--- Gap analysis ---")
idx_series = df.index.to_series()
gaps_s = idx_series.diff().dt.total_seconds().dropna()

median_gap = gaps_s.median()
log.info(f"Median bar gap : {median_gap:.0f}s  (expected 60s)")
log.info(f"Min bar gap    : {gaps_s.min():.0f}s")
log.info(f"Max bar gap    : {gaps_s.max():.0f}s")

LONG_GAP_THRESHOLD = 3600  # 60 minutes
long_gap_mask = gaps_s > LONG_GAP_THRESHOLD
long_gaps = gaps_s[long_gap_mask]
log.info(f"Gaps > 60 min  : {len(long_gaps):,}")

# Classify each long gap as weekend or intra-week
# The timestamp in gaps_s is the bar AFTER the gap
# The bar before the gap is the one at position (gap_idx - 1)
gap_end_times   = long_gaps.index          # bar that opens after the gap
gap_end_pos     = df.index.get_indexer(gap_end_times)
gap_start_times = df.index[gap_end_pos - 1]   # bar just before the gap

gap_df = pd.DataFrame({
    "gap_start": gap_start_times.values,   # numpy datetime64 → stored as object col
    "gap_end":   gap_end_times.values,
    "gap_hours": long_gaps.values / 3600,
}, index=range(len(long_gaps)))

# Convert to proper datetime so .dt accessor works
gap_df["gap_start"] = pd.to_datetime(gap_df["gap_start"], utc=True)
gap_df["gap_end"]   = pd.to_datetime(gap_df["gap_end"],   utc=True)

# A weekend gap: starts on Fri/Sat/Sun (dow 4/5/6)
gap_df["is_weekend"] = gap_df["gap_start"].dt.dayofweek >= 4

n_wknd_gaps    = int(gap_df["is_weekend"].sum())
n_intraweek_gaps = len(gap_df) - n_wknd_gaps
issues["weekend_gaps_over60min"]       = n_wknd_gaps
issues["intraweek_gaps_over60min"]     = n_intraweek_gaps

log.info(f"  Weekend gaps (>=60 min) : {n_wknd_gaps:,}")
log.info(f"  Intra-week gaps (>=60 min): {n_intraweek_gaps:,}")

if n_intraweek_gaps > 0:
    intra = gap_df[~gap_df["is_weekend"]].sort_values("gap_hours", ascending=False).head(20)
    log.info(f"\n  Top intra-week gaps:\n{intra.to_string(index=False)}")

issues["total_raw_rows"] = total_raw_rows

# ── 4. Enforce final dtypes (paranoia after all the filtering) ────────────────
df["open"]        = df["open"].astype("float32")
df["high"]        = df["high"].astype("float32")
df["low"]         = df["low"].astype("float32")
df["close"]       = df["close"].astype("float32")
df["tick_volume"] = df["tick_volume"].astype("int32")
df["volume"]      = df["volume"].astype("int64")
df["spread"]      = df["spread"].astype("int16")

# ── 5. Write Parquet ──────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_parquet(OUT_PATH, engine="pyarrow", compression="snappy", index=True)
file_size_mb = os.path.getsize(OUT_PATH) / 1024**2
log.info(f"\n--- Parquet written ---")
log.info(f"Path      : {OUT_PATH}")
log.info(f"File size : {file_size_mb:.2f} MB")

# ── 6. Final summary ──────────────────────────────────────────────────────────
mem_mb = df.memory_usage(deep=True).sum() / 1024**2

log.info("\n" + "="*65)
log.info("FINAL SUMMARY")
log.info("="*65)
log.info(f"Total rows after cleaning  : {len(df):,}")
log.info(f"Total raw rows (pre-clean) : {total_raw_rows:,}")
log.info(f"Rows removed               : {total_raw_rows - len(df):,}")
log.info(f"Date range                 : {df.index.min()}  →  {df.index.max()}")
log.info(f"DataFrame memory           : {mem_mb:.1f} MB")
log.info(f"Parquet file (snappy)      : {file_size_mb:.2f} MB")
log.info(f"\nData quality issues:")
for k, v in issues.items():
    log.info(f"  {k:<50} {v:>12,}")
log.info(f"\nColumn dtypes:")
for col, dtype in df.dtypes.items():
    log.info(f"  {col:<15} {dtype}")
log.info("="*65)

# ── 7. Verification read-back ─────────────────────────────────────────────────
log.info("\n--- Verification read-back ---")
import io
df_check = pd.read_parquet(OUT_PATH, engine="pyarrow")
sio = io.StringIO()
df_check.info(memory_usage="deep", buf=sio)
log.info(sio.getvalue())
log.info(f"\nDescribe:\n{df_check.describe().to_string()}")

log.info("Script completed successfully.")
