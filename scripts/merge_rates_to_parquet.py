"""Merge ECB/Fed interest rate features into the M1 parquet.

Reads swap_history_eurusd.csv (daily ECB/Fed rates), joins by calendar date
onto the M1 parquet, and computes derived fundamental features:

    fund_rate_diff             — ECB rate minus Fed rate
    fund_rate_diff_change      — day-over-day change in rate_diff
    fund_rate_diff_time_since  — days since last rate_diff change (fractional, min precision)
    fund_carry_rank            — rank of rate_diff over trailing 60 days [0, 1]

All rate columns are shifted by 1 day to avoid look-ahead (the rate for date D
is only known after the decision, so it's available from D+1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARQUET_PATH = "data/processed/eurusd_m1.parquet"
SWAP_CSV = "data/processed/swap_history_eurusd.csv"


def _compute_daily_features(rates: pd.DataFrame) -> pd.DataFrame:
    """Compute daily-level rate features on ~10k rows.

    Input rates has columns: date (datetime.date), ecb_rate, fed_rate.
    Returns DataFrame with date index and computed feature columns.
    """
    rates = rates.set_index("date").sort_index()
    # SHIFT(1) — rate for date D is available from D+1
    rates["rate_diff"] = (rates["ecb_rate"] - rates["fed_rate"]).shift(1)
    rates["rate_diff_change"] = rates["rate_diff"].diff().fillna(0.0)
    rates["rate_diff_change"] = np.where(
        rates["rate_diff_change"].abs() < 1e-10, 0.0, rates["rate_diff_change"]
    )

    # Time since last change (in days). Build from change dates, forward-fill.
    change_mask = rates["rate_diff_change"] != 0.0
    last_change_idx = pd.Series(
        pd.to_datetime(rates.index), index=rates.index
    ).where(change_mask).ffill()
    idx_dt = pd.to_datetime(rates.index)
    rates["time_since_change"] = ((idx_dt - last_change_idx).dt.days).fillna(0.0)

    # Carry rank over trailing 60 days
    roll = rates["rate_diff"].rolling(60, min_periods=1)
    rates["carry_rank"] = (
        (rates["rate_diff"] - roll.min()) / (roll.max() - roll.min() + 1e-10)
    ).fillna(0.5).clip(0.0, 1.0)

    return rates[["rate_diff", "rate_diff_change", "time_since_change", "carry_rank"]]


def merge_to_parquet(
    parquet_path: str = PARQUET_PATH,
    swap_csv: str = SWAP_CSV,
) -> int:
    """Merge rate features into M1 parquet in-place. Returns new column count."""
    # --- Load daily rates ---
    daily = pd.read_csv(
        swap_csv,
        parse_dates=["date"],
        usecols=["date", "ecb_rate", "fed_rate"],
    )
    daily["date"] = daily["date"].dt.date
    daily_features = _compute_daily_features(daily)
    # daily_features index is date (datetime.date)

    # --- Load M1 parquet ---
    df = pd.read_parquet(parquet_path)
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    idx = df.index

    # Broadcast daily features to M1 bars
    today = pd.Series(idx.date, index=idx, name="today")
    m1_with_date = today.to_frame().join(daily_features, on="today")

    features = pd.DataFrame(index=idx)
    features["fund_rate_diff"] = m1_with_date["rate_diff"]
    features["fund_rate_diff_change"] = m1_with_date["rate_diff_change"]
    features["fund_carry_rank"] = m1_with_date["carry_rank"]

    # Time since change: daily_base (days) + fractional day from bar position
    day_frac = idx.hour / 24.0 + idx.minute / 1440.0
    features["fund_rate_diff_time_since"] = (
        m1_with_date["time_since_change"] + day_frac
    ).fillna(0.0)

    # --- Merge into parquet ---
    existing_fund = [c for c in df.columns if c.startswith("fund_")]
    if existing_fund:
        df = df.drop(columns=existing_fund)

    df = df.join(features)
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

    n_new = len(features.columns)
    print(f"Merged {n_new} rate feature columns into {parquet_path}")
    return n_new


if __name__ == "__main__":
    merge_to_parquet()
