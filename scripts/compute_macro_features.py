"""Compute macro/economic event features for EURUSD M1 pipeline.

Reads processed economic calendar events and the M1 OHLCV parquet, then
produces per-M1-bar feature columns:

  Time-since per group    — days since last occurrence (capped 14d)
  Event flags             — True for 60 M1 bars (1 hour) after each event
  Standardised surprise   — (actual - forecast) / rolling_std(20 errors),
                            forward-filled for 60 bars
  Density (24h, 7d)       — count of high-impact events in look-back windows
  Time to next            — days until next scheduled high-impact event (capped 14d)

Usage:
    uv run python scripts/compute_macro_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT_PATH = Path("data/raw/economic_calendar.parquet")
PARQUET_PATH = Path("data/processed/eurusd_m1.parquet")
OUTPUT_PATH = Path("data/processed/fundamental_features.parquet")

MAJOR_GROUPS = ("fed_rate", "ecb_rate", "nfp", "cpi", "gdp")

IMPACT_HIGH = {"high impact expected", "high impact"}

FLAG_FORWARD_BARS = 60       # 1 hour at M1
ROLLING_ERROR_WINDOW = 20    # past events for surprise std
TIME_CAP_DAYS = 14.0


def _utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ensure *idx* is a UTC-aware DatetimeIndex."""
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def compute_features(
    event_path: str | Path = EVENT_PATH,
    m1_path: str | Path = PARQUET_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Compute all macro features and save to parquet."""
    # --- Load M1 index (UTC) ---
    m1_idx = _utc_index(pd.read_parquet(m1_path, columns=[]).index)
    print(f"M1 bars: {len(m1_idx):,}  ({m1_idx.min()} — {m1_idx.max()})")

    # --- Load events (already UTC from Phase 2) ---
    events = pd.read_parquet(event_path)
    high = events[events["impact"].isin(IMPACT_HIGH)].copy()
    print(f"Events: {len(events):,} total, {len(high):,} high-impact")

    result = pd.DataFrame({"timestamp": m1_idx})

    # ------------------------------------------------------------------
    # Per-group features: time_since, flag, surprise
    # ------------------------------------------------------------------
    for group in MAJOR_GROUPS:
        ge = events[events["event_group"] == group].copy()
        if ge.empty:
            continue
        ge = ge.dropna(subset=["timestamp"]).sort_values("timestamp")

        # -- Time-since via merge_asof (backward = most recent past event) --
        # No tolerance — we always want the most recent event, even if very old.
        ge_ts = ge[["timestamp"]].drop_duplicates().sort_values("timestamp")
        merged = pd.merge_asof(
            result[["timestamp"]].sort_values("timestamp"),
            ge_ts.rename(columns={"timestamp": "event_ts"}).sort_values("event_ts"),
            left_on="timestamp",
            right_on="event_ts",
            direction="backward",
        )
        delta_days = (
            (merged["timestamp"] - merged["event_ts"]).dt.total_seconds() / 86400
        )
        result[f"fund_ev_{group}_time_since"] = delta_days.clip(
            upper=TIME_CAP_DAYS
        ).astype(np.float32)

        # -- Surprise standardisation --
        ge_err = ge.dropna(subset=["actual", "forecast"]).copy()
        ge_err["error"] = ge_err["actual"] - ge_err["forecast"]
        ge_err["error_std"] = (
            ge_err["error"].rolling(ROLLING_ERROR_WINDOW, min_periods=2).std()
        )
        ge_err["surprise"] = ge_err["error"] / ge_err["error_std"].replace(0, np.nan)
        ge_err["surprise"] = ge_err["surprise"].clip(-5.0, 5.0)
        surprise_data = ge_err.dropna(subset=["surprise"])

        # Place values at the M1 bar *immediately after* each event
        # (conservative: event info only available from the next bar)
        surprise_col = pd.Series(np.nan, index=m1_idx, dtype=np.float32)
        flag_col = pd.Series(False, index=m1_idx, dtype=bool)

        for _, row in surprise_data.iterrows():
            loc = m1_idx.searchsorted(row["timestamp"])
            if loc < len(m1_idx):
                surprise_col.iloc[loc] = row["surprise"]

        for ts in ge["timestamp"]:
            loc = m1_idx.searchsorted(ts)
            if loc < len(m1_idx):
                flag_col.iloc[loc] = True

        result[f"fund_ev_{group}_flag"] = (
            flag_col.replace(False, np.nan)
            .ffill(limit=FLAG_FORWARD_BARS)
            .notna()
            .to_numpy()
        )
        result[f"fund_ev_{group}_surprise"] = surprise_col.ffill(
            limit=FLAG_FORWARD_BARS
        ).to_numpy()

        n_surprise = surprise_data.shape[0]
        n_flag = int(flag_col.sum())
        print(
            f"  {group:12s}  events={len(ge):5d}  "
            f"surprise={n_surprise:4d}  flags={n_flag:4d}  "
            f"time_since always populated"
        )

    # ------------------------------------------------------------------
    # Density: count of high-impact events in past 24h / 7d
    # ------------------------------------------------------------------
    markers = pd.Series(0.0, index=m1_idx, dtype=np.float64)
    # Vectorised searchsorted using nanosecond-epoch representation
    m1_epoch = m1_idx.asi8  # int64 nanosec since epoch, tz-free
    high_epoch = pd.DatetimeIndex(high["timestamp"]).asi8
    event_indices = np.searchsorted(m1_epoch, high_epoch)
    event_indices = event_indices[event_indices < len(m1_idx)]
    uidx, counts = np.unique(event_indices, return_counts=True)
    markers.iloc[uidx] = counts.astype(np.float64)

    result["fund_ev_density_24h"] = (
        markers.rolling("24h", min_periods=0).sum().astype(np.float32).to_numpy()
    )
    result["fund_ev_density_7d"] = (
        markers.rolling("7D", min_periods=0).sum().astype(np.float32).to_numpy()
    )
    print("  density_24h/7d   computed")

    # ------------------------------------------------------------------
    # Time to next high-impact event
    # ------------------------------------------------------------------
    high_ts = (
        pd.DataFrame({"next_ts": high["timestamp"].drop_duplicates()})
        .sort_values("next_ts")
    )
    merged_next = pd.merge_asof(
        result[["timestamp"]].sort_values("timestamp"),
        high_ts.sort_values("next_ts"),
        left_on="timestamp",
        right_on="next_ts",
        direction="forward",
        tolerance=pd.Timedelta(days=TIME_CAP_DAYS),
    )
    delta_next = (
        (merged_next["next_ts"] - merged_next["timestamp"])
        .dt.total_seconds() / 86400
    )
    result["fund_ev_time_to_next"] = delta_next.clip(upper=TIME_CAP_DAYS).astype(
        np.float32
    ).to_numpy()
    print("  time_to_next     computed")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    result = result.set_index("timestamp")
    result.to_parquet(output_path, engine="pyarrow", compression="snappy")
    print(f"\nSaved {len(result):,} rows x {len(result.columns)} cols -> {output_path}")
    return result


def main():
    compute_features()


if __name__ == "__main__":
    main()
