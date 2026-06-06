"""Orchestrator: merge all fundamental features into the M1 OHLCV parquet.

Reads the M1 parquet (already has fund_rate_* columns from Phase 1),
and merges in the economic-event features from Phase 3.

Usage:
    uv run python scripts/populate_fundamental_features.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PARQUET_PATH = Path("data/processed/eurusd_m1.parquet")
FEATURES_PATH = Path("data/processed/fundamental_features.parquet")


def merge_features(
    m1_path: str | Path = PARQUET_PATH,
    features_path: str | Path = FEATURES_PATH,
) -> None:
    """Merge fundamental feature columns into the M1 parquet in-place."""
    print(f"Loading M1 parquet: {m1_path}")
    m1 = pd.read_parquet(m1_path)
    print(f"  {len(m1):,} rows x {len(m1.columns)} cols")

    print(f"Loading features: {features_path}")
    features = pd.read_parquet(features_path)
    print(f"  {len(features):,} rows x {len(features.columns)} cols")

    # Check which feature columns don't already exist
    existing = set(m1.columns)
    new_cols = [c for c in features.columns if c not in existing]
    skip_cols = [c for c in features.columns if c in existing]
    if skip_cols:
        print(f"  Skipping ({len(skip_cols)} already exist): {skip_cols}")

    if not new_cols:
        print("  No new columns to merge - all already present.")
        return

    # Merge by index alignment (both have UTC DatetimeIndex)
    # Use .to_numpy() to avoid pandas index alignment issues
    m1[new_cols] = features[new_cols].to_numpy()

    print(f"Writing back {m1_path} ({len(m1.columns)} cols total)")
    m1.to_parquet(m1_path, engine="pyarrow", compression="snappy")

    print(f"  Added {len(new_cols)} new columns: {new_cols}")
    print("Done.")


def main():
    merge_features()


if __name__ == "__main__":
    main()
