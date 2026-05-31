"""Shared pytest fixtures for the Re-Diggn test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    """Minimal synthetic 1-minute OHLCV DataFrame for unit tests."""
    rng = np.random.default_rng(42)
    n = 500
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0001, n))
    spread = 0.00020
    noise = rng.uniform(0.00005, 0.00030, n)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-08 20:00", periods=n, freq="1min", tz="UTC"),
            "open": close - rng.uniform(0, 0.0002, n),
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
            "bid": close - spread / 2,
            "ask": close + spread / 2,
        }
    )


@pytest.fixture()
def ohlcv_df_with_targets(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV frame with synthetic target columns for target-stat tests."""
    df = ohlcv_df.copy()
    rng = np.random.default_rng(42)
    df["target_delta_60"] = rng.uniform(0.0001, 0.0030, len(df))
    df["target_max_eps_60"] = rng.uniform(0.0005, 0.0050, len(df))
    return df
