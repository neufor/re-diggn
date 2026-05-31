"""Technical indicator features for FX OHLCV data."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


def generate_indicators(
    df_input: pd.DataFrame,
    sma_periods: list[int] | None = None,
    minmax_periods: list[int] | None = None,
    stoch_periods: list[int] | None = None,
    stoch_smooth: list[int] | None = None,
    session_hours: list[int] | None = None,
    use_abs: bool = False,
    use_relative: bool = False,
) -> pd.DataFrame:
    """
    Generate technical indicators and session-time features.

    Indicators produced
    -------------------
    SMA
        One column per period: ``sma_{period}``

    Pairwise SMA differences (all i < j pairs)
        ``diff_sma_{i}_{j}``  [optionally abs or relative to close]

    MinMax range (Highest - Lowest)
        ``minmax_range_{period}``  [optionally relative to close]

    Flat-strength ratio
        Sum of |close diffs| over (period-1) bars divided by MinMax range.
        ``flat_strength_{period}``

    Level-100 proximity
        How close the price is to a round 100-pip level (0 = at level, 1 = midpoint).
        ``level_100``

    Stochastic K%, D%, and their diff (for every stoch_period × smooth combination)
        ``stoch_{period}_{smooth}_k``, ``_d``, ``diff_stoch_{period}_{smooth}``

    Session-time features (only when ``datetime`` column is present and
    ``session_hours`` is non-empty)
        ``session_hour_float`` — fractional position within the session (0→1)
        ``minute`` — minute of the hour
        One-hot columns for: ``weekday``, ``session_weekday``, ``quarter``,
        ``monthweek_business``

    Parameters
    ----------
    df_input : DataFrame
        Must have columns ``open``, ``high``, ``low``, ``close``.
    sma_periods : list[int], optional
    minmax_periods : list[int], optional
    stoch_periods : list[int], optional
    stoch_smooth : list[int], optional
    session_hours : list[int], optional
        Ordered list of trading hours that make up one session, e.g.
        ``[20, 21, 22, 23, 0, 1, 2, 3, 4, 5]`` for an Asian/Oceania session
        starting at 20:00 UTC.
    use_abs : bool
        Take absolute value of SMA diffs and stochastic columns.
    use_relative : bool
        Express SMA diffs and MinMax range as a fraction of close price.

    Returns
    -------
    DataFrame with all new columns appended (original columns preserved).
    """
    df = df_input.copy()

    sma_list = sorted(set(sma_periods or []))
    minmax_list = sorted(set(minmax_periods or []))
    stoch_list = sorted(set(stoch_periods or []))
    smooth_list = sorted(set(stoch_smooth or []))
    session_hours_list = list(session_hours or [])

    # ── SMAs ──────────────────────────────────────────────────────────────────
    for period in sma_list:
        df[f"sma_{period}"] = ta.sma(df["close"], length=period)  # type: ignore[attr-defined]

    # Pairwise SMA differences
    for p1, p2 in combinations(sma_list, 2):
        col = f"diff_sma_{p1}_{p2}"
        df[col] = df[f"sma_{p1}"] - df[f"sma_{p2}"]
        if use_abs:
            df[col] = df[col].abs()
        if use_relative:
            df[col] = df[col] / df["close"]

    # ── MinMax range and flat-strength ────────────────────────────────────────
    abs_diff = df["close"].diff().abs()
    for period in minmax_list:
        highest = df["high"].rolling(window=period).max()
        lowest = df["low"].rolling(window=period).min()
        range_col = f"minmax_range_{period}"
        df[range_col] = highest - lowest
        if use_relative:
            df[range_col] = df[range_col] / df["close"]

        num = abs_diff.rolling(window=period - 1, min_periods=period - 1).sum()
        df[f"flat_strength_{period}"] = (num / df[range_col]).replace(
            [np.inf, -np.inf], np.nan
        )

    # ── Round-level proximity ─────────────────────────────────────────────────
    # 0 when price is exactly on a 100-pip boundary; 1 when at the mid-point
    df["level_100"] = (
        abs(np.mod(df["close"], 0.0100) - 0.0050) / 0.0050
    )

    # ── Stochastic K%, D%, diff ───────────────────────────────────────────────
    for period in stoch_list:
        for smooth in smooth_list:
            stoch = ta.stoch(  # type: ignore[attr-defined]
                high=df["high"],
                low=df["low"],
                close=df["close"],
                k=period,
                d=smooth,
            )
            if stoch is None:
                continue
            prefix = f"stoch_{period}_{smooth}"
            k_col = f"{prefix}_k"
            d_col = f"{prefix}_d"
            diff_col = f"diff_{prefix}"
            df[k_col] = stoch.iloc[:, 0]
            df[d_col] = stoch.iloc[:, 1]
            df[diff_col] = stoch.iloc[:, 1] - stoch.iloc[:, 0]
            if use_abs:
                df[k_col] = df[k_col].abs()
                df[d_col] = df[d_col].abs()
                df[diff_col] = df[diff_col].abs()

    # ── Session-time features ─────────────────────────────────────────────────
    if "datetime" in df.columns and session_hours_list:
        start_h = session_hours_list[0]

        # Session start date for each bar (handles midnight-crossing sessions)
        df["session_start_date"] = df["datetime"].dt.normalize() + pd.to_timedelta(
            start_h, unit="h"
        )
        df.loc[df["datetime"] < df["session_start_date"], "session_start_date"] -= (
            pd.Timedelta(days=1)
        )
        # Skip Sunday sessions — roll back to Friday
        df.loc[
            df["session_start_date"].dt.weekday == 6, "session_start_date"
        ] -= pd.Timedelta(days=2)

        t = df["datetime"].dt
        frac_hour = t.hour + t.minute / 60.0
        df["session_hour_float"] = (
            np.mod(frac_hour - start_h, 24.0) / len(session_hours_list)
        )
        df["minute"] = t.minute
        df["weekday"] = t.weekday
        df["session_weekday"] = df["session_start_date"].dt.weekday
        df["quarter"] = t.quarter

        # Business week-of-month (week 0 = first Mon or 1st if Mon–Fri)
        month_start_weekday = df["datetime"].apply(lambda x: x.replace(day=1).weekday())
        bws = pd.to_datetime(df["datetime"].apply(lambda x: x.replace(day=1)))
        weekend_mask = month_start_weekday >= 5
        days_to_add = (7 - month_start_weekday) % 7
        bws = bws.copy()
        bws.loc[weekend_mask] = bws.loc[weekend_mask] + pd.to_timedelta(
            days_to_add.loc[weekend_mask], unit="D"
        )
        df["monthweek_business"] = (
            (df["datetime"] - bws).dt.days // 7
        ).astype(int)

        categorical_cols = ["weekday", "session_weekday", "quarter", "monthweek_business"]
        df = pd.get_dummies(
            df, columns=categorical_cols, prefix=categorical_cols, prefix_sep="_"
        )

    return df
