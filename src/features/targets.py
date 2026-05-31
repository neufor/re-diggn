"""Target variable construction for FX ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_target_delta(
    prices: pd.Series,
    window: int,
    kind: str = "min",
) -> np.ndarray:
    """
    For each bar i, compute the min or max absolute price move over the
    next `window` bars.

    kind="min"  → min(|future_max - price_i|, |price_i - future_min|)
    kind="max"  → max(|future_max - price_i|, |price_i - future_min|)

    The last `window` rows are left as 0 (no future data available).
    """
    if kind not in ("min", "max"):
        raise ValueError("kind must be 'min' or 'max'")

    arr = prices.to_numpy(dtype=float, copy=True)
    delta = np.zeros(len(arr))
    agg = min if kind == "min" else max

    for i in range(len(arr) - window):
        future = arr[i + 1 : i + window + 1]
        delta[i] = agg(
            abs(future.max() - arr[i]),
            abs(arr[i] - future.min()),
        )
    return delta


def create_target_stat(
    df_input: pd.DataFrame,
    session_hours: list[int] | None = None,
    target_prefixes: tuple[str, ...] = (
        "target_delta_",
        "target_min_eps_",
        "target_max_eps_",
    ),
) -> pd.DataFrame:
    """
    Append look-back statistics for every target column whose name starts
    with one of `target_prefixes`.

    For each target column the following features are added (all computed
    from *previous* sessions only — no leakage):

    Session-level stats (broadcast to every row of a session):
      - prev_session_mean
      - prev_5_sessions_mean
      - prev_10_sessions_mean
      - prev_20_sessions_mean
      - session_5_ago_mean

    Same-minute-of-session stats (across sessions, same offset):
      - prev_session_same_time
      - session_5_ago_same_time
      - prev_5_sessions_same_time_mean
      - prev_10_sessions_same_time_mean
      - prev_15_sessions_same_time_mean
      - prev_20_sessions_same_time_mean

    Requires either a `session_start_date` column already present, or both
    `datetime` and `session_hours` so the column can be derived.
    """
    df = df_input.copy()

    target_cols = [
        c for c in df.columns
        if any(str(c).startswith(p) for p in target_prefixes)
    ]
    if not target_cols:
        return df

    # Build session_start_date if missing
    if "session_start_date" not in df.columns:
        if "datetime" not in df.columns:
            raise ValueError(
                "create_target_stat requires 'session_start_date' or 'datetime' column"
            )
        if not session_hours:
            raise ValueError(
                "create_target_stat requires session_hours when 'session_start_date' is absent"
            )
        start_h = session_hours[0]
        df["session_start_date"] = df["datetime"].dt.normalize() + pd.to_timedelta(
            start_h, unit="h"
        )
        df.loc[df["datetime"] < df["session_start_date"], "session_start_date"] -= pd.Timedelta(
            days=1
        )
        # Skip Sunday sessions — roll back to Friday
        df.loc[
            df["session_start_date"].dt.weekday == 6, "session_start_date"
        ] -= pd.Timedelta(days=2)

    if "datetime" not in df.columns:
        raise ValueError("create_target_stat requires 'datetime' column")

    session_time_min = (
        (df["datetime"] - df["session_start_date"]).dt.total_seconds() // 60
    ).astype("int64")
    session_ids = df["session_start_date"]

    for tcol in target_cols:
        per_session_mean = df.groupby("session_start_date")[tcol].mean().sort_index()
        shifted = per_session_mean.shift(1)

        df[f"{tcol}_stat_prev_session_mean"] = session_ids.map(shifted)
        df[f"{tcol}_stat_prev_5_sessions_mean"] = session_ids.map(
            shifted.rolling(5).mean()
        )
        df[f"{tcol}_stat_prev_10_sessions_mean"] = session_ids.map(
            shifted.rolling(10).mean()
        )
        df[f"{tcol}_stat_prev_20_sessions_mean"] = session_ids.map(
            shifted.rolling(20).mean()
        )
        df[f"{tcol}_stat_session_5_ago_mean"] = session_ids.map(
            per_session_mean.shift(5)
        )

        tmp = pd.DataFrame(
            {
                "_session_time_min": session_time_min,
                "_session_start_date": df["session_start_date"],
                "_target": df[tcol],
            },
            index=df.index,
        ).sort_values(["_session_time_min", "_session_start_date"])

        g = tmp.groupby("_session_time_min", sort=False)["_target"]
        tmp["_prev_same"] = g.shift(1)
        tmp["_lag5_same"] = g.shift(5)
        tmp["_mean5_same"] = g.transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        tmp["_mean10_same"] = g.transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean()
        )
        tmp["_mean15_same"] = g.transform(
            lambda s: s.shift(1).rolling(15, min_periods=1).mean()
        )
        tmp["_mean20_same"] = g.transform(
            lambda s: s.shift(1).rolling(20, min_periods=1).mean()
        )

        aligned = tmp.sort_index()
        df[f"{tcol}_stat_prev_session_same_time"] = aligned["_prev_same"]
        df[f"{tcol}_stat_session_5_ago_same_time"] = aligned["_lag5_same"]
        df[f"{tcol}_stat_prev_5_sessions_same_time_mean"] = aligned["_mean5_same"]
        df[f"{tcol}_stat_prev_10_sessions_same_time_mean"] = aligned["_mean10_same"]
        df[f"{tcol}_stat_prev_15_sessions_same_time_mean"] = aligned["_mean15_same"]
        df[f"{tcol}_stat_prev_20_sessions_same_time_mean"] = aligned["_mean20_same"]

    return df.sort_index()
