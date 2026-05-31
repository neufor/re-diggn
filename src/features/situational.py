"""Short-horizon situational features for FX OHLCV data (prefix: ``situ_``)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


def generate_situational_features(
    df_input: pd.DataFrame,
    windows: tuple[int, ...] = (5, 15, 30, 60, 120),
    atr_periods: tuple[int, ...] = (14, 60, 120),
    rsi_periods: tuple[int, ...] = (7, 14, 28),
    bollinger_periods: tuple[int, ...] = (20, 60, 120),
    macd_params: tuple[int, int, int] = (12, 26, 9),
    adx_periods: tuple[int, ...] = (14, 30, 60),
    session_hours: list[int] | None = None,
    add_session_calm: bool = True,
    n_jobs: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Build situational, short-horizon features useful for forecasting the
    maximum price move over the next session (~8 h) in calm conditions.

    All new columns carry the ``situ_`` prefix.

    Feature groups
    --------------
    Bar microstructure
        ``situ_ret1m``, ``situ_body``, ``situ_upper_wick``,
        ``situ_lower_wick``, ``situ_true_range``

    Rolling-window stats (one set per window in ``windows``)
        Realized vol (std of returns), high-low range, price position in range,
        directional-change count, abs/signed return sums, return skew & kurtosis,
        Parkinson vol, Garman-Klass vol.

    ATR and Keltner-like channel width
        ``situ_atr_{p}`` and ``situ_keltner_like_width_{w}_over_atr_{p}``

    Bollinger Bands
        ``situ_bb_width_{p}``, ``situ_bb_pos_{p}``, ``situ_price_vs_bbmid_{p}``

    RSI and slope
        ``situ_rsi_{p}``, ``situ_rsi_slope_{p}``

    MACD
        ``situ_macd``, ``situ_macd_signal``, ``situ_macd_hist``,
        ``situ_macd_hist_slope``

    ADX, +DI, -DI and slope
        ``situ_di_plus_{p}``, ``situ_di_minus_{p}``, ``situ_adx_{p}``,
        ``situ_adx_slope_{p}``

    Price vs. SMA (scaled by ATR when available)
        ``situ_sma_{w}`` and ``situ_price_vs_sma_over_atr_{w}`` /
        ``situ_price_vs_sma_{w}``

    Session calmness (only when ``datetime`` is present and
    ``add_session_calm=True``)
        ``situ_calm_ratio`` — realized 60-min vol vs. 20-session median at the
        same minute-of-session.
        ``situ_calm_flag_070`` — 1 when calm_ratio < 0.70.

    Parameters
    ----------
    df_input : DataFrame
        Must have columns ``open``, ``high``, ``low``, ``close``.
    windows : tuple[int]
        Lookback windows (in bars) for rolling statistics.
    atr_periods, rsi_periods, bollinger_periods, adx_periods : tuple[int]
        Indicator look-back periods.
    macd_params : (fast, slow, signal)
    session_hours : list[int], optional
        Ordered trading hours of the session (UTC).  Required for calmness
        features if ``session_start_date`` is not already in the frame.
    add_session_calm : bool
    n_jobs : int, optional
        Worker threads for parallel indicator blocks.  Defaults to
        min(8, cpu_count − 1).
    progress : bool
        Show tqdm progress bars (requires tqdm; silently skipped if absent).

    Returns
    -------
    DataFrame with all ``situ_`` columns appended.
    """
    df = df_input.copy()

    if n_jobs is None:
        cpu = os.cpu_count() or 1
        n_jobs = max(1, min(8, cpu - 1))

    # ── progress bar helper ───────────────────────────────────────────────────
    class _DummyBar:
        def update(self, n: int = 1) -> None: ...
        def close(self) -> None: ...

    def _pbar(total: int, desc: str = "") -> object:
        if not progress:
            return _DummyBar()
        try:
            from tqdm.auto import tqdm  # type: ignore[import]
            return tqdm(total=total, desc=desc)
        except Exception:
            return _DummyBar()

    def _dispatch(fn, items, desc: str) -> None:
        """Run fn(item) for each item; collect results into df."""
        bar = _pbar(len(items), desc=desc)
        if n_jobs > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=min(n_jobs, len(items))) as ex:
                futures = {ex.submit(fn, item): item for item in items}
                for fut in as_completed(futures):
                    for k, v in fut.result().items():
                        df[k] = v
                    bar.update(1)  # type: ignore[union-attr]
        else:
            for item in items:
                try:
                    for k, v in fn(item).items():
                        df[k] = v
                except Exception:
                    pass
                bar.update(1)  # type: ignore[union-attr]
        bar.close()  # type: ignore[union-attr]

    # ── bar-level microstructure ──────────────────────────────────────────────
    ret1 = df["close"].pct_change()
    df["situ_ret1m"] = ret1
    df["situ_body"] = (df["close"] - df["open"]).astype(float)
    df["situ_upper_wick"] = (
        df["high"] - df[["close", "open"]].max(axis=1)
    ).clip(lower=0.0)
    df["situ_lower_wick"] = (
        df[["close", "open"]].min(axis=1) - df["low"]
    ).clip(lower=0.0)
    df["situ_true_range"] = (df["high"] - df["low"]).astype(float)

    # Pre-compute series shared across window blocks
    sign_ret = pd.Series(np.sign(ret1.values), index=df.index)
    sign_change = (sign_ret != sign_ret.shift(1)).astype(float)
    log_hl = (
        pd.Series(np.log((df["high"] / df["low"]).replace(0, np.nan).values), index=df.index)
        .replace([np.inf, -np.inf], np.nan)
    )
    log_co = (
        pd.Series(np.log((df["close"] / df["open"]).replace(0, np.nan).values), index=df.index)
        .replace([np.inf, -np.inf], np.nan)
    )
    log_hl2 = log_hl ** 2

    # ── rolling window blocks ─────────────────────────────────────────────────
    uniq_windows = sorted(set(int(w) for w in windows))

    def _window_block(w: int) -> dict[str, pd.Series]:
        coef_p = 1.0 / (4.0 * np.log(2.0))
        roll_max = df["high"].rolling(w).max()
        roll_min = df["low"].rolling(w).min()
        roll_range = roll_max - roll_min
        var_gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        return {
            f"situ_rv_std_{w}": ret1.rolling(w, min_periods=max(2, w // 3)).std(),
            f"situ_range_{w}": roll_range,
            f"situ_pos_in_range_{w}": (df["close"] - roll_min) / roll_range.replace(0, np.nan),
            f"situ_dir_changes_{w}": sign_change.rolling(w).sum(),
            f"situ_abs_ret_sum_{w}": ret1.abs().rolling(w).sum(),
            f"situ_ret_sum_{w}": ret1.rolling(w).sum(),
            f"situ_ret_skew_{w}": ret1.rolling(w, min_periods=max(3, w // 2)).skew(),
            f"situ_ret_kurt_{w}": ret1.rolling(w, min_periods=max(4, w // 2)).kurt(),
            f"situ_vol_parkinson_{w}": (coef_p * log_hl2.rolling(w).mean()).pow(0.5),
            f"situ_vol_gk_{w}": var_gk.rolling(w).mean().clip(lower=0).pow(0.5),
        }

    _dispatch(_window_block, uniq_windows, "situ:windows")

    # ── ATR ───────────────────────────────────────────────────────────────────
    def _atr_block(p: int) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        atr = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=p)  # type: ignore[attr-defined]
        if atr is None:
            return out
        out[f"situ_atr_{p}"] = atr
        for w in uniq_windows:
            width = df["high"].rolling(w).max() - df["low"].rolling(w).min()
            out[f"situ_keltner_like_width_{w}_over_atr_{p}"] = width / atr.replace(0, np.nan)
        return out

    _dispatch(_atr_block, sorted(set(int(p) for p in atr_periods)), "situ:atr")

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    def _bb_block(p: int) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        bb = ta.bbands(close=df["close"], length=p)  # type: ignore[attr-defined]
        if bb is None or bb.shape[1] < 3:
            return out
        mid, upper, lower = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        width = upper - lower
        out[f"situ_bb_width_{p}"] = width
        out[f"situ_bb_pos_{p}"] = (df["close"] - lower) / width.replace(0, np.nan)
        out[f"situ_price_vs_bbmid_{p}"] = df["close"] - mid
        return out

    _dispatch(_bb_block, sorted(set(int(p) for p in bollinger_periods)), "situ:bbands")

    # ── RSI ───────────────────────────────────────────────────────────────────
    def _rsi_block(p: int) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        rsi = ta.rsi(close=df["close"], length=p)  # type: ignore[attr-defined]
        if rsi is None:
            return out
        out[f"situ_rsi_{p}"] = rsi
        out[f"situ_rsi_slope_{p}"] = rsi.diff()
        return out

    _dispatch(_rsi_block, sorted(set(int(p) for p in rsi_periods)), "situ:rsi")

    # ── MACD ──────────────────────────────────────────────────────────────────
    try:
        fast, slow, sig = macd_params
        macd = ta.macd(close=df["close"], fast=int(fast), slow=int(slow), signal=int(sig))  # type: ignore[attr-defined]
        if macd is not None and macd.shape[1] >= 3:
            df["situ_macd"] = macd.iloc[:, 0]
            df["situ_macd_signal"] = macd.iloc[:, 1]
            df["situ_macd_hist"] = macd.iloc[:, 2]
            df["situ_macd_hist_slope"] = macd.iloc[:, 2].diff()
    except Exception:
        pass

    # ── ADX / DI ──────────────────────────────────────────────────────────────
    def _adx_block(p: int) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        adx_df = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=p)  # type: ignore[attr-defined]
        if adx_df is None or adx_df.shape[1] < 3:
            return out
        out[f"situ_di_plus_{p}"] = adx_df.iloc[:, 0]
        out[f"situ_di_minus_{p}"] = adx_df.iloc[:, 1]
        out[f"situ_adx_{p}"] = adx_df.iloc[:, 2]
        out[f"situ_adx_slope_{p}"] = adx_df.iloc[:, 2].diff()
        return out

    _dispatch(_adx_block, sorted(set(int(p) for p in adx_periods)), "situ:adx")

    # ── Price vs. SMA (ATR-scaled when available) ─────────────────────────────
    def _sma_block(w: int) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        sma = ta.sma(df["close"], length=w)  # type: ignore[attr-defined]
        out[f"situ_sma_{w}"] = sma
        atr_col = f"situ_atr_{w}"
        if atr_col in df.columns:
            out[f"situ_price_vs_sma_over_atr_{w}"] = (df["close"] - sma) / df[atr_col].replace(0, np.nan)
        else:
            out[f"situ_price_vs_sma_{w}"] = df["close"] - sma
        return out

    _dispatch(_sma_block, uniq_windows, "situ:sma")

    # ── Session calmness ──────────────────────────────────────────────────────
    if add_session_calm and "datetime" in df.columns:
        if "session_start_date" not in df.columns:
            if session_hours:
                start_h = session_hours[0]
                df["session_start_date"] = df["datetime"].dt.normalize() + pd.to_timedelta(
                    start_h, unit="h"
                )
                df.loc[df["datetime"] < df["session_start_date"], "session_start_date"] -= pd.Timedelta(days=1)
                df.loc[
                    df["session_start_date"].dt.weekday == 6, "session_start_date"
                ] -= pd.Timedelta(days=2)
            else:
                df["session_start_date"] = df["datetime"].dt.normalize()

        session_time_min = (
            (df["datetime"] - df["session_start_date"]).dt.total_seconds() // 60
        ).astype("int64")

        rv60 = ret1.rolling(60, min_periods=20).std()
        df["situ_rv_std_60"] = df.get("situ_rv_std_60", rv60)  # type: ignore[call-overload]

        tmp = pd.DataFrame(
            {"_t": session_time_min, "_s": df["session_start_date"], "_rv60": rv60},
            index=df.index,
        ).sort_values(["_t", "_s"])

        tmp["_same_time_med20"] = tmp.groupby("_t", sort=False)["_rv60"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=3).median()
        )
        base = tmp.sort_index()["_same_time_med20"]
        df["situ_calm_ratio"] = rv60 / base.replace(0, np.nan)
        df["situ_calm_flag_070"] = (df["situ_calm_ratio"] < 0.70).astype("float")

    return df
