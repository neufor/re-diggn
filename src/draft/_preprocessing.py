import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit

from itertools import combinations
import pandas_ta_classic as ta

# For visualization
from matplotlib import pyplot as plt
import seaborn as sns
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_target_delta(prices, window, type="min"):
    """
    For each point i, calculate the min or max absolute change in the 'close' price
    over the next 'window' records - to make sure it gets it in both directions for min, and max for any direction
    """
    delta = np.zeros(len(prices))
    for i in range(len(prices) - window):
        future_prices = prices[i+1:i+window+1]
        if type == "min":
            delta[i] = min(abs(future_prices.max() - prices[i]), 
                           abs(prices[i] - future_prices.min()))
        elif type == "max":
            delta[i] = max(abs(future_prices.max() - prices[i]), 
                           abs(prices[i] - future_prices.min()))
        else:
            raise ValueError("type must be 'min' or 'max'")
    return delta

def generate_indicators(df_input, 
                       sma_periods=None,
                       minmax_periods=None,
                       stoch_periods=None,
                       stoch_smooth=None,
                       session_hours = None,
                       use_abs=False,
                       use_relative = False):
    """
    Generate a comprehensive set of technical indicators using pandas-ta.

    Built indicators:
    - SMA over all `sma_periods`
    - All pairwise differences Diff(SMA_i, SMA_j) where i < j
    - MinMax range (Highest-Lowest) over all `minmax_periods`
    - Stochastic K% and D% for all `stoch_periods` and `stoch_smooth`
    
    Parameters:
    -----------
    df_input : DataFrame with 'open', 'high', 'low', 'close' columns
    sma_periods : list[int] or None -> uses global `sma_periods` when None
    minmax_periods : list[int] or None -> uses global `minmax_periods` when None
    stoch_periods : list[int] or None -> uses global `stoch_periods` when None
    stoch_smooth : list[int] or None -> uses global `stoch_smooth` when None
    
    Returns:
    --------
    DataFrame with added indicator columns
    """
    df = df_input.copy()

    # Resolve parameter grids from globals if not provided
    sma_list = sma_periods if sma_periods is not None else []
    minmax_list = minmax_periods if minmax_periods is not None else []
    stoch_list = stoch_periods if stoch_periods is not None else []
    smooth_list = stoch_smooth if stoch_smooth is not None else []
    session_hours_list = session_hours if session_hours is not None else []

    # SMA indicators on close price
    for period in sorted(set(sma_list)):
        col_name = f'sma_{period}'
        df[col_name] = ta.sma(df['close'], length=period) # type: ignore

    # Pairwise SMA differences: Diff(SMA_i, SMA_j) for i < j
    sma_periods_sorted = sorted(set(sma_list))
    for p1, p2 in combinations(sma_periods_sorted, 2):
        df[f'diff_sma_{p1}_{p2}'] = df[f'sma_{p1}'] - df[f'sma_{p2}']
        if (use_abs):
            df[f'diff_sma_{p1}_{p2}'] = abs(df[f'diff_sma_{p1}_{p2}'])
        if (use_relative):
            df[f'diff_sma_{p1}_{p2}'] = df[f'diff_sma_{p1}_{p2}'] / df['close']

    # MinMax (Highest - Lowest) indicators
    for period in sorted(set(minmax_list)):
        highest = df['high'].rolling(window=period).max()
        lowest = df['low'].rolling(window=period).min()
        df[f'minmax_range_{period}'] = highest - lowest
        if (use_relative):
            df[f'minmax_range_{period}'] = df[f'minmax_range_{period}'] / df['close']
    
    # Flat strength indicators: sum of abs(close diffs) over (period-1) divided by MinMax range over period
    abs_diff = df['close'].diff().abs()
    for period in sorted(set(minmax_list)):
        denom_col = f'minmax_range_{period}'
        num = abs_diff.rolling(window=period - 1, min_periods=period - 1).sum()
        df[f'flat_strength_{period}'] = (num / df[denom_col]).replace([np.inf, -np.inf], np.nan)
        if (use_relative):
            df[f'flat_strength_{period}'] = df[f'flat_strength_{period}'] / df['close']

    # Hour position of the session for example for range [20,21,22,23,0,1,2,3,4,5]: 20:00 -> 0.0, 05:00 -> 1.0
    if ('datetime' in df.columns) & (len(session_hours_list) > 0):
        
        # Session start datetime at the first minute of the configured session
        session_start_hour = session_hours_list[0]
        df['session_start_date'] = df['datetime'].dt.normalize() + pd.to_timedelta(session_start_hour, unit='h')
        df.loc[df['datetime'] < df['session_start_date'], 'session_start_date'] -= pd.Timedelta(days=1)
        df.loc[df['session_start_date'].dt.weekday == 6, 'session_start_date'] -= pd.Timedelta(days=2)

        t = df['datetime'].dt
        frac_hour = t.hour + t.minute / 60.0
        df['session_hour_float'] = np.mod(frac_hour - session_hours_list[0], 24.0) / len(session_hours_list)
        
        df["minute"] = t.minute
        
        df["weekday"] = t.weekday
        df["session_weekday"] = df['session_start_date'].dt.weekday
        df["quarter"] = t.quarter
        
        # Business monthweek: adjust based on whether month starts on working day
        month_start_weekday = df['datetime'].apply(lambda x: x.replace(day=1).weekday())  # 0=Mon, 6=Sun
        # If month starts on Sat(5) or Sun(6), week 0 starts on first Monday
        # Otherwise, week 0 starts on 1st
        business_week_start = df['datetime'].apply(lambda x: x.replace(day=1))
        business_week_start = pd.to_datetime(business_week_start)
        
        # Adjust for weekends
        weekend_mask = month_start_weekday >= 5  # Sat or Sun
        days_to_add = (7 - month_start_weekday) % 7
        business_week_start.loc[weekend_mask] = business_week_start.loc[weekend_mask] + pd.to_timedelta(days_to_add.loc[weekend_mask], unit='D')
        
        # Calculate business week number
        days_since_business_start = (df['datetime'] - business_week_start).dt.days
        df["monthweek_business"] = (days_since_business_start // 7).astype(int)

        # One-hot encode selected categorical date/time features
        categorical_cols = ["weekday", "session_weekday", "quarter", "monthweek_business"]
        df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, prefix_sep="_")

    # Stochastic indicators (K% and D%)
    for period in sorted(set(stoch_list)):
        for smooth in sorted(set(smooth_list)):
            stoch = ta.stoch(high=df['high'], low=df['low'], close=df['close'],  # type: ignore
                                    k=period, d=smooth)
            if stoch is not None:
                col_prefix = f'stoch_{period}_{smooth}'
                df[f'{col_prefix}_k'] = stoch.iloc[:, 0]
                df[f'{col_prefix}_d'] = stoch.iloc[:, 1]
                df[f'diff_{col_prefix}'] = stoch.iloc[:, 1] - stoch.iloc[:, 0]
                if (use_abs):
                   df[f'{col_prefix}_k'] = abs(df[f'{col_prefix}_k'])
                   df[f'{col_prefix}_d'] = abs(df[f'{col_prefix}_d'])
                   df[f'diff_{col_prefix}'] = abs(df[f'diff_{col_prefix}'])
    
    df["level_100"] = abs(np.mod(df["close"], 0.0100) - 0.0050) / 0.0050 # if price is 1.2300 or 1.2400 - then it will be 1, if 1.2350 - then 0
    
    # Hourly price indicators (avg of last 5 minutes in each hour) - deltas relative to -8h
    # if 'datetime' in df.columns:
    #     df['_hour'] = df['datetime'].dt.floor('h')
    #     hourly = df.groupby('_hour')['close'].apply(lambda x: x.tail(5).mean()).reset_index()
    #     hourly.columns = ['_hour', 'hour_price']
        
    #     # Get base price (-8h)
    #     base = hourly.copy()
    #     base['_hour'] = base['_hour'] - pd.Timedelta(hours=-8)
    #     base = base.rename(columns={'hour_price': 'hour_price_-8h'})
    #     df = df.merge(base, on='_hour', how='left')
        
    #     # Add deltas for other shifts
    #     for shift in [-7, -6, -5, -4, -3, -2, -1]:
    #         shifted = hourly.copy()
    #         shifted['_hour'] = shifted['_hour'] - pd.Timedelta(hours=shift)
    #         shifted = shifted.rename(columns={'hour_price': f'delta_{shift}h'})
    #         df = df.merge(shifted, on='_hour', how='left')
    #         df[f'delta_{shift}h'] = df[f'delta_{shift}h'] - df['hour_price_-8h']
        
    #     # Current hour delta
    #     df = df.merge(hourly.rename(columns={'hour_price': 'delta_current'}), on='_hour', how='left')
    #     df['delta_current'] = df['delta_current'] - df['hour_price_-8h']
        
    #     # Drop temporary columns
    #     df = df.drop(['_hour', 'hour_price_-8h'], axis=1)
   
    # Drop NaN values introduced by longest-period indicators

    return df


def create_target_stat(
    df_input: pd.DataFrame,
    session_hours=None,
    target_prefixes=("target_delta_", "target_min_eps_", "target_max_eps_"),
):
    """
    Create time-series statistics features based on existing target columns.

    For every target column matching `target_prefixes`, adds:
    - avg target over previous session
    - avg target over previous 5 sessions
    - avg target in session 5 sessions ago
    - target in previous session at the same time (within session)
    - target in session 5 sessions ago at the same time (within session)
    - avg target over previous 5 sessions at the same time (within session)
    - avg target over previous 20 sessions at the same time (within session)

    Notes:
    - All features are computed using only *previous sessions* (via shifts) to avoid leakage.
    - Requires either `session_start_date` column, or `datetime` + `session_hours` to derive it.
    """
    df = df_input.copy()

    target_cols = [
        c for c in df.columns
        if any(str(c).startswith(p) for p in target_prefixes)
    ]
    if not target_cols:
        return df

    if "session_start_date" not in df.columns:
        if "datetime" not in df.columns:
            raise ValueError("create_target_stat requires 'session_start_date' or 'datetime' in df")
        if session_hours is None or len(session_hours) == 0:
            raise ValueError("create_target_stat requires session_hours when 'session_start_date' is missing")

        session_start_hour = session_hours[0]
        df["session_start_date"] = df["datetime"].dt.normalize() + pd.to_timedelta(session_start_hour, unit="h")
        df.loc[df["datetime"] < df["session_start_date"], "session_start_date"] -= pd.Timedelta(days=1)
        df.loc[df["session_start_date"].dt.weekday == 6, "session_start_date"] -= pd.Timedelta(days=2)

    if "datetime" not in df.columns:
        raise ValueError("create_target_stat requires 'datetime' column to compute same-time-in-session features")

    # Minutes from session start (robust across midnight sessions)
    session_time_min = ((df["datetime"] - df["session_start_date"]).dt.total_seconds() // 60).astype("int64")

    # Ensure deterministic ordering
    #df = df.sort_values(["session_start_date", "datetime"]).copy()
    session_ids = df["session_start_date"]

    for tcol in target_cols:
        # ---- Session-level stats (mapped back to all rows of a session)
        per_session_mean = df.groupby("session_start_date")[tcol].mean().sort_index()

        df[f"{tcol}_stat_prev_session_mean"] = session_ids.map(per_session_mean.shift(1))
        df[f"{tcol}_stat_prev_5_sessions_mean"] = session_ids.map(per_session_mean.shift(1).rolling(5).mean())
        df[f"{tcol}_stat_session_5_ago_mean"] = session_ids.map(per_session_mean.shift(5))
        df[f"{tcol}_stat_prev_20_sessions_mean"] = session_ids.map(per_session_mean.shift(1).rolling(20).mean())
        df[f"{tcol}_stat_prev_10_sessions_mean"] = session_ids.map(per_session_mean.shift(1).rolling(10).mean())

        # ---- Same-time-in-session stats (per session_time_min across sessions)
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
        tmp["_mean5_same"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        tmp["_mean20_same"] = g.transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
        tmp["_mean10_same"] = g.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        tmp["_mean15_same"] = g.transform(lambda s: s.shift(1).rolling(15, min_periods=1).mean())

        tmp_aligned = tmp.sort_index()
        df[f"{tcol}_stat_prev_session_same_time"] = tmp_aligned["_prev_same"]
        df[f"{tcol}_stat_session_5_ago_same_time"] = tmp_aligned["_lag5_same"]
        df[f"{tcol}_stat_prev_5_sessions_same_time_mean"] = tmp_aligned["_mean5_same"]
        df[f"{tcol}_stat_prev_20_sessions_same_time_mean"] = tmp_aligned["_mean20_same"]
        df[f"{tcol}_stat_prev_10_sessions_same_time_mean"] = tmp_aligned["_mean10_same"]
        df[f"{tcol}_stat_prev_15_sessions_same_time_mean"] = tmp_aligned["_mean15_same"]

    # Restore original ordering/index
    df = df.sort_index()
    return df


def generate_situational_features(
    df_input: pd.DataFrame,
    windows=(5, 15, 30, 60, 120),
    atr_periods=(14, 60, 120),
    rsi_periods=(7, 14, 28),
    bollinger_periods=(20, 60, 120),
    macd_params=(12, 26, 9),
    adx_periods=(14, 30, 60),
    session_hours=None,
    add_session_calm: bool = True,
    n_jobs: int | None = None,
    progress: bool = True,
):
    """
    Build situational, short-horizon features that can help forecast the
    maximum price move over the next ~8 hours (T=480) for calm sessions.

    The features focus on the current state and small rolling windows:
    - Minute returns and their short-window statistics (std, range, entropy proxy)
    - Realized volatility variants (std of returns, Parkinson, Garman–Klass)
    - ATR, Bollinger Bands width/position, Keltner-like width via ATR
    - Price position within recent range and vs. moving averages
    - Momentum/oscillators and their slopes: RSI, MACD, Stochastic (if present)
    - Trend strength: ADX, +DI, -DI and short-term slopes
    - Candlestick microstructure: body/upper/lower wicks, fraction of green candles
    - Direction change counts and run-length features over small windows
    - Optional calmness ratio using per-minute-of-session historical baseline

    Parameters
    -----------
    df_input : DataFrame with columns: 'open', 'high', 'low', 'close' (and optional 'datetime')
    windows : iterable[int] of small lookbacks in minutes
    atr_periods, rsi_periods, bollinger_periods, macd_params, adx_periods : indicator params
    session_hours : list[int] representing ordered trading hours of the session (optional)
    add_session_calm : if True and datetime available, add calmness metrics vs. same-time baseline

    Returns
    --------
    DataFrame with new columns prefixed by 'situ_'
    """
    df = df_input.copy()

    # Resolve parallelism
    if n_jobs is None:
        try:
            cpu = os.cpu_count() or 1
            n_jobs = max(1, min(8, cpu - 1))
        except Exception:
            n_jobs = 1

    # Progress helper
    class _ProgressDummy:
        def update(self, n=1):
            pass
        def close(self):
            pass

    def _tqdm(total: int, desc: str = ""):
        if not progress:
            return _ProgressDummy()
        try:
            from tqdm.auto import tqdm as _t
            return _t(total=total, desc=desc)
        except Exception:
            return _ProgressDummy()

    # Basic 1m returns and candle parts
    ret1 = df['close'].pct_change()
    df['situ_ret1m'] = ret1
    body = (df['close'] - df['open']).astype(float)
    upper_wick = (df['high'] - df[['close', 'open']].max(axis=1)).clip(lower=0.0)
    lower_wick = (df[['close', 'open']].min(axis=1) - df['low']).clip(lower=0.0)
    true_range = (df['high'] - df['low']).astype(float)
    df['situ_body'] = body
    df['situ_upper_wick'] = upper_wick
    df['situ_lower_wick'] = lower_wick
    df['situ_true_range'] = true_range

    # Precompute shared series used across windows
    sign_ret = pd.Series(np.sign(ret1.values), index=df.index)
    sign_change = (sign_ret != sign_ret.shift(1)).astype(float)
    hl_ratio = (df['high'] / df['low']).replace(0, np.nan)
    log_hl = pd.Series(np.log(hl_ratio.values), index=df.index).replace([np.inf, -np.inf], np.nan)
    log_hl2 = (log_hl ** 2)
    co_ratio = (df['close'] / df['open']).replace(0, np.nan)
    log_co = pd.Series(np.log(co_ratio.values), index=df.index).replace([np.inf, -np.inf], np.nan)

    def _compute_window_block(w_int: int):
        res: dict[str, pd.Series] = {}
        # Realized volatility (std of returns)
        res[f'situ_rv_std_{w_int}'] = ret1.rolling(w_int, min_periods=max(2, w_int // 3)).std()
        # Range-based measures
        roll_max = df['high'].rolling(w_int).max()
        roll_min = df['low'].rolling(w_int).min()
        roll_range = (roll_max - roll_min)
        res[f'situ_range_{w_int}'] = roll_range
        res[f'situ_pos_in_range_{w_int}'] = (df['close'] - roll_min) / roll_range.replace(0, np.nan)
        # Direction changes in last w
        res[f'situ_dir_changes_{w_int}'] = sign_change.rolling(w_int).sum()
        # Run-length proxies
        # Use vectorized rolling sum of absolute returns (much faster than Python lambda)
        res[f'situ_abs_ret_sum_{w_int}'] = ret1.abs().rolling(w_int).sum()
        res[f'situ_ret_sum_{w_int}'] = ret1.rolling(w_int).sum()
        # Skew/Kurt
        res[f'situ_ret_skew_{w_int}'] = ret1.rolling(w_int, min_periods=max(3, w_int // 2)).skew()
        res[f'situ_ret_kurt_{w_int}'] = ret1.rolling(w_int, min_periods=max(4, w_int // 2)).kurt()
        # Parkinson
        coef_p = 1.0 / (4.0 * np.log(2.0))
        res[f'situ_vol_parkinson_{w_int}'] = (coef_p * log_hl2.rolling(w_int).mean()).pow(0.5)
        # Garman–Klass
        var_gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        res[f'situ_vol_gk_{w_int}'] = var_gk.rolling(w_int).mean().clip(lower=0).pow(0.5)
        return res

    # Dispatch window computations (parallel if n_jobs>1)
    uniq_windows = sorted(set(int(w) for w in windows))
    tasks_total = len(uniq_windows)
    pbar = _tqdm(tasks_total, desc="situ:windows")
    if n_jobs and n_jobs > 1 and len(uniq_windows) > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_compute_window_block, w): w for w in uniq_windows}
            for fut in as_completed(futures):
                res = fut.result()
                for k, v in res.items():
                    df[k] = v
                pbar.update(1)
    else:
        for w in uniq_windows:
            res = _compute_window_block(w)
            for k, v in res.items():
                df[k] = v
            pbar.update(1)
    pbar.close()

    # ATR and Keltner-like channel width
    def _compute_atr_block(p: int):
        out = {}
        atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=int(p))  # type: ignore[attr-defined]
        if atr is not None:
            out[f'situ_atr_{int(p)}'] = atr
            for w in uniq_windows:
                roll_max = df['high'].rolling(int(w)).max()
                roll_min = df['low'].rolling(int(w)).min()
                width = (roll_max - roll_min)
                out[f'situ_keltner_like_width_{int(w)}_over_atr_{int(p)}'] = width / atr.replace(0, np.nan)
        return out

    uniq_atr = sorted(set(int(p) for p in atr_periods))
    pbar = _tqdm(len(uniq_atr), desc="situ:atr")
    if n_jobs and n_jobs > 1 and len(uniq_atr) > 1:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(uniq_atr))) as ex:
            futures = {ex.submit(_compute_atr_block, p): p for p in uniq_atr}
            for fut in as_completed(futures):
                res = fut.result()
                for k, v in res.items():
                    df[k] = v
                pbar.update(1)
    else:
        for p in uniq_atr:
            try:
                res = _compute_atr_block(p)
                for k, v in res.items():
                    df[k] = v
            except Exception:
                pass
            pbar.update(1)
    pbar.close()

    # Bollinger Bands width and position
    def _compute_bb_block(p: int):
        out = {}
        bb = ta.bbands(close=df['close'], length=int(p))  # type: ignore[attr-defined]
        if bb is not None and bb.shape[1] >= 3:
            mid = bb.iloc[:, 0]
            upper = bb.iloc[:, 1]
            lower = bb.iloc[:, 2]
            width = (upper - lower)
            out[f'situ_bb_width_{int(p)}'] = width
            out[f'situ_bb_pos_{int(p)}'] = (df['close'] - lower) / width.replace(0, np.nan)
            out[f'situ_price_vs_bbmid_{int(p)}'] = (df['close'] - mid)
        return out

    uniq_bb = sorted(set(int(p) for p in bollinger_periods))
    pbar = _tqdm(len(uniq_bb), desc="situ:bbands")
    if n_jobs and n_jobs > 1 and len(uniq_bb) > 1:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(uniq_bb))) as ex:
            futures = {ex.submit(_compute_bb_block, p): p for p in uniq_bb}
            for fut in as_completed(futures):
                res = fut.result()
                for k, v in res.items():
                    df[k] = v
                pbar.update(1)
    else:
        for p in uniq_bb:
            try:
                res = _compute_bb_block(p)
                for k, v in res.items():
                    df[k] = v
            except Exception:
                pass
            pbar.update(1)
    pbar.close()

    # RSI and its slope
    def _compute_rsi_block(p: int):
        out = {}
        rsi = ta.rsi(close=df['close'], length=int(p))  # type: ignore[attr-defined]
        if rsi is not None:
            out[f'situ_rsi_{int(p)}'] = rsi
            out[f'situ_rsi_slope_{int(p)}'] = rsi.diff()
        return out

    uniq_rsi = sorted(set(int(p) for p in rsi_periods))
    pbar = _tqdm(len(uniq_rsi), desc="situ:rsi")
    if n_jobs and n_jobs > 1 and len(uniq_rsi) > 1:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(uniq_rsi))) as ex:
            futures = {ex.submit(_compute_rsi_block, p): p for p in uniq_rsi}
            for fut in as_completed(futures):
                res = fut.result()
                for k, v in res.items():
                    df[k] = v
                pbar.update(1)
    else:
        for p in uniq_rsi:
            try:
                res = _compute_rsi_block(p)
                for k, v in res.items():
                    df[k] = v
            except Exception:
                pass
            pbar.update(1)
    pbar.close()

    # MACD and histogram slope
    try:
        pbar = _tqdm(1, desc="situ:macd")
        fast, slow, sig = macd_params
        macd = ta.macd(close=df['close'], fast=int(fast), slow=int(slow), signal=int(sig))  # type: ignore[attr-defined]
        if macd is not None and macd.shape[1] >= 3:
            macd_line = macd.iloc[:, 0]
            signal_line = macd.iloc[:, 1]
            hist = macd.iloc[:, 2]
            df['situ_macd'] = macd_line
            df['situ_macd_signal'] = signal_line
            df['situ_macd_hist'] = hist
            df['situ_macd_hist_slope'] = hist.diff()
        pbar.update(1)
        pbar.close()
    except Exception:
        pass

    # ADX, +DI, -DI and their short slopes
    def _compute_adx_block(p: int):
        out = {}
        adx_df = ta.adx(high=df['high'], low=df['low'], close=df['close'], length=int(p))  # type: ignore[attr-defined]
        if adx_df is not None and adx_df.shape[1] >= 3:
            dmp = adx_df.iloc[:, 0]
            dmn = adx_df.iloc[:, 1]
            adx_val = adx_df.iloc[:, 2]
            out[f'situ_di_plus_{int(p)}'] = dmp
            out[f'situ_di_minus_{int(p)}'] = dmn
            out[f'situ_adx_{int(p)}'] = adx_val
            out[f'situ_adx_slope_{int(p)}'] = adx_val.diff()
        return out

    uniq_adx = sorted(set(int(p) for p in adx_periods))
    pbar = _tqdm(len(uniq_adx), desc="situ:adx")
    if n_jobs and n_jobs > 1 and len(uniq_adx) > 1:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(uniq_adx))) as ex:
            futures = {ex.submit(_compute_adx_block, p): p for p in uniq_adx}
            for fut in as_completed(futures):
                res = fut.result()
                for k, v in res.items():
                    df[k] = v
                pbar.update(1)
    else:
        for p in uniq_adx:
            try:
                res = _compute_adx_block(p)
                for k, v in res.items():
                    df[k] = v
            except Exception:
                pass
            pbar.update(1)
    pbar.close()

    # Price vs. short/medium SMAs (offset relative to ATR if available)
    def _compute_sma_block(w: int):
        out = {}
        sma = ta.sma(df['close'], length=int(w))  # type: ignore
        out[f'situ_sma_{int(w)}'] = sma
        if f'situ_atr_{int(w)}' in df.columns:
            atrw = df[f'situ_atr_{int(w)}']
            out[f'situ_price_vs_sma_over_atr_{int(w)}'] = (df['close'] - sma) / atrw.replace(0, np.nan)
        else:
            out[f'situ_price_vs_sma_{int(w)}'] = (df['close'] - sma)
        return out

    pbar = _tqdm(len(uniq_windows), desc="situ:sma")
    if n_jobs and n_jobs > 1 and len(uniq_windows) > 1:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(uniq_windows))) as ex:
            futures = {ex.submit(_compute_sma_block, w): w for w in uniq_windows}
            for fut in as_completed(futures):
                res = fut.result()
                for k, v in res.items():
                    df[k] = v
                pbar.update(1)
    else:
        for w in uniq_windows:
            res = _compute_sma_block(w)
            for k, v in res.items():
                df[k] = v
            pbar.update(1)
    pbar.close()

    # Optional: calmness metrics vs. historical same-time baseline across sessions
    if add_session_calm and ('datetime' in df.columns):
        # Build session_start_date if needed
        if 'session_start_date' not in df.columns:
            if session_hours is None or len(session_hours) == 0:
                # Fallback: treat calendar day start as session start
                df['session_start_date'] = df['datetime'].dt.normalize()
            else:
                session_start_hour = session_hours[0]
                df['session_start_date'] = df['datetime'].dt.normalize() + pd.to_timedelta(session_start_hour, unit='h')
                df.loc[df['datetime'] < df['session_start_date'], 'session_start_date'] -= pd.Timedelta(days=1)
                df.loc[df['session_start_date'].dt.weekday == 6, 'session_start_date'] -= pd.Timedelta(days=2)

        # Minutes from session start
        session_time_min = ((df['datetime'] - df['session_start_date']).dt.total_seconds() // 60).astype('int64')
        # Realized vol over last 60 minutes
        rv60 = ret1.rolling(60, min_periods=20).std()
        df['situ_rv_std_60'] = df.get('situ_rv_std_60', rv60)

        tmp = pd.DataFrame({
            '_t': session_time_min,
            '_s': df['session_start_date'],
            '_rv60': rv60
        }, index=df.index).sort_values(['_t', '_s'])

        g = tmp.groupby('_t', sort=False)['_rv60']
        # historical baseline at same minute-of-session using previous sessions only
        tmp['_same_time_med20'] = g.transform(lambda s: s.shift(1).rolling(20, min_periods=3).median())
        tmp_aligned = tmp.sort_index()
        base = tmp_aligned['_same_time_med20']
        df['situ_calm_ratio'] = (rv60 / base.replace(0, np.nan))
        df['situ_calm_flag_070'] = (df['situ_calm_ratio'] < 0.70).astype('float')

    return df