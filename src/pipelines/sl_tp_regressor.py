#!/usr/bin/env python3
"""
SL/TP Regression pipeline: predict per-entry optimal stop-loss and take-profit
to maximise combined BUY+SELL PnL within a 480-bar window.

Strategy
--------
- Entry    : open BUY + SELL simultaneously at each 21:00 UTC session bar.
- Window   : T = 480 M1 bars (8 h).
- Target   : per-entry optimal (sl, tp) found by exhaustive grid search in
             hindsight (maximises buy_pnl + sell_pnl for that specific trade).
- Features : situational indicators (vol, ATR, BB, RSI, MACD, ADX) computed
             from the 120 bars preceding the entry bar.  No look-ahead bias.
- Model    : two independent LightGBM regressors (sl, tp).  Same hyperparams,
             tuned jointly by an Optuna study whose objective is *simulated
             PnL* on the TimeSeriesSplit validation folds (not MSE).

Output
------
  models/sl_tp_regressor/
    metrics.json           -- best params + train/test PnL overview
    optuna_trials.csv      -- full Optuna landscape
    model_sl.pkl           -- trained SL LightGBM regressor
    model_tp.pkl           -- trained TP LightGBM regressor
    predictions_test.csv   -- per-entry test predictions vs. oracle targets
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tqdm.auto import tqdm

from typing import Literal

from src.features import generate_indicators, generate_situational_features

FeatureSet = Literal["situational", "indicators"]

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH         = Path("data/processed/eurusd_m1.parquet")
ORACLE_CACHE_PATH = Path("data/processed/oracle_targets_sl_tp.parquet")
ORACLE_META_PATH  = Path("data/processed/oracle_targets_sl_tp_meta.json")

# Switch between "situational" (situ_ features) and "indicators" (SMA/Stoch/MinMax).
# RESULTS_DIR is derived automatically so both model variants stay separate.
FEATURE_SET: FeatureSet = "situational"
RESULTS_DIR = Path(f"models/sl_tp_regressor_{FEATURE_SET}")

SESSION_HOURS = [21, 22, 23, 0, 1, 2, 3]   # UTC hours; entry taken at HH:00 on weekdays
MAX_BARS      = 480
MT5_POINT        = 0.00001
PIP              = 0.0001

SL_MIN, SL_MAX = 10, 30
TP_MIN, TP_MAX = 4, 16

FEATURE_LOOKBACK  = 120   # bars of pre-entry history used by situ_ features
N_CV_SPLITS       = 5
N_TRIALS          = 100
SEED              = 42
# Oracle grid parallelism: entries per worker chunk.
# Smaller → less RAM per chunk, more chunks, finer progress.
# ~1 000 entries × 480 bars × 4 intermediates ≈ 40 MB per worker.
ORACLE_CHUNK_SIZE = 1000
TEST_FRAC        = 0.20

# Phase-2 device: "gpu" (OpenCL, works with standard pip wheel on Windows/Linux)
#                 "cuda" (requires LightGBM built with -DUSE_CUDA=1, Linux only)
#                 "cpu"
LGBM_DEVICE    = "gpu"
LGBM_DEVICE_ID = 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_m1() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.lower()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=False)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("Etc/GMT-3")
    df.index = df.index.tz_convert("UTC")
    df = df.sort_index().loc["2005":"2025"]
    return df


# ---------------------------------------------------------------------------
# Entry positions
# ---------------------------------------------------------------------------

def session_entry_positions(df: pd.DataFrame, hours: list[int] = SESSION_HOURS) -> np.ndarray:
    """Return sorted integer positions of all HH:00 bars for every hour in ``hours``."""
    idx = pd.DatetimeIndex(df.index)
    masks = [(idx.hour == h) & (idx.dayofweek < 5) for h in hours]
    combined = np.zeros(len(idx), dtype=bool)
    for m in masks:
        combined |= m
    return np.sort(np.where(combined)[0])


# ---------------------------------------------------------------------------
# Forward-look matrices (same as session_baseline)
# ---------------------------------------------------------------------------

def build_forward_matrices(
    df: pd.DataFrame,
    entry_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (H, L, C_exit, entry_px, spread_pip) for each entry position."""
    n = len(entry_pos)
    N = len(df)

    highs   = df["high"].to_numpy(dtype=np.float32)
    lows    = df["low"].to_numpy(dtype=np.float32)
    closes  = df["close"].to_numpy(dtype=np.float32)
    spreads = df["spread"].to_numpy(dtype=np.float32) * MT5_POINT / PIP

    H      = np.empty((n, MAX_BARS), dtype=np.float32)
    L      = np.empty((n, MAX_BARS), dtype=np.float32)
    C_exit = np.empty(n, dtype=np.float32)
    entry_px   = closes[entry_pos]
    spread_pip = spreads[entry_pos]

    for i, pos in enumerate(entry_pos):
        start  = pos + 1
        end    = min(start + MAX_BARS, N)
        actual = end - start

        H[i, :actual] = highs[start:end]
        L[i, :actual] = lows[start:end]

        if actual < MAX_BARS:
            fill_h = highs[end - 1] if actual > 0 else entry_px[i]
            fill_l = lows[end - 1]  if actual > 0 else entry_px[i]
            H[i, actual:] = fill_h
            L[i, actual:] = fill_l

        C_exit[i] = closes[end - 1] if actual > 0 else entry_px[i]

    return H, L, C_exit, entry_px, spread_pip


# ---------------------------------------------------------------------------
# Trade simulation — fixed SL/TP for all entries (baseline / oracle eval)
# ---------------------------------------------------------------------------

def simulate(
    H: np.ndarray,
    L: np.ndarray,
    C_exit: np.ndarray,
    entry: np.ndarray,
    spread_pip: np.ndarray,
    sl_pips: int,
    tp_pips: int,
) -> np.ndarray:
    """Vectorised BUY+SELL. Returns (n, 2) pnl array (net of spread)."""
    tp_pts = np.float32(tp_pips * PIP)
    sl_pts = np.float32(sl_pips * PIP)
    e = entry[:, None]

    buy_tp_mask = H >= e + tp_pts
    buy_sl_mask = L <= e - sl_pts
    buy_tp_any  = buy_tp_mask.any(axis=1)
    buy_sl_any  = buy_sl_mask.any(axis=1)
    buy_tp_idx  = np.where(buy_tp_any,  np.argmax(buy_tp_mask, axis=1), MAX_BARS)
    buy_sl_idx  = np.where(buy_sl_any,  np.argmax(buy_sl_mask, axis=1), MAX_BARS)

    buy_pnl = np.where(
        ~buy_tp_any & ~buy_sl_any,
        (C_exit - entry) / PIP,
        np.where(buy_tp_idx < buy_sl_idx, float(tp_pips), float(-sl_pips)),
    ).astype(np.float32) - spread_pip

    sell_tp_mask = L <= e - tp_pts
    sell_sl_mask = H >= e + sl_pts
    sell_tp_any  = sell_tp_mask.any(axis=1)
    sell_sl_any  = sell_sl_mask.any(axis=1)
    sell_tp_idx  = np.where(sell_tp_any, np.argmax(sell_tp_mask, axis=1), MAX_BARS)
    sell_sl_idx  = np.where(sell_sl_any, np.argmax(sell_sl_mask, axis=1), MAX_BARS)

    sell_pnl = np.where(
        ~sell_tp_any & ~sell_sl_any,
        (entry - C_exit) / PIP,
        np.where(sell_tp_idx < sell_sl_idx, float(tp_pips), float(-sl_pips)),
    ).astype(np.float32) - spread_pip

    return np.column_stack([buy_pnl, sell_pnl])


# ---------------------------------------------------------------------------
# Trade simulation — per-entry SL/TP arrays (model eval)
# ---------------------------------------------------------------------------

def simulate_per_trade(
    H: np.ndarray,
    L: np.ndarray,
    C_exit: np.ndarray,
    entry: np.ndarray,
    spread_pip: np.ndarray,
    sl_arr: np.ndarray,   # int array, shape (n,)
    tp_arr: np.ndarray,   # int array, shape (n,)
) -> np.ndarray:
    """
    Vectorised BUY+SELL where each entry has its own SL/TP.
    Returns (n, 2) pnl array (net of spread).
    """
    sl_arr = sl_arr.astype(np.float32)
    tp_arr = tp_arr.astype(np.float32)
    tp_pts = (tp_arr * PIP).astype(np.float32)[:, None]
    sl_pts = (sl_arr * PIP).astype(np.float32)[:, None]
    e = entry[:, None]

    buy_tp_mask = H >= e + tp_pts
    buy_sl_mask = L <= e - sl_pts
    buy_tp_any  = buy_tp_mask.any(axis=1)
    buy_sl_any  = buy_sl_mask.any(axis=1)
    buy_tp_idx  = np.where(buy_tp_any,  np.argmax(buy_tp_mask, axis=1), MAX_BARS)
    buy_sl_idx  = np.where(buy_sl_any,  np.argmax(buy_sl_mask, axis=1), MAX_BARS)

    buy_pnl = np.where(
        ~buy_tp_any & ~buy_sl_any,
        (C_exit - entry) / PIP,
        np.where(buy_tp_idx < buy_sl_idx, tp_arr, -sl_arr),
    ).astype(np.float32) - spread_pip

    sell_tp_mask = L <= e - tp_pts
    sell_sl_mask = H >= e + sl_pts
    sell_tp_any  = sell_tp_mask.any(axis=1)
    sell_sl_any  = sell_sl_mask.any(axis=1)
    sell_tp_idx  = np.where(sell_tp_any, np.argmax(sell_tp_mask, axis=1), MAX_BARS)
    sell_sl_idx  = np.where(sell_sl_any, np.argmax(sell_sl_mask, axis=1), MAX_BARS)

    sell_pnl = np.where(
        ~sell_tp_any & ~sell_sl_any,
        (entry - C_exit) / PIP,
        np.where(sell_tp_idx < sell_sl_idx, tp_arr, -sl_arr),
    ).astype(np.float32) - spread_pip

    return np.column_stack([buy_pnl, sell_pnl])


# ---------------------------------------------------------------------------
# Oracle: per-entry optimal SL/TP via exhaustive grid search
# ---------------------------------------------------------------------------

def _oracle_chunk_worker(
    H_c: np.ndarray,
    L_c: np.ndarray,
    C_c: np.ndarray,
    E_c: np.ndarray,
    SP_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full (sl, tp) grid search for a chunk of entries.

    Each worker receives a small slice of the entry arrays so intermediate
    boolean masks are proportional to chunk_size × MAX_BARS, not n × MAX_BARS.
    This keeps peak RAM constant regardless of core count.
    """
    nc    = len(E_c)
    b_sl  = np.full(nc, SL_MIN, dtype=np.int16)
    b_tp  = np.full(nc, TP_MIN, dtype=np.int16)
    b_pnl = np.full(nc, -np.inf, dtype=np.float32)
    for sl in range(SL_MIN, SL_MAX + 1):
        for tp in range(TP_MIN, TP_MAX + 1):
            pnls     = simulate(H_c, L_c, C_c, E_c, SP_c, sl, tp)
            combined = pnls.sum(axis=1)
            better   = combined > b_pnl
            b_pnl[better] = combined[better]
            b_sl[better]  = sl
            b_tp[better]  = tp
    return b_sl, b_tp, b_pnl


def find_optimal_sl_tp(
    H: np.ndarray,
    L: np.ndarray,
    C_exit: np.ndarray,
    entry_px: np.ndarray,
    spread_pip: np.ndarray,
    chunk_size: int = ORACLE_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each entry find the (sl, tp) pair that maximises buy_pnl + sell_pnl.

    Parallelised over entry chunks so peak RAM = n_jobs × chunk_size × MAX_BARS
    (constant regardless of total entries), not n_total × MAX_BARS × n_jobs.
    Returns (opt_sl, opt_tp, opt_pnl), each shape (n,).
    """
    from joblib import Parallel, delayed

    n      = len(entry_px)
    starts = list(range(0, n, chunk_size))

    best_sl  = np.empty(n, dtype=np.int16)
    best_tp  = np.empty(n, dtype=np.int16)
    best_pnl = np.empty(n, dtype=np.float32)

    with tqdm(total=len(starts), desc="Oracle grid", unit="chunk", dynamic_ncols=True) as pbar:
        def _run(s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            sl = slice(s, s + chunk_size)
            result = _oracle_chunk_worker(H[sl], L[sl], C_exit[sl], entry_px[sl], spread_pip[sl])
            pbar.update(1)
            return result

        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = Parallel(
            n_jobs=-1, prefer="threads"
        )(delayed(_run)(s) for s in starts)

    for i, (sl_arr, tp_arr, pnl_arr) in enumerate(chunks):
        s = starts[i]
        best_sl[s : s + chunk_size]  = sl_arr
        best_tp[s : s + chunk_size]  = tp_arr
        best_pnl[s : s + chunk_size] = pnl_arr

    return best_sl, best_tp, best_pnl


# ---------------------------------------------------------------------------
# Oracle cache — compute once for the full dataset, reuse for any subset
# ---------------------------------------------------------------------------

def _oracle_meta() -> dict:
    """Cache key: grid params only — independent of date range or subset size."""
    return {
        "sl_min":        SL_MIN,
        "sl_max":        SL_MAX,
        "tp_min":        TP_MIN,
        "tp_max":        TP_MAX,
        "max_bars":      MAX_BARS,
        "session_hours": SESSION_HOURS,
    }


def load_or_compute_oracle_targets(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Return oracle targets for every session-hour entry in ``df_full``.

    The cache is keyed only on grid parameters (SL/TP ranges, MAX_BARS,
    SESSION_HOURS).  Changing the analysis time window does NOT invalidate
    the cache — callers filter the returned DataFrame by timestamp.

    Returns a DataFrame with columns: timestamp, oracle_sl, oracle_tp, oracle_pnl.
    Only recomputes when SL_MIN/MAX, TP_MIN/MAX, MAX_BARS, or SESSION_HOURS change.
    """
    current_meta = _oracle_meta()

    if ORACLE_CACHE_PATH.exists() and ORACLE_META_PATH.exists():
        with open(ORACLE_META_PATH) as f:
            cached_meta = json.load(f)
        if cached_meta == current_meta:
            print("  Oracle targets loaded from cache.")
            return pd.read_parquet(ORACLE_CACHE_PATH)
        print("  Oracle cache: grid params changed — recomputing full dataset.")

    all_pos = session_entry_positions(df_full, SESSION_HOURS)
    n_combos = (SL_MAX - SL_MIN + 1) * (TP_MAX - TP_MIN + 1)
    print(
        f"  Running oracle grid search: {len(all_pos):,} entries × "
        f"{n_combos:,} combos ({SL_MIN}–{SL_MAX} × {TP_MIN}–{TP_MAX} pips)…"
    )
    H, L, C_exit, entry_px, spread_pip = build_forward_matrices(df_full, all_pos)
    opt_sl, opt_tp, opt_pnl = find_optimal_sl_tp(H, L, C_exit, entry_px, spread_pip)

    all_ts = pd.DatetimeIndex(df_full.index[all_pos])
    result = pd.DataFrame(
        {
            "timestamp":  all_ts,
            "oracle_sl":  opt_sl.astype(np.int16),
            "oracle_tp":  opt_tp.astype(np.int16),
            "oracle_pnl": opt_pnl.astype(np.float32),
        }
    )
    ORACLE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(ORACLE_CACHE_PATH, index=False)
    with open(ORACLE_META_PATH, "w") as f:
        json.dump(current_meta, f, indent=2)
    print(f"  Oracle targets saved → {ORACLE_CACHE_PATH}  ({len(result):,} rows)")
    return result


# ---------------------------------------------------------------------------
# Feature matrix — two backends, one dispatch function
# ---------------------------------------------------------------------------

# Columns that exist in the raw dataframe and must never be used as features.
_RAW_COLS = {"open", "high", "low", "close", "spread", "volume",
             "tick_volume", "real_volume", "datetime", "session_start_date"}


def _entry_scalars(df: pd.DataFrame, entry_pos: np.ndarray, spread_pip: np.ndarray) -> pd.DataFrame:
    """Scalar features that are always appended regardless of feature set."""
    idx = pd.DatetimeIndex(df.index[entry_pos])
    return pd.DataFrame(
        {
            "spread_pip":     spread_pip,
            "entry_hour_utc": idx.hour.astype(np.float32),
            "entry_dow":      idx.dayofweek.astype(np.float32),
            "entry_month":    idx.month.astype(np.float32),
        },
        index=np.arange(len(entry_pos)),
    )


def _build_situational(
    df: pd.DataFrame,
    entry_pos: np.ndarray,
    spread_pip: np.ndarray,
) -> pd.DataFrame:
    df_feat = df.copy()
    if "datetime" not in df_feat.columns:
        df_feat["datetime"] = df_feat.index.tz_convert(None)

    print("  [situational] computing situ_ features on full dataframe...")
    df_feat = generate_situational_features(
        df_feat,
        windows=(5, 15, 30, 60, 120),
        atr_periods=(14, 60),
        rsi_periods=(7, 14),
        bollinger_periods=(20, 60),
        macd_params=(12, 26, 9),
        adx_periods=(14, 30),
        add_session_calm=False,
        progress=True,
    )

    feat_cols = [c for c in df_feat.columns if c.startswith("situ_")]
    X = df_feat.iloc[entry_pos][feat_cols].copy()
    X.index = np.arange(len(entry_pos))
    return pd.concat([X, _entry_scalars(df, entry_pos, spread_pip)], axis=1)


def _build_indicators(
    df: pd.DataFrame,
    entry_pos: np.ndarray,
    spread_pip: np.ndarray,
) -> pd.DataFrame:
    df_feat = df.copy()
    if "datetime" not in df_feat.columns:
        df_feat["datetime"] = df_feat.index.tz_convert(None)

    print("  [indicators] computing SMA/MinMax/Stoch features on full dataframe...")
    df_feat = generate_indicators(
        df_feat,
        sma_periods=[10, 20, 50, 100, 200],
        minmax_periods=[10, 30, 60, 120, 240],
        stoch_periods=[14, 28],
        stoch_smooth=[3, 5],
        session_hours=SESSION_HOURS,
        use_abs=False,
        use_relative=True,
    )

    feat_cols = [c for c in df_feat.columns if c not in _RAW_COLS]
    X = df_feat.iloc[entry_pos][feat_cols].copy()
    # one-hot bool columns → float32
    X = X.astype(np.float32)
    X.index = np.arange(len(entry_pos))
    return pd.concat([X, _entry_scalars(df, entry_pos, spread_pip)], axis=1)


def build_feature_matrix(
    df: pd.DataFrame,
    entry_pos: np.ndarray,
    spread_pip: np.ndarray,
    feature_set: FeatureSet = FEATURE_SET,
) -> pd.DataFrame:
    """
    Build the feature matrix for the given entry positions.

    ``feature_set`` selects the backend:
      "situational"  — situ_ volatility / momentum / microstructure features
      "indicators"   — SMA, MinMax, Stochastic, session-time one-hot features

    All variants append: spread_pip, entry_hour_utc, entry_dow, entry_month.
    Features are computed on the full dataframe (no look-ahead) and sliced at
    each entry position.
    """
    if feature_set == "situational":
        return _build_situational(df, entry_pos, spread_pip).astype(np.float32)
    elif feature_set == "indicators":
        return _build_indicators(df, entry_pos, spread_pip).astype(np.float32)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")


# ---------------------------------------------------------------------------
# Optuna objective — simulated PnL on val folds
# ---------------------------------------------------------------------------

def _clip_predictions(pred: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.clip(np.round(pred).astype(int), lo, hi)


def make_lgbm_objective(
    X_tr: np.ndarray,
    y_sl_tr: np.ndarray,
    y_tp_tr: np.ndarray,
    H_tr: np.ndarray,
    L_tr: np.ndarray,
    C_tr: np.ndarray,
    E_tr: np.ndarray,
    SP_tr: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
):
    """
    Optuna objective.  For each trial:
      1. Train two LGBMs (SL, TP) on each CV train fold.
      2. Predict on the val fold, round to integer, clip to valid range.
      3. Simulate BUY+SELL with per-entry predicted SL/TP.
      4. Return mean val-fold total PnL.
    """
    def objective(trial: optuna.Trial) -> float:
        params: dict = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "device":            LGBM_DEVICE,
            "gpu_device_id":     LGBM_DEVICE_ID,
            "random_state": SEED,
            # n_jobs=1 on GPU (avoids LightGBM warning); use all cores on CPU
            "n_jobs": 1 if LGBM_DEVICE != "cpu" else -1,
            "verbose": -1,
        }

        fold_pnls: list[float] = []
        for tr_idx, vi_idx in cv_splits:
            m_sl = lgb.LGBMRegressor(**params)
            m_tp = lgb.LGBMRegressor(**params)
            m_sl.fit(X_tr[tr_idx], y_sl_tr[tr_idx])
            m_tp.fit(X_tr[tr_idx], y_tp_tr[tr_idx])

            pred_sl = _clip_predictions(m_sl.predict(X_tr[vi_idx]), SL_MIN, SL_MAX)
            pred_tp = _clip_predictions(m_tp.predict(X_tr[vi_idx]), TP_MIN, TP_MAX)

            pnl = simulate_per_trade(
                H_tr[vi_idx], L_tr[vi_idx], C_tr[vi_idx],
                E_tr[vi_idx], SP_tr[vi_idx],
                pred_sl, pred_tp,
            )
            fold_pnls.append(float(pnl.sum()))

        return float(np.mean(fold_pnls))

    return objective


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _stats(pnls: np.ndarray) -> dict:
    total = float(pnls.sum())
    n     = pnls.size
    gp    = float(pnls[pnls > 0].sum())
    gl    = float(abs(pnls[pnls < 0].sum()))
    return {
        "sessions":      len(pnls),
        "trades":        n,
        "net_pips":      round(total, 1),
        "buy_pips":      round(float(pnls[:, 0].sum()), 1),
        "sell_pips":     round(float(pnls[:, 1].sum()), 1),
        "win_rate_pct":  round(float((pnls > 0).mean()) * 100, 1),
        "avg_pnl_pip":   round(total / n if n else 0, 2),
        "profit_factor": round(gp / gl, 3) if gl > 0 else float("inf"),
    }


def _print_overview(label: str, pnls: np.ndarray, period: str) -> dict:
    s = _stats(pnls)
    sep = "=" * 56
    print(f"\n{sep}\n  {label}\n{sep}")
    print(f"  Period        : {period}")
    print(f"  Sessions      : {s['sessions']:,}  |  Trades: {s['trades']:,}")
    print(f"  Net profit    : {s['net_pips']:>10,.1f} pips")
    print(f"    BUY         : {s['buy_pips']:>10,.1f} pips")
    print(f"    SELL        : {s['sell_pips']:>10,.1f} pips")
    print(f"  Win rate      : {s['win_rate_pct']:.1f}%")
    print(f"  Avg P&L/trade : {s['avg_pnl_pip']:>9.2f} pips")
    print(f"  Profit factor : {s['profit_factor']:.3f}")
    print(sep)
    return {"period": period, **s}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Feature set  : {FEATURE_SET!r}  →  saving to {RESULTS_DIR}")
    print("Loading EURUSD M1 data...")
    df = load_m1()
    print(f"  {len(df):,} bars | {df.index[0].date()} → {df.index[-1].date()}")

    # Session entry positions — all SESSION_HOURS, weekdays only, chronological
    pos = session_entry_positions(df)
    print(f"  {len(pos):,} session entries across hours {SESSION_HOURS} UTC")

    # Chronological 80/20 split
    split   = int(len(pos) * (1.0 - TEST_FRAC))
    tr_pos  = pos[:split]
    te_pos  = pos[split:]
    tr_dates = f"{df.index[tr_pos[0]].date()} → {df.index[tr_pos[-1]].date()}"
    te_dates = f"{df.index[te_pos[0]].date()} → {df.index[te_pos[-1]].date()}"
    print(f"  Train: {len(tr_pos):,}  ({tr_dates})")
    print(f"  Test : {len(te_pos):,}   ({te_dates})")

    # Forward-look matrices
    print("\nBuilding forward-look matrices...")
    H_all, L_all, C_all, E_all, SP_all = build_forward_matrices(df, pos)
    H_tr, L_tr, C_tr, E_tr, SP_tr = H_all[:split], L_all[:split], C_all[:split], E_all[:split], SP_all[:split]
    H_te, L_te, C_te, E_te, SP_te = H_all[split:], L_all[split:], C_all[split:], E_all[split:], SP_all[split:]

    # Feature matrices (no look-ahead: situ_ features at entry bar)
    print("\nBuilding feature matrices...")
    X_tr_df = build_feature_matrix(df, tr_pos, SP_tr, FEATURE_SET)
    X_te_df = build_feature_matrix(df, te_pos, SP_te, FEATURE_SET)

    feature_names = list(X_tr_df.columns)
    X_tr = X_tr_df.to_numpy(dtype=np.float32)
    X_te = X_te_df.to_numpy(dtype=np.float32)
    print(f"  Feature count: {len(feature_names)}")

    # Oracle targets — loaded from parquet cache (computed once for the full dataset)
    print(f"\nOracle targets (grid {SL_MIN}–{SL_MAX} × {TP_MIN}–{TP_MAX} pips)…")
    oracle_df  = load_or_compute_oracle_targets(df)
    entry_ts   = pd.DatetimeIndex(df.index[pos])
    oracle_sub = oracle_df.set_index("timestamp").loc[entry_ts]
    y_sl_all   = oracle_sub["oracle_sl"].to_numpy(dtype=np.int16)
    y_tp_all   = oracle_sub["oracle_tp"].to_numpy(dtype=np.int16)
    pnl_all    = oracle_sub["oracle_pnl"].to_numpy(dtype=np.float32)
    y_sl_tr, y_tp_tr, oracle_pnl_tr = y_sl_all[:split], y_tp_all[:split], pnl_all[:split]
    y_sl_te, y_tp_te, oracle_pnl_te = y_sl_all[split:], y_tp_all[split:], pnl_all[split:]
    print(f"  Train oracle avg combined PnL : {oracle_pnl_tr.mean():.2f} pips/trade")
    print(f"  Test  oracle avg combined PnL : {oracle_pnl_te.mean():.2f} pips/trade")
    print(f"  Train SL — mean={y_sl_tr.mean():.1f}  std={y_sl_tr.std():.1f}")
    print(f"  Train TP — mean={y_tp_tr.mean():.1f}  std={y_tp_tr.std():.1f}")

    # TimeSeriesSplit
    tscv      = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_splits = list(tscv.split(np.arange(len(tr_pos))))

    # Optuna study — objective = simulated PnL on val folds
    print(f"\nOptuna hyperparameter search ({N_TRIALS} trials, {N_CV_SPLITS}-fold TSCV)")
    print("  Objective: mean val-fold simulated PnL with predicted SL/TP\n")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    pbar = tqdm(total=N_TRIALS, desc="Optuna", unit="trial", dynamic_ncols=True)

    def _update_pbar(s: optuna.Study, t: optuna.trial.FrozenTrial) -> None:
        best = s.best_value if s.best_trial else float("nan")
        pbar.set_postfix(
            cv=f"{t.value:+.0f}" if t.value is not None else "—",
            best=f"{best:+.0f}",
        )
        pbar.update(1)

    study.optimize(
        make_lgbm_objective(
            X_tr, y_sl_tr, y_tp_tr,
            H_tr, L_tr, C_tr, E_tr, SP_tr,
            cv_splits,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=False,
        callbacks=[_update_pbar],
    )
    pbar.close()

    best_params = study.best_params
    print(f"\nBest CV PnL : {study.best_value:,.1f} pips (mean over {N_CV_SPLITS} folds)")
    print(f"Best params : {best_params}")

    # Retrain on full training set with best hyperparams
    print("\nRetraining on full training set with best params...")
    final_params = {
        **best_params,
        "device":        LGBM_DEVICE,
        "gpu_device_id": LGBM_DEVICE_ID,
        "random_state":  SEED,
        "n_jobs":        1 if LGBM_DEVICE != "cpu" else -1,
        "verbose":       -1,
    }
    model_sl = lgb.LGBMRegressor(**final_params)
    model_tp = lgb.LGBMRegressor(**final_params)
    model_sl.fit(X_tr, y_sl_tr)
    model_tp.fit(X_tr, y_tp_tr)

    # Predictions on train + test
    pred_sl_tr = _clip_predictions(model_sl.predict(X_tr), SL_MIN, SL_MAX)
    pred_tp_tr = _clip_predictions(model_tp.predict(X_tr), TP_MIN, TP_MAX)
    pred_sl_te = _clip_predictions(model_sl.predict(X_te), SL_MIN, SL_MAX)
    pred_tp_te = _clip_predictions(model_tp.predict(X_te), TP_MIN, TP_MAX)

    # Simulate with predicted SL/TP
    pnl_tr = simulate_per_trade(H_tr, L_tr, C_tr, E_tr, SP_tr, pred_sl_tr, pred_tp_tr)
    pnl_te = simulate_per_trade(H_te, L_te, C_te, E_te, SP_te, pred_sl_te, pred_tp_te)

    # For comparison: simulate with session_baseline best fixed SL/TP
    # (load from metrics.json if available, else use round defaults)
    baseline_metrics_path = Path("models/session_baseline/metrics.json")
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            bl = json.load(f)
        bl_sl = int(bl["best_sl_pips"])
        bl_tp = int(bl["best_tp_pips"])
    else:
        bl_sl, bl_tp = 20, 30
    pnl_baseline_te = simulate(H_te, L_te, C_te, E_te, SP_te, bl_sl, bl_tp)

    # Overviews
    train_m    = _print_overview("TRAIN — model predicted SL/TP", pnl_tr,          tr_dates)
    test_m     = _print_overview("TEST  — model predicted SL/TP", pnl_te,          te_dates)
    baseline_m = _print_overview(
        f"TEST  — baseline fixed SL={bl_sl} TP={bl_tp}", pnl_baseline_te, te_dates
    )
    oracle_te_pnl_mat = simulate_per_trade(
        H_te, L_te, C_te, E_te, SP_te,
        y_sl_te.astype(int), y_tp_te.astype(int),
    )
    oracle_m = _print_overview("TEST  — oracle optimal SL/TP (ceiling)", oracle_te_pnl_mat, te_dates)

    # Feature importances
    imp_sl = pd.Series(model_sl.feature_importances_, index=feature_names).sort_values(ascending=False)
    imp_tp = pd.Series(model_tp.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop-10 SL features:")
    print(imp_sl.head(10).to_string())
    print("\nTop-10 TP features:")
    print(imp_tp.head(10).to_string())

    # Persist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "strategy":      "sl_tp_regressor",
        "feature_set":   FEATURE_SET,
        "session_hours": SESSION_HOURS,
        "max_bars":   MAX_BARS,
        "sl_range":   [SL_MIN, SL_MAX],
        "tp_range":   [TP_MIN, TP_MAX],
        "best_lgbm_params":  best_params,
        "cv_best_pnl_pips":  float(study.best_value),
        "oracle_train_avg_pnl_per_trade": float(oracle_pnl_tr.mean()),
        "oracle_test_avg_pnl_per_trade":  float(oracle_pnl_te.mean()),
        "train":    train_m,
        "test":     test_m,
        "baseline": baseline_m,
        "oracle":   oracle_m,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)

    with open(RESULTS_DIR / "model_sl.pkl", "wb") as f:
        pickle.dump(model_sl, f)
    with open(RESULTS_DIR / "model_tp.pkl", "wb") as f:
        pickle.dump(model_tp, f)

    # Per-entry test predictions CSV
    te_timestamps = df.index[te_pos]
    pred_df = pd.DataFrame(
        {
            "timestamp":  te_timestamps,
            "oracle_sl":  y_sl_te,
            "oracle_tp":  y_tp_te,
            "oracle_pnl": oracle_pnl_te,
            "pred_sl":    pred_sl_te,
            "pred_tp":    pred_tp_te,
            "pred_pnl":   pnl_te.sum(axis=1),
        }
    )
    pred_df.to_csv(RESULTS_DIR / "predictions_test.csv", index=False)

    imp_df = pd.DataFrame({"feature": feature_names,
                           "importance_sl": model_sl.feature_importances_,
                           "importance_tp": model_tp.feature_importances_})
    imp_df.to_csv(RESULTS_DIR / "feature_importances.csv", index=False)

    print(f"\nSaved → {RESULTS_DIR}/")
    print("  metrics.json           -- hyperparams + PnL overview vs oracle + baseline")
    print("  optuna_trials.csv      -- full Optuna landscape")
    print("  model_sl.pkl           -- trained SL LightGBM")
    print("  model_tp.pkl           -- trained TP LightGBM")
    print("  predictions_test.csv   -- per-entry test-set predictions")
    print("  feature_importances.csv")


if __name__ == "__main__":
    main()
