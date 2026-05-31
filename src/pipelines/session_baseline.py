#!/usr/bin/env python3
"""
Baseline model: fixed SL/TP optimization over the 21:00 UTC session.

Strategy
--------
- Entry  : open BUY + SELL simultaneously at the close of the first bar
           of each UTC hour inside the session window (21:00-03:00 UTC),
           one entry per hour per weekday.
- Exit   : whichever comes first -- TP hit, SL hit, or 480-bar time-limit.
- Spread : deducted per trade at entry (spread column is in MT5 points;
           1 pip = 10 points for EURUSD 5-decimal pricing).
- Optuna : TPESampler, 200 trials, objective = mean fold net-profit in pips
           evaluated on the *validation* folds of a 5-fold TimeSeriesSplit
           using ONLY the 21:00 UTC entry (the canonical session open).
- Split  : first 80% of sessions = train+CV; last 20% = held-out test.

Output
------
  models/session_baseline/
    metrics.json          -- best params + train/test overview
    optuna_trials.csv     -- full Optuna landscape
    breakdown_by_hour.csv -- per-entry-hour stats (full date range)
    breakdown_by_dow.csv  -- per-day-of-week stats at 21:00 entry
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- constants ----------------------------------------------------------------
DATA_PATH   = Path("data/processed/eurusd_m1.parquet")
RESULTS_DIR = Path("models/session_baseline")

SESSION_UTC_HOUR = 21        # canonical entry hour (used for optimisation)
SESSION_HOURS    = [21, 22, 23, 0, 1, 2, 3]  # full window for breakdown
MAX_BARS         = 480       # order lifetime (1-min bars)
MT5_POINT        = 0.00001  # EURUSD 5-decimal point size
PIP              = 0.0001   # 1 pip = 10 MT5 points

SL_MIN, SL_MAX = 5, 50    # search range, pips
TP_MIN, TP_MAX = 5, 100   # search range, pips

N_CV_SPLITS = 5
N_TRIALS    = 200
SEED        = 42
TEST_FRAC   = 0.20

DOW_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


# -- data loading -------------------------------------------------------------

def load_m1() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.lower()

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=False)

    # Data is stored as +03:00 broker time; convert to UTC
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("Etc/GMT-3")
    df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    df = df.loc["2005":"2025"]
    return df


# -- session-entry positions --------------------------------------------------

def session_entry_positions(df: pd.DataFrame, hour: int = SESSION_UTC_HOUR) -> np.ndarray:
    """Integer positions of bars at `hour`:00 UTC on weekdays."""
    idx = df.index
    mask = (idx.hour == hour) & (idx.minute == 0) & (idx.dayofweek < 5)
    return np.where(mask)[0]


# -- forward-look matrices (precomputed once) ----------------------------------

def build_forward_matrices(
    df: pd.DataFrame,
    entry_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute fixed-width look-ahead windows for every session entry.

    Returns (H, L, C_exit, entry_px, spread_pip), each (n,) or (n, MAX_BARS).
    """
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


# -- vectorized trade simulation ----------------------------------------------

def simulate(
    H: np.ndarray,
    L: np.ndarray,
    C_exit: np.ndarray,
    entry: np.ndarray,
    spread_pip: np.ndarray,
    sl_pips: int,
    tp_pips: int,
) -> np.ndarray:
    """
    Fully vectorized BUY+SELL simulation.
    Returns (n, 2) pnl array: col 0 = BUY pips, col 1 = SELL pips (net of spread).
    Tie-break: if TP and SL triggered in the same bar, SL wins (conservative).
    """
    tp_pts = np.float32(tp_pips * PIP)
    sl_pts = np.float32(sl_pips * PIP)
    e = entry[:, None]

    # BUY
    buy_tp_mask = H >= e + tp_pts
    buy_sl_mask = L <= e - sl_pts
    buy_tp_any  = buy_tp_mask.any(axis=1)
    buy_sl_any  = buy_sl_mask.any(axis=1)
    buy_tp_idx  = np.where(buy_tp_any, np.argmax(buy_tp_mask, axis=1), MAX_BARS)
    buy_sl_idx  = np.where(buy_sl_any, np.argmax(buy_sl_mask, axis=1), MAX_BARS)

    buy_pnl = np.where(
        ~buy_tp_any & ~buy_sl_any,
        (C_exit - entry) / PIP,
        np.where(buy_tp_idx < buy_sl_idx, float(tp_pips), float(-sl_pips)),
    ).astype(np.float32) - spread_pip

    # SELL
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


# -- Optuna objective ---------------------------------------------------------

def make_objective(
    H_tr, L_tr, C_tr, E_tr, SP_tr,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
):
    def objective(trial: optuna.Trial) -> float:
        sl = trial.suggest_int("sl_pips", SL_MIN, SL_MAX)
        tp = trial.suggest_int("tp_pips", TP_MIN, TP_MAX)
        fold_profits = [
            float(simulate(H_tr[vi], L_tr[vi], C_tr[vi], E_tr[vi], SP_tr[vi], sl, tp).sum())
            for _, vi in cv_splits
        ]
        return float(np.mean(fold_profits))
    return objective


# -- reporting helpers --------------------------------------------------------

def compute_stats(pnls: np.ndarray) -> dict:
    total    = float(pnls.sum())
    n_trades = pnls.size
    win_rate = float((pnls > 0).mean())
    avg_pnl  = total / n_trades if n_trades else 0.0
    gp = float(pnls[pnls > 0].sum())
    gl = float(abs(pnls[pnls < 0].sum()))
    return {
        "sessions":         len(pnls),
        "trades":           n_trades,
        "net_pips":         round(total, 1),
        "buy_pips":         round(float(pnls[:, 0].sum()), 1),
        "sell_pips":        round(float(pnls[:, 1].sum()), 1),
        "win_rate_pct":     round(win_rate * 100, 1),
        "avg_pnl_pip":      round(avg_pnl, 2),
        "profit_factor":    round(gp / gl, 3) if gl > 0 else float("inf"),
    }


def print_overview(label: str, pnls: np.ndarray, period_start: str, period_end: str) -> dict:
    s = compute_stats(pnls)
    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")
    print(f"  Period        : {period_start} -> {period_end}")
    print(f"  Sessions      : {s['sessions']:,}  |  Trades: {s['trades']:,}")
    print(f"  Net profit    : {s['net_pips']:>10,.1f} pips")
    print(f"    BUY         : {s['buy_pips']:>10,.1f} pips")
    print(f"    SELL        : {s['sell_pips']:>10,.1f} pips")
    print(f"  Win rate      : {s['win_rate_pct']:.1f}%")
    print(f"  Avg P&L/trade : {s['avg_pnl_pip']:>9.2f} pips")
    print(f"  Profit factor : {s['profit_factor']:.3f}")
    print(f"{'='*54}")
    return {"period_start": period_start, "period_end": period_end, **s}


def group_stats(pnls: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Aggregate compute_stats per unique label value."""
    rows = []
    for lab in sorted(set(labels.tolist())):
        mask = labels == lab
        p = pnls[mask]
        if len(p) == 0:
            continue
        rows.append({"label": lab, **compute_stats(p)})
    return pd.DataFrame(rows)


def print_breakdown(df_stats: pd.DataFrame, title: str, label_col: str = "label") -> None:
    cols = [label_col, "sessions", "net_pips", "buy_pips", "sell_pips",
            "win_rate_pct", "avg_pnl_pip", "profit_factor"]
    hdr = f"{'':>10}  {'sess':>5}  {'net pip':>9}  {'buy':>9}  {'sell':>9}  {'win%':>5}  {'avg':>6}  {'PF':>6}"
    print(f"\n--- {title} ---")
    print(hdr)
    print("-" * len(hdr))
    for _, row in df_stats.iterrows():
        print(
            f"  {str(row[label_col]):>8}  {int(row['sessions']):>5}  "
            f"{row['net_pips']:>9,.1f}  {row['buy_pips']:>9,.1f}  {row['sell_pips']:>9,.1f}  "
            f"{row['win_rate_pct']:>5.1f}  {row['avg_pnl_pip']:>6.2f}  {row['profit_factor']:>6.3f}"
        )


# -- main ---------------------------------------------------------------------

def main() -> None:
    print("Loading EURUSD M1 data (2005-2025)...")
    df = load_m1()
    print(f"  {len(df):,} bars | {df.index[0].date()} -> {df.index[-1].date()}")

    # Collect entry positions for ALL session hours, sorted chronologically
    pos = np.sort(np.concatenate([
        session_entry_positions(df, hour=h) for h in SESSION_HOURS
    ]))
    pos_hours = pd.DatetimeIndex(df.index[pos]).hour.to_numpy(dtype=np.int8)
    pos_dows  = pd.DatetimeIndex(df.index[pos]).dayofweek.to_numpy(dtype=np.int8)
    print(f"  {len(pos):,} session entries across hours {SESSION_HOURS} (weekdays)")

    # 80/20 chronological split
    split = int(len(pos) * (1.0 - TEST_FRAC))
    train_pos = pos[:split]
    test_pos  = pos[split:]

    train_start = str(df.index[train_pos[0]].date())
    train_end   = str(df.index[train_pos[-1]].date())
    test_start  = str(df.index[test_pos[0]].date())
    test_end    = str(df.index[test_pos[-1]].date())

    print(f"  Train: {len(train_pos):,} entries ({train_start} -> {train_end})")
    print(f"  Test : {len(test_pos):,} entries  ({test_start} -> {test_end})")

    # Precompute forward-look matrices
    print("\nBuilding forward-look matrices...")
    H_tr, L_tr, C_tr, E_tr, SP_tr = build_forward_matrices(df, train_pos)
    H_te, L_te, C_te, E_te, SP_te = build_forward_matrices(df, test_pos)
    print(f"  Matrix size: {H_tr.shape} x 2 ~ {(H_tr.nbytes + L_tr.nbytes) / 1e6:.0f} MB")

    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_splits = list(tscv.split(np.arange(len(train_pos))))

    # Optuna optimisation (all session hours)
    print(f"\nOptimising: {N_TRIALS} Optuna trials, {N_CV_SPLITS}-fold TimeSeriesSplit CV")
    print(f"  SL range: {SL_MIN}-{SL_MAX} pips   TP range: {TP_MIN}-{TP_MAX} pips")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        make_objective(H_tr, L_tr, C_tr, E_tr, SP_tr, cv_splits),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_sl = study.best_params["sl_pips"]
    best_tp = study.best_params["tp_pips"]
    print(f"\nBest params  : SL = {best_sl} pips | TP = {best_tp} pips")
    print(f"CV avg-fold profit : {study.best_value:,.1f} pips")

    # Train / test overview
    train_pnls = simulate(H_tr, L_tr, C_tr, E_tr, SP_tr, best_sl, best_tp)
    test_pnls  = simulate(H_te, L_te, C_te, E_te, SP_te, best_sl, best_tp)

    train_m = print_overview("TRAIN SET (in-sample)",     train_pnls, train_start, train_end)
    test_m  = print_overview("TEST SET  (out-of-sample)", test_pnls,  test_start,  test_end)

    # -------------------------------------------------------------------------
    # Breakdowns — reuse already-simulated full-dataset pnls
    # -------------------------------------------------------------------------
    print("\nComputing hour/DOW breakdowns (full date range, optimised SL/TP)...")

    H_all, L_all, C_all, E_all, SP_all = build_forward_matrices(df, pos)
    all_pnls = simulate(H_all, L_all, C_all, E_all, SP_all, best_sl, best_tp)

    hour_stats = group_stats(all_pnls, pos_hours)
    hour_stats["label"] = hour_stats["label"].apply(lambda h: f"{h:02d}:00")
    print_breakdown(hour_stats, "By entry hour (UTC) -- full date range")

    dow_stats = group_stats(all_pnls, pos_dows)
    dow_stats["label"] = dow_stats["label"].apply(lambda d: DOW_NAMES[int(d)])
    print_breakdown(dow_stats, "By day of week -- full date range, all session hours")

    # -------------------------------------------------------------------------
    # Persist
    # -------------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "strategy": "session_baseline",
        "session_hours_utc": SESSION_HOURS,
        "max_bars": MAX_BARS,
        "best_sl_pips": best_sl,
        "best_tp_pips": best_tp,
        "cv_avg_fold_profit_pips": float(study.best_value),
        "train": train_m,
        "test":  test_m,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    trials_df = study.trials_dataframe()[
        ["number", "params_sl_pips", "params_tp_pips", "value"]
    ].rename(columns={"value": "cv_profit_pips"})
    trials_df.to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)

    hour_stats.to_csv(RESULTS_DIR / "breakdown_by_hour.csv", index=False)
    dow_stats.to_csv(RESULTS_DIR / "breakdown_by_dow.csv", index=False)

    print(f"\nSaved -> {RESULTS_DIR}/")
    print("  metrics.json            -- best params + train/test overview")
    print("  optuna_trials.csv       -- full Optuna landscape")
    print("  breakdown_by_hour.csv   -- per-entry-hour stats")
    print("  breakdown_by_dow.csv    -- per-day-of-week stats")


if __name__ == "__main__":
    main()
