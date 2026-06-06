#!/usr/bin/env python3
"""
Compute oracle SL/TP targets for every M1 bar on EURUSD.

For each bar (entry) with a T=480 forward window, for each SL level
(5..50 pips, step 5) and each direction (buy, sell), find the optimal
take-profit level (in pips) that maximises realised P&L including swap costs.

Appends 40 target columns to data/processed/eurusd_m1.parquet:
  target_opt_tp_buy_sl_5   …  target_opt_tp_buy_sl_50   (int16, 10 cols)
  target_opt_tp_sell_sl_5  …  target_opt_tp_sell_sl_50  (int16, 10 cols)
  target_swap_buy_sl_5     …  target_swap_buy_sl_50     (bool,  10 cols)
  target_swap_sell_sl_5    …  target_swap_sell_sl_50    (bool,  10 cols)

Key assumptions
  - OHLC is BID price
  - BUY  enters at ASK (= close + spread), SL/TP checked against bid H/L
  - SELL enters at BID (= close),       SL/TP checked against ask (H+spread, L+spread)
  - Rollover at 21:00 UTC (broker midnight under +3h correction)
  - Swap cost in pips, 3× on Wednesday (UTC dayofweek = 2)
  - Swap does NOT affect SL/TP triggering — added to P&L separately
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/processed/eurusd_m1.parquet")

T                = 480       # forward window (1-min bars)
MT5_POINT        = 0.00001   # 5-decimal point size for EURUSD
PIP              = 0.0001    # 1 pip = 10 MT5 points
SL_LEVELS        = list(range(5, 55, 5))    # [5, 10, …, 50]
TP_MAX           = 200        # upper bound for TP search (pips)
ROLLOVER_HOUR    = 21         # UTC hour of broker-midnight rollover
CHUNK_SIZE       = 50_000     # entries per chunk (memory / progress granularity)

SESSION_HOURS    = [21, 22, 23, 0, 1, 2, 3]  # unused here — kept for reference

SEED             = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_m1() -> pd.DataFrame:
    """Load EURUSD M1 data in UTC (same as session_baseline.py)."""
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
# Forward-matrix builder (chunked, per-entry loop — same pattern as
# session_baseline.build_forward_matrices)
# ---------------------------------------------------------------------------

def build_forward_chunk(
    df: pd.DataFrame,
    entry_positions: np.ndarray,
    window: int = T,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Build forward-looking matrices for a chunk of CONSECUTIVE entry positions.

    Uses sliding_window_view for a fully vectorised construction (no Python
    entry loop).

    Returns
    -------
    H           (n, T)  float32 — forward highs (bid)
    L           (n, T)  float32 — forward lows  (bid)
    spread_mtx  (n, T)  float32 — spread at each forward bar (MT5 points)
    sw_long_mtx (n, T)  float32 — swap_long at each forward bar
    sw_short_mtx(n, T)  float32 — swap_short at each forward bar
    ts_fwd      (n, T)  datetime64[ns] — forward bar timestamps
    C_bid       (n,)    float32 — bid close at timeout bar
    entry_close (n,)    float32 — close at entry bar (bid)
    entry_spread(n,)    float32 — spread at entry bar (MT5 points)
    entry_ts    (n,)    datetime64[ns] — entry timestamps (not forward)
    """
    from numpy.lib.stride_tricks import sliding_window_view as swv

    highs   = df["high"].to_numpy(dtype=np.float32)
    lows    = df["low"].to_numpy(dtype=np.float32)
    closes  = df["close"].to_numpy(dtype=np.float32)
    spreads = df["spread"].to_numpy(dtype=np.float32)
    sw_l    = df["swap_long"].to_numpy(dtype=np.float32)
    sw_s    = df["swap_short"].to_numpy(dtype=np.float32)
    ts      = df.index.to_numpy(dtype="datetime64[ns]")

    N = len(df)
    cs = int(entry_positions[0])
    ce = int(entry_positions[-1]) + 1
    n  = ce - cs

    # For entry at position pos, forward bars are pos+1 .. pos+window.
    # sliding_window_view(arr, window)[pos+1] gives exactly that.
    offset = cs + 1
    H          = swv(highs,   window)[offset:offset + n].copy()
    L          = swv(lows,    window)[offset:offset + n].copy()
    spread_mtx = swv(spreads, window)[offset:offset + n].copy()
    sw_long_mtx  = swv(sw_l, window)[offset:offset + n].copy()
    sw_short_mtx = swv(sw_s, window)[offset:offset + n].copy()
    ts_fwd     = swv(ts,      window)[offset:offset + n].copy()

    C_bid          = closes[cs + window:cs + window + n].copy()
    entry_close_arr = closes[cs:ce].copy()
    entry_spread_arr = spreads[cs:ce].copy()
    entry_ts_arr    = ts[cs:ce].copy()

    return (H, L, spread_mtx, sw_long_mtx, sw_short_mtx, ts_fwd,
            C_bid, entry_close_arr, entry_spread_arr, entry_ts_arr)


# ---------------------------------------------------------------------------
# Cumulative swap matrix
# ---------------------------------------------------------------------------

def compute_cumulative_swap(
    ts_fwd: np.ndarray,           # (n, T) datetime64[ns]
    sw_long_fwd: np.ndarray,      # (n, T) float32
    sw_short_fwd: np.ndarray,     # (n, T) float32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-bar cumulative swap cost (in pips) for each entry.

    Swap is incurred on the bar that crosses ROLLOVER_HOUR UTC (broker
    midnight).  3× on Wednesday (UTC dayofweek == 2).

    Returns (cumul_buy, cumul_sell), both (n, T) float32.
    """
    n, T = ts_fwd.shape

    # --- rollover detection: which forward bars cross ROLLOVER_HOUR UTC? ---
    # timestamp as nanoseconds since epoch
    ts_ns = ts_fwd.astype(np.int64)                     # (n, T)
    # seconds since start of UTC day
    day_seconds = ts_ns % (86_400 * 10**9)              # (n, T)
    hours = (day_seconds // (3600 * 10**9)).astype(np.int32)   # (n, T)
    mins  = ((day_seconds % (3600 * 10**9)) // (60 * 10**9)).astype(np.int32)

    # A bar that lands exactly on ROLLOVER_HOUR:00 is the first bar of the
    # new trading day — swap is incurred.
    rollover_bar = (hours == ROLLOVER_HOUR) & (mins == 0)   # (n, T) bool

    # --- weekday detection (UTC) ---
    # 1970-01-01 was Thursday (= 3 in Python weekday convention).
    days_since_epoch = ts_ns // (86_400 * 10**9)
    weekday = ((days_since_epoch + 3) % 7).astype(np.int32)  # 0=Mon … 6=Sun
    is_wednesday = (weekday == 2)                             # Wed = 2

    # --- swap cost per rollover event ---
    swap_cost_buy  = np.zeros((n, T), dtype=np.float32)
    swap_cost_sell = np.zeros((n, T), dtype=np.float32)

    ro_idx = np.where(rollover_bar)
    if len(ro_idx[0]) > 0:
        factor = np.where(is_wednesday[ro_idx], 3.0, 1.0)
        swap_cost_buy[ro_idx]  = sw_long_fwd[ro_idx] * factor
        swap_cost_sell[ro_idx] = sw_short_fwd[ro_idx] * factor

    # --- cumulative ---
    cumul_buy  = np.cumsum(swap_cost_buy,  axis=1)
    cumul_sell = np.cumsum(swap_cost_sell, axis=1)

    return cumul_buy, cumul_sell


# ---------------------------------------------------------------------------
# Optimal TP for a single (SL, direction) for a chunk of entries
# ---------------------------------------------------------------------------

def find_optimal_tp(
    H: np.ndarray,            # (n, T) — price series for SL/TP check
    L: np.ndarray,            # (n, T) — price series for SL/TP check
    entry_px: np.ndarray,     # (n,)   — entry price (ask for buy, bid for sell)
    swap_cumul: np.ndarray,   # (n, T) — cumulative swap (pips)
    C_exit: np.ndarray,       # (n,)   — exit price at timeout
    sl_pips: int,
    direction: str,           # "buy" or "sell"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fully-vectorised optimal TP search via running-extremum breakpoints.

    For each entry:
      1. Find the SL exit bar.
      2. Compute cumulative max (buy) or min (sell) of price over the forward
         window (running extremum).
      3. At every bar where the running extremum advances, the hypothetical
         TP is the excursion in pips.  Profit = TP + cumulative swap at that
         bar.
      4. Also considers the no-TP timeout case.
      5. Picks the TP with the highest profit.

    Returns
    -------
    opt_tp   (n,) int16  — optimal TP in pips (0 = timeout was better)
    has_swap (n,) bool   — True if swap incurred at the chosen exit bar
    """
    n, T_ = H.shape
    sl_pts = np.float32(sl_pips * PIP)

    if direction == "buy":
        sl_trig = L <= (entry_px[:, None] - sl_pts)
    else:
        sl_trig = H >= (entry_px[:, None] + sl_pts)

    any_sl = sl_trig.any(axis=1)
    sl_bar = np.where(any_sl, sl_trig.argmax(axis=1).astype(np.int32), T_)

    # Timeout profit (no TP hit before SL / T expiry)
    if direction == "buy":
        timeout_pnl = (C_exit - entry_px) / PIP
    else:
        timeout_pnl = (entry_px - C_exit) / PIP
    last_bar = np.where(sl_bar < T_, sl_bar, T_ - 1)
    timeout_profit = timeout_pnl + swap_cumul[np.arange(n), last_bar]

    # Running extremum
    if direction == "buy":
        running = np.maximum.accumulate(H, axis=1)
        tp_all = np.round((running - entry_px[:, None]) / PIP).astype(np.int16)
    else:
        running = np.minimum.accumulate(L, axis=1)
        tp_all = np.round((entry_px[:, None] - running) / PIP).astype(np.int16)

    # Valid TP requires TP > 0 AND bar is before SL
    bar_idx = np.arange(T_)[None, :]              # (1, T)
    before_sl = bar_idx < sl_bar[:, None]         # (n, T)
    valid = (tp_all > 0) & before_sl

    # Profit at each valid candidate, sentinel for others
    profit_all = tp_all.astype(np.float32) + swap_cumul
    profit_all[~valid] = np.float32(-1e9)

    # Best TP per entry (argmax picks first occurrence on tie → earliest bar)
    best_bar = profit_all.argmax(axis=1)           # (n,)
    best_profit = profit_all[np.arange(n), best_bar]
    best_tp = tp_all[np.arange(n), best_bar]
    best_swap = swap_cumul[np.arange(n), best_bar] != 0.0

    # If timeout beats even the best TP, prefer TP=0 (no TP)
    better_timeout = timeout_profit > best_profit
    best_tp[better_timeout] = 0
    best_swap[better_timeout] = False

    return best_tp.astype(np.int16), best_swap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 56)
    print("  Oracle SL/TP targets - per (SL, direction) with swap")
    print("=" * 56)

    # -- Load --------------------------------------------------------------
    print("\nLoading EURUSD M1 data...")
    df = load_m1()
    # Drop any previous target columns (safe for re-runs)
    drop_cols = [c for c in df.columns if c.startswith("target_")]
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} existing target columns from dataframe")
        df = df.drop(columns=drop_cols)
    N = len(df)
    N_valid = N - T
    print(f"  {N:,} bars total, {N_valid:,} have a full {T}-bar forward window")

    # -- Pre-allocate output arrays ----------------------------------------
    n_sl = len(SL_LEVELS)

    opt_tp_buy  = np.zeros((n_sl, N), dtype=np.int16)
    opt_tp_sell = np.zeros((n_sl, N), dtype=np.int16)
    swp_buy     = np.zeros((n_sl, N), dtype=bool)
    swp_sell    = np.zeros((n_sl, N), dtype=bool)

    # Mark last T bars as invalid (shorter forward window)
    opt_tp_buy[:, N_valid:]  = -1
    opt_tp_sell[:, N_valid:] = -1

    # -- Chunked processing ------------------------------------------------
    chunk_starts = list(range(0, N_valid, CHUNK_SIZE))
    n_chunks = len(chunk_starts)
    print(f"  Processing {n_chunks} chunk(s) of ~{CHUNK_SIZE:,} entries each")

    for chunk_idx, cs in enumerate(chunk_starts):
        ce = min(cs + CHUNK_SIZE, N_valid)
        entry_positions = np.arange(cs, ce, dtype=np.int64)
        n_chunk = ce - cs
        print(f"\n  Chunk {chunk_idx + 1}/{n_chunks}  (entries {cs:,}-{ce - 1:,}, n={n_chunk:,})")

        # -- Build forward matrices (fully vectorised via sliding_window_view)
        (H, L, spread_mtx, sw_long_mtx, sw_short_mtx, ts_fwd,
         C_bid, entry_close, entry_spread, entry_ts) = build_forward_chunk(
            df, entry_positions, window=T,
        )

        # -- Cumulative swap -----------------------------------------------
        cumul_buy, cumul_sell = compute_cumulative_swap(
            ts_fwd, sw_long_mtx, sw_short_mtx,
        )

        # -- Per-SL (parallel) ------------------------------------------------
        # BUY: entry at ask, SL/TP checked against bid H/L
        entry_ask = entry_close + entry_spread * MT5_POINT

        # SELL: entry at bid, SL/TP checked against ask (H+spread, L+spread)
        spread_pts = spread_mtx * MT5_POINT
        H_ask = H + spread_pts
        L_ask = L + spread_pts
        C_ask = C_bid + spread_mtx[:, -1] * MT5_POINT
        entry_bid = entry_close

        def _run_sl(sl: int) -> tuple:
            tp_buy, sw_buy = find_optimal_tp(
                H, L, entry_ask, cumul_buy, C_bid, sl, "buy",
            )
            tp_sell, sw_sell = find_optimal_tp(
                H_ask, L_ask, entry_bid, cumul_sell, C_ask, sl, "sell",
            )
            return sl, tp_buy, sw_buy, tp_sell, sw_sell

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_run_sl)(sl) for sl in SL_LEVELS
        )

        for sl, tp_buy, sw_buy, tp_sell, sw_sell in results:
            sl_idx = SL_LEVELS.index(sl)
            opt_tp_buy[sl_idx, cs:ce]  = tp_buy
            opt_tp_sell[sl_idx, cs:ce] = tp_sell
            swp_buy[sl_idx, cs:ce]     = sw_buy
            swp_sell[sl_idx, cs:ce]    = sw_sell

        print(f"    -> SL {SL_LEVELS[0]}..{SL_LEVELS[-1]} done")

    # -- Assemble output DataFrame -----------------------------------------
    print("\nAssembling output DataFrame...")
    columns: dict[str, np.ndarray] = {}
    for sl_idx, sl in enumerate(SL_LEVELS):
        columns[f"target_opt_tp_buy_sl_{sl}"]   = opt_tp_buy[sl_idx]
        columns[f"target_opt_tp_sell_sl_{sl}"]  = opt_tp_sell[sl_idx]
        columns[f"target_swap_buy_sl_{sl}"]     = swp_buy[sl_idx]
        columns[f"target_swap_sell_sl_{sl}"]    = swp_sell[sl_idx]

    target_df = pd.DataFrame(columns, index=df.index)

    # -- Append to parquet -------------------------------------------------
    print(f"Appending {len(columns)} columns to {DATA_PATH} ...")
    df_out = pd.concat([df, target_df], axis=1)

    # Set dtypes explicitly
    for c in target_df.columns:
        if c.startswith("target_opt_tp"):
            df_out[c] = df_out[c].astype(np.int16)   # -1 sentinel for invalid
        elif c.startswith("target_swap"):
            df_out[c] = df_out[c].astype(bool)

    # Write via temp file — never corrupt the original on partial write
    tmp = DATA_PATH.with_suffix(".parquet.tmp")
    df_out.to_parquet(tmp)
    tmp.replace(DATA_PATH)
    print(f"  Done -> {DATA_PATH}")
    print(f"  Total columns: {len(df_out.columns)}")

    # -- Quick stats -------------------------------------------------------
    print("\n-- Quick stats --")
    valid = target_df.iloc[:N_valid]
    for sl in SL_LEVELS:
        bc = f"target_opt_tp_buy_sl_{sl}"
        sc = f"target_opt_tp_sell_sl_{sl}"
        print(
            f"  {bc:>30s}:  mean={valid[bc].mean():.1f}  std={valid[bc].std():.1f}"
            f"  nonzero={float((valid[bc] > 0).mean()):.1%}"
        )
        print(
            f"  {sc:>30s}:  mean={valid[sc].mean():.1f}  std={valid[sc].std():.1f}"
            f"  nonzero={float((valid[sc] > 0).mean()):.1%}"
        )

    # Swap hit rate
    for sl in SL_LEVELS:
        bc = f"target_swap_buy_sl_{sl}"
        sc = f"target_swap_sell_sl_{sl}"
        print(
            f"  {bc:>30s}:  swap%={float(valid[bc].mean()):.1%}"
        )
        print(
            f"  {sc:>30s}:  swap%={float(valid[sc].mean()):.1%}"
        )


if __name__ == "__main__":
    main()
