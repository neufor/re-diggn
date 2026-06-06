"""
Reconstruct historical EURUSD swap rates (long/short) for Alpari.

Uses hardcoded ECB deposit facility rate and Fed funds effective rate
histories (from official ECB page and Wikipedia). Forward-fills between
rate-change dates to produce a daily series.

Current Alpari swap rates (June 2026):
  swap_long  = -0.1041  pips / 1 lot / day   (from alpari.com)
  swap_short = +0.0476  pips / 1 lot / day

Formula (per Alpari FAQ):
  SWAP = Contract × (InterestRateDiff + Markup) / 100 × Price / DaysPerYear
"""
from __future__ import annotations

import csv
import datetime
import os

import numpy as np
import pandas as pd

# ── configuration ──────────────────────────────────────────────────
CURRENT_SWAP_LONG = -0.1041
CURRENT_SWAP_SHORT = 0.0476

EURUSD_CONTRACT = 100_000
EURUSD_TICK_VALUE = 10.0          # USD per pip per standard lot
DAYS_PER_YEAR = 360

OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "swap_history_eurusd.csv")

# ECB and Fed rates are referenced to the approximate snapshot date
# when the current Alpari swap rates were observed on their site.
SNAPSHOT_DATE = datetime.date(2026, 6, 6)

# ── hardcoded rate histories ──────────────────────────────────────

# ECB Deposit Facility Rate (%)  — change dates from ecb.europa.eu
ECB_RATE_CHANGES: list[tuple[str, float]] = [
    ("1999-01-01", 2.00),
    ("1999-04-09", 1.50),
    ("1999-11-05", 2.00),
    ("2000-02-04", 2.25),
    ("2000-03-17", 2.50),
    ("2000-04-28", 2.75),
    ("2000-06-09", 3.25),
    ("2000-09-01", 3.50),
    ("2000-10-06", 3.75),
    ("2001-05-11", 3.50),
    ("2001-08-31", 3.25),
    ("2001-09-18", 2.75),
    ("2001-11-09", 2.25),
    ("2002-12-06", 1.75),
    ("2003-03-07", 1.50),
    ("2003-06-06", 1.00),
    ("2005-12-06", 1.25),
    ("2006-03-08", 1.50),
    ("2006-06-15", 1.75),
    ("2006-08-09", 2.00),
    ("2006-10-11", 2.25),
    ("2006-12-13", 2.50),
    ("2007-03-14", 2.75),
    ("2007-06-13", 3.00),
    ("2008-07-09", 3.25),
    ("2008-10-08", 2.75),
    ("2008-11-12", 2.25),
    ("2008-12-10", 2.00),
    ("2009-01-21", 1.00),
    ("2009-03-11", 0.50),
    ("2009-04-08", 0.25),
    ("2011-04-13", 0.50),
    ("2011-07-13", 0.75),
    ("2011-11-09", 0.50),
    ("2011-12-14", 0.25),
    ("2012-07-11", 0.00),
    ("2014-06-11", -0.10),
    ("2014-09-10", -0.20),
    ("2015-12-09", -0.30),
    ("2016-03-16", -0.40),
    ("2019-09-18", -0.50),
    ("2022-07-27", 0.00),
    ("2022-09-14", 0.75),
    ("2022-10-27", 1.25),
    ("2022-12-21", 2.00),
    ("2023-02-08", 2.50),
    ("2023-03-22", 3.00),
    ("2023-05-10", 3.25),
    ("2023-06-21", 3.50),
    ("2023-07-26", 3.75),
    ("2023-09-20", 4.00),
    ("2024-06-12", 3.75),
    ("2024-09-18", 3.50),
    ("2024-10-23", 3.25),
    ("2024-12-18", 3.00),
    ("2025-02-05", 2.75),
    ("2025-03-12", 2.50),
    ("2025-04-23", 2.25),
    ("2025-06-11", 2.00),
    ("2026-06-06", 2.00),   # held / snapshot
]

# Fed Funds Effective Rate (%) — change dates from Wikipedia (FOMC actions)
# Midpoint used for target range periods
FED_RATE_CHANGES: list[tuple[str, float]] = [
    ("1999-01-01", 4.75),
    ("1999-06-30", 5.00),
    ("1999-08-24", 5.25),
    ("1999-11-16", 5.50),
    ("2000-02-02", 5.75),
    ("2000-03-21", 6.00),
    ("2000-05-16", 6.50),
    ("2001-01-03", 6.00),
    ("2001-01-31", 5.50),
    ("2001-03-20", 5.00),
    ("2001-04-18", 4.50),
    ("2001-05-15", 4.00),
    ("2001-06-27", 3.75),
    ("2001-08-21", 3.50),
    ("2001-09-17", 3.00),
    ("2001-10-02", 2.50),
    ("2001-11-06", 2.00),
    ("2001-12-11", 1.75),
    ("2002-11-06", 1.25),
    ("2003-06-25", 1.00),
    ("2004-06-30", 1.25),
    ("2004-08-10", 1.50),
    ("2004-09-21", 1.75),
    ("2004-11-10", 2.00),
    ("2004-12-14", 2.25),
    ("2005-02-02", 2.50),
    ("2005-03-22", 2.75),
    ("2005-05-03", 3.00),
    ("2005-06-30", 3.25),
    ("2005-08-09", 3.50),
    ("2005-09-20", 3.75),
    ("2005-11-01", 4.00),
    ("2005-12-13", 4.25),
    ("2006-01-31", 4.50),
    ("2006-03-28", 4.75),
    ("2006-05-10", 5.00),
    ("2006-06-29", 5.25),
    ("2007-09-18", 4.75),
    ("2007-10-31", 4.50),
    ("2007-12-11", 4.25),
    ("2008-01-22", 3.50),
    ("2008-01-30", 3.00),
    ("2008-03-18", 2.25),
    ("2008-04-30", 2.00),
    ("2008-10-08", 1.50),
    ("2008-10-29", 1.00),
    ("2008-12-16", 0.125),   # ZIRP: 0-0.25% target → use mid
    ("2015-12-16", 0.375),   # 0.25-0.50%
    ("2016-12-14", 0.625),   # 0.50-0.75%
    ("2017-03-15", 0.875),   # 0.75-1.00%
    ("2017-06-14", 1.125),   # 1.00-1.25%
    ("2017-12-13", 1.375),   # 1.25-1.50%
    ("2018-03-21", 1.625),   # 1.50-1.75%
    ("2018-06-13", 1.875),   # 1.75-2.00%
    ("2018-09-26", 2.125),   # 2.00-2.25%
    ("2018-12-19", 2.375),   # 2.25-2.50%
    ("2019-07-31", 2.125),   # 2.00-2.25%
    ("2019-09-18", 1.875),   # 1.75-2.00%
    ("2019-10-30", 1.625),   # 1.50-1.75%
    ("2020-03-03", 1.125),   # 1.00-1.25%
    ("2020-03-15", 0.125),   # 0-0.25%
    ("2022-03-16", 0.375),   # 0.25-0.50%
    ("2022-05-04", 0.875),   # 0.75-1.00%
    ("2022-06-15", 1.625),   # 1.50-1.75%
    ("2022-07-27", 2.375),   # 2.25-2.50%
    ("2022-09-21", 3.125),   # 3.00-3.25%
    ("2022-11-02", 3.875),   # 3.75-4.00%
    ("2022-12-14", 4.375),   # 4.25-4.50%
    ("2023-02-01", 4.625),   # 4.50-4.75%
    ("2023-03-22", 4.875),   # 4.75-5.00%
    ("2023-05-03", 5.125),   # 5.00-5.25%
    ("2023-07-26", 5.375),   # 5.25-5.50%
    ("2024-09-18", 4.875),   # 4.75-5.00%
    ("2024-11-07", 4.625),   # 4.50-4.75%
    ("2024-12-18", 4.375),   # 4.25-4.50%
    ("2025-01-29", 4.375),   # held
    ("2025-03-19", 4.375),   # held
    ("2025-05-07", 4.375),   # held
    ("2025-06-18", 4.375),   # held
    ("2025-07-30", 4.375),   # 4.25-4.50% held
    ("2025-09-17", 4.125),   # 4.00-4.25%
    ("2025-10-29", 3.875),   # 3.75-4.00%
    ("2025-12-10", 3.625),   # 3.50-3.75%
    ("2026-01-28", 3.625),   # held
    ("2026-03-18", 3.625),   # held
    ("2026-04-29", 3.625),   # held  (3.50-3.75%)
]


# ── helpers ─────────────────────────────────────────────────────────
def _build_daily_map(
    changes: list[tuple[str, float]],
    end_date: datetime.date,
) -> dict[datetime.date, float]:
    """Convert change-point list → daily map with forward-fill."""
    parsed = []
    for ds, val in changes:
        d = datetime.datetime.strptime(ds, "%Y-%m-%d").date()
        parsed.append((d, val))
    parsed.sort()

    result: dict[datetime.date, float] = {}
    current_val = parsed[0][1]
    idx = 0

    date = parsed[0][0]
    while date <= end_date:
        # Check if there's a change today
        while idx < len(parsed) and parsed[idx][0] <= date:
            current_val = parsed[idx][1]
            idx += 1
        result[date] = current_val
        date += datetime.timedelta(days=1)

    return result


"""
Simple model: swap rates are proportional to the interest rate differential.

  raw_diff = (eur_rate - usd_rate) * FACTOR   # theoretical pips for 1 lot
  swap_long = raw_diff * PASS_THROUGH_LONG
  swap_short = -raw_diff * PASS_THROUGH_SHORT

Long and short ALWAYS have opposite signs.
PASS_THROUGH factors (< 1) absorb broker spread, reference rates used, etc.
"""
PASS_THROUGH_LONG = None   # set during calibration
PASS_THROUGH_SHORT = None

FACTOR = EURUSD_CONTRACT / 100.0 / DAYS_PER_YEAR / EURUSD_TICK_VALUE


def calibrate(eur_rate: float, usd_rate: float, target_long: float, target_short: float):
    """Calibrate pass-through factors from current snapshot."""
    global PASS_THROUGH_LONG, PASS_THROUGH_SHORT
    raw = (eur_rate - usd_rate) * FACTOR
    PASS_THROUGH_LONG = target_long / raw if abs(raw) > 1e-9 else 0.0
    PASS_THROUGH_SHORT = target_short / (-raw) if abs(raw) > 1e-9 else 0.0


def swap_pips(eur_rate: float, usd_rate: float) -> tuple[float, float]:
    """Simple: proportional to rate differential, opposite signs."""
    raw = (eur_rate - usd_rate) * FACTOR
    return raw * PASS_THROUGH_LONG, -raw * PASS_THROUGH_SHORT


# ── main ────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  EURUSD Swap History Reconstruction")
    print("=" * 55)

    # 1. Build daily rate maps
    print("\n[1] Building daily ECB & Fed rate maps …")
    ecb_map = _build_daily_map(ECB_RATE_CHANGES, SNAPSHOT_DATE)
    fed_map = _build_daily_map(FED_RATE_CHANGES, SNAPSHOT_DATE)

    print(f"  ECB deposit rate: {len(ecb_map)} days  ({min(ecb_map)} … {max(ecb_map)})")
    print(f"  Fed funds rate:   {len(fed_map)} days  ({min(fed_map)} … {max(fed_map)})")

    # 2. Calibrate markup from current rates
    latest_ecb = ecb_map[SNAPSHOT_DATE]
    latest_fed = fed_map[SNAPSHOT_DATE]
    print(f"\n  Snapshot ({SNAPSHOT_DATE}) ECB: {latest_ecb}%  Fed: {latest_fed}%")
    print(f"  Current Alpari: long={CURRENT_SWAP_LONG}  short={CURRENT_SWAP_SHORT}")

    calibrate(latest_ecb, latest_fed, CURRENT_SWAP_LONG, CURRENT_SWAP_SHORT)
    print(f"\n[2] Calibrated:")
    print(f"    PASS_THROUGH_LONG  = {PASS_THROUGH_LONG:.4f}  "
          f"(Alpari passes {PASS_THROUGH_LONG*100:.1f}% of raw diff to long)")
    print(f"    PASS_THROUGH_SHORT = {PASS_THROUGH_SHORT:.4f}  "
          f"(Alpari passes {PASS_THROUGH_SHORT*100:.1f}% of raw diff to short)")

    v_long, v_short = swap_pips(latest_ecb, latest_fed)
    print(f"    Verification:  long={v_long:.4f}  short={v_short:.4f}  "
          f"(target: {CURRENT_SWAP_LONG:.4f} / {CURRENT_SWAP_SHORT:.4f})")

    # 3. Reconstruct full history
    print(f"\n[3] Reconstructing daily swap rates …")
    all_dates = sorted(set(ecb_map) & set(fed_map))
    print(f"  Date range: {all_dates[0]} … {all_dates[-1]}  ({len(all_dates)} days)")

    records: list[dict] = []
    for d in all_dates:
        sw_long, sw_short = swap_pips(ecb_map[d], fed_map[d])
        records.append({
            "date": d.isoformat(),
            "swap_long": round(sw_long, 4),
            "swap_short": round(sw_short, 4),
            "ecb_rate": round(ecb_map[d], 4),
            "fed_rate": round(fed_map[d], 4),
        })

    # 4. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "swap_long", "swap_short",
                                          "ecb_rate", "fed_rate"])
        w.writeheader()
        w.writerows(records)
    print(f"\n[4] Saved => {OUTPUT_FILE}  ({len(records)} rows)")

    # 5. Summary
    sw_longs = [r["swap_long"] for r in records]
    sw_shorts = [r["swap_short"] for r in records]
    median_l = sorted(sw_longs)[len(sw_longs) // 2]
    median_s = sorted(sw_shorts)[len(sw_shorts) // 2]
    print(f"\n  Summary (full history):")
    print(f"    Swap long:  min={min(sw_longs):.4f}  max={max(sw_longs):.4f}  median={median_l:.4f}")
    print(f"    Swap short: min={min(sw_shorts):.4f}  max={max(sw_shorts):.4f}  median={median_s:.4f}")

    # Period breakdowns
    zirp = [r for r in records if "2015-01" <= r["date"] < "2020-01"]
    if zirp:
        print(f"    2015-2019 avg:  long={np.mean([r['swap_long'] for r in zirp]):.4f}  "
              f"short={np.mean([r['swap_short'] for r in zirp]):.4f}")

    hiking = [r for r in records if "2022-06" <= r["date"] <= "2023-12"]
    if hiking:
        print(f"    2022-06 to 2023-12 avg:  long={np.mean([r['swap_long'] for r in hiking]):.4f}  "
              f"short={np.mean([r['swap_short'] for r in hiking]):.4f}")

    print("\nDone.")
    return records


# ── merge into eurusd_m1.parquet ────────────────────────────────────
PARQUET_PATH = os.path.join("data", "processed", "eurusd_m1.parquet")


def merge_swap_to_parquet(
    swap_csv: str = OUTPUT_FILE,
    parquet_path: str = PARQUET_PATH,
) -> int:
    """Merge swap_long / swap_short columns into M1 parquet by date.

    Each M1 bar gets the swap rate corresponding to its calendar date
    (swap rates are constant within a day).  Writes back in-place.

    Returns the number of rows with missing swap data (0 if full coverage).
    """
    if not os.path.exists(parquet_path):
        print(f"  [skip] Parquet not found: {parquet_path}")
        return 0

    print(f"\n[{4}] Merging swap columns into M1 parquet …")

    bars = pd.read_parquet(parquet_path)
    swaps = pd.read_csv(swap_csv, parse_dates=["date"])

    # Build a swap lookup by date
    swaps["date"] = swaps["date"].dt.date
    swap_lookup = swaps.set_index("date")[["swap_long", "swap_short"]]

    # Create date-only column from M1 DatetimeIndex and join — preserves the index
    bars["date"] = bars.index.date
    bars = bars.join(swap_lookup, on="date")
    bars = bars.drop(columns=["date"])

    n_nan = int(bars["swap_long"].isna().sum())
    if n_nan:
        print(f"  WARNING: {n_nan} rows with no swap data (before swap history starts)")
    else:
        print(f"  All {len(bars):,} rows matched — full coverage")

    # Order columns: original first, then swap
    original_cols = [c for c in bars.columns if c not in ("swap_long", "swap_short")]
    bars = bars[original_cols + ["swap_long", "swap_short"]]

    bars.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=True)

    file_size_mb = os.path.getsize(parquet_path) / 1024**2
    print(f"  Wrote back => {parquet_path}  ({file_size_mb:.2f} MB)")
    print(f"  New columns: swap_long ({bars['swap_long'].dtype}), "
          f"swap_short ({bars['swap_short'].dtype})")
    return n_nan


if __name__ == "__main__":
    main()
    merge_swap_to_parquet()
