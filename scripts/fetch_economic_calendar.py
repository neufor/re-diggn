"""Process ForexFactory economic calendar data for EUR/USD-relevant events.

Downloads the Hugging Face dataset (Ehsanrs2/Forex_Factory_Calendar) which
covers Jan 2007 — Apr 2025, filters to EUR/USD high/medium impact events,
normalizes event names to a canonical taxonomy, converts timestamps to UTC,
and saves to data/raw/economic_calendar.parquet.

Usage:
    uv run python scripts/fetch_economic_calendar.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
OUTPUT_PATH = RAW_DIR / "economic_calendar.parquet"
HF_URL = (
    "https://huggingface.co/datasets/Ehsanrs2/"
    "Forex_Factory_Calendar/resolve/main/forex_factory_cache.csv"
)

# Impact mapping from the dataset's text labels
IMPACT_HIGH = {"high impact expected", "high impact"}
IMPACT_MEDIUM = {"medium impact expected", "medium impact"}
TARGET_IMPACTS = IMPACT_HIGH | IMPACT_MEDIUM

# Canonical event groups — ordered by priority (first match wins)
EVENT_GROUPS: list[tuple[re.Pattern, str]] = [
    # Fed rate decisions (named "FOMC Statement" in ForexFactory)
    (re.compile(r"fomc\s*statement", re.I), "fed_rate"),
    (re.compile(r"fed\s*interest\s*rate\s*decision", re.I), "fed_rate"),
    (re.compile(r"fed\s*monetary\s*policy\s*statement", re.I), "fed_rate"),
    # FOMC meeting minutes (separate from the statement/decision)
    (re.compile(r"fomc\s*meeting\s*minutes", re.I), "fomc_minutes"),
    (re.compile(r"fomc\s*minutes", re.I), "fomc_minutes"),
    (re.compile(r"beige\s*book", re.I), "fomc_minutes"),
    # ECB rate decisions (named "ECB Press Conference" in ForexFactory)
    (re.compile(r"ecb\s*press\s*conference", re.I), "ecb_rate"),
    (re.compile(r"ecb\s*interest\s*rate\s*decision", re.I), "ecb_rate"),
    (re.compile(r"ecb\s*monetary\s*policy\s*statement", re.I), "ecb_rate"),
    # Wages (separate from NFP — different units / scale)
    (re.compile(r"average\s*hourly\s*earnings", re.I), "wages"),
    # ADP (must precede NFP patterns to avoid matching "ADP Non-Farm Employment Change")
    (re.compile(r"adp\s*non.farm", re.I), "adp"),
    (re.compile(r"adp\s*employment", re.I), "adp"),
    # NFP (jobs count, unit = thousands)
    (re.compile(r"non.farm\s*(payrolls|employment\s*change)", re.I), "nfp"),
    (re.compile(r"nonfarm\s*payrolls", re.I), "nfp"),
    # CPI
    (re.compile(r"cpi\b", re.I), "cpi"),
    (re.compile(r"consumer\s*price\s*index", re.I), "cpi"),
    # GDP
    (re.compile(r"gdp\b", re.I), "gdp"),
    (re.compile(r"gross\s*domestic\s*product", re.I), "gdp"),
    # PMI
    (re.compile(r"ism\s*manufacturing\s*pm[i]", re.I), "pmi"),
    (re.compile(r"ism\s*services\s*pm[i]", re.I), "pmi"),
    (re.compile(r"(manufacturing|services)\s*pm[i]", re.I), "pmi"),
    # Retail sales
    (re.compile(r"retail\s*sales", re.I), "retail_sales"),
    # Unemployment/labour
    (re.compile(r"unemployment\s*(rate|claims)", re.I), "unemployment"),
    (re.compile(r"jobless\s*claims", re.I), "unemployment"),
    # Consumer sentiment
    (re.compile(r"michigan\s*consumer\s*sentiment", re.I), "consumer_sentiment"),
    (re.compile(r"consumer\s*confidence", re.I), "consumer_sentiment"),
    # JOLTS
    (re.compile(r"jolts", re.I), "jolts"),
    # PPI
    (re.compile(r"ppi\b", re.I), "ppi"),
    (re.compile(r"producer\s*price\s*index", re.I), "ppi"),
    # Industrial production
    (re.compile(r"industrial\s*production", re.I), "industrial_production"),
    (re.compile(r"factory\s*orders", re.I), "industrial_production"),
    # Trade balance
    (re.compile(r"trade\s*balance", re.I), "trade_balance"),
    # Catch-all for rate decisions that didn't match above
    (re.compile(r"interest\s*rate\s*decision", re.I), "fed_rate"),
    (re.compile(r"monetary\s*policy", re.I), "fed_rate"),
]

FALLBACK_GROUP = "other"

# Groups that get their own feature columns
MAJOR_GROUPS = {"fed_rate", "ecb_rate", "nfp", "cpi", "gdp"}


def _normalize_event(name: str) -> str:
    """Map an event name to a canonical group."""
    for pattern, group in EVENT_GROUPS:
        if pattern.search(name):
            return group
    return FALLBACK_GROUP


def _parse_value(val: object) -> float | None:
    """Parse a numeric value from the dataset, handling strings like '172K', '2.5%', etc."""
    if val is None or (isinstance(val, str) and val.strip() in ("", "-", "N/A")):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    for ch in (",", "%", "$", "€", "£"):
        text = text.replace(ch, "")
    # Handle K suffix (thousands)
    multiplier = 1.0
    if text.upper().endswith("K"):
        text = text[:-1]
        multiplier = 1000.0
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def process(
    csv_path: str | None = None,
    output_path: str | Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Download and process the ForexFactory calendar dataset.

    Steps:
        1. Download CSV from Hugging Face (or load local)
        2. Filter to EUR/USD with high/medium impact
        3. Normalize event names to canonical groups
        4. Convert timestamps from +03:30 to UTC
        5. Save as snappy-compressed parquet
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=["DateTime"])
    else:
        local_path = RAW_DIR / "forex_factory_cache.csv"
        if local_path.exists():
            df = pd.read_csv(local_path, parse_dates=["DateTime"])
        else:
            print(f"Downloading from Hugging Face: {HF_URL}")
            df = pd.read_csv(HF_URL, parse_dates=["DateTime"])
            df.to_csv(local_path, index=False)
            print(f"Cached raw data to {local_path}")

    print(f"Loaded {len(df):,} total events ({df['DateTime'].min()} — {df['DateTime'].max()})")

    # --- Filter ---
    df = df[df["Currency"].isin(["EUR", "USD"])].copy()
    df["impact"] = df["Impact"].str.lower().str.strip()
    df = df[df["impact"].isin(TARGET_IMPACTS)].copy()
    print(f"After EUR/USD high/medium filter: {len(df):,} events")

    # --- Normalize event groups ---
    df["event_group"] = df["Event"].apply(_normalize_event)
    df["event_name"] = df["Event"]

    # --- Parse numeric values ---
    df["actual"] = df["Actual"].apply(_parse_value)
    df["forecast"] = df["Forecast"].apply(_parse_value)
    df["previous"] = df["Previous"].apply(_parse_value)

    # --- Convert timestamp to UTC ---
    # Dataset uses Iran time (+03:30/+04:30 DST). Convert to UTC.
    df["timestamp"] = pd.to_datetime(df["DateTime"], utc=True)

    # --- Build output ---
    result = pd.DataFrame({
        "timestamp": df["timestamp"],
        "currency": df["Currency"],
        "event_group": df["event_group"],
        "event_name": df["event_name"],
        "impact": df["impact"],
        "is_major": df["event_group"].isin(MAJOR_GROUPS),
        "actual": df["actual"],
        "forecast": df["forecast"],
        "previous": df["previous"],
    })
    result = result.sort_values("timestamp").reset_index(drop=True)
    result.to_parquet(output_path, engine="pyarrow", compression="snappy")

    n_major = result["is_major"].sum()
    print(f"\nSaved {len(result):,} events to {output_path}")
    print(f"  Major groups: {n_major:,} events")
    print("  Group distribution:")
    print(f"    {result['event_group'].value_counts().to_string()}")

    return result


def main():
    process()


if __name__ == "__main__":
    main()
