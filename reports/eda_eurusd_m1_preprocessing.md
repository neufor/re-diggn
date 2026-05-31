# EURUSD M1 — Raw Data Preprocessing

**Date:** 2026-05-31  
**Script:** [`scripts/clean_eurusd_m1.py`](../scripts/clean_eurusd_m1.py)  
**Output:** [`data/processed/eurusd_m1.parquet`](../data/processed/eurusd_m1.parquet)

---

## Source files

| File | Period | Size |
|------|--------|------|
| `EURUSD_i_M1_200001030001_200412311901.csv` | 2000-01-03 → 2004-12-31 | 101.4 MB |
| `EURUSD_i_M1_200501030000_200912311859.csv` | 2005-01-03 → 2009-12-31 | 93.2 MB |
| `EURUSD_i_M1_201001040000_201412311959.csv` | 2010-01-04 → 2014-12-31 | 117.5 MB |
| `EURUSD_i_M1_201501020900_201912312059.csv` | 2015-01-02 → 2019-12-31 | 108.9 MB |
| `EURUSD_i_M1_202001020600_202505230134.csv` | 2020-01-02 → 2025-05-23 | 116.3 MB |

**Format:** Tab-separated, no header row. Columns: `date` (`YYYY.MM.DD`), `time` (`HH:MM:SS`), `open`, `high`, `low`, `close`, `tick_volume`, `volume`, `spread`. Source: MetaTrader 5 broker export (bid-based prices).

---

## Processing steps

1. Read all 5 CSVs with typed columns (float32 prices, int32 tick_volume, int64 volume, int16 spread).
2. Combine `date` + `time` → single UTC `DatetimeIndex` (`datetime64[us, UTC]`).
3. Deduplicate on timestamp (`keep="last"` — MT5 may write a bar twice on update).
4. Sort ascending and check monotonicity.
5. Drop rows with zero or negative OHLC prices.
6. Drop bars where `high < low` (hard OHLC corruption). Flag soft violations (`high < max(O,C)` or `low > min(O,C)`) — kept, noted below.
7. Drop Saturday and Sunday bars (`dayofweek >= 5`).
8. Report gaps > 60 minutes — classified as weekend vs. intra-week (holiday). No gap-filling.
9. Write Parquet with pyarrow + snappy compression.

---

## Data quality results

| Check | Count | Action |
|-------|------:|--------|
| Duplicate timestamps | 0 | — |
| Out-of-order rows | 0 | — |
| Zero / negative prices | 0 | — |
| OHLC hard violations (`high < low`) | 0 | — |
| OHLC soft violations | 0 | — |
| Weekend bars | 0 | MT5 already excludes them |
| Zero tick_volume bars | 0 | — |
| **Rows removed total** | **0** | Data was clean on export |

The raw MT5 export was already clean — no rows were dropped.

---

## Gap analysis

| Category | Count |
|----------|------:|
| Gaps > 60 min (weekend) | 1,310 |
| Gaps > 60 min (intra-week) | 84 |

All 84 intra-week gaps are legitimate market holidays. The 20 largest:

| Gap start (UTC) | Gap end (UTC) | Hours | Note |
|-----------------|---------------|------:|------|
| 2009-12-24 18:59 | 2009-12-28 00:00 | 77.0 | Christmas 2009 |
| 2009-12-31 18:59 | 2010-01-04 00:00 | 77.0 | New Year 2010 |
| 2015-12-24 19:59 | 2015-12-28 00:06 | 76.1 | Christmas 2015 |
| 2015-12-31 19:59 | 2016-01-04 00:05 | 76.1 | New Year 2016 |
| 2020-12-31 20:59 | 2021-01-04 00:05 | 75.1 | New Year 2021 |
| 2020-12-24 20:59 | 2020-12-28 00:05 | 75.1 | Christmas 2020 |
| 2000-04-13 22:54 | 2000-04-17 00:01 | 73.1 | Easter 2000 |

No gaps were filled. Features that require a continuous index (e.g. rolling windows) must handle these boundaries explicitly — see critical rules in `CLAUDE.md`.

---

## Output schema

```
DatetimeIndex: 9,085,974 entries
  dtype: datetime64[us, UTC]  (UTC, no conversion needed)

open         float32   # bid-based, 5-decimal precision
high         float32
low          float32
close        float32
tick_volume  int32     # number of ticks in the bar
volume       int64     # broker-aggregated volume (see note below)
spread       int16     # spread in points (1 point = 0.00001)
```

**Output:** 130.62 MB (snappy Parquet), down from ~537 MB raw CSV (~76% compression).

---

## Known caveats

### Volume regime split
`volume` is zero for **82.4% of bars** (2000–2014) — MT5 did not provide real volume for FX in that era. From 2015 onward, `volume` contains broker-aggregated tick counts (max observed ~80B units). Do not use `volume` as a feature across the full date range without era segmentation; use `tick_volume` instead, which is consistent throughout.

### Spread evolution
Average spread narrows significantly over the dataset lifetime:
- ~50 points (5.0 pips) in 2000–2005
- ~10–16 points (1.0–1.6 pips) in 2020–2025

Spread-based features (e.g. cost-adjusted return) must be computed per-bar, not with a fixed assumption. When backtesting, use the bar's own `spread` value.

### Prices are bid-based
All OHLC prices reflect the bid. For realistic trade simulation: use `close + spread * 0.00001` for market-buy fills, `close` for market-sell fills.

---

## Reproducing

```powershell
cd C:\Projects\My\Re-Diggn
uv run python scripts\clean_eurusd_m1.py
```

Full run log is written to `scripts\clean_eurusd_m1.log`.

## Loading the output

```python
import pandas as pd
df = pd.read_parquet("data/processed/eurusd_m1.parquet")
# DatetimeIndex is UTC; slice example:
df_2020 = df["2020":"2024"]
```

Via DuckDB (no Python needed for SQL exploration):

```sql
ATTACH 'data/processed/eurusd_m1.parquet' AS eurusd (TYPE PARQUET);
SELECT date_trunc('year', timestamp) AS year, count(*) AS bars
FROM eurusd GROUP BY ALL ORDER BY year;
```
