# Feature Reference — EURUSD M1 Pipeline

## Feature generation pipeline

New features follow a strict order to avoid look-ahead bias. Each stage reads the
previous stage's output, computes derived columns, and writes back to the parquet.

```
swap_history_eurusd.csv  +  economic_calendar.parquet
   (daily ECB/Fed rates)    (events 2007-2025)
            |                         |
            v                         v
    merge_rates_to_parquet.py    compute_macro_features.py
            |                         |
            |         +---------------+  (18 fund_ev_* columns)
            |         |
            v         v
     populate_fundamental_features.py
            |
            v
    eurusd_m1.parquet (71 cols)
            |
            v
    sl_tp_regressor.py  (FeatureSet = "situational" | "indicators" |
                          "fundamental" | "all")
```

### Stage order

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `scripts/merge_rates_to_parquet.py` | Reads `swap_history_eurusd.csv`, computes daily rate-diff features on ~10k rows, broadcasts to 7.3M M1 bars. Writes `fund_rate_*` columns. |
| 2 | `scripts/fetch_economic_calendar.py` | Downloads/processes Hugging Face ForexFactory dataset. Filters EUR+USD high/medium impact events. Maps raw event names to ~17 canonical groups. Converts timestamps to UTC. Outputs `data/raw/economic_calendar.parquet`. |
| 3 | `scripts/compute_macro_features.py` | Reads economic calendar + M1 index. Computes per-M1-bar features: time-since-event, event flags (60-bar window), standardised surprise, event density, time-to-next. Outputs `data/processed/fundamental_features.parquet`. |
| 4 | `scripts/populate_fundamental_features.py` | Merges fundamental feature columns into the M1 parquet in-place. |

### Adding a new fundamental feature

1. Write a script that computes the feature column(s) at M1 resolution.
2. If the computation is expensive, process at a coarser resolution (daily/hourly) then broadcast.
3. Use `.to_numpy()` when assigning into the parquet DataFrame to avoid pandas index-alignment issues.
4. Add the column name to `_RAW_COLS` in `src/pipelines/sl_tp_regressor.py` so it is excluded from raw-column filtering in the indicators feature set.
5. Run `populate_fundamental_features.py` (or merge directly).

---

## Column reference

### Rate differential features (4 columns)

| Column | Type | Description |
|--------|------|-------------|
| `fund_rate_diff` | FLOAT32 | ECB rate minus Fed rate (`ecb_rate - fed_rate`). Shifted by 1 day: the rate for date D is available from D+1 bars onward, avoiding look-ahead on decision-day bars. |
| `fund_rate_diff_change` | FLOAT32 | Day-over-day change in `fund_rate_diff`. Non-zero only on days where either ECB or Fed changed rates. Zero-fill on non-change days. |
| `fund_rate_diff_time_since` | FLOAT32 | Days (fractional) since the last rate-diff change. 0.0 at the first M1 bar following a rate change. Increases by 1/1440 per minute. Capped at 14 days. |
| `fund_carry_rank` | FLOAT32 | Rank of `fund_rate_diff` over the trailing 60-day window, scaled to [0, 1]. 1.0 = rate diff is at its 60-day maximum, 0.0 = at 60-day minimum. Captures whether carry is historically attractive. |

**Source**: `swap_history_eurusd.csv` (reconstructed from ECB Deposit Facility Rate and Fed Funds Effective Rate).

---

### Economic event features (18 columns)

#### Per-group time-since-event (5 columns)

| Column | Type | Description |
|--------|------|-------------|
| `fund_ev_fed_rate_time_since` | FLOAT32 | Days since the last Fed rate decision (FOMC Statement). NaN if no event within the lookback. Capped at 14 days. |
| `fund_ev_ecb_rate_time_since` | FLOAT32 | Days since the last ECB rate decision (ECB Press Conference). Capped at 14 days. |
| `fund_ev_nfp_time_since` | FLOAT32 | Days since the last NFP release (Non-Farm Employment Change). Capped at 14 days. |
| `fund_ev_cpi_time_since` | FLOAT32 | Days since the last CPI release. Capped at 14 days. |
| `fund_ev_gdp_time_since` | FLOAT32 | Days since the last GDP release. Capped at 14 days. |

Measured as `(current_bar_timestamp - most_recent_event_timestamp)` in fractional days.
Always populated (forward-filled from the earliest event in the series; bars before the first
event of that group have NaN).

#### Per-group event flags (5 columns)

| Column | Type | Description |
|--------|------|-------------|
| `fund_ev_fed_rate_flag` | BOOL | True at the bar containing a Fed rate decision AND for the next 59 M1 bars (60 bars = 1 hour total). |
| `fund_ev_ecb_rate_flag` | BOOL | Same for ECB rate decisions. |
| `fund_ev_nfp_flag` | BOOL | Same for NFP releases. |
| `fund_ev_cpi_flag` | BOOL | Same for CPI releases. |
| `fund_ev_gdp_flag` | BOOL | Same for GDP releases. |

The flag is True starting from the M1 bar that *contains* the event timestamp. Forward-filled
with a limit of 60 bars. Bars more than 1 hour after the event return to False.

#### Per-group standardised surprise (5 columns)

| Column | Type | Description |
|--------|------|-------------|
| `fund_ev_fed_rate_surprise` | FLOAT32 | Always NaN (rate decisions have no numeric actual/forecast in ForexFactory data). |
| `fund_ev_ecb_rate_surprise` | FLOAT32 | Always NaN (same reason). |
| `fund_ev_nfp_surprise` | FLOAT32 | Standardised surprise: `(actual - forecast) / rolling_std(past 20 forecast errors)`. Forward-filled for 60 bars after release. Clipped to [-5, 5]. |
| `fund_ev_cpi_surprise` | FLOAT32 | Same formula for CPI. |
| `fund_ev_gdp_surprise` | FLOAT32 | Same formula for GDP. |

Standardised surprise makes values comparable across event types (a +0.5 NFP surprise ≈ same
magnitude as a +0.5 CPI surprise). The rolling standard deviation uses 20 prior forecast errors
within the same event group (minimum 2 periods required).

Surprise values are only available after the event timestamp. Non-event bars have NaN
(forward-filled with a 60-bar limit).

#### Aggregate features (3 columns)

| Column | Type | Description |
|--------|------|-------------|
| `fund_ev_density_24h` | FLOAT32 | Count of high-impact events (any group) in the trailing 24 hours. Rolling window at M1 resolution. |
| `fund_ev_density_7d` | FLOAT32 | Count of high-impact events in the trailing 7 days. |
| `fund_ev_time_to_next` | FLOAT32 | Days (fractional) until the next scheduled high-impact event. Computed from actual occurrence timestamps, not forecast schedules. Capped at 14 days. 0 at bars that coincide with an event. |

---

### Event taxonomy

Raw ForexFactory event names are mapped to canonical groups. Order matters — first match wins.

| Canonical group | Matches (regex) | Actual events |
|----------------|-----------------|---------------|
| `fed_rate` | FOMC Statement, Fed Interest Rate Decision, "interest rate decision", "monetary policy" | 269 events, 0 with numeric actual/forecast |
| `ecb_rate` | ECB Press Conference, ECB Interest Rate Decision | 178 events, 0 with numeric actual/forecast |
| `nfp` | Non-Farm Employment Change, Nonfarm Payrolls | 220 events, 219 with surprise data |
| `cpi` | CPI, Consumer Price Index | 1,414 events, 1,407 with surprise data |
| `gdp` | GDP, Gross Domestic Product | 516 events, 515 with surprise data |
| `fomc_minutes` | FOMC Meeting Minutes, Beige Book | 209 events, 0 with surprise data |
| `pmi` | ISM Manufacturing/Services PMI, "PMI" | 2,078 events |
| `wages` | Average Hourly Earnings | 220 events (separated from NFP — different unit scale) |
| `adp` | ADP Non-Farm Employment Change | 218 events |
| `retail_sales` | Retail Sales m/m, Retail Sales y/y | 670 events |
| `unemployment` | Unemployment Rate, Jobless Claims | 1,176 events |
| `consumer_sentiment` | Michigan Consumer Sentiment, Consumer Confidence | 218 events |
| `ppi` | PPI, Producer Price Index | 504 events |
| `industrial_production` | Industrial Production, Factory Orders | 712 events |
| `trade_balance` | Trade Balance | 159 events |
| `jolts` | JOLTS Job Openings | 107 events |
| `other` | Everything else | 10,397 events |

Data covers **2007-01-01 to 2025-04-04** (from Hugging Face `Ehsanrs2/Forex_Factory_Calendar`).

---

### Look-ahead safeguards

1. **Rate features**: `shift(1)` on the daily rate series — the rate for date D is only available from D+1 bars onward.
2. **Event flags**: Placed at the M1 bar containing the event timestamp, then forward-filled. No backward-looking component.
3. **Surprise**: Placed at the M1 bar *following* the event timestamp (conservative — the number is not known until the very next bar). Forward-filled but never back-filled.
4. **Time-since**: `merge_asof(direction="backward")` — always looks back at past events only.
5. **Density**: Rolling window over past events only.
6. **Time-to-next**: Uses `merge_asof(direction="forward")` — references future events. This is valid because release schedules are pre-announced; the feature represents "market knows an event is coming in X days."
