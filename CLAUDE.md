# ML Project — Claude Context

## Domain
**Forex trading**: ML-driven FX signals, risk models, position sizing, market microstructure for currency pairs (majors, minors, crosses).

## Sub-agents — when to use each

| Agent | Invoke when |
|-------|-------------|
| `data-analyst` | First look at a new dataset, EDA, data quality, feature distributions on FX tick/bar data |
| `ml-trainer` | Building / tuning / evaluating ML models for FX signals |
| `backtesting-quant` | Backtesting FX strategies, risk metrics, walk-forward validation |
| `docs` | Generate/refresh ONBOARDING.md, update CLAUDE.md, populate memory MCP |
| `experiment-tracker` | Log/compare MLflow runs, manage model registry, suggest next experiments |
| `code-reviewer` | Review code for ML antipatterns: leakage, look-ahead bias, wrong CV, missing seeds |

Example: "Use the data-analyst agent to profile `data/raw/eurusd_m5.parquet`"

## Project structure
```
data/
  raw/               # source FX data (MT5 CSV exports), never modified
  processed/
    eurusd_m1.parquet    # cleaned M1 OHLCV bars + swap_long/swap_short
    swap_history_eurusd.csv   # reconstructed daily swap rates (CSV)
models/
  session_baseline/   # SL/TP optimisation results (Optuna)
    metrics.json
    optuna_trials.csv
    breakdown_by_hour.csv
    breakdown_by_dow.csv
  <name>/             # future models follow the same layout
    config.yaml
    model.pkl or model.ckpt
    metrics.json
notebooks/
  exploration/  # scratch EDA notebooks (not version-controlled)
  reports/      # clean, reproducible notebooks
scripts/
  clean_eurusd_m1.py           # MT5 CSV → parquet preprocessing script
  reconstruct_swap_history.py  # reconstruct daily swap rates + merge into parquet
src/
  features/           # feature engineering package (production)
    __init__.py       # re-exports: generate_indicators, generate_situational_features,
                      #             create_target_delta, create_target_stat
    indicators.py     # generate_indicators() — SMAs, MinMax, Stochastic, session-time OHE
    situational.py    # generate_situational_features() — vol, ATR, BB, RSI, MACD, ADX (situ_ prefix)
    targets.py        # create_target_delta(), create_target_stat() — forward-look targets + stat lookbacks
  pipelines/          # end-to-end runnable scripts
    session_baseline.py  # fixed SL/TP optimisation via Optuna on EURUSD M1 21:00 UTC session
  models/             # model definitions (not yet implemented)
  utils/              # shared utilities (not yet implemented)
  draft/              # prototype/scratch code (not imported elsewhere)
    _preprocessing.py # origin draft from which src/features/ was derived
reports/
  figures/
  eda_*.md
  model_card_*.md
  tearsheet_*.md
tests/
```

## Datasets

### EURUSD M1 OHLCV (`data/processed/eurusd_m1.parquet`)
- **Source**: MT5 export CSV files from Alpari broker (`data/raw/EURUSD_i_M1_*.csv`, tab-separated)
- **Columns**: open, high, low, close, tick_volume, volume, spread, **swap_long, swap_short**
- **Rows**: ~9M (2000-01-03 — 2025-05-23, M1 bars, UTC)
- **Generation**: `scripts/clean_eurusd_m1.py` — concatenates CSV files, parses timestamps, drops duplicates, removes weekend bars, validates OHLC integrity, writes snappy-compressed Parquet
- **Note**: `volume=0` before ~2015 (MT5 FX limitation for that era); `volume>0` after

### Swap history (`data/processed/swap_history_eurusd.csv`)
- **Method**: Reconstructed from ECB Deposit Facility Rate + Fed Funds Effective Rate histories (hardcoded from ecb.europa.eu and Wikipedia FOMC tables)
- **Model**: Proportional pass-through: swap_long = raw_diff × 0.2306, swap_short = -raw_diff × 0.1055
  - `raw_diff = (eur_rate - usd_rate) × CONTRACT / 100 / 360 / TICK_VALUE`
  - Pass-through factors calibrated to current Alpari EURUSD swap rates (long=-0.1041, short=+0.0476 as of 2026-06-06)
  - Long and short always opposite signs (guaranteed by model design)
- **Range**: 1999-01-01 — 2026-06-06 (10,019 daily rows)
- **Alpari swap formula**: SWAP = Contract × (InterestRateDiff + Markup) / 100 × Price / DaysPerYear (360-day year)
- **Limitations**: Alpari may use EURIBOR/SOFR rather than ECB/Fed reference rates; pass-through may not be perfectly linear across all rate environments
- **Merged into parquet**: `reconstruct_swap_history.py` joins by date via `merge_swap_to_parquet()`, writing back in-place. Each M1 bar within a day shares the same swap values.

## MCP servers available
- **filesystem** — read/write project files
- **memory** — persist experiment results and domain facts across sessions
- **context7** — look up library docs (pandas, torch, vectorbt, backtrader, MetaTrader5, etc.)
- **brave-search** — search papers, FX market data docs, broker APIs
- **jupyter** — execute Python in a live kernel (start with `jupyter lab --no-browser`)
- **serena** — code navigation, symbol search, cross-file refactoring
- **duckdb** — analytical SQL on parquet/CSV files; `ATTACH 'data/processed/eurusd_m5.parquet'`
- **github** — issues, PRs, experiment tracking as comments
- **openproject** — work packages, tasks, sprints, time tracking

## Key conventions
- Random seed: **42** everywhere
- Python version: **3.11+**
- Package manager: **uv** (use `uv add` not `pip install`)
- Linter / formatter: **ruff** (`ruff check . && ruff format .`)
- Type checking: **mypy --strict** on `src/`
- Tests: **pytest** with fixtures in `tests/conftest.py`
- Data versioning: **DVC** for datasets and model artifacts
- Notebooks: never commit notebooks with cell outputs (use `nbstripout`)
- Timestamps: store as UTC; FX sessions reasoned about explicitly (Sydney/Tokyo/London/NY)
- Prices: keep bid/ask separate where available; compute mid only when needed

## Critical rules
1. **No look-ahead bias** in any FX feature — always verified by data-analyst before training
2. **No random K-fold** on time-series FX data — use TimeSeriesSplit or walk-forward with embargo
3. **Backtest assumptions must be explicit**: spread, commission, slippage, swap/rollover, execution lag
4. **Respect session boundaries and weekend gaps** — FX market closes Fri 22:00 UTC, reopens Sun 22:00 UTC
5. **Use bid for sells, ask for buys** — never assume mid-price execution in backtests
6. **Account for swap (overnight financing)** on positions held past 22:00 UTC (3x on Wednesdays for most pairs)

## Environment variables required
See `.env.example` for required keys (broker API keys, data provider tokens). Copy to `.env` (gitignored).
