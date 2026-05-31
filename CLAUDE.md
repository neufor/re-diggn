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
  raw/          # source FX data (tick, OHLCV bars), never modified
  processed/    # cleaned, resampled, feature-engineered
models/
  <name>/
    config.yaml
    model.pkl or model.ckpt
    metrics.json
notebooks/
  exploration/  # scratch EDA notebooks (not version-controlled)
  reports/      # clean, reproducible notebooks
src/
  features/     # feature engineering (technicals, microstructure, sessions)
  models/       # model definitions
  pipelines/    # end-to-end pipelines
  utils/        # shared utilities
reports/
  figures/
  eda_*.md
  model_card_*.md
  tearsheet_*.md
tests/
```

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
