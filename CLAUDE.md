# ML Project — Claude Context

## Domain
This project is one of:
- **Fintech / Trading**: ML-driven trading signals, risk models, portfolio optimization, market microstructure
- **Cheminformatics**: QSAR, virtual screening, molecular property prediction, hit-to-lead

Update the line above when you clone this template.

## Sub-agents — when to use each

| Agent | Invoke when |
|-------|-------------|
| `data-analyst` | First look at a new dataset, EDA, data quality, feature distributions |
| `ml-trainer` | Building / tuning / evaluating ML models |
| `backtesting-quant` | Backtesting trading strategies, risk metrics, walk-forward validation |
| `molecule-analyst` | Any task involving chemical structures, SMILES, RDKit, ADMET |
| `docs` | Generate/refresh ONBOARDING.md, update CLAUDE.md, populate memory MCP |
| `experiment-tracker` | Log/compare MLflow runs, manage model registry, suggest next experiments |
| `code-reviewer` | Review code for ML antipatterns: leakage, look-ahead bias, wrong CV, missing seeds |

Example: "Use the data-analyst agent to profile `data/raw/ohlcv.parquet`"

## Project structure
```
data/
  raw/          # source data, never modified
  processed/    # cleaned, feature-engineered
  docking/      # docking inputs/outputs (cheminformatics)
models/
  <name>/
    config.yaml
    model.pkl or model.ckpt
    metrics.json
notebooks/
  exploration/  # scratch EDA notebooks (not version-controlled)
  reports/      # clean, reproducible notebooks
src/
  features/     # feature engineering
  models/       # model definitions
  pipelines/    # end-to-end pipelines
  utils/        # shared utilities
reports/
  figures/
  eda_*.md
  model_card_*.md
  tearsheet_*.md  (trading)
  chem_*.md       (cheminformatics)
tests/
```

## MCP servers available
- **filesystem** — read/write project files
- **memory** — persist experiment results and domain facts across sessions
- **context7** — look up library docs (pandas, torch, rdkit, backtrader, etc.)
- **brave-search** — search papers, market data docs, chemical databases
- **jupyter** — execute Python in a live kernel (start with `jupyter lab --no-browser`)
- **serena** — code navigation, symbol search, cross-file refactoring
- **duckdb** — analytical SQL on parquet/CSV files; `ATTACH 'data/processed/ohlcv.parquet'`
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

## Critical rules
1. **No look-ahead bias** in any trading feature — always verified by data-analyst before training
2. **No random K-fold** on time-series data — use TimeSeriesSplit or walk-forward
3. **No random split** on molecular data — use scaffold split
4. **Backtest assumptions must be explicit**: costs, slippage, execution lag
5. **Standardize SMILES** before any computation — use the standardization pipeline in molecule-analyst

## Environment variables required
See `.env.example` for required keys. Copy to `.env` (gitignored).
