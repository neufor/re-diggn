# Project Onboarding — Re-Diggn

## What this project does

Re-Diggn is an ML-driven FX trading signal research project focused on EURUSD M1 data.
The current goal is to forecast the maximum price move available within a session (~8 hours)
so that profitable SL/TP levels can be identified. A rule-based session baseline (Optuna-optimised
fixed SL/TP) has been built and validated; an ML layer using the feature engineering package is
the next stage.

## Domain

**Pair**: EURUSD
**Timeframe**: M1 bars (1-minute OHLCV + spread, broker timezone GMT+3, converted to UTC)
**Target**: Maximum forward price move over a configurable look-ahead window; session-level
SL/TP optimisation for a simultaneous BUY+SELL straddle entered at the 21:00 UTC session open.

## Quick start

```powershell
# 1. Install dependencies
uv sync

# 2. Preprocess raw MT5 CSV files into a single parquet
python scripts/clean_eurusd_m1.py

# 3. Run the session baseline (Optuna SL/TP optimisation)
python -m src.pipelines.session_baseline

# 4. Use the feature engineering package in a notebook or script
python - <<'EOF'
from src.features import generate_indicators, generate_situational_features
from src.features import create_target_delta, create_target_stat
import pandas as pd
df = pd.read_parquet("data/processed/eurusd_m1.parquet")
df = generate_indicators(df, sma_periods=[20, 50, 200], session_hours=[21,22,23,0,1,2])
df = generate_situational_features(df, session_hours=[21,22,23,0,1,2])
EOF
```

## Architecture

```
data/raw/  (MT5 CSV)
    |
    v  scripts/clean_eurusd_m1.py
data/processed/eurusd_m1.parquet
    |
    +---> src/pipelines/session_baseline.py
    |         Vectorised BUY+SELL simulation, Optuna TPE, TimeSeriesSplit CV
    |         Output: models/session_baseline/metrics.json + breakdowns
    |
    +---> src/features/  (feature engineering package)
              indicators.py       -- trend/momentum/session-time features
              situational.py      -- vol regime, oscillators, microstructure (situ_ prefix)
              targets.py          -- forward-look targets + session-level stat features
              |
              v  (ML pipeline — not yet implemented)
          src/models/ + src/pipelines/
              Output: models/<name>/model.pkl + metrics.json
```

## Key files

| File | Purpose |
|------|---------|
| `src/features/__init__.py` | Public API — import all four functions from here |
| `src/features/indicators.py` | `generate_indicators()` — SMAs, MinMax range, flat-strength, Stochastic K%/D%, level-100 proximity, session-time one-hot encoding |
| `src/features/situational.py` | `generate_situational_features()` — ~100 `situ_` prefixed columns: realized vol, ATR, BB, RSI, MACD, ADX, Keltner-like width, session calmness ratio; uses ThreadPoolExecutor for parallelism |
| `src/features/targets.py` | `create_target_delta()` — min/max future price move; `create_target_stat()` — cross-session look-back statistics (prev/5/10/20 sessions, same-minute-of-session) |
| `src/pipelines/session_baseline.py` | Standalone pipeline: loads M1 data, optimises SL/TP with Optuna, saves results to `models/session_baseline/` |
| `scripts/clean_eurusd_m1.py` | Merges raw MT5 CSV files, normalises columns, converts to UTC, saves parquet |
| `data/processed/eurusd_m1.parquet` | Primary dataset: EURUSD M1 2005–2025, UTC-indexed |
| `models/session_baseline/metrics.json` | Best SL/TP params + train/test performance overview |
| `src/draft/_preprocessing.py` | Origin prototype — not imported; kept for reference |
| `pyproject.toml` | Package manifest; use `uv add` to add dependencies |
| `CLAUDE.md` | Agent context — domain, conventions, critical rules |

## Sub-agents — when to use each

| Agent | Use for |
|-------|---------|
| `data-analyst` | EDA on `data/processed/eurusd_m1.parquet`, data quality checks, feature distribution analysis, look-ahead bias audits |
| `ml-trainer` | Building and tuning ML models consuming the `src/features/` output; evaluating signal quality |
| `backtesting-quant` | Extending or replacing `src/pipelines/session_baseline.py`; walk-forward validation; risk metrics (Sharpe, max drawdown) |
| `experiment-tracker` | Logging and comparing MLflow/Optuna runs; managing `models/` registry |
| `code-reviewer` | Reviewing `src/features/` for look-ahead bias, missing seeds, incorrect time-series splits |
| `docs` | Re-generating this file; updating CLAUDE.md; populating the memory MCP after structural changes |

## MCP servers

| Server | Purpose |
|--------|---------|
| `filesystem` | Read/write all project files |
| `memory` | Persist experiment results, discovered facts, and module summaries across sessions |
| `serena` | Symbol search, cross-file refactoring, get_symbols_overview on src/ |
| `context7` | Library docs: pandas, numpy, pandas_ta_classic, optuna, sklearn, xgboost, vectorbt |
| `brave-search` | FX market microstructure papers, broker API docs, ICT/SMC methodology references |
| `jupyter` | Execute Python in a live kernel (start: `jupyter lab --no-browser`) |
| `duckdb` | Analytical SQL on parquet; `ATTACH 'data/processed/eurusd_m1.parquet'` |
| `github` | Issues, PRs, experiment notes as comments |
| `openproject` | Work packages, sprints, time tracking |

## Conventions

- All timestamps are UTC. Raw data arrives as GMT+3 (broker) and is converted in `scripts/clean_eurusd_m1.py`.
- `session_hours` is always an ordered list of UTC hours, e.g. `[21, 22, 23, 0, 1, 2, 3]` for the overnight session starting 21:00 UTC. Pass it consistently to all three feature functions.
- `session_start_date` is a derived column (datetime of session-open for that bar). It is written by `generate_indicators()` and consumed by `create_target_stat()` — run `generate_indicators` first or supply the column yourself.
- `situ_` prefix is reserved for `generate_situational_features()` outputs. Do not create columns with this prefix elsewhere.
- Target column names must match one of the prefixes in `target_prefixes` (`"target_delta_"`, `"target_min_eps_"`, `"target_max_eps_"`) for `create_target_stat()` to pick them up.
- The session baseline pipeline uses `seed=42`, `TimeSeriesSplit(n_splits=5)`, `TEST_FRAC=0.20`.
- Sunday sessions are folded back to the preceding Friday — this logic lives in all three feature modules and in the baseline pipeline.

## Known gotchas

- **pandas_ta_classic** is the TA library in use (not `pandas_ta`). The import is `import pandas_ta_classic as ta`. It is not declared in `pyproject.toml` yet — add it with `uv add pandas_ta_classic`.
- **tqdm** is an optional dependency for progress bars in `generate_situational_features()`. The function silently falls back to a no-op if tqdm is not installed.
- `create_target_delta()` leaves the last `window` rows as `0` (no future data). Filter those rows before training.
- `generate_indicators()` appends `session_start_date` to the DataFrame as a side-effect when `session_hours` is provided. This is intentional — downstream functions depend on it.
- `data/processed/eurusd_m1.parquet` covers 2005–2025 but the raw CSVs have a separate file for 2000–2004 which is excluded from the pipeline (pre-Euro float era, data quality concerns).
- `src/models/` and `src/utils/` directories exist in the project structure convention but contain no code yet — do not import from them.
- `model/` (singular, at the repo root) is an empty directory — ignore it; the canonical output location is `models/` (plural).
