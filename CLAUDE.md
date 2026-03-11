# Cardinality-Constrained Portfolio Optimisation

## Project Overview

Solves the cardinality-constrained portfolio selection problem: find an optimal subset of N ETFs from a universe of M ETFs that maximises the Sharpe ratio, subject to position count, weight, and budget constraints. Uses genetic algorithms for ETF selection and SLSQP for weight optimisation. Validated via out-of-sample backtesting.

## Tech Stack

- **Python 3.7+** — primary language
- **C++** — high-performance parallel GA (`cpp/optimisation.cpp`, compiled to `cpp/optimisation` binary)
- **Key Python libs**: numpy, pandas, scipy, pygad, arch, copulae, pmdarima, yfinance, matplotlib, pulp
- **C++ deps** (header-only submodules): Eigen (linear algebra), csv-parser

## Project Structure

```
src/                         # Python source package
├── __init__.py
├── optimisation.py          # Core: GA selection + SLSQP weights + copula correlation + risk parity
├── simple_ga_optimisation.py # Simpler parallel island-based GA using InvestNow data
├── backtest.py              # Forward-walk backtesting with Sharpe/Sortino/Calmar/drawdown stats
├── forecast.py              # ARIMA returns + GARCH variance forecasting
├── db.py                    # SQLite database module (schema, save/load functions, CSV migration)
├── monte_carlo_optimisation.py  # Monte Carlo brute-force baseline
├── mip_optimisation.py      # Mixed Integer Linear Programming alternative
├── portfolio_utils.py       # Shared utility functions
├── download_data.py         # Yahoo Finance data downloader
├── list_of_stocks.py        # ETF/stock universe definitions
└── prices_EDA.py            # Exploratory data analysis / visualisation
tests/                       # Unit tests
├── __init__.py
├── test_optimisation.py     # Tests for optimisation module
├── test_backtest.py         # Tests for backtest module
├── test_db.py               # Tests for database module
├── test_portfolio_utils.py  # Tests for portfolio utilities
└── test_securities.py       # Tests for security universe/download
cpp/                         # C++ parallel island GA implementation
├── optimisation.cpp         # Source code
└── optimisation             # Compiled binary
benchmark/                   # Benchmarking framework package
├── __init__.py
├── adapters.py              # Adapter wrappers for all optimisation methods
├── runner.py                # Orchestrates parallel benchmark runs
├── analysis.py              # Result analysis and reporting
└── results.py               # Data structures for benchmark results
Data/                        # CSV price data, ETF lists, forecast outputs (~112 MB)
├── portfolio.db             # SQLite database (gitignored, created by db.py)
├── ETF_Prices.csv           # Daily adjusted close for ~1792 ETFs
└── ...                      # Other CSV data files
Images/                      # Visualisation outputs
benchmark_results/           # Benchmark run outputs (JSON/PKL)
run_benchmark.py             # CLI entry point for benchmarking
requirements.txt
CLAUDE.md
README.md
```

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main GA optimisation (InvestNow data)
python -m src.simple_ga_optimisation

# Run the full optimisation with copulas/forecasts
python -m src.optimisation

# Run backtests
python -m src.backtest

# Generate ARIMA/GARCH forecasts
python -m src.forecast

# Download fresh ETF price data
python -m src.download_data

# Database
python -m src.db                # Create empty database with schema
python -m src.db migrate        # Import existing CSVs into database

# Run benchmarks
python run_benchmark.py

# Run tests
python -m unittest discover tests
```

## Key Concepts

- **Sharpe ratio** = E(R) / Std(R) is the primary objective function
- **Cardinality constraint**: typically 3-20 ETFs from a universe of 1700+
- **Genetic algorithm** selects which ETFs; **SLSQP** optimises portfolio weights
- **Island-based parallel GA** with migration for better convergence (`src/simple_ga_optimisation.py`, `cpp/optimisation.cpp`)
- **CCC model** (Bollerslev 1990): forecast variances via GARCH, historical correlations for covariance
- **Copula-GARCH**: AR(1)-GARCH residuals fitted with skew-t copulas for better correlation estimation
- **Backtesting**: forward-walk out-of-sample evaluation; hypothesis testing confirms optimised portfolios significantly outperform random selection

## Data

- **ETF_Prices.csv** (98 MB): daily adjusted close for ~1792 ETFs (2014-2025)
- **time_series_20251016_113257.csv** (13 MB): InvestNow NZ managed fund data
- **expected_returns.csv** / **variances.csv**: ARIMA/GARCH forecast outputs
- NZ-specific and leveraged ETF subsets also available
- Data files are gitignored where large; do not commit raw price CSVs without checking size

## Conventions

- Log returns are used for fitness calculations in the GA
- Missing price data is forward-filled then backward-filled
- Portfolio weights sum to 1.0 (fully invested, no leverage)
- No short selling by default (weights >= 0)
- Tests use `unittest` in `tests/`; run `python -m unittest discover tests` to verify changes
- When modifying optimisation parameters (population size, generations, mutation rate, migration), document the rationale — small changes significantly affect convergence

## Database

All optimisation results, backtest metrics, and data provenance are stored in `Data/portfolio.db` (SQLite). CSVs remain for backward compatibility.

### Schema (12 tables)

| Table | Purpose |
|-------|---------|
| `exchanges` | Market groupings: US, NZX, ASX |
| `tickers` | Master list of instruments (unique per exchange) |
| `prices` | Daily close prices (one row per ticker per date) |
| `forecast_runs` | Metadata for each ARIMA/GARCH forecast generation |
| `expected_returns` | Forecasted returns linked to a forecast run |
| `variances` | Forecasted variances linked to a forecast run |
| `data_sources` | Tracks each data download event |
| `optimisation_runs` | GA/MIP/Monte Carlo run parameters and results |
| `portfolio_holdings` | ETF selections + weights per optimisation run |
| `backtest_sessions` | One row per backtest execution |
| `backtest_results` | Per-portfolio metrics within a backtest session |

### Key relationships

- `tickers.exchange_id` → `exchanges.id`
- `prices.ticker_id` → `tickers.id`
- `expected_returns/variances.ticker_id` → `tickers.id`
- `expected_returns/variances.forecast_run_id` → `forecast_runs.id`
- `portfolio_holdings.run_id` → `optimisation_runs.id`
- `backtest_results.session_id` → `backtest_sessions.id`

### Python usage

```python
from src import db

conn = db.get_connection()                    # Opens DB, creates tables if needed
db.save_prices(conn, df, exchange='US')       # Save wide-format price DataFrame
prices = db.load_prices(conn, exchange='US')  # Load as wide-format DataFrame
db.save_optimisation_run(conn, params, results, holdings)
db.save_forecast_results(conn, er_series, var_series, n_periods=252)
conn.close()
```

### Exchange codes

- **US** — United States (ETFs, stocks from Yahoo Finance)
- **NZX** — New Zealand Exchange (ETFs, InvestNow managed funds)
- **ASX** — Australian Securities Exchange
