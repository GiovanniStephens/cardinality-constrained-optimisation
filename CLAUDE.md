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
├── optimisers/              # All portfolio optimisation algorithms
│   ├── __init__.py          # Re-exports all optimisers + BaseOptimiser
│   ├── base.py              # BaseOptimiser ABC
│   ├── pygad_ga.py          # PyGAD GA + SLSQP + copula/CCC correlation
│   ├── island_ga.py         # Parallel island-based GA
│   ├── monte_carlo.py       # Monte Carlo brute-force baseline
│   └── mip.py               # Mixed Integer Linear Programming
├── backtest.py              # Forward-walk backtesting with Sharpe/Sortino/Calmar/drawdown stats
├── forecast.py              # ARIMA returns + GARCH variance forecasting
├── db.py                    # SQLite database module (schema, save/load functions, CSV migration)
├── portfolio_utils.py       # Shared utility functions + OptimisationResult
├── config.py                # Centralised algorithm/pipeline configuration
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
data/                        # CSV price data, ETF lists, forecast outputs (~112 MB)
├── portfolio.db             # SQLite database (gitignored, created by db.py)
├── ETF_Prices.csv           # Daily adjusted close for ~1792 ETFs
└── ...                      # Other CSV data files
images/                      # Visualisation outputs
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
python -m src.optimisers.island_ga

# Run the full optimisation with copulas/forecasts
python -m src.optimisers.pygad_ga

# Run backtests
python -m src.backtest

# Generate ARIMA/GARCH forecasts
python -m src.forecast

# Download price data (ETFs + stocks)
python -m src.download_data --asset-types equities etfs
python -m src.download_data --incremental   # only new dates

# Validate data quality (flag bad tickers)
python -m src.data_quality                  # validate and flag
python -m src.data_quality --dry-run        # preview without writing

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
- **Cardinality constraint**: typically 10-20 instruments from a universe of thousands
- **Genetic algorithm** selects which instruments; **SLSQP** optimises portfolio weights
- **Island-based parallel GA** with migration for better convergence (`src/optimisers/island_ga.py`, `cpp/optimisation.cpp`)
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
- **Covariance estimation rule:** NEVER compute the full M×M covariance matrix when M > ~30. The sample covariance from ~1260 daily observations is rank-deficient and noise-dominated at large M. Always compute the sub-covariance matrix for each candidate portfolio (n ≤ 30 securities), where the estimation problem is well-conditioned (T >> n). The equal-weight column-sum shortcut (`||Xc @ s||² / (n² × (T-1))`) is permitted as a fast path during GA search since it is mathematically equivalent to `w^T @ Σ_sub @ w` for equal weights.

## Sharpe Ratio Overfitting — Critical Awareness

### The Pitfall

When we optimise a portfolio on historical data (maximising the in-sample Sharpe ratio), the reported Sharpe is **biased upward**. The GA searches over a vast combinatorial space of portfolio selections (choosing 10-20 from 1700+ instruments), and the best-found solution will almost certainly exploit noise patterns in the training data that do not persist out-of-sample. This is compounded by SLSQP weight optimisation, which further tailors weights to historical noise.

**This is not a bug — it is an inherent property of optimisation on finite samples.**

### Evidence in This Project

- Benchmark in-sample Sharpe ratios: **2.3 to 5.7** (see `benchmark_results/`)
- Realistic annual equity portfolio Sharpe ratios: **0.3 to 1.0**
- Typical in-sample to out-of-sample degradation: **30-50%** (median ~44% per academic literature)
- A Sharpe ratio of 5.7 on annual equity data is not a sign of a great strategy — it is a sign of overfitting

### Academic References

1. **Bailey & López de Prado (2014)**, "The Deflated Sharpe Ratio" (*Journal of Portfolio Management*, Vol. 40, No. 5, pp. 94-107) — corrects observed Sharpe for selection bias when multiple strategies are tested and for non-normality (skewness, kurtosis) of returns. The DSR gives the probability that an observed Sharpe is genuine after accounting for how many trials were run.

2. **Harvey, Liu, Zhu (2016)**, "…and the Cross-Section of Expected Returns" (*Review of Financial Studies*, Vol. 29, No. 1, pp. 5-68) — argues the t-statistic threshold for significance should be ~3.0, not the traditional 2.0, to account for multiple testing across hundreds of strategies/factors.

3. **López de Prado (2015)**, "The Probability of Backtest Overfitting" — provides a model-free, metric-agnostic framework for computing the probability that a backtested strategy is overfit, using Combinatorially Symmetric Cross-Validation (CSCV).

### Key Formula: Variance of the Sharpe Ratio Estimator

```
Var(SR) = [1 - skewness * SR + (excess_kurtosis / 4) * SR^2] / T
```

Where `T` is the number of observations. For normal returns (skewness=0, excess kurtosis=0), this simplifies to `1/T`. Fat tails and negative skew (common in equities) inflate the variance further.

### Mitigations in Place

- Forward-walk backtesting with non-overlapping OOS windows (5yr train + 1yr test)
- Multiple evaluation metrics beyond Sharpe (Sortino, Calmar, max drawdown)
- Hypothesis testing (paired t-tests, Friedman tests) across windows
- Cardinality constraints (10-20 holdings) limiting parameter space
- Weight bounds (5-45%) preventing extreme concentration

### Future Work

- Ledoit-Wolf shrinkage estimators for the covariance matrix
- Full Deflated Sharpe Ratio computation gating results
- Combinatorially Purged Cross-Validation (CPCV) for more robust OOS evaluation
- Transaction cost modelling to further deflate apparent performance
- Regime-aware validation (compare training vs test period volatility)

### Rules for Interpreting Results

- **Never trust an in-sample Sharpe above 2.0** for annual equity portfolios without strong OOS confirmation
- **Sharpe above 3.0** on annual data is almost certainly overfit (Harvey et al. 2016 threshold)
- **Always report IS and OOS Sharpe side-by-side** — the gap is the overfitting signal
- The benchmark framework reports **in-sample** fitness values; these measure optimisation quality, not expected real-world performance
- When in doubt, apply a 50% haircut to any in-sample Sharpe as a rough OOS estimate

## Database

All optimisation results, backtest metrics, and data provenance are stored in `data/portfolio.db` (SQLite). CSVs remain for backward compatibility.

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

- **US** — United States (ETFs, stocks, ADRs — all FinanceDatabase instruments are US-listed)
- **NZX** — New Zealand Exchange (ETFs, InvestNow managed funds)
- **ASX** — Australian Securities Exchange

Note: Geographic exposure is tracked via the `country` column on `tickers`, not via exchange codes. FinanceDatabase instruments are all US-listed; the `country` field reflects the company's domicile.

## Investment Universe

### Investor Profile

- **Broker**: Interactive Brokers (US + international markets)
- **Tax jurisdiction**: New Zealand — FIF rules apply (5% FDR on offshore holdings over $50k cost)
- **Existing exposure**: Significant NZ/AU via KiwiSaver — IB portfolio should diversify away from this
- **Portfolio size**: $50k–$200k NZD

### Universe Scope

All instruments sourced from FinanceDatabase (US-listed). Configuration in `src/universe_config.py`.

- **Equities**: ~22k across 27 countries (US, Canada, UK, Japan, Australia, Germany, France, Switzerland, Netherlands, Sweden, Norway, Denmark, Finland, Ireland, Belgium, Austria, Singapore, Hong Kong, Israel, New Zealand, Brazil, Mexico, India, South Korea, Taiwan, South Africa, Thailand). Foreign companies are available as US-listed ADRs.
- **ETFs**: ~2,900 covering equities, bonds/treasuries, commodities, REITs, crypto, managed futures/CTA
- **Excluded**: China, Russia (geopolitical risk)
- **History filter**: 5+ years of daily price data required (~1,260 trading days)

### Asset Classes of Interest

| Class | Examples | Diversification role |
|-------|---------|---------------------|
| Bonds/Treasuries | TLT, IEF, AGG, TIPS | Negative equity correlation |
| Commodities | GLD, SLV, DBC, DBA | Low equity correlation |
| REITs | VNQ, VNQI | Different return drivers |
| Managed futures/CTA | DBMF, KMLM | Designed for low/negative correlation |
| Crypto ETFs | BITO, IBIT, ETHE | Low traditional-asset correlation |
| International equities | ADRs + country ETFs | Geographic diversification |

### Portfolio Constraints

- **Positions**: 10–20 (cardinality constraint)
- **Rebalancing**: Quarterly
- **Objective**: Maximise Sharpe ratio with maximal inter-holding decorrelation
