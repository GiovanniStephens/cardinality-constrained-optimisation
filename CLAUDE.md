# Cardinality-Constrained Portfolio Optimisation

## Project Overview

Solves the cardinality-constrained portfolio selection problem: find an optimal subset of N ETFs from a universe of M ETFs that maximises the Sharpe ratio, subject to position count, weight, and budget constraints. Uses genetic algorithms for ETF selection and SLSQP for weight optimisation. Validated via out-of-sample backtesting.

## Tech Stack

- **Python 3.10+** — primary language
- **C++** — high-performance parallel GA (`cpp/optimisation.cpp`, compiled to `cpp/optimisation` binary)
- **Metal (GPU)** — Apple Silicon GPU-accelerated fitness evaluation via Metal compute shaders (`cpp/metal_fitness.mm`)
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
├── backtest.py              # Forward-walk backtesting orchestrator (portfolio creation, OOS eval)
├── backtest_types.py        # Dataclasses: WindowSpec, PortfolioResult, MethodResults, WindowResult
├── backtest_windows.py      # Window generation, data slicing, cross-window aggregation
├── backtest_simulation.py   # Portfolio simulation: get_random_weights, run_portfolio, get_statistics
├── backtest_statistics.py   # Hypothesis tests: difference_of_means, paired_t_test, friedman_test
├── forecast.py              # ARIMA returns + GARCH variance forecasting
├── db.py                    # SQLite database module (schema, migrations, save/load functions)
├── portfolio_utils.py       # OptimisationResult dataclass + save_optimisation_result DB helper
├── returns.py               # Log returns, expected returns, variances
├── covariance.py            # Ledoit-Wolf, copula-CCC, shrinkage covariance estimation
├── metrics.py               # Sharpe, Sortino, Calmar, drawdown, overfitting detection
├── weights.py               # SLSQP weight optimisation, risk-parity, portfolio variance
├── data_loading.py          # DB-first / CSV-fallback price data loading
├── binary_io.py             # Binary data format for C++ optimiser
├── group_constraints.py     # Group allocation constraints (country, sector) for SLSQP
├── config.py                # Centralised algorithm/pipeline/universe configuration
├── download_data.py         # Yahoo Finance data downloader
├── pipeline.py              # Orchestrates data download, quality checks, and forecasting
├── data_quality.py          # Data validation and bad-ticker flagging
└── logging_config.py        # Centralised logging setup
tests/                       # Unit and integration tests
├── __init__.py
├── helpers.py               # Shared test utilities + base classes (BaseDBTest, BaseTmpDirTest, OptimiserTestMixin)
├── test_optimisers.py       # Tests for optimisation algorithms
├── test_backtest.py         # Tests for backtest module
├── test_db.py               # Tests for database module
├── test_portfolio_utils.py  # Tests for portfolio utilities
├── test_securities.py       # Tests for security universe/download
├── test_forecast.py         # Tests for ARIMA/GARCH forecasting
├── test_group_constraints.py # Tests for group allocation constraints
├── test_data_quality.py     # Tests for data validation
├── test_pipeline.py         # Tests for pipeline orchestration
├── test_cpp_equivalence.py  # Tests for C++/Python parity + Metal GPU equivalence
├── test_backtest_integration.py     # Integration tests for backtesting
├── test_benchmark_integration.py    # Integration tests for benchmarks
├── test_forecast_integration.py     # Integration tests for forecasting
└── test_pipeline_integration.py     # Integration tests for pipeline
cpp/                         # C++ parallel island GA implementation
├── CMakeLists.txt           # CMake build config (auto-detects Metal on macOS)
├── optimisation.cpp         # GA/MC source — `--gpu` flag enables Metal path
├── metal_fitness.h          # PIMPL header for GPU evaluator (pure C++, no ObjC)
├── metal_fitness.mm         # ObjC++ Metal implementation (shader + host code)
└── optimisation             # Compiled binary (gitignored, rebuild via cmake)
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
run_throughput_benchmark.py  # CLI entry point for throughput benchmarking
Makefile                     # Build automation (install, test, build-cpp, clean)
pyproject.toml               # Project metadata, dependencies, console_scripts
.github/workflows/test.yml   # CI: runs tests on push/PR
CLAUDE.md
README.md
```

## Common Commands

```bash
# Install (editable mode, includes all dependencies from pyproject.toml)
pip install -e ".[dev]"
# or
make install

# Run the main GA optimisation (InvestNow data)
python -m src.optimisers.island_ga

# Run the full optimisation with copulas/forecasts
python -m src.optimisers.pygad_ga

# Run backtests (also available as: portfolio-backtest)
python -m src.backtest

# Generate ARIMA/GARCH forecasts (also: portfolio-forecast)
python -m src.forecast

# Download price data (also: portfolio-download)
python -m src.download_data --asset-types equities etfs
python -m src.download_data --incremental   # only new dates

# Validate data quality (flag bad tickers)
python -m src.data_quality                  # validate and flag
python -m src.data_quality --dry-run        # preview without writing

# Database
python -m src.db                # Create empty database with schema
python -m src.db migrate        # Import existing CSVs into database

# Run benchmarks (also: portfolio-benchmark)
python run_benchmark.py

# Build C++ optimiser (requires CMake 3.14+, Eigen and csv-parser submodules)
# On macOS with Metal, GPU support is auto-detected and compiled in.
make build-cpp

# Run C++ GA with Metal GPU acceleration (Apple Silicon only)
./cpp/optimisation --binary --data data.bin --gpu --time-budget 10

# Run tests
python -m unittest discover tests
# or
make test
```

## Metal GPU Acceleration

The `--gpu` flag enables Metal compute shader fitness evaluation on Apple Silicon. The GPU evaluates entire populations in parallel while the CPU handles GA operators (selection, crossover, mutation).

**Architecture**: each threadgroup evaluates one portfolio. 64 threads per group parallelize over the time dimension (T=1260 rows) with coalesced memory access. Thread 0 extracts selected ETF indices via bit-scan, all threads accumulate column values for their time slice, then a tree reduction combines partial squared norms for the Sharpe ratio.

**Build**: CMake auto-detects Metal/Foundation frameworks on macOS. On other platforms the binary compiles without GPU support; `--gpu` prints a warning and falls back to CPU.

**Correctness**: GPU uses FP32 during the GA search (sufficient for ranking — max 0.00002% Sharpe error vs FP64). After the GA completes, each island's best solution is re-evaluated with FP64 on CPU for precise final reporting. Tests in `tests/test_cpp_equivalence.py::TestMetalGpuEquivalence` verify GPU/CPU/Python parity.

**Performance** (M4, M=1800, T=1260, n=15, pop=1000):

| Config | Evals/sec |
|--------|-----------|
| CPU 1 island | 268K |
| GPU 1 island | 1.2M (4.5x) |
| CPU 10 islands | 1.6M |
| GPU 10 islands | 5.8M (3.6x) |

**Limitations**: `maxETFs` must be ≤ 64 (shader `selectedIndices` array size). `THREADS_PER_GROUP` must be a power of 2 (tree reduction correctness). Multiple islands share one GPU evaluator — command buffers are thread-safe but GPU dispatches serialize on the command queue.

## Key Concepts

- **Sharpe ratio** = E(R) / Std(R) is the primary objective function
- **Cardinality constraint**: typically 10-20 instruments from a universe of thousands
- **Genetic algorithm** selects which instruments; **SLSQP** optimises portfolio weights
- **Island-based parallel GA** with migration for better convergence (`src/optimisers/island_ga.py`, `cpp/optimisation.cpp`)
- **CCC model** (Bollerslev 1990): forecast variances via GARCH, historical correlations for covariance
- **Copula-GARCH**: AR(1)-GARCH residuals fitted with skew-t copulas for better correlation estimation
- **Backtesting**: forward-walk out-of-sample evaluation; hypothesis testing confirms optimised portfolios significantly outperform random selection

## Data

- **Prices.csv** (~287 MB): daily adjusted close for equities + ETFs (2014-2025)
- **Securities.csv**: ticker metadata (name, country, asset type) from FinanceDatabase
- **expected_returns.csv** / **variances.csv**: ARIMA/GARCH forecast outputs
- **known_bad_tickers.csv**: cached tickers that failed download validation
- **portfolio.db**: SQLite database (gitignored, created by `src/db.py`)
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
- Ledoit-Wolf shrinkage covariance estimation (default for all covariance paths)
- T >> N observation ratio guards (error if T/N < 1, warn if T/N < 10)

### Future Work
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

All instruments sourced from FinanceDatabase (US-listed). Configuration in `src/config.py`.

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

### Group Allocation Constraints

Configured in `src/config.py` `GROUP_CONSTRAINTS`. Enforced as linear constraints in SLSQP weight optimisation. Implementation in `src/group_constraints.py` + `src/weights.py`.

| Dimension | Group | Max | Rationale |
|-----------|-------|-----|-----------|
| `country` | United States | 60% | Prevent US dominance from recent outperformance |
| `country` | Each non-US country | 20% | Prevent single non-US country concentration |
| `sector` | Each of 11 GICS sectors | 50% | Prevent sector concentration (especially tech) |

Supported dimensions: `country`, `sector`, `asset_type`, `category_group`. Tickers with NULL metadata are unconstrained. Run `python -m src.db backfill` to populate metadata from FinanceDatabase.
