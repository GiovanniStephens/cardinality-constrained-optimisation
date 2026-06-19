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
│   ├── base.py              # BaseOptimiser ABC + OptimisationResult dataclass
│   ├── pygad_ga.py          # PyGAD GA + SLSQP + copula/CCC correlation
│   ├── island_ga.py         # Parallel island-based GA
│   ├── monte_carlo.py       # Monte Carlo brute-force baseline
│   └── mip.py               # Mixed Integer Linear Programming
├── backtest/                # Forward-walk backtesting package
│   ├── __init__.py          # Re-exports public API for `from src.backtest import ...`
│   ├── __main__.py          # CLI entry point (`python -m src.backtest`)
│   ├── runner.py            # Orchestrator: evaluate_window(), main()
│   ├── types.py             # Dataclasses: WindowSpec, PortfolioResult, MethodResults, WindowResult
│   ├── windows.py           # Window generation, data slicing
│   ├── simulation.py        # Portfolio simulation: get_random_weights, run_portfolio, get_statistics
│   └── statistics.py        # Hypothesis tests, cross-window aggregation
├── forecast.py              # ARIMA returns + GARCH variance forecasting
├── db/                      # SQLite database package (schema, migrations, save/load functions)
│   ├── __init__.py          # Re-exports all public functions for `from src import db` usage
│   ├── __main__.py          # CLI entry point (create DB, migrate CSVs, backfill metadata)
│   ├── schema.py            # Schema DDL, version, default seed data
│   ├── connection.py        # Connection management (get_connection, DB_PATH)
│   ├── tickers.py           # Ticker CRUD, metadata backfill, exclusion flags
│   ├── prices.py            # Price data storage and retrieval
│   ├── bad_tickers.py       # Known-bad ticker cache for download retry logic
│   ├── forecasts.py         # Forecast data storage and retrieval
│   ├── optimisation.py      # Optimisation run storage and retrieval
│   ├── backtest.py          # Backtest session and result storage
│   ├── metadata.py          # Data source and metadata functions
│   └── migrations.py        # CSV migration functions for importing legacy data
├── portfolio_utils.py       # save_optimisation_result DB helper (OptimisationResult re-exported for compat)
├── returns.py               # Log returns, expected returns, variances
├── covariance.py            # Ledoit-Wolf, copula-CCC, shrinkage covariance estimation
├── metrics.py               # Sharpe, Sortino, Calmar, drawdown, overfitting detection
├── weights.py               # SLSQP weight optimisation, risk-parity, portfolio variance
├── data_loading.py          # DB-first / CSV-fallback price data loading
├── binary_io.py             # Binary data format for C++ optimiser
├── group_constraints.py     # Group allocation constraints (country, sector) for SLSQP
├── config.py                # Centralised algorithm/pipeline/universe configuration
├── universe.py              # Security universe building (FinanceDatabase queries, ticker filtering)
├── download/                # Data download package (Yahoo Finance)
│   ├── __init__.py          # Re-exports public API
│   ├── __main__.py          # CLI entry point (`python -m src.download`)
│   ├── core.py              # Download primitives, download_and_save
│   ├── cli.py               # Argparse CLI, logging, summary reporting
│   ├── session.py           # Proxy/Tor session management, circuit rotation
│   ├── validate.py          # Ticker validation with resumable caching
│   └── workers.py           # Multi-worker concurrent download (threads, subprocesses)
├── exceptions.py            # Domain exception hierarchy (PortfolioError base)
├── pipeline.py              # Orchestrates data download, quality checks, and forecasting
├── data_quality.py          # Data validation and bad-ticker flagging
└── logging_config.py        # Centralised logging setup
tests/                       # Unit and integration tests
├── __init__.py
├── helpers.py               # Shared test utilities + base classes (BaseDBTest, BaseTmpDirTest, OptimiserTestMixin)
├── test_optimisers.py       # Tests for optimisation algorithms
├── test_backtest.py         # Tests for backtest module
├── test_backtest_runner.py  # Tests for src/backtest/runner.py
├── test_db.py               # Tests for database module
├── test_portfolio_utils.py  # Tests for portfolio utilities
├── test_securities.py       # Tests for security universe/download
├── test_forecast.py         # Tests for ARIMA/GARCH forecasting
├── test_group_constraints.py # Tests for group allocation constraints
├── test_download_session.py  # Tests for download session/proxy management
├── test_download_validate.py # Tests for ticker validation
├── test_download_workers.py  # Tests for multi-worker concurrent download
├── test_data_quality.py     # Tests for data validation
├── test_pipeline.py         # Tests for pipeline orchestration
├── test_cli_entrypoints.py  # Tests that every console_script / `python -m` entry still imports
├── test_logging_config.py   # Tests for centralised logging setup
├── test_benchmark_analysis.py       # Tests for benchmark/analysis.py
├── test_cpp_equivalence.py  # Tests for C++/Python parity + Metal GPU equivalence
├── test_backtest_integration.py     # Integration tests for backtesting
├── test_benchmark_integration.py    # Integration tests for benchmarks
├── test_forecast_integration.py     # Integration tests for forecasting
└── test_pipeline_integration.py     # Integration tests for pipeline
cpp/                         # C++ parallel island GA implementation
├── CMakeLists.txt           # CMake build config (auto-detects Metal on macOS)
├── optimisation.cpp         # GA orchestrator — CLI, island dispatch, result reporting
├── data_io.h                # Data I/O and preprocessing (header)
├── data_io.cpp              # Data I/O and preprocessing (implementation)
├── ga_types.h               # GA types, operators, fitness (header-only)
├── monte_carlo.h            # Monte Carlo worker (header)
├── monte_carlo.cpp          # Monte Carlo worker (implementation)
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
├── portfolio.db             # SQLite database (gitignored, primary store, created by src/db)
├── Prices.csv               # Daily adjusted close, equities + ETFs (gitignored, ~287 MB)
├── Securities.csv           # Ticker metadata from FinanceDatabase
├── ETF_Prices.csv           # Legacy ETF-only CSV; used by migration path in src/db/migrations.py
└── ...                      # Other CSV data files (forecast outputs, bad-ticker caches, etc.)
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
python -m src.backtest                          # walk-forward (default): 5y train + 6mo OOS, 17 windows
python -m src.backtest --mode cpcv              # Combinatorially Purged CV: 66 splits with PBO + CIs

# Generate ARIMA/GARCH forecasts (also: portfolio-forecast)
python -m src.forecast

# Download price data (also: portfolio-download) — see docs/DATA_REFRESH.md
make refresh                                # full refresh + auto-validate + health check
make health-check                           # verify the liquid core is current (no download)
make retry-core                             # recovery: force-refresh core watchlist, bypass cache
python -m src.download --asset-types equities etfs   # (what `make refresh` runs)
python -m src.download --incremental        # only new dates (now promotes correctly)

# Validate data quality (flag bad tickers) — now AUTO-runs after every promotion
python -m src.data_quality                  # validate and flag
python -m src.data_quality --dry-run        # preview without writing

# Database
python -m src.db                # Create empty database with schema
python -m src.db migrate        # Import existing CSVs into database

# Run benchmarks (also: portfolio-benchmark)
python run_benchmark.py
python run_benchmark.py --quick                                  # fast smoke run
python run_benchmark.py --algorithms pygad_ga island_ga --runs 10 --time-budget 30
python run_throughput_benchmark.py                               # evals/sec throughput comparison

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
- **portfolio.db**: SQLite database (gitignored, created by `src/db`)
- **core_etfs.csv**: ~87 liquid IB-tradeable ETFs — the *protected watchlist* (never bad-cached) and the health-check verification list
- Data files are gitignored where large; do not commit raw price CSVs without checking size

### Data-refresh gotchas (full runbook: `docs/DATA_REFRESH.md`)

1. **Trust the health check, not the summary.** A refresh isn't done until `make health-check` shows `✅ all core tickers current`. The "Totals: X saved" line can be green while `SPY` silently never refreshed.
2. **The bad-ticker cache can blacklist good tickers.** `known_bad_tickers` skips failed tickers on every run. Now mitigated: `core_etfs.csv` is protected (never skipped), entries expire after 30 days (`PIPELINE_BAD_TICKER_TTL_DAYS`), and it takes 3 failures (`PIPELINE_BAD_CACHE_MIN_FAILURES`) to skip. Inspect with `sqlite3 data/portfolio.db "SELECT * FROM known_bad_tickers WHERE symbol='SPY'"`; bypass with `--ignore-bad-cache`.
3. **A full refresh takes hours** (Yahoo throttles the proxy; the ~36k universe is mostly dead/foreign tickers). More workers won't help — capped at 24 and throttle-bound. Incremental refresh now promotes correctly (min_history skipped on staging; prod re-validated post-promotion).

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

When we optimise a portfolio on historical data (maximising the in-sample Sharpe ratio), the reported Sharpe is **biased upward**. The GA searches over a vast combinatorial space of portfolio selections (choosing 8-20 from ~860 long-history instruments after the 95% coverage filter), and the best-found solution will almost certainly exploit noise patterns in the training data that do not persist out-of-sample. This is compounded by SLSQP weight optimisation, which further tailors weights to historical noise.

**This is not a bug — it is an inherent property of optimisation on finite samples.**

### Empirical Findings (May 2026 backtest series)

The walk-forward + CPCV results materially refined the priors. Key findings:

- **Walk-forward backtest (v6, 17 windows × 16 strategies)**: top method `cc_ccc_baseline` reported mean OOS Sharpe **+1.86** with std 3.4. Other GA-selected methods (`cc_inverse_vol` +1.76, `cc_copulae` +1.72, `cc_min_variance` +1.58) clustered nearby. `mc_optimised` was rank 10 at +0.95 with much lower variance.

- **CPCV backtest** (n_groups=12, k=2 → 66 splits, full run, ~9h50m wall): the ranking **flipped**. `cc_copulae` rose to rank 1 at +1.08 (std 0.67, widest of any method); `mc_optimised` was rank 2 at +1.04 with the tightest distribution (std 0.29). `cc_ccc_baseline` collapsed to +0.96 (−48% from walk-forward). `cc_min_variance` collapsed to +0.81 (−49%, biggest IS-OOS gap pre-split-66). **PBO = 0.909** — the in-sample-best method ranks below median OOS in 91% of CPCV splits, a strong overfitting signal at the strategy-family level. PBO climbed monotonically with sample size (0.79 at 28 splits, 0.85 at 65 splits, 0.909 at 66 splits): the partial-run numbers underestimated the overfitting.

- **The GA-selection / fancy-weighting edge in walk-forward was largely an artefact of regime concentration** between 5-year train and 6-month test windows. Once CPCV breaks that regime correlation, the simpler MC + equal-weight methods dominate.

- **Heavier weight-optimisation doesn't pay off OOS — and the lever is the covariance/weighting choice, not the expected-return input.** All four max-Sharpe SLSQP variants (`cc_optimised`, `cc_copulae`, `cc_ccc_baseline`, `mc_optimised`) feed on the *same* noisy sample-mean ER; what separates them is the covariance construction and the selection. `cc_optimised` (Ledoit-Wolf sample cov) sits mid-pack (rank 5), while the correlation-focused `cc_copulae` leads and the ER-free heuristic weightings (equal-weight, inverse-vol) travel well. Swapping sample-mean ER for an ARIMA forecast (`cc_arima_er`) was only a marginal lift. *(Genuinely ER-free methods: min-variance, inverse-vol, risk-parity, max-diversification, equal-weight. `cc_copulae` and `cc_ccc_baseline` still use sample-mean ER — they differ from `cc_optimised` only in how the covariance is built.)*

- **GARCH variance forecasting hurts**. `cc_garch_var` and `cc_arima_garch` rank near the bottom of cc_* methods in both walk-forward and CPCV (~50-62% IS-OOS degradation). The GARCH inputs add noise instead of signal.

- **ARIMA on returns is a marginal lift** (`cc_arima_er` rank 6-9 in both runs). The ER forecast adds something but not enough to justify complexity.

- **Realistic OOS Sharpe ceiling for this universe**: ~1.0-1.2 net of costs. Not 1.5+. Anything above 1.5 OOS sustained over 10+ years is a red flag for selection bias, regime concentration, or undisclosed leverage. See DeMiguel/Garlappi/Uppal (2009) and the live track records of multi-strategy funds (AQR Style Premia delivered 0.1 net 2013-2024 vs 0.9 backtest).

### Academic References

1. **Bailey & López de Prado (2014)**, "The Deflated Sharpe Ratio" (*Journal of Portfolio Management*, Vol. 40, No. 5, pp. 94-107) — corrects observed Sharpe for selection bias when multiple strategies are tested and for non-normality (skewness, kurtosis) of returns. The DSR gives the probability that an observed Sharpe is genuine after accounting for how many trials were run. **Implemented in `src/metrics.deflated_sharpe_ratio` and gated per-window in `src/backtest/runner.py`.**

2. **Harvey, Liu, Zhu (2016)**, "…and the Cross-Section of Expected Returns" (*Review of Financial Studies*, Vol. 29, No. 1, pp. 5-68) — argues the t-statistic threshold for significance should be ~3.0, not the traditional 2.0, to account for multiple testing across hundreds of strategies/factors.

3. **López de Prado (2018)**, *Advances in Financial Machine Learning*, Wiley, Ch. 12 — Combinatorially Purged Cross-Validation (CPCV) and Probability of Backtest Overfitting (PBO). **Implemented in `src/backtest/cpcv.py`; runnable via `python -m src.backtest --mode cpcv`.**

4. **DeMiguel, Garlappi, Uppal (2009)**, "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" (*RFS*, 22(5), 1915-1953) — across 14 methods × 7 datasets, none reliably beat 1/N OOS. Empirically validated by our CPCV run: simpler methods (MC selection, equal-weight, inverse-vol) generalise better than the GA + sophisticated weighting.

### Key Formula: Variance of the Sharpe Ratio Estimator

```
Var(SR) = [1 - skewness * SR + (excess_kurtosis / 4) * SR^2] / T
```

Where `T` is the number of observations. For normal returns (skewness=0, excess kurtosis=0), this simplifies to `1/T`. Fat tails and negative skew (common in equities) inflate the variance further.

### Mitigations in Place

- Forward-walk backtesting with non-overlapping OOS windows (5yr train + 6mo OOS, 17 windows)
- **Combinatorially Purged Cross-Validation** (`--mode cpcv`, 66 splits with purge+embargo, PBO computation)
- **Deflated Sharpe Ratio** logged per (method, window) with PASS/WEAK/FAIL gating thresholds at 0.95/0.5
- Multiple evaluation metrics beyond Sharpe (Sortino, Calmar, max drawdown)
- Hypothesis testing (paired t-tests, Friedman tests) across windows
- Cardinality constraints (research/backtest: 8-20 holdings; production rebalance: capped at 10) limiting parameter space
- Weight bounds preventing extreme concentration (research config 5-45%; production rebalance 5-25% per holding via `run_rebalance.py --max-weight 0.25`)
- Ledoit-Wolf shrinkage covariance estimation (default for all covariance paths)
- Gaussian copula correlation (default; t-copula gated behind `COPULA_TYPE='t'` due to super-cubic scaling in dim)
- Per-ticker GARCH residual cache (`src/covariance.py:_garch_residuals_cache`) to avoid redundant fits
- T >> N observation ratio guards (error if T/N < 1, warn if T/N < 10)

### Future Work

- **Hierarchical Risk Parity** (López de Prado 2016, *J. Portfolio Management*) — clustering-based weighting that avoids covariance inversion; targeted next addition. Likely beats `cc_inverse_vol` while remaining simple.
- **Tu-Zhou shrinkage to 1/N** (Tu & Zhou 2011, *JFE*) — `w = δ·w_method + (1−δ)·w_1/N` as a cheap variance reducer to bolt onto any weighting method.
- **Turnover penalty** in SLSQP objective (`λ·||w_new − w_old||₁`, Boyd et al. 2017 *FnT Optimization*) — reduces real costs and damps IS-noise-driven reweighting.
- ~~**Pre-filtered universe** (~30-60 hand-curated ETFs by asset-class + factor exposure) — single biggest PBO reducer~~ **— TESTED June 2026, it BACKFIRED.** A data-driven 142-ETF curated universe (correlation clustering, most-liquid representative) *underperformed* the broad universe on a clean common-window A/B (mean per-window OOS Sharpe **+0.17 vs +1.30**, broad won 16/22). Shrinking the universe forced concentration (~10 holdings vs ~23) and flipped return skew negative. See *Lessons Learned (June 2026)* below. Curation is only worth revisiting jointly with constraint re-tuning (relax the return floor, raise min-holdings) — not as a standalone PBO lever.
- **Harvey-Liu haircut** applied automatically: with K methods tested, expected true Sharpe ≈ S × √(1 − 2·ln(K)/T).
- **Trend-following sleeve** (DBMF or DIY 10-futures TSMOM at 12mo) — orthogonal premium, crisis alpha. ~2-3 day implementation.
- **Defined-risk short SPX vol sleeve** — VRP harvesting via 30-45 DTE 10-15 delta put spreads. Requires options data + IB Portfolio Margin.
- **Levered risk parity wrapper** (NTSX/NTSI or DIY ES + ZN/ZB futures, 1.4-1.5x notional) — captures leverage aversion premium.
- Combined multi-strategy stack target: realistic OOS Sharpe **1.4-1.7** with 3-5 uncorrelated sleeves (vs 1.0-1.2 long-only ceiling).

### Rules for Interpreting Results

- **Never trust an in-sample Sharpe above 2.0** for annual equity portfolios without strong OOS confirmation.
- **Sharpe above 3.0** on annual data is almost certainly overfit (Harvey et al. 2016 threshold).
- **Always report IS and OOS Sharpe side-by-side** — the gap is the overfitting signal.
- **Trust CPCV results over walk-forward.** Walk-forward Sharpes are systematically inflated by regime correlation between train and test windows; CPCV's combinatorial structure breaks this. Our v6 walk-forward winner (cc_ccc_baseline, 1.86) collapsed by 51% in CPCV.
- **PBO > 0.5 means the strategy family is overfit.** Our 16-method stack on the broad universe ran PBO = 0.909 (66 splits) — strong evidence that the apparent ranking from walk-forward isn't robust. (We *expected* pre-filtering the universe to be the highest-leverage PBO fix; the June 2026 curation test contradicted that — see *Lessons Learned* below.)
- The benchmark framework reports **in-sample** fitness values; these measure optimisation quality, not expected real-world performance.
- When in doubt, apply a 50% haircut to any in-sample Sharpe as a rough OOS estimate, *then* apply Harvey-Liu's multiple-testing correction on top.

## Lessons Learned (consolidated, June 2026)

Distilled across the May–June 2026 work (walk-forward v5/v6, CPCV, category caps,
must-have/return-floor, pooled lev/inv, and the curated-universe experiment). The
through-line: **breadth and diversification are the dominant drivers of OOS robustness
here; almost every attempt to "improve" the book by narrowing it or chasing return has
hurt.**

**1. Diversification beats cleverness — repeatedly.**
Every result points the same way: more holdings + a broader candidate set generalise
better than concentrated, "optimised" books. Heavier weight-optimisation hurt — the
ER-free or correlation-focused weightings (equal-weight, inverse-vol, min-var, plus the
copula/CCC *covariance* variants) all beat the full sample-mean-ER + Ledoit-Wolf
max-Sharpe SLSQP (`cc_optimised`) OOS (note copula/CCC still use sample-mean ER, so the
lever is the covariance/weighting choice, not dropping ER); GARCH variance forecasting
hurt (ranks 9–10); min-variance had the worst IS-OOS gap; and the curated universe hurt
by forcing concentration. The 1/N / broad-diversification result
(DeMiguel-Garlappi-Uppal 2009) keeps winning. **Default to broad + simple + many holdings.**

**2. A smaller universe forced concentration and worsened the tail (the curation test).**
A 142-ETF data-driven curated universe (US-listed + 5y history + correlation clustering,
most-liquid representative) was beaten by the broad universe on a clean **common-window**
A/B: mean per-window OOS Sharpe **+0.17 (curated) vs +1.30 (broad)**, broad won 16/22.
Cause: with only ~142 candidates the **12% return floor binds hard**, so the GA could
only assemble ~10 high-beta holdings (vs ~23 for broad), which blew up in crisis quarters
(2020 COVID quarter: −110% annualised). Curation *lowered* excess kurtosis (67→39.6) but
**flipped return skew from +2.40 to −2.23** — i.e. it traded an upside fat tail for a
downside one. Net: strictly worse risk profile. Machinery lives in `curate_universe.py`
+ `--curated`; the volume/ADV plumbing (schema v4, `backfill-volume`) is reusable.

**3. Constraints do not transfer across universe sizes.**
The 12% return floor was tuned for the broad universe; on a small one it became a
concentration trap. **Any universe change must be re-paired with constraint re-tuning**
(return floor, min/max holdings). The untested-but-likely fix for curation: relax the
floor to ~0–8% and/or raise min-holdings so a small liquid set is forced to diversify.

**4. The return floor is a recurring footgun.** 15% got gamed by FX-collapse trades →
lowered to 12%; 12% then forced concentration on the curated universe. A hard return
floor repeatedly pushes the optimiser toward fragile, high-beta corners. Prefer letting
the Sharpe objective + diversification constraints do the work over flooring return.

**5. Walk-forward Sharpes are inflated; trust CPCV.** Train/test regime correlation lifts
walk-forward numbers (v6 winner cc_ccc_baseline 1.86 → collapsed ~48% in CPCV; PBO=0.909).
Realistic long-only OOS ceiling remains **~1.0–1.2** net of costs.

**6. Pooling lev/inv at 20% helped only marginally** (0.97→1.08 Sharpe, within noise) and
left the fat tails unchanged — kept *split* in production per design.

### Methodology lessons (how to test, not just what)

- **Always compare on COMMON windows.** The curated run looked catastrophic (−0.25 vs
  +0.97) on raw aggregates purely because a *today*-curated static list has no pre-2019
  history, so it silently skipped calm early windows and was judged on a crisis-heavy
  2020–2026 sample. The real story only appeared after aligning to the 22 shared windows.
- **A static "currently-liquid" allow-list cannot be cleanly walk-forward-tested into the
  past** — its constituents didn't all exist. Such a list is a *forward* tool; validate it
  on the period where it has full history, against the broad set over the same period.
- **Watch the holdings count, not just the Sharpe.** "Always hits the min-holdings floor"
  was the tell that a constraint (not the universe) was binding and driving the result.
- **Reduced kurtosis is not automatically good — check skew.** A lower fat-tail moment with
  negative skew (downside tail) is worse than a higher one with positive skew (upside tail).

### Managed-futures sleeve & the path past 1.0 (June 18, 2026)

**7. A managed-futures sleeve is the first lever that *helps* OOS instead of backfiring.**
Built a synthetic, **parameter-free** time-series-momentum sleeve (TSMOM, Moskowitz-Ooi-Pedersen
2012 — 12-month trend, long/short, vol-targeted) on a fixed basket of long-history liquid ETFs
(SPY/IEF/DBC/GLD/VNQ), blended into the book at the *portfolio* level (`(1−α)·book + α·sleeve`).
Machinery in `src/sleeves/` (`trend.py` engine, `overlay.py` full-history cache), gated behind
`config.BACKTEST_RUN_SLEEVE_STRATEGIES` (default off), A/B-registered as `<base>_trend15/25/35`
arms; run via `run_sleeve_experiment.py`. Reality-check vs real CTAs: the synthetic stream
correlates **+0.48 with DBMF** at a matching ~10.5% annualised vol — a faithful proxy. The
"futures" of managed futures = a long/short, vol-targeted *trend rule*; we model that return
stream on cash-ETF price tape rather than holding the ETFs long.

**8. The sleeve's value only appears under CPCV — walk-forward actively masks it.** Indicative
(reduced-param) runs: in *walk-forward* the sleeve appeared to HURT the inflated high-base
methods (`cc_inverse_vol`, `cc_copulae`) while helping the honest low-base ones (`mc_optimised`,
`cc_equal_weight`). In *CPCV* it **lifts all four bases** — confirmed at **full fidelity** (66 splits, 8000 pop × 30
portfolios, ~10.2h): mean **+0.02 to +0.07** (best `mc_optimised_trend35` **+0.907** vs +0.873
base), with **lower std for 3 of 4** → a clearer **risk-adjusted (mean/std)** gain (`mc_optimised`
2.64→3.02; `cc_copulae` 2.72→3.21). Same mechanism as lesson 5: walk-forward regime-inflation
hides a diversifier; CPCV reveals it. **Two honest caveats from the deciding run:** (a) the mean
lift stays **WITHIN the 95% CIs** — consistent and reliable, but **not statistically decisive**;
(b) the family-level **PBO went the wrong way at full fidelity** (0.606 with the sleeve vs **0.515**
base-only — the *reduced* run had it the other way, 0.591 vs 0.636), most likely a
correlated-near-duplicate-arms artifact (4 bases × 3 α are near-identical, inflating PBO rank-noise),
but by that metric the sleeve does **not** reduce overfitting. **Net: a genuine, small,
variance-reducing lift — exactly what the `√(ΣSᵢ²)` math (lesson 11) predicts for *one* ETF-proxy
sleeve (~0.4 standalone S). The compounding toward 1.4+ comes from the multi-market futures upgrade
+ additional orthogonal sleeves, not from this proxy alone.**

**9. A diversifier shows up in variance and PBO before it shows up in mean.** The cleanest,
most consistent sleeve effect was *tighter* OOS distributions and lower PBO — not a big mean
bump. Judge a low-correlation sleeve on mean/std and tail, not mean alone; raw Sharpe
under-credits it.

**10. The 12% return floor is correctly a WHOLE-PORTFOLIO constraint** (verified across all four
code paths: `island_ga.py:77,82`; `cpp/ga_types.h:226-227`; `weights.py:171-175`;
`run_rebalance.py:286,290`) — never per-asset. And because the sleeve blends at the book level
*after* optimisation, the floor never constrains it; it governs only the equity book.

**11. 1.6+ Sharpe comes from a multi-sleeve stack, not a better equity book.** For uncorrelated
streams optimally combined, squared Sharpes add: `SR_stack = √(ΣSᵢ²)`. So the dominant lever is
**orthogonality (low correlation), not any single stream's standalone Sharpe** — a 0.4-Sharpe
*uncorrelated* sleeve beats grinding the 1.0 equity book to 1.1. With the full futures/options/
margin toolkit (multi-market trend + VRP + carry + relative-value) ~1.4-1.6 net is reachable, at
which point it is an **execution + tail-risk** problem, not a strategy-availability one. Two
honesty checks: (a) **trend has positive skew and hedges the negative-skew sleeves (VRP/carry) —
their tails offset**, so combine them; (b) **Sharpe flatters negative-skew sleeves** — haircut
VRP/carry with DSR/tail-aware metrics before believing the stack number. Equity *factor tilting*
(value/quality/low-vol/cross-sectional momentum, à la Crack) is a 1.0→1.1 lever — useful but
still long equity beta, so the smallest marginal diversification of the candidate sleeves. The
long-only ceiling (~1.0-1.2) is unchanged; the stack is the only way past it.

**12. Broadening the trend sleeve from 5 to 15 ETF markets was a WASH (June 19, 2026) — lesson 11
in the data.** Tested a 15-market, 4-cluster basket (`config.TSMOM_BASKET_MULTI`, gated behind
`config.TSMOM_USE_MULTI_MARKET`, default OFF) with **cluster-balanced two-level weighting**
(equal-weight legs within a cluster, then equal-weight across clusters, so equities can't dominate
by leg count). A cheap **Stage-0 diagnostic** (build the sleeve on full history; check standalone
Sharpe + correlation-to-book + skew + the optimal-combine stack — no GA/CPCV) killed it in minutes:
standalone Sharpe **0.61→0.54** (a *loss*), orthogonality genuinely *improved* (corr-to-book
0.28→0.21, corr-to-SPY 0.19→0.13), skew ~unchanged (−0.47→−0.51, both negative on a *daily*
measure — trend's "positive skew" is a monthly/crisis-horizon property), and the
**optimally-combined stack was a dead wash: 1.056→1.054.** *Why:* equal-weighting clusters
**dilutes the gold/commodity trend** — the engine of the 5-ETF sleeve's Sharpe — to **12% of book
variance**, handing 33% each to equity and REIT (correlated, higher-vol, less diversifying vs the
book). More ETF "markets" added correlated redundancy + thin-liquidity noise, not independent
trend. **A single trend sleeve broadened is still ONE stream — it does not add an orthogonal
premium** (lesson 11); recovering the Sharpe by tilting cluster weights toward commodity would just
reintroduce a tuned/overfitting surface. **Verdict: keep the 5-ETF proxy in production.** The
multi-market machinery is sound, tested (14/14 sleeve tests), and reversible (flag off → existing
runs byte-identical) — kept documented like the curated-universe machinery, not enabled. The real
lever remains a *second* uncorrelated premium (VRP/carry), not a fatter trend sleeve. *(Caveat: the
faithful CTA reality-check is `sleeve_reality_check.py`'s vol-matched +0.48 vs DBMF; a raw-daily
correlation proxy reads ~0.21 for both baskets — same method both sides, so no regression, but
don't read the raw-daily number as the proxy breaking.)*

## Strategy Taxonomy & Empirical Verdicts (May 2026)

The 16 weighting/selection strategies tested, sorted by CPCV OOS Sharpe (n=66 splits, full run; PBO=0.909):

| Rank | Method | CPCV OOS | Std | 95% CI | Walk-forward OOS | Verdict |
|---|---|---|---|---|---|---|
| 1 | `cc_copulae` | +1.081 | 0.67 | [+0.92, +1.24] | +1.72 | GA + Gaussian copula correlation. Highest mean but widest CI; tied with rank 2-5 statistically. |
| 2 | `mc_optimised` | +1.037 | 0.29 | [+0.97, +1.11] | +0.95 | MC selection + SLSQP. **Most robust**: tightest distribution, only +3% IS-OOS gap. |
| 3 | `mc_random_weights` | +1.002 | 0.28 | [+0.93, +1.07] | +0.93 | MC selection + random weights. Tightest std after rank 2; surprisingly competitive. |
| 4 | `cc_equal_weight` | +0.998 | 0.40 | [+0.90, +1.09] | +1.03 | GA + 1/N. Travels well; minimal estimation. |
| 5 | `cc_optimised` | +0.992 | 0.41 | [+0.89, +1.09] | +1.35 | GA + max-Sharpe SLSQP. Walk-forward inflated. |
| 6 | `cc_ccc_baseline` | +0.963 | 0.58 | [+0.82, +1.10] | **+1.86** | GA + Bollerslev CCC. Walk-forward winner; collapsed −48% in CPCV. |
| 7 | `cc_inverse_vol` | +0.951 | 0.73 | [+0.78, +1.13] | +1.76 | GA + 1/σ. Largest std among working methods. |
| 8 | `cc_random_weights` | +0.916 | 0.36 | [+0.83, +1.00] | +0.94 | GA + random weights. |
| 9 | `cc_garch_var` | +0.900 | 0.51 | [+0.78, +1.02] | +0.71 | GA + GARCH variance + sample R. Forecasting hurts. |
| 10 | `cc_arima_garch` | +0.877 | 0.52 | [+0.75, +1.00] | +0.74 | GA + ARIMA + GARCH. Combined forecasting hurts more. |
| 11 | `cc_arima_er` | +0.870 | 0.39 | [+0.77, +0.96] | +1.17 | GA + ARIMA ER. Marginal forecast lift. |
| 12 | `cc_max_diversification` | +0.863 | 0.49 | [+0.74, +0.98] | +1.13 | Choueifaty-Coignard DR objective. Overfits IS variance structure. |
| 13 | `cc_risk_parity` | +0.859 | 0.43 | [+0.76, +0.96] | +0.99 | ERC. Underperforms inverse-vol; full ERC machinery doesn't earn its cost. |
| 14 | `cc_min_variance` | +0.813 | 0.61 | [+0.67, +0.96] | +1.58 | GA + min-var SLSQP. Largest IS-OOS gap (+48% pre-split-66; mean inflated by anomalous final fold). |
| 15 | `random_optimised` | +0.640 | 0.30 | [+0.57, +0.71] | +0.83 | Random pick + SLSQP. Pure baseline. |
| 16 | `random_random` | +0.449 | 0.29 | [+0.38, +0.52] | +0.69 | Random pick + random weights. Floor. |

**Key takeaways from this matrix**:

1. **The top 7 methods all have overlapping 95% CIs.** `cc_copulae` leads on the mean but with std 0.67, vs `mc_optimised` at std 0.29. By any robustness-adjusted criterion (mean / std), `mc_optimised` is the most reliable choice.
2. **Simpler weighting beats sophisticated weighting OOS.** The top 5 CPCV methods are all "low parameter count": copula (correlation-focused covariance), MC search (no estimation), MC + random weights, GA + 1/N, GA + max-Sharpe.
3. **Walk-forward winners are not CPCV winners.** The 5 cc_* methods that ranked top in walk-forward all dropped 4-7 ranks in CPCV.
4. **Forecasting hurts more often than it helps.** GARCH-variance methods rank 9-10. Drop GARCH variance forecasting in this universe.
5. **Min-variance is the worst overfitter on the IS-OOS gap dimension** — IS Sharpe ~1.44, OOS ~0.81, despite being rank 14 in OOS mean. The objective rewards quirky low-variance combinations that don't generalise.
6. **Final-split (test=10,11) was anomalously favourable**, pushing several methods' OOS means up by 0.05-0.12 each (`cc_inverse_vol` recorded OOS +5.56 on that single split, `cc_copulae` +4.48, `cc_min_variance` +4.58). The standard deviations reflect this as inflated tail-mass; the rankings would be more pessimistic without it.

## Backtest CLI

```bash
python -m src.backtest                          # Walk-forward (default): 5y train + 6mo OOS, 17 windows
python -m src.backtest --mode cpcv              # Combinatorially Purged CV: 66 splits with PBO + CIs
python -m src.backtest --mode cpcv \
    --n-groups 8 --k-test 2                     # Faster CPCV: 28 splits (~5h instead of ~10h)
python -m src.backtest --mode cpcv \
    --purge-days 10 --embargo-days 10           # Tighter purge/embargo for stronger leakage guards
```

Per-window log format includes DSR gating:
```
Window 2014-2018/2018 results (294s):
  cc_optimised   IS_sharpe=+2.41  OOS_sharpe=+2.41  degradation=0%  DSR=1.000 [PASS] (M=3.6e+07, n=1260)
  cc_copulae     IS_sharpe=+2.39  OOS_sharpe=+3.04  degradation=-27%  DSR=1.000 [PASS] (M=3.6e+07, n=1260)
  ...
```

CPCV `_report_cpcv_results` prints: per-method OOS mean + 95% CI across splits, plus method-level PBO.

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

**Policy: ETFs only — no single stocks.** Production portfolios are built exclusively from ETFs, by deliberate design choice (June 2026): single names carry idiosyncratic blow-up risk and cut against the diversification objective. Single-stock equities were trialled as an experiment and are **excluded from production runs**. The rebalance loads with `asset_type='etf'` by default (`run_rebalance.py` defaults to `--asset-type etf`; pass `--asset-type all` only to reproduce the equities experiment).

All instruments sourced from FinanceDatabase (US-listed). Configuration in `src/config.py`.

- **ETFs (production universe)**: ~2,900 (≈3,900 incl. foreign-listed) covering equities, bonds/treasuries, commodities, REITs, crypto, managed futures/CTA. The only asset class used for live portfolios.
- **Equities (excluded from production)**: ~22k single names across 27 countries remain in the DB for research only; the ETF-only policy filters them out of the optimisation.
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

- **Instruments**: ETFs only — no single stocks (see Universe Scope policy)
- **Positions**: up to 10 holdings (cardinality cap; `run_rebalance.py --max-etfs 10`). Earlier runs used 10–20; capped to 10 (June 2026) for a focused, lower-turnover book.
- **Per-holding weight**: 5–25% (June 2026; `run_rebalance.py --max-weight 0.25`). Tightened from 45% to limit single-holding concentration; forces ≥4 holdings to fill the book.
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
