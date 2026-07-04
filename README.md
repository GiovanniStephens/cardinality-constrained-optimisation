# Cardinality-Constrained Portfolio Optimisation

Select N instruments from a universe of ~25,000 equities and ETFs to maximise the Sharpe Ratio, subject to constraints on holdings count, weights, and optionally return/risk targets. The core idea: use a genetic algorithm to choose *which* instruments to hold (the cardinality problem), then use SLSQP to optimise *how much* of each to hold (the weight problem).

## Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Key Configuration](#key-configuration)
- [Group Allocation Constraints](#group-allocation-constraints)
- [Covariance Estimation Methods](#covariance-estimation-methods)
- [Alternative Optimisation Approaches](#alternative-optimisation-approaches)
- [Backtest](#backtest)
- [Building the C++ Optimiser](#building-the-c-optimiser)
- [Methodology Notes](#methodology-notes)
- [Running Tests](#running-tests)

## Quick Start

```bash
pip install -e ".[dev]"

# 1. Download price data from Yahoo Finance into the SQLite DB (equities + ETFs)
python -m src.download --asset-types equities etfs

# 2. (Optional) Forecast returns and variances
python -m src.forecast

# 3. Run the optimisation
#    pygad_ga is the single-process default; use island_ga for the parallel variant.
python -m src.optimisers.pygad_ga

# 4. Run the backtest to validate performance
python -m src.backtest
```

**Data storage:** All price data, forecasts, optimisation runs, and backtest results live in `data/portfolio.db` (SQLite). CSVs (`data/Prices.csv`, `data/Securities.csv`, etc.) are retained as a fallback for legacy code paths and can be imported with `python -m src.db migrate`.

## Project Structure

```
.
├── src/
│   ├── optimisers/                    # All optimisation algorithms
│   │   ├── base.py                    # BaseOptimiser ABC + OptimisationResult
│   │   ├── pygad_ga.py                # PyGAD GA + SLSQP + copula/CCC correlation
│   │   ├── island_ga.py               # Parallel island-model GA
│   │   ├── monte_carlo.py             # Random search (10M+ trials)
│   │   └── mip.py                     # Mixed integer programming (PuLP)
│   ├── backtest/                      # Forward-walk backtesting package
│   │   ├── __main__.py                # `python -m src.backtest`
│   │   ├── runner.py                  # Orchestrator: evaluate_window(), main()
│   │   ├── types.py                   # Dataclasses: WindowSpec, PortfolioResult, ...
│   │   ├── windows.py                 # Window generation, data slicing
│   │   ├── simulation.py              # Portfolio simulation (buy-and-hold)
│   │   └── statistics.py              # Hypothesis tests (t-test, Friedman)
│   ├── download/                      # Yahoo Finance data download package
│   │   ├── __main__.py                # `python -m src.download`
│   │   ├── cli.py                     # Argparse CLI + logging
│   │   ├── core.py                    # Download primitives, download_and_save
│   │   ├── session.py                 # Proxy/Tor session management
│   │   ├── validate.py                # Ticker validation with resumable caching
│   │   └── workers.py                 # Multi-worker concurrent download
│   ├── db/                            # SQLite database package
│   │   ├── __main__.py                # `python -m src.db [migrate|backfill]`
│   │   ├── schema.py                  # Schema DDL, version, seed data
│   │   ├── connection.py              # Connection management
│   │   ├── tickers.py                 # Ticker CRUD and metadata backfill
│   │   ├── prices.py                  # Price storage/retrieval
│   │   ├── forecasts.py               # Forecast storage/retrieval
│   │   ├── optimisation.py            # Optimisation run persistence
│   │   ├── backtest.py                # Backtest session/result persistence
│   │   ├── bad_tickers.py             # Known-bad ticker cache
│   │   ├── metadata.py                # Data source metadata
│   │   └── migrations.py              # CSV → DB migration functions
│   ├── forecast.py                    # ARIMA returns + GARCH variance forecasts
│   ├── data_loading.py                # DB-first / CSV-fallback price loading
│   ├── data_quality.py                # Data validation and bad-ticker flagging
│   ├── pipeline.py                    # Orchestrates download, quality checks, forecasting
│   ├── portfolio_utils.py             # OptimisationResult + DB save helper
│   ├── group_constraints.py           # Group allocation constraints (country, sector)
│   ├── config.py                      # Centralised configuration
│   ├── universe.py                    # Security universe building (FinanceDatabase)
│   ├── covariance.py                  # Ledoit-Wolf, copula-CCC, shrinkage covariance
│   ├── metrics.py                     # Sharpe, Sortino, Calmar, drawdown, DSR
│   ├── weights.py                     # SLSQP weight optimisation, risk-parity
│   ├── returns.py                     # Log returns, expected returns, variances
│   ├── binary_io.py                   # Binary data format for C++ optimiser
│   ├── exceptions.py                  # Domain exception hierarchy
│   └── logging_config.py              # Centralised logging setup
├── cpp/                               # C++ parallel island GA (CPU + Metal GPU)
│   ├── optimisation.cpp               # GA orchestrator, CLI, island dispatch
│   ├── ga_types.h                     # GA types, operators, fitness (header-only)
│   ├── data_io.{h,cpp}                # Data I/O and preprocessing
│   ├── monte_carlo.{h,cpp}            # Monte Carlo worker
│   ├── metal_fitness.h                # PIMPL header for GPU evaluator
│   ├── metal_fitness.mm               # ObjC++ Metal compute shader implementation
│   └── CMakeLists.txt                 # Build config (auto-detects Metal on macOS)
├── benchmark/                         # Benchmarking framework
│   ├── adapters.py                    # Adapter wrappers for all optimisation methods
│   ├── runner.py                      # Parallel benchmark orchestration
│   ├── analysis.py                    # Result analysis and reporting
│   └── results.py                     # Data structures for benchmark results
├── tests/                             # Unit and integration tests
├── run_benchmark.py                   # Benchmark CLI entry point
├── run_throughput_benchmark.py        # Throughput benchmark CLI entry point
└── data/
    ├── portfolio.db                   # SQLite database (gitignored, primary store)
    ├── Prices.csv                     # Daily adjusted close (gitignored; legacy/regenerable)
    ├── Securities.csv                 # Ticker metadata from FinanceDatabase
    ├── expected_returns.csv           # Forecast output (ARIMA)
    ├── variances.csv                  # Forecast output (GARCH)
    └── ...
```

Console scripts (installed by `pip install -e .`, defined in `pyproject.toml`):

| Command | Equivalent `python -m` |
|---|---|
| `portfolio-download`  | `python -m src.download` |
| `portfolio-forecast`  | `python -m src.forecast` |
| `portfolio-backtest`  | `python -m src.backtest` |
| `portfolio-benchmark` | `python run_benchmark.py` |

## How It Works

### The Two-Stage Optimisation

**Stage 1: ETF Selection (Genetic Algorithm).**
Each chromosome is a binary vector over the ETF universe (1 = include, 0 = exclude). The GA evaluates fitness by optimising weights for each candidate subset and computing its Sharpe Ratio. Penalties are applied if the number of selected ETFs falls outside `[MIN_NUM_STOCKS, MAX_NUM_STOCKS]`.

**Stage 2: Weight Optimisation (SLSQP).**
Given the selected ETFs, `scipy.optimize.minimize` finds weights that maximise the Sharpe Ratio (or minimise risk budget error for risk parity), subject to:
- Weights sum to 1 (fully invested, no leverage)
- Each weight in `[MIN_WEIGHT, MAX_WEIGHT]` (no shorting)
- Optional: target return or target risk constraint

### Data Pipeline

```
src.download              src.forecast             src.optimisers.pygad_ga
Yahoo Finance  ──>  portfolio.db    ──>  expected_returns   ──>  Portfolio
                    (+ Prices.csv)       variances              selection
                                         (optional)             + weights
```

1. **`src.download`**. Downloads adjusted close prices from Yahoo Finance (2014-2025) into `data/portfolio.db` (and optionally `data/Prices.csv`). Supports equities, ETFs, and funds. Filters instruments with <90% data availability.

2. **`src.forecast`**. Generates forward-looking inputs:
   - **Returns**: Auto-ARIMA per ETF, projects price 252 days out, computes log return.
   - **Variances**: GARCH(1,1) with skew-t innovations, annualised.
   - Outputs saved to `data/expected_returns.csv` and `data/variances.csv`.

3. **`src.optimisers.pygad_ga`**. Runs the two-stage optimisation. Can use either historical averages or forecasted values depending on `use_forecasts` flag.

4. **`src.backtest`**. Validates the approach out-of-sample (see Backtest section below).

## Key Configuration

These parameters in `src/config.py` control the optimisation behaviour:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GA_MAX_SECURITIES` | 15 | Maximum ETFs in portfolio |
| `GA_MIN_SECURITIES` | 3 | Minimum ETFs in portfolio |
| `GA_TARGET_RETURN` | 0.15 | Minimum annualised return constraint |
| `GA_MAX_WEIGHT` | 0.45 | Maximum allocation to any single ETF |
| `GA_MIN_WEIGHT` | 0.05 | Minimum allocation to any single ETF |
| `GROUP_CONSTRAINTS` | See config.py | Group allocation caps by country/sector (see [Group Allocation Constraints](#group-allocation-constraints)) |

Backtest configuration (also in `src/config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BACKTEST_NUM_PORTFOLIOS` | 20 | Number of portfolios to generate per group |
| `BACKTEST_NUM_CHILDREN` | 8000 | GA population size (passed as `--pop-size` to the C++ island GA) |
| `BACKTEST_NUM_DAYS_OOS` | 126 | Out-of-sample period (~6 trading months) |
| `BACKTEST_RUN_FORECAST_STRATEGIES` | True | Enable the three "fast" forecast methods (`cc_arima_er`, `cc_garch_var`, `cc_arima_garch`) |
| `BACKTEST_RUN_FORECAST_COPULA_STRATEGIES` | False | Gate the two expensive copula+forecast methods (off by default) |
| `COPULA_TYPE` | `'gaussian'` | Copula family for `cc_copulae` (`'gaussian'` or `'t'`; t-copula scales super-cubically in dim) |
| `CPCV_N_GROUPS` | 12 | Number of disjoint time groups for CPCV |
| `CPCV_K_TEST` | 2 | Test groups per CPCV split (12 choose 2 = 66 splits) |
| `CPCV_PURGE_DAYS` | 5 | Days dropped between train and test to break leakage |
| `CPCV_EMBARGO_DAYS` | 5 | Days dropped after test for embargoed train rows |

## Group Allocation Constraints

Beyond position count and weight bounds, the optimiser supports **group allocation constraints** that cap how much of the portfolio can be allocated to any single country, sector, or other grouping dimension. These are enforced as additional linear constraints in the SLSQP weight optimisation step.

Configured in `src/config.py` via `GROUP_CONSTRAINTS`. Supported dimensions:

| Dimension | Source | Applies to |
|-----------|--------|------------|
| `country` | `tickers.country` column | Stocks (company domicile) and ETFs with country metadata |
| `sector` | `tickers.sector` column | Equities (FinanceDatabase sectors) |
| `asset_type` | `tickers.asset_type` column | All instruments (`stock`, `etf`) |
| `category_group` | `tickers.category_group` column | ETFs (e.g. `Fixed Income`, `Equities`) |

### Active Constraints

| Dimension | Group | Max | Rationale |
|-----------|-------|-----|-----------|
| `country` | United States | 60% | Prevent US dominance from recent outperformance |
| `country` | Each non-US country | 20% | Prevent single non-US country concentration |
| `sector` | Each of 11 GICS sectors | 50% | Prevent sector concentration (especially tech) |

Tickers with NULL metadata for a dimension are unconstrained in that dimension. Run `python -m src.db backfill` to populate sector and category_group metadata from FinanceDatabase.

### Customising Constraints

```python
# src/config.py
GROUP_CONSTRAINTS = {
    'country': {
        'United States': (0.0, 0.40),   # min 0%, max 40% US
    },
    'sector': {
        'Information Technology': (0.0, 0.30),  # max 30% tech
    },
    'category_group': {
        'Fixed Income': (0.10, 0.40),   # 10-40% bonds
    },
}
```

Set `GROUP_CONSTRAINTS = {}` to disable all group constraints.

## Why Cardinality-Constrained?

Three reasons this constraint matters:

1. **Covariance matrix invertibility.** The variance-covariance matrix requires at least N observations for N assets. With 1700+ ETFs, you'd need 1700+ days of data (~7 years) for a reliable estimate. Constraining to 10-15 holdings sidesteps this.

2. **Transaction costs.** Entering and rebalancing 50+ positions is impractical without significant capital. A concentrated portfolio is cheaper to manage.

3. **Estimation error.** Fewer assets means fewer parameters to estimate, reducing the impact of estimation noise on portfolio construction.

## Covariance Estimation Methods

The project implements three approaches to building the variance-covariance matrix, in increasing sophistication:

### 1. Historical Sample Covariance (default)
Standard sample covariance scaled to annual (`cov * 252`). Simple but assumes stationarity.

### 2. CCC Model (with forecast variances)
Bollerslev's (1990) Constant Conditional Correlation model. Uses GARCH-forecasted variances on the diagonal with historical correlations:

```
Cov = D * R * D
```
where D is a diagonal matrix of forecast standard deviations and R is the historical correlation matrix. Enabled when `forecast.py` outputs are available.

### 3. Copula-GARCH (most sophisticated)
After the GA selects ETFs, weights can be re-optimised using correlations estimated from skew-t copulas fitted to AR(1)-GARCH(1,1) residuals. More accurate but slower; it is only used for the final weight optimisation, not during the GA search. Set `use_copulae=True` in `optimize()`.

## Alternative Optimisation Approaches

The repo includes three alternative solvers beyond the primary PyGAD-based approach:

| File | Method | When to use | Pros | Cons |
|------|--------|-------------|------|------|
| `optimisers/pygad_ga.py` | PyGAD genetic algorithm | Default single-run; interactive use | Flexible constraints, good results | ~100s per portfolio |
| `optimisers/island_ga.py` | Island-model parallel GA | Production benchmarks; large universes | Multi-threaded, 8000 population with migration; pairs with the C++/Metal binary for the fastest path | More complex, harder to tune |
| `optimisers/monte_carlo.py` | Random search (10M+ trials) | Baseline for comparison only | Dead simple, embarrassingly parallel | Inefficient convergence |
| `optimisers/mip.py` | Mixed integer linear program (PuLP) | Smoke-test an exact solution on a small universe | Exact solution | Linear approximation of Sharpe Ratio |

All optimisers implement `BaseOptimiser.optimise(prices) -> OptimisationResult`. The PyGAD approach gives the best balance of result quality and complexity for single runs; the island GA is the fastest when benchmarking at scale.

## Backtest

The backtest validates whether the cardinality-constrained approach adds value beyond random portfolio selection, and, more importantly, quantifies how much in-sample Sharpe survives out-of-sample. Two modes are supported:

```bash
python -m src.backtest                    # Walk-forward (default): 5y train + 6mo OOS, 17 windows
python -m src.backtest --mode cpcv        # Combinatorially Purged CV: 66 splits with PBO + CIs
python -m src.backtest --mode cpcv \
    --n-groups 8 --k-test 2               # Faster CPCV: 28 splits
```

### Strategies compared (16)

The runner sweeps a matrix of **selection × weighting** strategies. Each method differs along three axes: how holdings are *selected*, how they are *weighted*, and which *expected-return* and *covariance/correlation* inputs feed the weighting step.

| Method | Selection | Weighting | Expected-return input | Covariance / correlation input |
|---|---|---|---|---|
| `cc_optimised` | GA | max-Sharpe SLSQP | sample mean | Ledoit-Wolf sample cov |
| `cc_copulae` | GA | max-Sharpe SLSQP | sample mean | sample vols × Gaussian-copula correlation |
| `cc_ccc_baseline` | GA | max-Sharpe SLSQP | sample mean | CCC: historical vols × sample correlation |
| `cc_arima_er` | GA | max-Sharpe SLSQP | **ARIMA forecast** | Ledoit-Wolf sample cov |
| `cc_garch_var` | GA | max-Sharpe SLSQP | sample mean | CCC: **GARCH-forecast** vols × sample correlation |
| `cc_arima_garch` | GA | max-Sharpe SLSQP | **ARIMA forecast** | CCC: **GARCH-forecast** vols × sample correlation |
| `cc_min_variance` | GA | min-variance SLSQP | — (ignored) | Ledoit-Wolf sample cov |
| `cc_inverse_vol` | GA | inverse-vol (1/σ) | — | per-asset vol only |
| `cc_risk_parity` | GA | risk parity (ERC) | — | sample cov |
| `cc_max_diversification` | GA | max diversification ratio | — | sample cov |
| `cc_equal_weight` | GA | equal (1/N) | — | — |
| `cc_random_weights` | GA | random | — | — |
| `mc_optimised` | Monte Carlo | max-Sharpe SLSQP | sample mean | Ledoit-Wolf sample cov |
| `mc_random_weights` | Monte Carlo | random | — | — |
| `random_optimised` | random | max-Sharpe SLSQP | sample mean | Ledoit-Wolf sample cov |
| `random_random` | random | random | — | — |

- **Selection**: GA = cardinality-constrained island GA (8-15 holdings); Monte Carlo = best Sharpe of 100k random draws; random = unoptimised baseline.
- **Weighting**: `max-Sharpe SLSQP` maximises in-sample Sharpe; the rest are heuristic or ER-free objectives. Only the max-Sharpe variants consume an expected-return input.
- **Benchmarks**: SPY 100% and a 60/40 SPY/AGG split (when those tickers are in the universe).
- Two further copula+GARCH variants (`cc_garch_copula`, `cc_arima_garch_copula`) are gated off by default behind `BACKTEST_RUN_FORECAST_COPULA_STRATEGIES` (super-cubic t-copula cost).

See [Empirical findings](#empirical-findings-may-2026) below for how these rank out-of-sample — the short version: simpler weighting and ER-free objectives travel best.

Each portfolio is held without rebalancing across the OOS window. Per-window log lines include in-sample Sharpe, out-of-sample Sharpe, IS/OOS degradation %, and a [Deflated Sharpe Ratio](#robustness-gating-deflated-sharpe-and-cpcv) gate (`PASS` / `WEAK` / `FAIL`).

Metrics collected per portfolio: annualised return + volatility, Sharpe, Sortino (downside deviation), Calmar (return / max drawdown), max drawdown.

### Robustness gating: Deflated Sharpe and CPCV

Single in-sample Sharpe values are biased upward by selection (16 strategies × N portfolios per window all compete on the same data). Two corrections are applied:

1. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014). Corrects for selection bias and non-normality given the number of trials. Logged per (method, window) with `PASS / WEAK / FAIL` thresholds at DSR 0.95 / 0.5. Implemented in `src/metrics.deflated_sharpe_ratio` and gated in `src/backtest/runner.py`.

2. **Combinatorially Purged Cross-Validation** (López de Prado 2018, Ch. 12). Splits the time index into 12 disjoint groups, holds out 2 at a time (66 combinations), and inserts a 5-day purge plus 5-day embargo around each test region to prevent label leakage. The full CPCV run yields a confidence interval per method *and* a method-level **Probability of Backtest Overfitting (PBO)**. Implemented in `src/backtest/cpcv.py`; runnable via `python -m src.backtest --mode cpcv`.

### Empirical findings (May 2026)

The walk-forward and CPCV runs disagree, and the disagreement is itself the story. Final CPCV: 66 splits, ~9h50m wall-clock, PBO = 0.909.

| Rank | Method | CPCV OOS | Std | 95% CI | Walk-forward OOS |
|---|---|---|---|---|---|
| 1 | `cc_copulae` | +1.081 | 0.67 | [+0.92, +1.24] | +1.72 |
| 2 | `mc_optimised` | +1.037 | 0.29 | [+0.97, +1.11] | +0.95 |
| 3 | `mc_random_weights` | +1.002 | 0.28 | [+0.93, +1.07] | +0.93 |
| 4 | `cc_equal_weight` | +0.998 | 0.40 | [+0.90, +1.09] | +1.03 |
| 5 | `cc_optimised` | +0.992 | 0.41 | [+0.89, +1.09] | +1.35 |
| 6 | `cc_ccc_baseline` | +0.963 | 0.58 | [+0.82, +1.10] | **+1.86** |
| 14 | `cc_min_variance` | +0.813 | 0.61 | [+0.67, +0.96] | +1.58 |
| 15 | `random_optimised` | +0.640 | 0.30 | [+0.57, +0.71] | +0.83 |
| 16 | `random_random` | +0.449 | 0.29 | [+0.38, +0.52] | +0.69 |

**Method-level PBO = 0.909** in CPCV: the in-sample-best method ranks below median OOS in 91% of splits, a strong overfitting signal at the strategy-family level. PBO climbed monotonically with sample size (0.79 at 28 splits, 0.85 at 65 splits, 0.909 at 66 splits); the partial-run numbers underestimated the overfitting. The remaining methods omitted from the table above (`cc_inverse_vol`, `cc_random_weights`, `cc_garch_var`, `cc_arima_garch`, `cc_arima_er`, `cc_max_diversification`, `cc_risk_parity`) cluster between +0.86 and +0.95 OOS, with the GARCH/ARIMA forecast variants showing the largest IS-OOS drops.

Key takeaways:

- **The top 7 methods have overlapping 95% CIs.** `cc_copulae` leads on the mean (widest std, 0.67 in this run); `mc_optimised` is half a step behind (+1.04) with the tightest distribution (std 0.29). The two are statistically tied. The production deployable pick is **`cc_copulae`** (chosen July 2026 for consistency and its correlation-focused covariance — the June CPCV with trend arms showed it with the tightest per-fold distribution); `mc_optimised` is the mean-robustness alternative.
- **Walk-forward Sharpes are systematically inflated.** The walk-forward winner (`cc_ccc_baseline`, +1.86) collapsed to +0.96 in CPCV (−48%). Trust CPCV over walk-forward.
- **Simpler weighting beats sophisticated weighting OOS.** The top 5 CPCV methods are all "low parameter count": copula (correlation-focused covariance), MC search, MC + random weights, GA + 1/N, GA + max-Sharpe.
- **Forecasting hurts more often than it helps.** GARCH-variance methods (`cc_garch_var`, `cc_arima_garch`) rank 9-10 in CPCV.
- **Realistic OOS Sharpe ceiling for this universe is ~1.0-1.1** net of costs, not 1.5+. Anything above 1.5 OOS sustained over 10+ years is a red flag.

### Academic references

- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", *Journal of Portfolio Management* 40(5).
- Harvey, Liu, Zhu (2016), "...and the Cross-Section of Expected Returns", *Review of Financial Studies* 29(1). Argues the t-statistic threshold for significance should be roughly 3.0, not 2.0, after multiple-testing correction.
- López de Prado (2018), *Advances in Financial Machine Learning*, Wiley, Ch. 12. CPCV and PBO.
- DeMiguel, Garlappi, Uppal (2009), "Optimal Versus Naive Diversification", *RFS* 22(5). Across 14 methods on 7 datasets, none reliably beat 1/N OOS. Empirically validated by our CPCV run.

### Proposed paths to higher Sharpe

- **Hierarchical Risk Parity** (López de Prado 2016): clustering-based weighting that avoids covariance inversion.
- **Tu-Zhou shrinkage to 1/N** (Tu & Zhou 2011, *JFE*): `w = δ·w_method + (1−δ)·w_1/N` as a cheap variance reducer.
- **Turnover penalty** in SLSQP (`λ·||w_new − w_old||₁`): reduces real costs and damps IS-noise reweighting.
- ~~**Pre-filtered universe** (~30 to 60 hand-curated ETFs): single biggest PBO-reduction lever.~~ **Tested June 2026 — it backfired.** A 142-ETF data-driven curated universe lost to the broad set on a clean common-window A/B (mean per-window OOS Sharpe +0.17 vs +1.30): shrinking the universe forced concentration under the 12% return floor and flipped return skew negative. Revisit only jointly with constraint re-tuning, not as a standalone PBO lever (see CLAUDE.md → *Lessons Learned (June 2026)*).
- **Trend-following sleeve** (DBMF or DIY futures TSMOM): orthogonal premium, crisis alpha. **A synthetic, parameter-free TSMOM proxy is now implemented** (`src/sleeves/`, gated behind `BACKTEST_RUN_SLEEVE_STRATEGIES`, run via `run_sleeve_experiment.py`); reality-checks at +0.48 correlation to DBMF. **Full-fidelity CPCV (66 splits, ~10h) lifts all four base methods** (+0.02–0.07 mean, best `mc_optimised_trend35` +0.91 vs +0.87) and **tightens the OOS distribution** (clearer risk-adjusted/`mean÷std` gain), but the mean lift **stays within the 95% CIs** (not statistically decisive) and family PBO nudged up (0.606 vs 0.515 base-only). A real, small, variance-reducing lift — not transformative. The production upgrade is a multi-market futures book (see the stack below). **Broadening the *ETF-proxy* sleeve from 5 to 15 markets was tested (June 2026) and is a wash** — cluster-balanced weighting improved orthogonality (corr-to-book 0.28→0.21) but *diluted* the gold/commodity trend that drives the sleeve's Sharpe (standalone 0.61→0.54), so the optimal-combine stack barely moved (1.056→1.054). Machinery kept but disabled (`config.TSMOM_BASKET_MULTI`, flag `TSMOM_USE_MULTI_MARKET`, default off); production stays on the 5-ETF proxy. A *real* 50-100-market **futures** book (not ETF proxies) is a different, data-heavier proposition — see the stack below.
- **Defined-risk short SPX vol sleeve**: VRP harvesting via 30-45 DTE 10-15 delta put spreads.
- **Levered risk parity wrapper** (NTSX/NTSI or DIY ES + ZN/ZB futures): captures leverage aversion premium.

Combined multi-strategy stack target: realistic OOS Sharpe **1.4-1.7** with 3-5 uncorrelated sleeves vs. the 1.0-1.2 long-only ceiling.

### The multi-strategy stack: what actually reaches 1.6+

Nothing done to the *equity* book breaks the ~1.0-1.2 long-only ceiling — better selection, weighting, or factor tilting moves 1.0 → ~1.1. **1.6+ comes only from stacking uncorrelated return streams.** For uncorrelated streams optimally combined, squared Sharpes add:

```
SR_stack = √(S₁² + S₂² + … + Sₙ²)
```

so the dominant lever is **orthogonality, not standalone Sharpe**: a 0.4-Sharpe *uncorrelated* sleeve does more than grinding the equity book to 1.1. With the full **futures / options / margin** toolkit the menu opens up and ~1.4-1.6 net becomes reachable — at which point it is an **execution and tail-risk** problem, not a strategy-availability one.

| Sleeve | Vehicle | Net S | Skew |
|---|---|---|---|
| Core book (equity or levered risk parity) | ETFs / NTSX / ES+ZN futures | ~1.0 | − |
| Multi-market trend (50-100 markets) | futures | 0.5-0.7 | **+ (crisis alpha)** |
| Volatility risk premium | defined-risk options | 0.4-0.7 | − (fat left) |
| Cross-asset carry | FX / rates / commodity futures | 0.4-0.6 | − |
| Market-neutral relative value | long/short | 0.4-0.6 | mixed |

**Two design rules that fall out of the skew column:**

1. **Tails offset, not just correlations.** Trend has *positive* skew (it profits in crashes); VRP and carry have *negative* skew (they sell insurance). The trend sleeve literally pays out when the short-vol/carry sleeves detonate — combine them.
2. **Sharpe flatters the negative-skew sleeves.** Size VRP/carry on a tail-aware / Deflated-Sharpe basis, defined-risk only, and haircut their standalone Sharpes before trusting the stack number.

**Build order (by diversification value, not standalone Sharpe):** (1) upgrade the trend sleeve from the 5-ETF proxy to a multi-market **futures** book — biggest single move, intended to raise trend's own Sharpe ~0.4→0.6 and supply the positive-skew engine. *Note (June 2026): doing this with **ETF proxies** does NOT deliver the lift — a 15-ETF-market cluster-balanced version was tested and washed out (standalone Sharpe fell 0.61→0.54; the liquid distinct-market ETFs cluster and carry no roll/carry). The 0.4→0.6 requires genuine futures across 50-100 markets, which needs a futures-data pipeline — not in scope yet.* (2) defined-risk VRP; (3) cross-asset carry; (4) modest leverage (futures) to scale the finished stack to ~12-15% vol; (5) CPCV the **joint stack** with a cost model and tail-aware metrics — the combination is the only number that counts. **Realistic landing: 1.4-1.6 net with excellent execution, 1.3-1.5 robust; treat any backtest above ~1.6 as a red flag until it survives CPCV + DSR + costs.**

## Margin Leverage Analysis

**Verdict (July 2026): margin leverage is uneconomic at current financing rates — the
production book stays at the 12% return floor, unlevered.** The full
unconstrained-max-Sharpe + leverage recipe was built and evaluated;
`run_leverage_analysis.py` sizes how much IB margin a saved book can safely carry
(`--min-return 0` on the rebalance disables the floor to produce the tangency book):

```bash
python run_leverage_analysis.py                    # latest cc_copulae book
python run_leverage_analysis.py --run-id 68 --seed 42 --borrow-rate 0.08
```

The recommendation is the **minimum of independent caps** — half-Kelly with financing
(`L* = (μ−r_b)/σ²`), a vol target, the Reg-T liquidation-drop identity
(`d* = (1−mL)/(L(1−m))`; at 25% maintenance L=1.5 survives a 56% book drop, L=2.0 only
33%), a CVaR budget, a stationary-bootstrap **first-passage P(liquidation) < 1%/yr**
simulation (IB auto-liquidates in real time — first passage, not terminal VaR, is the
binding statistic), and a 2.0 hard ceiling. Sizing uses haircut inputs (μ×0.5, σ
inflated to stressed levels); with honest numbers the answer at 2026 rates is
**"L = 1.0 — don't lever"** at every financing tested (USD margin ~5.14%, NZD
~3.75–4.75%, box spreads ~4.2%) — the financing spread over the haircut return binds,
never liquidation risk. Without cheap leverage, Tobin separation says to move along the
frontier instead — which is what the reinstated 12% return floor does. Machinery in
`src/leverage.py`; tests in `tests/test_leverage.py`; re-run the analysis when a new
sleeve raises the stack's Sharpe or financing falls.

## Building the C++ Optimiser

The `cpp/` directory contains a parallel island-model GA written in C++ that is the fastest path for large universes. On Apple Silicon, fitness evaluation can optionally run on the GPU via Metal compute shaders; CMake auto-detects Metal and compiles it in when available. On other platforms the binary compiles CPU-only and `--gpu` falls back with a warning.

```bash
# Requires CMake 3.14+. Eigen and csv-parser are header-only git submodules.
git submodule update --init --recursive
make build-cpp

# Run the binary directly
./cpp/optimisation --binary --data data.bin --time-budget 10
./cpp/optimisation --binary --data data.bin --gpu --time-budget 10   # Apple Silicon
```

**Metal GPU notes.** Each threadgroup evaluates one portfolio with 64 threads parallelising over the time dimension. The GA search runs in FP32, which is sufficient for ranking (max 0.00002% Sharpe error vs FP64), and each island's best solution is re-evaluated in FP64 on the CPU for precise final reporting. On an M4 with M=1800, T=1260, n=15, pop=1000: roughly 1.2M evals/sec single-island GPU (4.5× CPU) and 5.8M evals/sec across 10 islands (3.6× CPU). Limitations: `maxETFs ≤ 64` (shader array size) and `THREADS_PER_GROUP` must be a power of 2.

## Methodology Notes

### Objective Function

The primary objective is the Sharpe Ratio:

```
Sharpe = E(R_p) / Std(R_p)
```

The `sharpe_ratio()` function returns the *negative* Sharpe Ratio because `scipy.optimize.minimize` minimises, so minimising the negative Sharpe maximises it.

**Risk parity** is also supported as an alternative objective (`risk_parity=True` in `optimize()`). This minimises the squared difference between each asset's risk contribution and an equal target, producing a portfolio where all holdings contribute equally to total risk.

### Implicit Assumptions

Mean-variance optimisation implicitly assumes:
- The variance-covariance structure remains constant going forward
- Historical averages are good estimators of future returns and (co)variances

Both assumptions are known to be flawed: returns, variances, and correlations all change over time. The forecasting module (`forecast.py`) partially mitigates this by using ARIMA and GARCH models instead of raw historical averages.

### Return Forecasting

Auto-ARIMA models (minimising AIC) project prices 252 trading days forward. The forecast return is the log ratio of the final to first forecast price. Wild forecasts can occur for illiquid or volatile ETFs.

### Variance Forecasting

GARCH(1,1) with GJR leverage term and skew-t innovations. The forecast variance horizon is 252 days, annualised. Some assets produce extreme variance forecasts.

> **Empirical caveat (May 2026):** GARCH-variance methods rank near the bottom in both walk-forward and CPCV. The forecast inputs add noise rather than signal in this universe; see the [Backtest § Empirical findings](#empirical-findings-may-2026) table.

## Running Tests

```bash
python -m unittest discover tests
```

Tests cover data loading, return calculations, Sharpe Ratio computation, weight constraints, and covariance matrix generation.

## Dependencies

```
arch            # GARCH volatility models
copulae         # Copula-based correlation estimation
curl_cffi       # HTTP client (yfinance backend)
financedatabase # Instrument universe sourcing
matplotlib      # Plotting
muarch          # Multivariate ARCH models
numpy           # Numerical computing
pandas          # Data manipulation
pmdarima        # Auto-ARIMA forecasting
pulp            # Mixed integer linear programming
pygad           # Genetic algorithm framework
scikit-learn    # Ledoit-Wolf covariance shrinkage
scikit-posthocs # Post-hoc statistical tests
scipy           # Optimisation (SLSQP) and statistics
seaborn         # Statistical visualisation
tqdm            # Progress bars
yfinance        # Yahoo Finance data download
```

See `pyproject.toml` for exact version constraints.

## Todo

- [x] Add risk parity portfolios
- [x] Maximum drawdown, Calmar ratio, Sortino ratio
- [x] Test optimisation against portfoliovisualizer.com
- [x] Download full stock + ETF universe (~25k tickers from FinanceDatabase)
- [x] Data quality validation (`src/data_quality.py`)
- [x] Deflated Sharpe Ratio gating (Bailey & López de Prado 2014)
- [x] Combinatorially Purged Cross-Validation with PBO (López de Prado 2018, Ch. 12)
- [ ] Hierarchical Risk Parity (López de Prado 2016): clustering-based weighting that avoids covariance inversion
- [ ] Tu-Zhou shrinkage to 1/N (`w = δ·w_method + (1−δ)·w_1/N`) as a cheap variance reducer
- [ ] Turnover penalty in SLSQP objective (`λ·||w_new − w_old||₁`) to damp IS-noise reweighting
- [x] Pre-filtered universe (~30 to 60 hand-curated ETFs) — tested June 2026; backfired (forced concentration, lost the common-window A/B). Machinery shipped as `curate_universe.py` + `--curated`; not adopted.
- [~] Trend-following / managed futures sleeve — synthetic parameter-free TSMOM proxy built; full-fidelity CPCV (66 splits) done: consistent small lift on all four bases (+0.02–0.07) + lower std, but within CIs (not decisive) and PBO nudged up. Multi-market futures upgrade pending (`src/sleeves/`, `run_sleeve_experiment.py`)
- [x] Multi-market *ETF-proxy* trend broadening (5→15 markets, cluster-balanced) — tested June 2026; a **wash** (orthogonality improved 0.28→0.21 but standalone Sharpe fell 0.61→0.54 as cluster-balancing diluted the gold/commodity trend; optimal-combine stack 1.056→1.054). Machinery shipped as `config.TSMOM_BASKET_MULTI` + flag `TSMOM_USE_MULTI_MARKET`; not adopted (production stays on the 5-ETF proxy).
- [ ] Multi-market **futures** trend upgrade (50-100 real futures, not ETF proxies) — intended to raise the trend sleeve's own Sharpe ~0.4→0.6; the positive-skew engine of the stack. Needs a futures-data pipeline (the ETF-proxy route above doesn't deliver the lift).
- [ ] Defined-risk short SPX vol sleeve (30-45 DTE 10-15 delta put spreads)
- [ ] Cross-asset carry sleeve (FX / rates / commodity futures)
- [ ] Market-neutral relative-value sleeve
- [ ] CPCV the **joint** multi-strategy stack with a cost model + tail-aware (Deflated-Sharpe) metrics
- [ ] Portfolio beta and alpha (requires benchmark specification)
- [ ] Verify weights match an independent optimisation engine
