# Cardinality-Constrained Portfolio Optimisation

Select N ETFs from a universe of 1700+ to maximise the Sharpe Ratio, subject to constraints on holdings count, weights, and optionally return/risk targets. The core idea: use a genetic algorithm to choose *which* ETFs to hold (the cardinality problem), then use SLSQP to optimise *how much* of each to hold (the weight problem).

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download ETF price data from Yahoo Finance
python -m src.download_data

# 2. (Optional) Forecast returns and variances
python -m src.forecast

# 3. Run the optimisation
python -m src.optimisers.pygad_ga

# 4. Run the backtest to validate performance
python -m src.backtest
```

## Project Structure

```
.
├── src/
│   ├── optimisers/              # All optimisation algorithms (BaseOptimiser ABC)
│   │   ├── pygad_ga.py          # PyGAD GA + SLSQP + copula/CCC correlation
│   │   ├── island_ga.py         # Parallel island-model GA
│   │   ├── monte_carlo.py       # Random search (10M+ trials)
│   │   └── mip.py               # Mixed integer programming (PuLP)
│   ├── backtest.py              # Out-of-sample backtesting framework
│   ├── forecast.py              # ARIMA return + GARCH variance forecasts
│   ├── download_data.py         # Fetches prices from Yahoo Finance
│   ├── portfolio_utils.py       # Shared analytics + OptimisationResult
│   ├── config.py                # Centralised algorithm/pipeline/universe configuration
│   ├── db.py                    # SQLite database module
│   ├── pipeline.py              # Orchestrates download, quality checks, forecasting
│   ├── data_quality.py          # Data validation and bad-ticker flagging
│   └── logging_config.py        # Centralised logging setup
├── cpp/
│   └── optimisation.cpp         # C++ parallel island GA
├── benchmark/                   # Benchmarking framework
│   ├── adapters.py              # Adapter wrappers for all optimisation methods
│   ├── runner.py                # Orchestrates parallel benchmark runs
│   ├── analysis.py              # Result analysis and reporting
│   └── results.py               # Data structures for benchmark results
├── tests/                       # Unit and integration tests
│
└── data/
    ├── portfolio.db             # SQLite database (gitignored)
    ├── ETF_Prices.csv           # Daily adjusted close for ~1792 ETFs (98 MB)
    ├── NZ_ETF_Prices.csv        # NZ ETF prices (smaller dataset)
    ├── expected_returns.csv     # Output from forecast (ARIMA)
    ├── variances.csv            # Output from forecast (GARCH)
    └── ...
```

## How It Works

### The Two-Stage Optimisation

**Stage 1 -- ETF Selection (Genetic Algorithm):**
Each chromosome is a binary vector over the ETF universe (1 = include, 0 = exclude). The GA evaluates fitness by optimising weights for each candidate subset and computing its Sharpe Ratio. Penalties are applied if the number of selected ETFs falls outside `[MIN_NUM_STOCKS, MAX_NUM_STOCKS]`.

**Stage 2 -- Weight Optimisation (SLSQP):**
Given the selected ETFs, `scipy.optimize.minimize` finds weights that maximise the Sharpe Ratio (or minimise risk budget error for risk parity), subject to:
- Weights sum to 1 (fully invested, no leverage)
- Each weight in `[MIN_WEIGHT, MAX_WEIGHT]` (no shorting)
- Optional: target return or target risk constraint

### Data Pipeline

```
src.download_data         src.forecast             src.optimisers.pygad_ga
Yahoo Finance  ──>  ETF_Prices.csv  ──>  expected_returns.csv  ──>  Portfolio
                                    ──>  variances.csv              selection
                                         (optional)                 + weights
```

1. **`src.download_data`** -- Downloads adjusted close prices from Yahoo Finance (2014-2025). Filters ETFs with <90% data availability.

2. **`src.forecast`** -- Generates forward-looking inputs:
   - **Returns**: Auto-ARIMA per ETF, projects price 252 days out, computes log return.
   - **Variances**: GARCH(1,1) with skew-t innovations, annualised.
   - Outputs saved to `data/expected_returns.csv` and `data/variances.csv`.

3. **`src.optimisers.pygad_ga`** -- Runs the two-stage optimisation. Can use either historical averages or forecasted values depending on `use_forecasts` flag.

4. **`src.backtest`** -- Validates the approach out-of-sample (see Backtest section below).

## Key Configuration

These parameters in `src/config.py` control the optimisation behaviour:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GA_MAX_SECURITIES` | 15 | Maximum ETFs in portfolio |
| `GA_MIN_SECURITIES` | 3 | Minimum ETFs in portfolio |
| `GA_TARGET_RETURN` | 0.15 | Minimum annualised return constraint |
| `GA_MAX_WEIGHT` | 0.45 | Maximum allocation to any single ETF |
| `GA_MIN_WEIGHT` | 0.05 | Minimum allocation to any single ETF |

Backtest configuration (also in `src/config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BACKTEST_NUM_PORTFOLIOS` | 20 | Number of portfolios to generate per group |
| `BACKTEST_NUM_CHILDREN` | 100 | GA population size |
| `BACKTEST_NUM_DAYS_OOS` | 252 | Out-of-sample period (~1 trading year) |

## Why Cardinality-Constrained?

Three reasons this constraint matters:

1. **Covariance matrix invertibility** -- The variance-covariance matrix requires at least N observations for N assets. With 1700+ ETFs, you'd need 1700+ days of data (~7 years) for a reliable estimate. Constraining to 10-15 holdings sidesteps this.

2. **Transaction costs** -- Entering and rebalancing 50+ positions is impractical without significant capital. A concentrated portfolio is cheaper to manage.

3. **Estimation error** -- Fewer assets means fewer parameters to estimate, reducing the impact of estimation noise on portfolio construction.

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
After the GA selects ETFs, weights can be re-optimised using correlations estimated from skew-t copulas fitted to AR(1)-GARCH(1,1) residuals. More accurate but slower -- only used for the final weight optimisation, not during the GA search. Set `use_copulae=True` in `optimize()`.

## Alternative Optimisation Approaches

The repo includes three alternative solvers beyond the primary PyGAD-based approach:

| File | Method | Pros | Cons |
|------|--------|------|------|
| `optimisers/pygad_ga.py` | PyGAD genetic algorithm | Flexible constraints, good results | ~100s per portfolio |
| `optimisers/island_ga.py` | Island-model parallel GA | Multi-threaded, 8000 population with migration | More complex, harder to tune |
| `optimisers/monte_carlo.py` | Random search (10M+ trials) | Dead simple, embarrassingly parallel | Inefficient convergence |
| `optimisers/mip.py` | Mixed integer linear program (PuLP) | Exact solution | Linear approximation of Sharpe Ratio |

All optimisers implement `BaseOptimiser.optimise(prices) -> OptimisationResult`. The PyGAD approach gives the best balance of result quality and complexity.

## Backtest

The backtest validates whether the cardinality-constrained approach adds value beyond random portfolio selection. It compares six groups:

1. **CC + optimised weights** -- GA-selected ETFs with SLSQP-optimised weights
2. **CC + copula weights** -- GA-selected ETFs with copula-estimated covariance for weight optimisation
3. **CC + forecast weights** -- GA-selected ETFs using ARIMA/GARCH forecasts
4. **CC + random weights** -- GA-selected ETFs with random allocations
5. **Random + optimised weights** -- Randomly selected ETFs with SLSQP weights
6. **Random + random weights** -- Fully random baseline

Each portfolio is run forward for 252 days out-of-sample without rebalancing. Performance metrics collected:
- Annualised return and volatility
- Sharpe Ratio
- Maximum drawdown
- Calmar Ratio (return / max drawdown)
- Sortino Ratio (return / downside deviation)

### Backtest Results

Configuration: 100 portfolios, 1000 GA children, 252 OOS days, max 10 holdings, max 20% weight.

| Portfolio Type | Sharpe Mean | Sharpe Std |
|---|---|---|
| CC + optimised | 2.95 | 0.47 |
| CC + copulae | 3.06 | 0.48 |
| CC + forecasts | 2.98 | 0.40 |
| CC + random weights | 2.57 | 0.88 |
| Random + optimised | 0.63 | 0.41 |
| Random + random | 0.10 | 0.48 |

The t-statistics confirm the GA-selected portfolios significantly outperform random selection (t ~ -37 to -42, p effectively 0). The copula-based weight optimisation gives a slight edge (~3.06 vs 2.95) but is much slower.

![Out-of-sample Sharpe Ratio Distributions by Portfolio Construction Method](https://github.com/GiovanniStephens/cardinality-constrained-optimisation/blob/main/images/Out-of-sample%20Sharpe%20Ratio%20Distributions%20by%20Portfolio%20Construction%20Method.png)

## Methodology Notes

### Objective Function

The primary objective is the Sharpe Ratio:

```
Sharpe = E(R_p) / Std(R_p)
```

The `sharpe_ratio()` function returns the *negative* Sharpe Ratio because `scipy.optimize.minimize` minimises -- so minimising the negative Sharpe maximises it.

**Risk parity** is also supported as an alternative objective (`risk_parity=True` in `optimize()`). This minimises the squared difference between each asset's risk contribution and an equal target, producing a portfolio where all holdings contribute equally to total risk.

### Implicit Assumptions

Mean-variance optimisation implicitly assumes:
- The variance-covariance structure remains constant going forward
- Historical averages are good estimators of future returns and (co)variances

Both assumptions are known to be flawed -- returns, variances, and correlations all change over time. The forecasting module (`forecast.py`) partially mitigates this by using ARIMA and GARCH models instead of raw historical averages.

### Return Forecasting

Auto-ARIMA models (minimising AIC) project prices 252 trading days forward. The forecast return is the log ratio of the final to first forecast price. Wild forecasts can occur for illiquid or volatile ETFs.

### Variance Forecasting

GARCH(1,1) with GJR leverage term and skew-t innovations. The forecast variance horizon is 252 days, annualised. Some assets produce extreme variance forecasts -- this is a known issue noted in the README history.

## Running Tests

```bash
python -m unittest discover tests
```

Tests cover data loading, return calculations, Sharpe Ratio computation, weight constraints, and covariance matrix generation.

## Dependencies

```
arch            # GARCH volatility models
copulae         # Copula-based correlation estimation
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

## Todo

- [x] Add risk parity portfolios
- [x] Maximum drawdown, Calmar ratio, Sortino ratio
- [x] Test optimisation against portfoliovisualizer.com
- [x] Download full stock + ETF universe (~25k tickers from FinanceDatabase)
- [x] Data quality validation (`src/data_quality.py`)
- [ ] Portfolio beta and alpha (requires benchmark specification)
- [ ] Verify weights match an independent optimisation engine
- [ ] Run optimisation on the combined stocks + ETFs universe
