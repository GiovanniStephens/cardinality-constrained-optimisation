"""
Centralised configuration for the portfolio optimisation pipeline.

Single source of truth for algorithm parameters, data processing
thresholds, universe filters, and backtest settings. All instruments
are US-listed (equities trade as ADRs, ETFs are US-domiciled).
"""

import os

# ─── Geographic scope ────────────────────────────────────────────────────────

DEVELOPED_MARKETS = [
    'United States', 'Canada', 'United Kingdom', 'Japan', 'Australia',
    'Germany', 'France', 'Switzerland', 'Netherlands', 'Sweden',
    'Norway', 'Denmark', 'Finland', 'Ireland', 'Belgium',
    'Austria', 'Singapore', 'Hong Kong', 'Israel', 'New Zealand',
]

EMERGING_MARKETS = [
    'Brazil', 'Mexico', 'India', 'South Korea', 'Taiwan',
    'South Africa', 'Thailand',
]

EXCLUDED_COUNTRIES = ['China', 'Russia']

INCLUDED_COUNTRIES = DEVELOPED_MARKETS + EMERGING_MARKETS

INCLUDED_MARKET_CAPS = ['Small Cap', 'Mid Cap', 'Large Cap', 'Mega Cap']

# US exchanges for equity filtering (excludes OTC/Pink Sheets)
INCLUDED_EXCHANGES = ['NMS', 'NYQ', 'ASE', 'NCM', 'NGM']

# ─── History & data filters ──────────────────────────────────────────────────

MIN_HISTORY_YEARS = 5
MIN_HISTORY_DAYS = 1260   # 5 * 252 trading days

# ─── Data quality thresholds ─────────────────────────────────────────────────

MAX_STALENESS_DAYS = 30          # last trade must be within 30 calendar days
MIN_ANNUAL_VOLATILITY = 0.001    # 0.1% annualised vol minimum
MAX_CONSECUTIVE_SAME_PRICE = 20  # flag if 20+ identical consecutive closes
MAX_EXTREME_RETURN_PCT = 0.05    # flag if >5% of days have >10 std dev returns

# ─── Data processing ─────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 252
DATA_MIN_COVERAGE = 0.95         # keep columns with >= 95% non-null rows
DATA_FFILL_LIMIT = 5             # max consecutive NaN rows to forward-fill
DATA_LOOKBACK_DAYS = 730         # 2 years of calendar days

# ─── Data paths ────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DB_PATH = os.path.join(DATA_DIR, 'portfolio.db')
ETF_PRICES_CSV = os.path.join(DATA_DIR, 'ETF_Prices.csv')
NZ_ETF_PRICES_CSV = os.path.join(DATA_DIR, 'NZ_ETF_Prices.csv')
INVESTNOW_PRICES_CSV = os.path.join(DATA_DIR, 'time_series_20251016_113257.csv')
EXPECTED_RETURNS_CSV = os.path.join(DATA_DIR, 'expected_returns.csv')
VARIANCES_CSV = os.path.join(DATA_DIR, 'variances.csv')

# ─── Portfolio constraints ───────────────────────────────────────────────────

TARGET_POSITIONS = (10, 20)      # (min, max) cardinality constraint
REBALANCE_FREQUENCY = 'quarterly'

# ─── Pygad GA parameters ────────────────────────────────────────────────────

GA_MIN_SECURITIES = 3
GA_MAX_SECURITIES = 15
GA_MIN_WEIGHT = 0.05
GA_MAX_WEIGHT = 0.45
GA_TARGET_RETURN = 0.15
GA_NUM_GENERATIONS = 6
GA_POPULATION_SIZE = 1000
GA_CROSSOVER_PROBABILITY = 0.85
GA_THREAD_POOL_SIZE = 10

# ─── Island GA parameters ───────────────────────────────────────────────────

ISLAND_GA_NUM_GENERATIONS = 70
ISLAND_GA_POPULATION_SIZE = 8000
ISLAND_GA_NUM_ELITES = 100
ISLAND_GA_MIGRATION_INTERVAL = 10
ISLAND_GA_MIGRATION_RATE = 0.1
ISLAND_GA_MIN_SECURITIES = 8
ISLAND_GA_MAX_SECURITIES = 20
ISLAND_GA_MIN_RETURN = 0.12

# ─── Backtest parameters ────────────────────────────────────────────────────

BACKTEST_NUM_PORTFOLIOS = 20
BACKTEST_NUM_CHILDREN = 100
BACKTEST_NUM_DAYS_OOS = 252
BACKTEST_MC_TRIALS = 100_000
BACKTEST_MAX_WEIGHT_FLOOR = 0.3

# ─── Rolling backtest parameters ───────────────────────────────────────────

BACKTEST_TRAIN_YEARS = 5
BACKTEST_TEST_DAYS = 252       # 1 year OOS per window
BACKTEST_STEP_DAYS = 252       # non-overlapping yearly windows
BACKTEST_FORECAST_WINDOWS = [] # window labels that run forecast-based GA, e.g. ['2015-2019/2020']

# ─── Pipeline defaults ───────────────────────────────────────────────────────

PIPELINE_BATCH_SIZE = 500
PIPELINE_RATE_LIMIT_DELAY = 2.0    # seconds between batches (Yahoo throttles below this)
PIPELINE_BYTES_PER_ROW = 55        # empirical, for disk space estimation
PIPELINE_DISK_HEADROOM = 1.5       # require 50% extra free space
PIPELINE_MAX_RETRIES = 5           # retries per batch (includes empty-result retries)
PIPELINE_CIRCUIT_BREAKER_THRESHOLD = 5   # consecutive failed batches before abort
PIPELINE_MAX_RATE_LIMIT_DELAY = 60.0     # max adaptive inter-batch delay (seconds)
PIPELINE_BATCH_TIMEOUT = 300             # seconds per batch download timeout

# ─── Forecast parameters ──────────────────────────────────────────────────

FORECAST_MIN_OBSERVATIONS = 30
FORECAST_ARIMA_START_P = 1
FORECAST_ARIMA_START_Q = 1
FORECAST_ARIMA_MAX_P = 5
FORECAST_ARIMA_MAX_Q = 5
FORECAST_GARCH_P = 1
FORECAST_GARCH_O = 1
FORECAST_GARCH_Q = 1
FORECAST_GARCH_DIST = 'skewt'
FORECAST_GARCH_VOL = 'Garch'

# ─── MIP parameters ──────────────────────────────────────────────────────

MIP_DEFAULT_RISK_AVERSION = 0.8
MIP_DEFAULT_MAX_ETFS = 10
