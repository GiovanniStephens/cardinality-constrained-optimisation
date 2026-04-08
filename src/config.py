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
DATA_QUALITY_MIN_OBS = 30                      # minimum observations for vol/return checks
DATA_QUALITY_MIN_OBS_EXTREME = 60              # minimum observations for extreme-return check
DATA_QUALITY_EXTREME_RETURN_MULTIPLIER = 10    # flag returns > N * robust_std

# ─── Covariance estimation ─────────────────────────────────────────────────
COV_SHRINKAGE_ENABLED = True        # Use Ledoit-Wolf shrinkage by default
COV_MIN_OBS_RATIO = 10              # Warn if T/N < this
COV_MIN_OBS_RATIO_ERROR = 1.0       # Error if T/N < this (singular matrix)
COPULA_GARCH_SCALE = 10             # MUArch scale factor for numerical stability
COPULA_DIAGNOSTIC_LAGS = 10         # Ljung-Box lag count for residual diagnostics

# ─── General / tolerances ─────────────────────────────────────────────────────

NUMERICAL_TOLERANCE = 1e-10
RISK_FREE_RATE = 0.0
STATISTICAL_SIGNIFICANCE_LEVEL = 0.05

# ─── Data processing ─────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 252
DATA_MIN_COVERAGE = 0.95         # keep columns with >= 95% non-null rows
DATA_FFILL_LIMIT = 5             # max consecutive NaN rows to forward-fill
DATA_LOOKBACK_DAYS = 730         # 2 years of calendar days
DATA_MIN_COVERAGE_PERMISSIVE = 0.10  # permissive coverage for load_data()

# ─── Data paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_PATH = os.path.join(DATA_DIR, 'portfolio.db')
ETF_PRICES_CSV = os.path.join(DATA_DIR, 'ETF_Prices.csv')
NZ_ETF_PRICES_CSV = os.path.join(DATA_DIR, 'NZ_ETF_Prices.csv')
INVESTNOW_PRICES_CSV = os.path.join(DATA_DIR, 'time_series_20251016_113257.csv')
EXPECTED_RETURNS_CSV = os.path.join(DATA_DIR, 'expected_returns.csv')
VARIANCES_CSV = os.path.join(DATA_DIR, 'variances.csv')

# ─── C++ binary ──────────────────────────────────────────────────────────

CPP_BINARY_PATH = os.path.join(PROJECT_ROOT, 'cpp', 'optimisation')

# ─── Pygad GA parameters ────────────────────────────────────────────────────

GA_MIN_SECURITIES = 3
GA_MAX_SECURITIES = 15
GA_MIN_WEIGHT = 0.05
GA_MAX_WEIGHT = 0.45
GA_TARGET_RETURN = 0.15
GA_NUM_GENERATIONS = 100           # Was 6; far too few for meaningful search
GA_POPULATION_SIZE = 1000
GA_CROSSOVER_PROBABILITY = 0.85
GA_THREAD_POOL_SIZE = 10
GA_ELITISM_FRACTION = 0.10         # 10% elitism (Eiben & Smith 2003 sweet spot)
GA_EARLY_STOP_SATURATE = 15        # Stop after 15 gens without improvement
GA_SELECTION_TYPE = 'tournament'   # Tournament selection (Jalota & Thakur 2018)
GA_TOURNAMENT_SIZE = 5             # K=5 tournament size
GA_PARENT_FRACTION = 0.30          # 30% of population eligible as parents

# ─── Island GA parameters ───────────────────────────────────────────────────

ISLAND_GA_NUM_GENERATIONS = 150    # Was 70; more convergence time (cheap fitness)
ISLAND_GA_POPULATION_SIZE = 8000
ISLAND_GA_NUM_ELITES = 100
ISLAND_GA_MIGRATION_INTERVAL = 10
ISLAND_GA_MIGRATION_RATE = 0.1
ISLAND_GA_MIN_SECURITIES = 8
ISLAND_GA_MAX_SECURITIES = 20
ISLAND_GA_MIN_RETURN = 0.12
ISLAND_GA_MUTATION_RATE = 0.005    # Floor: ~8 flips per individual for 1700 ETFs
ISLAND_GA_ADAPTIVE_MUTATION = True # Linear decay from initial to final rate
ISLAND_GA_MUTATION_RATE_INITIAL = 0.01   # High early exploration
ISLAND_GA_MUTATION_RATE_FINAL = 0.002    # Low late exploitation
ISLAND_GA_STAGNATION_LIMIT = 30   # Per-island early stopping on stagnation

MC_NUM_TRIALS = 10_000_000         # Default Monte Carlo trial count

# ─── Backtest parameters ────────────────────────────────────────────────────

BACKTEST_NUM_PORTFOLIOS = 20
BACKTEST_NUM_CHILDREN = 100
BACKTEST_NUM_DAYS_OOS = 252
BACKTEST_MC_TRIALS = 100_000
BACKTEST_MAX_WEIGHT_FLOOR = 0.3
BACKTEST_MIN_METHODS_FOR_STATS = 3   # minimum methods for Friedman test
SHARPE_WARN_THRESHOLD = 2.0          # in-sample Sharpe above this is suspicious
SHARPE_CRITICAL_THRESHOLD = 3.0      # in-sample Sharpe above this is almost certainly overfit

# ─── Rolling backtest parameters ───────────────────────────────────────────

BACKTEST_TRAIN_YEARS = 5
BACKTEST_TEST_DAYS = 252       # 1 year OOS per window
BACKTEST_STEP_DAYS = 252       # non-overlapping yearly windows
BACKTEST_FORECAST_WINDOWS = [] # window labels that run forecast-based GA, e.g. ['2015-2019/2020']

# ─── Pipeline defaults ───────────────────────────────────────────────────────

PIPELINE_BATCH_SIZE = 200
PIPELINE_RATE_LIMIT_DELAY = 4.0    # seconds between batches (Yahoo throttles below this)
PIPELINE_BYTES_PER_ROW = 55        # empirical, for disk space estimation
PIPELINE_DISK_HEADROOM = 1.5       # require 50% extra free space
PIPELINE_MAX_RETRIES = 5           # retries per batch (includes empty-result retries)
PIPELINE_CIRCUIT_BREAKER_THRESHOLD = 10  # consecutive failed batches before trip
PIPELINE_CIRCUIT_BREAKER_MAX_TRIPS = 3   # hard abort after this many trips
PIPELINE_CIRCUIT_BREAKER_COOLDOWN = 300.0  # seconds to pause on trip before retry
PIPELINE_MAX_RATE_LIMIT_DELAY = 600.0    # max adaptive inter-batch delay (10 min)
PIPELINE_BATCH_TIMEOUT = 300             # seconds per batch download timeout
PIPELINE_MIN_SUB_BATCH_SIZE = 10         # minimum tickers per sub-batch split
PIPELINE_INTER_TYPE_COOLDOWN = 60.0      # seconds between asset type pipelines

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
GARCH_SCALE = 100           # Multiplier for GARCH fitting (daily returns ~ 0.001)
FORECAST_MIN_PRICE_FLOOR = 0.0001  # Floor for ARIMA forecast price predictions

# ─── Pipeline / download ──────────────────────────────────────────────────

DOWNLOAD_THREADS = 2
DOWNLOAD_TIMEOUT = 30

# ─── MIP parameters ──────────────────────────────────────────────────────

MIP_DEFAULT_RISK_AVERSION = 0.8
MIP_DEFAULT_MAX_ETFS = 10

# ─── Universe pre-filters ─────────────────────────────────────────────────
# Regex matching unwanted suffixes: warrants (-WT), units (-UN),
# preferred (-PA), rights (-RT), etc.
TICKER_EXCLUDE_SUFFIXES = r'[-.](?:WT[A-Z]?|WS|UN|RT|RI|P[A-Z])$'

# Name patterns for SPACs/shell companies (case-insensitive)
TICKER_EXCLUDE_NAME_PATTERNS = [
    r'Acquisition Corp',
    r'Blank Check',
    r'\bSPAC\b',
    r'Special Purpose Acquisition',
    r'Merger Corp',
]
