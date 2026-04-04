"""
Centralised configuration for the portfolio optimisation pipeline.

Single source of truth for algorithm parameters, data processing
thresholds, universe filters, and backtest settings. All instruments
are US-listed (equities trade as ADRs, ETFs are US-domiciled).
"""

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
DATA_LOOKBACK_DAYS = 730         # 2 years of calendar days

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

# ─── Island GA parameters ───────────────────────────────────────────────────

ISLAND_GA_NUM_GENERATIONS = 70
ISLAND_GA_POPULATION_SIZE = 8000
ISLAND_GA_NUM_ELITES = 100
ISLAND_GA_MIGRATION_INTERVAL = 10
ISLAND_GA_MIGRATION_RATE = 0.1

# ─── Backtest parameters ────────────────────────────────────────────────────

BACKTEST_NUM_PORTFOLIOS = 20
BACKTEST_NUM_CHILDREN = 100
BACKTEST_NUM_DAYS_OOS = 252
BACKTEST_MC_TRIALS = 100_000

# ─── Rolling backtest parameters ───────────────────────────────────────────

BACKTEST_TRAIN_YEARS = 5
BACKTEST_TEST_DAYS = 252       # 1 year OOS per window
BACKTEST_STEP_DAYS = 252       # non-overlapping yearly windows
BACKTEST_FORECAST_WINDOWS = [] # window labels that run forecast-based GA, e.g. ['2015-2019/2020']
