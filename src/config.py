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
# Copula family for estimate_corr_using_copulas. 'gaussian' has a closed-form
# correlation MLE so it scales O(N²); 't' fits the tail-dependence parameter
# iteratively and scales super-cubically (199s for N=20 vs <50ms for gaussian).
# GARCH on the marginals already captures heavy tails, so 'gaussian' is the
# pragmatic default. Set to 't' to recover the original copula-CCC behaviour.
COPULA_TYPE = 'gaussian'

# ─── General / tolerances ─────────────────────────────────────────────────────

NUMERICAL_TOLERANCE = 1e-10
RISK_FREE_RATE = 0.0
STATISTICAL_SIGNIFICANCE_LEVEL = 0.05

# ─── Data processing ─────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 252
DATA_MIN_COVERAGE = 0.95         # keep columns with >= 95% non-null rows
DATA_FFILL_LIMIT = 5             # max consecutive NaN rows to forward-fill
DATA_LOOKBACK_DAYS = 1825        # 5 years of calendar days (~1260 trading days)
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
# Data-driven curated ETF allow-list (built by curate_universe.py). Opt-in via
# the --curated flag on run_rebalance.py / backtest_rebalance.py; restricts the
# search universe to one liquid representative per correlation cluster.
CURATED_UNIVERSE_CSV = os.path.join(DATA_DIR, 'curated_universe.csv')

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

# ─── Group allocation constraints (optional) ────────────────────────────────
# Dict of group dimension -> {group_name: (min_weight, max_weight)}.
# Weight fractions are of total portfolio (0.0 to 1.0).
# Empty dict = no constraints (default). Only listed groups are constrained.

# Non-US country caps: 20% each to prevent single-country dominance
_NON_US_COUNTRY_CAPS = {
    c: (0.0, 0.20) for c in INCLUDED_COUNTRIES if c != 'United States'
}

# Standard FinanceDatabase equity sectors — cap at 50% each
CONSTRAINED_SECTORS = [
    'Information Technology', 'Health Care', 'Financials',
    'Consumer Discretionary', 'Industrials', 'Communication Services',
    'Consumer Staples', 'Energy', 'Materials', 'Real Estate', 'Utilities',
]

GROUP_CONSTRAINTS = {
    'country': {
        'United States': (0.0, 0.60),   # max 60% US
        **_NON_US_COUNTRY_CAPS,
    },
    'sector': {s: (0.0, 0.50) for s in CONSTRAINED_SECTORS},
}

# ─── Asset-class category limits for the ETF rebalance (run_rebalance.py) ─────
# FinanceDatabase sector/category metadata is ~blank for ETFs, so these caps are
# driven by the crude name classifier (src.categorise). Tuned for a young,
# high-risk, decorrelation-seeking investor whose NZ/AU equity is already held
# via KiwiSaver (see CLAUDE.md "Investment Universe").
REBALANCE_EXCLUDE_CATEGORIES = []      # none excluded — all categories kept, just capped
REBALANCE_CATEGORY_CAPS = {            # max fraction of the portfolio per cap-group
    'Equity': 0.70,                    # growth core, but forces >=30% into diversifiers
    'Fixed Income': 0.30,
    'Cash/FX': 0.10,                   # Currency + Cash combined — dry powder only
    'Commodity': 0.25,
    'Real Estate': 0.20,
    'Alternatives': 0.30,              # managed futures / CTA — crisis alpha
    'Inverse': 0.10,                   # short/inverse hedges (incl 2x/3x short) — tightened
                                       # 0.15→0.10 July 2026: the 25% managed-futures sleeve
                                       # is the designated crisis-alpha allocation; paying
                                       # inverse-fund decay on top of it is double-hedging
    'Leveraged': 0.05,                 # long-amplifying — reduced; get leverage from account margin instead
    'Crypto': 0.15,
    'Unknown': 0.15,                   # backstop for unclassified names (raised
                                       # 0.10→0.15 July 2026, user-approved: allows 3
                                       # legs at the ~4.7% relaxed floor now that the
                                       # classifier hardening shrank this bucket)
}
# Category minimums (same dimension as the caps; enforced as SLSQP lower bounds
# and in the candidate compliance scan). RETIRED as a market-participation lever
# (July 2026): a name-based 'Equity >= 50%' floor was Goodhart-gamed twice — the
# optimiser filled it with a misclassified bond fund, then with a fully-buffered
# defined-protection fund (beta ~0.1). Superseded by the beta floor below, which
# is measured on return covariance and cannot be satisfied by look-alikes.
# Machinery retained: any {'<group>': min_frac} entry here is enforced.
REBALANCE_CATEGORY_MINS = {}

# Market-participation floor: minimum portfolio beta to the benchmark over the
# training window (beta is linear in weights -> one SLSQP inequality, same form
# as the return floor). July 2026: without cheap leverage max-Sharpe holds ~25%
# plain equity at beta ~0.07 — an absolute-return profile. The beta floor forces
# genuine market exposure; shorts/hedges count negative, buffered or cash-like
# "equity" funds contribute almost nothing, so it cannot be gamed by labels.
# Applies to the *equity book* (the deployable book's beta is ~0.75x this).
# CLI: run_rebalance.py --min-beta; <= 0 disables.
REBALANCE_MIN_BETA = 0.50
REBALANCE_BETA_BENCHMARK = 'SPY'
# Conviction "must-have" holdings — forced into every rebalance portfolio at the
# selection stage (held at >= the per-position floor, still counted toward the
# category caps). Override per-run with `run_rebalance.py --must-have SMH,VOO`.
REBALANCE_MUST_HAVE = ['SMH']

# Production-rebalance liquidity floor: minimum average daily dollar volume (USD)
# for a US-listed ETF to be selectable. Foreign (dot-suffix) listings carry no
# stored volume and are always dropped; US ETFs below this ADV are excluded so the
# book stays IB-tradeable. Applied in run_rebalance.py via src.liquidity. Run
# `python -m src.db backfill-volume` to keep ADV coverage current.
# 500k (July 2026, was 1M): a small patient book (~$3-12k positions, quarterly,
# limit orders worked at the spread) is ~1-2% of daily volume at this floor.
REBALANCE_MIN_ADV_USD = 500_000

# Managed-futures ETFs that make up the trend-sleeve capital allocation (alpha).
# The sleeve is a FIXED strategic allocation split equally across these funds — a
# tradeable proxy for the validated synthetic TSMOM stream, held by hand because
# (a) these funds are too young for the GA's 5y history filter and (b) a fixed
# allocation protects the diversifier the in-sample optimiser would underweight.
# Several funds diversify single-manager risk (though they are mutually correlated
# trend-followers, so it is manager diversification, not a new premium). Override
# with `run_rebalance.py --sleeve-etfs DBMF,KMLM`. Three distinct trend models:
# DBMF (SG-CTA index replication), KMLM (fixed-weight Mount Lucas index), CTA
# (Simplify actively-managed adaptive trend).
REBALANCE_SLEEVE_ETFS = ['DBMF', 'KMLM', 'CTA']

# Production-rebalance history window (calendar days). Research/backtest keeps
# the 5y standard (DATA_LOOKBACK_DAYS / MIN_HISTORY_DAYS: T >> N for CPCV and
# stable covariance); the LIVE allocation admits ETFs with ~2 years of history
# so the book isn't restricted to pre-2021 funds. min_history:* quality flags
# are treated as ADVISORY on this path (hard flags still exclude); the 95%
# coverage test over this window does the real admission.
REBALANCE_LOOKBACK_DAYS = 730

# Margin-leverage analysis defaults (run_leverage_analysis.py / src.leverage).
# Borrow rate: IB Pro USD tier under $100k is benchmark + 1.5% (~5.1% as of
# mid-2026) — 0.055 adds headroom; it floats, so stress it, don't trust it.
# Maintenance: Reg-T 25% on broad 1x ETFs; IB auto-liquidates on breach.
REBALANCE_BORROW_RATE = 0.055
REBALANCE_MAINTENANCE_MARGIN = 0.25

# Classifier label -> cap-group (merges Currency+Cash; Leveraged/Inverse split by direction).
REBALANCE_CAP_GROUP = {
    'Equity': 'Equity', 'Fixed Income': 'Fixed Income',
    'Currency': 'Cash/FX', 'Cash': 'Cash/FX',
    'Commodity': 'Commodity', 'Real Estate': 'Real Estate',
    'Alternatives': 'Alternatives',
    'Inverse': 'Inverse', 'Leveraged': 'Leveraged',
    'Crypto': 'Crypto', 'Unknown': 'Unknown',
}

# ─── Backtest parameters ────────────────────────────────────────────────────

BACKTEST_NUM_PORTFOLIOS = 30
BACKTEST_NUM_CHILDREN = 8000   # GA population per portfolio (matches ISLAND_GA_POPULATION_SIZE)
BACKTEST_NUM_DAYS_OOS = 252
BACKTEST_MC_TRIALS = 100_000
BACKTEST_MAX_WEIGHT_FLOOR = 0.3
BACKTEST_MIN_METHODS_FOR_STATS = 3   # minimum methods for Friedman test
SHARPE_WARN_THRESHOLD = 2.0          # in-sample Sharpe above this is suspicious
SHARPE_CRITICAL_THRESHOLD = 3.0      # in-sample Sharpe above this is almost certainly overfit

# ─── Rolling backtest parameters ───────────────────────────────────────────

BACKTEST_TRAIN_YEARS = 5
BACKTEST_TEST_DAYS = 126       # 6 months OOS per window
BACKTEST_STEP_DAYS = 126       # non-overlapping 6-month windows

# Toggle the ARIMA / GARCH / Copula-CCC forecast strategy family. When
# enabled, each window's training period is used to fit fresh forecasts
# for the union of GA-selected tickers (no leakage). Disable to skip the
# extra ~5–15 wall-min/window for forecast fits during development.
BACKTEST_RUN_FORECAST_STRATEGIES = True
# The two copula+forecast SLSQP variants (cc_garch_copula,
# cc_arima_garch_copula) are super-cubic in selection size — single
# windows blew past 50 min on these. They're gated behind their own flag
# so the cheap forecast strategies (cc_arima_er, cc_garch_var,
# cc_arima_garch) can run by default.
BACKTEST_RUN_FORECAST_COPULA_STRATEGIES = False

# ─── Managed-futures / trend-following sleeve (research experiment) ──────────
# A separately-managed return stream blended into the book at the portfolio
# level: combined = (1-alpha)*book + alpha*sleeve. The sleeve runs canonical
# time-series momentum (Moskowitz-Ooi-Pedersen 2012) on a fixed basket of
# long-history liquid ETFs. The spec is FROZEN (parameter-free) so it adds no
# overfitting surface; only `alpha` is swept, at the fixed levels below.
# See CLAUDE.md "Future Work" (trend-following sleeve) and docs.
#
# Default OFF → existing walk-forward / CPCV runs are byte-identical. Flip to
# True to register the sleeve A/B arms.
BACKTEST_RUN_SLEEVE_STRATEGIES = False

# One long-history liquid ETF per asset class. Each value is a fallback chain
# tried in order at load time (first with usable history wins); resolved once
# and logged. NEVER tuned to results.
TSMOM_BASKET = {
    'equity':    ['SPY', 'IVV', 'VOO'],
    'bond':      ['IEF', 'TLT', 'AGG'],
    'commodity': ['DBC', 'GSG', 'DJP'],
    'gold':      ['GLD', 'IAU'],
    'reit':      ['VNQ', 'IYR', 'SCHH'],
}

# Multi-market basket (research experiment, opt-in via TSMOM_USE_MULTI_MARKET).
# A broader managed-futures proxy: more markets → more independent trends.
# Nested {cluster: {slot: [fallback_chain]}}. Aggregation is two-level EQUAL
# weight — equal-weight legs within a cluster, then equal-weight ACROSS clusters
# — so each asset class caps at 1/N_clusters of book risk regardless of leg
# count, and a numerous-but-correlated cluster (esp. equities) cannot dominate
# and turn the sleeve into long equity beta (which would kill its orthogonality
# to the book it diversifies). 15 markets, 4 clusters. Gold folds into commodity
# (a single instrument shouldn't own a full cluster). Thin FX wrappers (FXA
# ~$1m/day ADV) and broad-commodity baskets (DBC/GSG/DJP — they double-count the
# single-commodity legs) are deliberately screened out on tradeability + signal
# quality. FROZEN/parameter-free, same discipline as TSMOM_BASKET. NEVER tuned.
TSMOM_BASKET_MULTI = {
    'equity':    {'us_lc': ['SPY', 'IVV', 'VOO'], 'us_growth': ['QQQ'],
                  'us_sc': ['IWM'], 'em': ['EEM'], 'dev_exus': ['EFA']},
    'rates':     {'short': ['SHY'], 'mid': ['IEF'], 'long': ['TLT', 'AGG'],
                  'ig': ['LQD'], 'hy': ['HYG']},
    'commodity': {'gold': ['GLD', 'IAU'], 'oil': ['USO'], 'ags': ['DBA']},
    'reit':      {'us': ['VNQ', 'IYR', 'SCHH'], 'intl': ['VNQI']},
}

# Selector: when True the sleeve is built from the nested TSMOM_BASKET_MULTI;
# when False from the original flat 5-ETF TSMOM_BASKET. Default False → every
# existing run (and the committed 5-ETF results) stays byte-identical.
TSMOM_USE_MULTI_MARKET = False

TSMOM_LOOKBACK_DAYS    = 252    # 12-month trailing-return sign (MOP-2012)
TSMOM_VOL_LOOKBACK     = 60     # ~3-month ex-ante vol for inverse-vol sizing
TSMOM_TARGET_VOL_INSTR = 0.10   # 10% annualised per-instrument vol target
TSMOM_TARGET_VOL_BOOK  = 0.10   # 10% annualised book-level vol target
TSMOM_REBALANCE_DAYS   = 21     # refresh signal+sizing monthly, hold between
TSMOM_ALLOW_SHORT      = True   # longs+shorts — the crisis-alpha source

# Swept knob: fraction of the book allocated to the sleeve. Kept to 3 fixed
# levels to limit multiple-testing inflation (PBO sensitivity).
TSMOM_ALPHAS = (0.15, 0.25, 0.35)

# Base methods that get sleeve-overlay variants (the best-travelling OOS
# methods per the CLAUDE.md strategy taxonomy). 4 bases × 3 alphas = 12 arms.
BACKTEST_SLEEVE_BASE_METHODS = (
    'cc_copulae', 'cc_equal_weight', 'cc_inverse_vol', 'mc_optimised',
)

# ─── Combinatorially Purged CV (López de Prado 2018) ────────────────────────
# Used by `python -m src.backtest --mode cpcv`. With 12 years of data and
# n_groups=12, k=2 → C(12,2)=66 splits. Smaller n_groups = fewer splits but
# more train/test data per split. purge_days handles short-run autocorrelation
# at test boundaries; embargo_days handles forward-looking leakage.
CPCV_N_GROUPS = 12
CPCV_K_TEST_GROUPS = 2
CPCV_PURGE_DAYS = 5
CPCV_EMBARGO_DAYS = 5

# ─── Download defaults ───────────────────────────────────────────────────────

DOWNLOAD_DEFAULT_START = "2014-04-30"
DOWNLOAD_DEFAULT_END = "2025-04-30"
DOWNLOAD_DEFAULT_BATCH_SIZE = 500
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_BACKOFF_BASE = 2  # seconds, for exponential backoff

# ─── Pipeline defaults ───────────────────────────────────────────────────────

PIPELINE_BATCH_SIZE = 20             # was 50 — full 11yr download needs smaller batches
PIPELINE_PROXY_BATCH_SIZE = 10       # amortise cookie+crumb fetch over ~10 tickers per session; 2.5x fewer requests to Akamai than batch=1
PIPELINE_RATE_LIMIT_DELAY = 4.0    # inter-batch delay, widened jitter in the caller
PIPELINE_INTRA_BATCH_DELAY = (0.3, 1.5)   # random range sleep before each yf.download call inside a batch loop — breaks synchronised 16-worker bursts
PIPELINE_BYTES_PER_ROW = 55        # empirical, for disk space estimation
PIPELINE_DISK_HEADROOM = 1.5       # require 50% extra free space
PIPELINE_MAX_RETRIES = 5           # retries per batch (includes empty-result retries)
PIPELINE_CIRCUIT_BREAKER_THRESHOLD = 10  # consecutive failed batches before trip
PIPELINE_CIRCUIT_BREAKER_MAX_TRIPS = 10  # hard abort after this many trips (was 3 — too aggressive against Yahoo's transient throttling; full ETF run aborted every worker with 60%+ of partition unprocessed)
PIPELINE_CIRCUIT_BREAKER_COOLDOWN = 300.0  # seconds to pause on trip before retry
PIPELINE_MAX_RATE_LIMIT_DELAY = 600.0    # max adaptive inter-batch delay (10 min)
PIPELINE_BATCH_TIMEOUT = 300             # seconds per batch download timeout
PIPELINE_MIN_SUB_BATCH_SIZE = 10         # minimum tickers per sub-batch split
PIPELINE_INTER_TYPE_COOLDOWN = 600.0     # 10 min — let Akamai's per-IP throttle counters drain between ETF/fund pipelines
PIPELINE_DEFAULT_WORKERS = 1              # sequential by default (1 = use download_and_save)
PIPELINE_MAX_WORKERS = 24                 # cap for --workers
PIPELINE_SUBPROCESS_STAGGER = (15, 30)    # random range seconds between subprocess launches — 16 workers now spread over ~5 min instead of ~80s
PIPELINE_SESSION_ROTATE_INTERVAL = 50     # rotate proxy session every N batches

# ─── Bad-ticker cache (known_bad_tickers) ────────────────────────────────────
# Hardening after June 2026 incident: a single transient Yahoo failure on
# 2026-04-16 blacklisted SPY/VOO/AGG (failure_count=1, no expiry), and the
# pipeline silently skipped them on every subsequent run for ~2 months.
PIPELINE_BAD_CACHE_MIN_FAILURES = 3       # require N failures before skipping a ticker (one blip no longer blacklists)
PIPELINE_BAD_TICKER_TTL_DAYS = 30         # cache entries expire after N days so transient failures self-heal and get retried
PIPELINE_PROTECTED_TICKERS_CSV = os.path.join(DATA_DIR, 'core_etfs.csv')  # liquid core that must NEVER be cached/skipped

# ─── Fingerprint rotation (anti-Akamai) ──────────────────────────────────────
# curl_cffi impersonation targets. Rotating these breaks the "N workers with
# identical JA3/JA4" signal that Akamai uses to correlate our residential
# proxy traffic. Each new session in _make_session() picks one pseudorandomly.
IMPERSONATE_TARGETS = [
    'chrome124', 'chrome120', 'chrome119', 'chrome116', 'chrome110',
    'safari17_0', 'safari17_2_ios', 'safari16_5', 'edge101', 'edge99',
]
# Accept-Language header variation per session. Yahoo's Akamai treats
# uniform Accept-Language across thousands of requests as a bot signal.
ACCEPT_LANGUAGE_POOL = [
    'en-US,en;q=0.9',
    'en-GB,en;q=0.9',
    'en-US,en;q=0.9,es;q=0.8',
    'en-AU,en;q=0.9',
    'en-CA,en-US;q=0.9,en;q=0.8',
    'en;q=0.9',
]
# Warm-up URL fetched on each new session before hitting the API. Gives
# the session legitimate Akamai cookies + establishes "this IP browsed
# the site" rather than "this IP only calls query1 endpoints".
WARMUP_URL = 'https://finance.yahoo.com/'
# Use Yahoo's v8 chart endpoint directly, bypassing yfinance's cookie+crumb
# dance. Cuts HTTP requests per ticker from 3 (cookie + crumb + chart) to 1
# (chart). Probe-validated 2026-04 — endpoint serves public daily data
# without auth. When False, falls back to the yfinance code path.
DOWNLOAD_USE_DIRECT_V8 = True

# ─── Ticker validation pass ──────────────────────────────────────────────
# Pre-download validation: download 1 week of data per ticker to check if
# yfinance can serve it. Dramatically reduces wasted retries on invalid tickers.
#
# Recommended asset types: equities + etfs (~42k tickers, ~2h).
# Funds (~58k) are NOT recommended — most use foreign exchange suffixes
# (.F, .MU, .L, .MC, .MX, .VI) that Yahoo barely supports, causing
# individual "possibly delisted" probes that take 17+ hours.
#
# Examples of excluded funds (from FinanceDatabase):
#   ES0137434009.MC  CAIXABANK VALOR 97/50 EUROSTOXX (Spanish guarantee fund)
#   0P0000CV0Z.F     APO Vivace INKA R (German alt fund, Frankfurt)
#   0P00011UY5.F     ProfitlichSchmidlin Fonds UI I (German allocation fund)
#   ES0118844002.MC  BANKINTER IBEX 2025 II GARANTIZ (Spanish guarantee fund)
#   I+CORPB-MG1.MX   unnamed Mexican corporate bond fund
#   FJKLWX           The First Trust Combined Series (US closed-end trust)
#   JXBCX            JPMorgan Access Balanced Fund (US balanced fund)
#   OBIOX            Oberweis International Opportunities Fund (US intl equity)
#   GFACX            American Funds Growth Fund of America C (US growth fund)
#   PXEAX            Pax Global Environmental Markets Fund A (US ESG fund)
#
# US-listed funds (5-letter tickers like FJKLWX, JXBCX) work fine on Yahoo;
# the problem is the ~50k foreign-exchange funds that dominate the list.
VALIDATION_WINDOWS = [('2024-07-01', '2024-07-08')]
VALIDATION_BATCH_SIZE = 50           # was 500 — Yahoo throttles large ticker-count requests
VALIDATION_MAX_RETRIES = 1          # low retries — just identifying valid tickers
VALIDATION_DELAY = 2.0              # was 4.0 — smaller batches can go faster
VALIDATION_TIMEOUT = 60             # seconds per validation batch
VALIDATION_CACHE_FILE = 'validated_tickers.json'
VALIDATION_CACHE_HOURS = 168        # 1 week — re-validate weekly

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

# ─── Tor proxy rotation ─────────────────────────────────────────────────
TOR_SOCKS_PROXY = 'socks5://127.0.0.1:9050'
TOR_CONTROL_PORT = 9051
TOR_CONTROL_PASSWORD = ''           # empty = cookie auth (default Homebrew config)
TOR_ROTATE_EVERY_N_BATCHES = 1     # new circuit every N batches (1 = every batch)
