"""
Investment universe configuration.

Single source of truth for which markets, asset types, and filters
to apply when building the investable universe. All instruments are
US-listed (equities trade as ADRs, ETFs are US-domiciled).
"""

# ─── Geographic scope ─────────────────────────────────────────────────────────

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

# ─── History & data filters ───────────────────────────────────────────────────

MIN_HISTORY_YEARS = 5
MIN_HISTORY_DAYS = 1260   # 5 * 252 trading days

# ─── Data quality thresholds ──────────────────────────────────────────────────

MAX_STALENESS_DAYS = 30          # last trade must be within 30 calendar days
MIN_ANNUAL_VOLATILITY = 0.001    # 0.1% annualised vol minimum
MAX_CONSECUTIVE_SAME_PRICE = 20  # flag if 20+ identical consecutive closes
MAX_EXTREME_RETURN_PCT = 0.05    # flag if >5% of days have >10 std dev returns

# ─── Portfolio constraints ────────────────────────────────────────────────────

TARGET_POSITIONS = (10, 20)     # (min, max) cardinality constraint
REBALANCE_FREQUENCY = 'quarterly'
