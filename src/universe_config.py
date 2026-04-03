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

# ─── Portfolio constraints ────────────────────────────────────────────────────

TARGET_POSITIONS = (10, 20)     # (min, max) cardinality constraint
REBALANCE_FREQUENCY = 'quarterly'
