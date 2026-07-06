"""Crude keyword-based ETF classification from the ``name`` field.

A starting-point classifier for when the FinanceDatabase ``category`` /
``category_group`` metadata is blank or corrupted (~30% of active ETFs).
It maps a fund name to a broad asset-class bucket using ordered keyword rules:
the first matching bucket wins, so more specific / risk-overriding buckets
(leveraged-inverse, crypto, commodity) are checked before the broad Equity
catch-all.

This is deliberately crude — it gets most names right and gives every ticker a
label so category caps have something to bite on, but it WILL mislabel some.
Use it as a fallback where real metadata is missing, and treat ``Unknown`` as
its own capped bucket rather than trusting it to be empty.

Keyword choices here are the product of a spot-check audit (June 2026) that
caught false positives like "Netherlands"→ether, "France"→franc, "Currency
Hedged equity"→currency, resource *equities* (gold miners, oil services)→
commodity, "Enhanced Income"→leveraged, and plain "Short <index>" inverse funds
slipping into Equity. Edit with care and re-audit.

July 2026 hardening (driven by the 596 Unknowns left after the Nasdaq-directory
name backfill): added preferred/convertible/CLO/MBS/structured-credit → Fixed
Income, buffer-family/buywrite/tail-risk/rate-hedge overlays → Alternatives, and
a long thematic/sector/brand/country equity tail. Deliberately NOT added, after
hazard analysis: bare 'income' (equity-income vs bond income), 'alternative'
(would misroute "Alternative Energy"), 'enhanced' (prior audit false positive),
'allocation'/'multi-asset' (balanced funds stay Unknown — capped, not trusted),
'tactical', and 'hedged equity' (collides with "Currency Hedged Equity").
Substring paddings matter: 'clo ' not 'clo' (cloud/close/clock), ' media' not
'media' (intermediate), ' mbs', ' ipo', 'adr '.

Buckets: Leveraged, Inverse, Crypto, Real Estate, Commodity, Fixed Income,
Cash, Currency, Alternatives, Equity, Unknown.
"""

CATEGORIES = [
    'Leveraged', 'Inverse', 'Crypto', 'Real Estate', 'Commodity', 'Fixed Income',
    'Cash', 'Currency', 'Alternatives', 'Equity', 'Unknown',
]

# Unambiguous leverage/inverse markers.
_LEVERAGE_TERMS = (
    'inverse', 'leverage', 'ultrapro', 'ultrashort', '2x', '3x', '-1x',
    '1.5x', 'bull 2', 'bull 3', 'bear ', ' bear', 'daily target',
)

# "Short" means *duration* (a bond fund), not *inverse*, in these contexts —
# don't treat them as leveraged/inverse.
_SHORT_DURATION_CTX = (
    'short-term', 'short term', 'short duration', 'short-duration',
    'short maturity', 'short-maturity', 'ultra short', 'ultra-short',
    'short dated', 'short-dated', 'short bond', 'short treasury',
    'short gilt', 'short government', 'short corporate', 'short core',
    'short federal', 'short provincial', 'shorter', 'short-dur',
)


def _is_leveraged(name_lc: str) -> bool:
    if any(t in name_lc for t in _LEVERAGE_TERMS):
        return True
    if 'daily' in name_lc and ('bull' in name_lc or 'bear' in name_lc):
        return True
    # 'ultra' as a leverage brand, but not the "ultra short-term" bond usage.
    if 'ultra' in name_lc and 'short' not in name_lc and 'term' not in name_lc:
        return True
    # Plain "-1x Short <index>" inverse funds — "short" outside a duration context.
    if 'short' in name_lc and not any(c in name_lc for c in _SHORT_DURATION_CTX):
        return True
    return False


# Within a leveraged/inverse fund, these mark *short/inverse* (negative-beta)
# exposure — a hedge. Their absence means a long-amplifying (leveraged) product.
_INVERSE_TERMS = ('inverse', 'short', 'bear', 'ultrashort', '-1x', '-2x', '-3x')


def _leverage_direction(name_lc: str) -> str:
    """Split a leveraged/inverse fund by direction: short/inverse exposure (a
    crisis hedge, incl. 2x/3x short) -> 'Inverse'; long-amplifying -> 'Leveraged'.
    Account margin can replace 'Leveraged' but not 'Inverse'."""
    if any(t in name_lc for t in _INVERSE_TERMS):
        return 'Inverse'
    return 'Leveraged'


# A commodity keyword + one of these means a resource *equity* (miners, oil
# services), not the commodity itself → classify as Equity. Also includes brand
# words that collide with commodity substrings ("Gold"man, "Corn"erstone).
_RESOURCE_EQUITY_TERMS = (
    'miner', 'mining', 'producer', 'exploration', 'services', 'equipment',
    'companies', 'drill', 'refiner', 'agribusiness', 'goldman', 'cornerstone',
    # resource *sector* equity (companies), not the physical commodity:
    'oil & gas', '& gas', 'nuclear', 'bugs', 'global agriculture',
)


# A real-estate keyword + one of these is actually an equity fund that merely
# excludes or names real estate ("ex A-REIT", "Property & Casualty Insurance").
_REALESTATE_EQUITY_TERMS = (
    'ex a-reit', 'ex-reit', 'ex reit', 'property & casualty',
    'property and casualty', 'casualty',
)

# Ordered (label, keyword-tuple). First match wins. Fixed Income sits before
# Cash/Currency so "Treasury Bond ... Dollar" resolves to Fixed Income, while a
# bare "US Dollar" falls through to Currency.
_RULES = [
    ('Crypto', ('bitcoin', 'ethereum', 'crypto', 'blockchain', 'digital asset')),
    ('Real Estate', ('reit', 'real estate', 'property')),
    ('Commodity', ('gold', 'silver', 'bullion', 'platinum', 'palladium',
                   'copper', 'crude', ' oil', 'natural gas', 'commodit',
                   'precious metal', 'agriculture', 'wheat', 'corn', 'uranium',
                   'rhodium', 'soybean', 'sugar', 'cocoa', 'cotton', 'nickel',
                   'zinc', 'aluminium', 'aluminum', 'base metal', 'gasoline')),
    ('Fixed Income', ('bond', 'treasury', 'govt', 'government', 'gilt', 'bund',
                      'sovereign', 'aggregate', 'corporate', 'high yield',
                      'investment grade', 'municipal', ' muni', 'tips',
                      'inflation-protected', 'inflation linked', 'fixed income',
                      'senior loan', 'floating rate', 'maturity', 'duration',
                      ' debt', 'credit', ' ktb', 'iboxx',
                      # July 2026: hybrid/securitised income families
                      'preferred', 'convertible', 'clo ', 'cmbs', ' mbs',
                      'mortgage', 'securitiz', 'structured credit',
                      'structured product', 'collateralized')),
    ('Cash', ('money market', 'money mkt', 't-bill', 'treasury bill',
              'overnight', 'liquidity fund', 'ultra short-term')),
    ('Currency', ('currencyshares', 'dollar', ' yen', 'swiss franc', 'sterling',
                  'renminbi', 'yuan', 'rupee', ' fx ')),
    ('Alternatives', ('managed futures', 'trend', 'macro', 'hedge fund',
                      'hedge replication', ' vix', 'merger', 'arbitrage',
                      'market neutral', 'multi-strateg', 'long/short',
                      'long short', 'absolute return', 'risk premia', 'buffer',
                      'covered call', 'put write', 'option',
                      # July 2026: defined-outcome / overlay / hedge families
                      'buywrite', 'buy-write', 'structured outcome',
                      'defined outcome', 'target outcome', 'defined risk',
                      'collared', 'tail risk', 'swan sos', 'barrier',
                      'interest rate hedge', 'inflation expectations')),
    ('Equity', ('equity', 'stock', 's&p', 'sp 500', 'nasdaq', 'msci', 'ftse',
                'russell', 'stoxx', 'dax', 'nikkei', 'topix', 'kospi',
                'dividend', 'index', 'large cap', 'large-cap', 'mid cap',
                'mid-cap', 'small cap', 'small-cap', 'mega cap', 'value',
                'growth', 'quality', 'momentum', 'esg', 'world', 'emerging',
                'developed', 'europe', 'japan', 'china', 'usa', 'sector',
                'technology', 'financial', 'health', 'consumer', 'industrial',
                'utilit', 'communication', 'materials', 'factor', 'minimum vol',
                'low volatility', 'energy', 'water', 'cyber', 'battery',
                'robot', 'semiconductor', 'infrastructure', 'clean energy',
                'global ', 'csi300', 'csi 300', 'set50', ' 500', ' 400',
                ' 600', ' 100', ' 200', '50 ',
                # July 2026: thematic / sector / brand tail from the Unknowns
                'biotech', 'pharma', 'medical', 'insurance', 'bank',
                'aerospace', 'defense', 'transport', 'retail', 'steel',
                'solar', 'software', 'internet', 'cloud computing', ' media',
                'entertainment', 'leisure', 'beverage', 'restaurant',
                'home construction', 'homebuild', 'building', 'micro-cap',
                'micro cap', 'moat', 'buyback', 'earnings fund', 'alphadex',
                'dorsey wright', 'dow jones', 'dow 30', 'qqq', ' ipo', 'adr ',
                'shareholder yield', 'natural resources', 'environmental',
                'rare earth', 'strategic metals', 'mlp', 'bdc',
                'broker-dealer', 'gaming', 'esports', 'betting', 'innovation',
                'disruptive',
                # single-country equity funds observed in the Unknown set
                'india', 'vietnam', 'israel', 'switzerland', 'germany',
                'united kingdom', 'latin america', 'eurozone', 'canada',
                'brazil', 'mexico', 'taiwan', 'australia')),
]


# Hard fixed-income markers — their presence means a real bond fund, so the
# "dividend equity" override below must not fire.
_HARD_BOND_TERMS = (
    'bond', 'corporate', 'treasury', 'muni', 'gilt', 'sovereign', 'govt',
    'government', 'loan', ' debt', 'tips', 'aggregate', 'bund',
)


def _is_dividend_equity(name_lc: str) -> bool:
    """True for high-dividend *equity* funds that falsely trip an FI keyword
    (e.g. 'high yield' in "High Yield Equity Dividend")."""
    return (('equity' in name_lc or 'dividend' in name_lc)
            and not any(t in name_lc for t in _HARD_BOND_TERMS))


def classify_etf(name) -> str:
    """Classify an ETF into a broad asset-class bucket from its name."""
    if not name:
        return 'Unknown'
    n = str(name).lower()
    if _is_leveraged(n):
        return _leverage_direction(n)
    for label, terms in _RULES:
        if any(t in n for t in terms):
            if label == 'Commodity' and any(t in n for t in _RESOURCE_EQUITY_TERMS):
                return 'Equity'            # resource equity, not the commodity
            if label == 'Real Estate' and any(t in n for t in _REALESTATE_EQUITY_TERMS):
                return 'Equity'            # excludes/names RE but is an equity fund
            if label == 'Fixed Income' and _is_dividend_equity(n):
                return 'Equity'            # high-dividend equity, not a bond fund
            return label
    return 'Unknown'


def categorise_universe(conn, exchange='US', asset_type='etf'):
    """Return {symbol: bucket} for an exchange/asset_type using crude name rules."""
    from src import db
    ex_id = db._get_exchange_id(conn, exchange)
    rows = conn.execute(
        "SELECT symbol, name FROM tickers "
        "WHERE exchange_id = ? AND asset_type = ?", (ex_id, asset_type)).fetchall()
    return {r['symbol']: classify_etf(r['name']) for r in rows}
