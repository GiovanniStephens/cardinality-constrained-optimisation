"""Full-history TSMOM sleeve: computed once, memoised, sliced per window.

The sleeve is the SAME overlay for every portfolio and every backtest window,
so it is computed once on the FULL price history — loaded directly from the DB,
bypassing the universe min-history/coverage filters that would otherwise drop
the basket members — and then *sliced* per window by the caller (by reindexing
the returned Series onto each window's date index).

Computing on full history then slicing — never slicing then recomputing — is the
causal-correctness keystone: it keeps the trailing-vol normaliser identical
across windows, and because the series is causal per-date (see
``src.sleeves.trend``), any sliced window's OOS portion only ever depends on
prices within that OOS portion or earlier, exactly what a live strategy sees.
"""

import logging

import pandas as pd

from src.config import TSMOM_BASKET, DATA_FFILL_LIMIT
from src.sleeves.trend import compute_tsmom_returns

logger = logging.getLogger(__name__)

# Memoised full-history sleeve return Series (default basket only). One compute
# per process; the series is window-independent.
_sleeve_cache = None


def _basket_candidates(basket):
    """Flatten the basket's fallback chains into a single ticker list."""
    out = []
    for candidates in basket.values():
        out.extend(candidates)
    return out


def get_cached_sleeve_series(conn, basket=None, force=False):
    """Return the full-history TSMOM sleeve return Series, memoised per process.

    :param conn: sqlite3 connection (the one ``evaluate_split`` already holds).
    :param basket: optional basket override (tests). When given, the result is
        neither read from nor written to the module cache.
    :param force: recompute even if a cached default-basket series exists.
    :return: ``pd.Series`` of daily sleeve log returns indexed by datetime.
    """
    global _sleeve_cache
    use_cache = basket is None
    if use_cache and _sleeve_cache is not None and not force:
        return _sleeve_cache

    from src import db
    basket = TSMOM_BASKET if basket is None else basket
    prices = db.load_prices(
        conn, exchange='US', tickers=_basket_candidates(basket),
        exclude_flagged=False, min_coverage=None, ffill_limit=DATA_FFILL_LIMIT,
    )
    if prices.empty:
        raise ValueError(
            "TSMOM sleeve: no basket prices found in the DB (tried %s)."
            % _basket_candidates(basket))
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    sleeve = compute_tsmom_returns(prices, basket=basket)
    logger.info("TSMOM sleeve built: %d days, %s to %s",
                len(sleeve), sleeve.index.min(), sleeve.index.max())

    if use_cache:
        _sleeve_cache = sleeve
    return sleeve


def reset_cache():
    """Clear the memoised sleeve series (mainly for tests)."""
    global _sleeve_cache
    _sleeve_cache = None
