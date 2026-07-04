"""Liquidity / tradeability filtering for the production rebalance universe.

The broad ETF universe stored under the ``US`` exchange is polluted with
foreign-listed micro-caps (``329660.KS``, ``GLD.BK``, ``USDTR.IS``, …) that are
not tradeable on the user's Interactive Brokers account and carry no stored
share volume. ``filter_by_liquidity`` removes them and any thinly-traded US ETF
below an average-dollar-volume (ADV) floor, so the optimiser can only pick names
that are actually deployable.

Two independent stages:

1. **Foreign listings** — any symbol containing ``'.'`` is a non-US Yahoo listing
   (``.KS`` Korea, ``.TO`` Toronto, ``.L`` London, …). US listings have no dot.
   This is the codebase convention (see ``build_dedup_map.is_us_listed``,
   ``curate_universe._load_candidates``, ``volume_backfill.candidate_symbols``).
   Data-independent, so it removes the foreign names even before ADV is known.

2. **ADV floor** — reuse :func:`src.db.load_avg_dollar_volume` (mean of
   ``close * volume`` over the recent ``adv_window`` days with stored volume) and
   drop US ETFs below ``min_adv_usd``. Symbols with no stored volume have unknown
   liquidity and are dropped too (tradeability-safe); run
   ``python -m src.db backfill-volume`` first to maximise ADV coverage.

Must-have tickers passed via ``protect`` are never dropped.
"""

import logging

import pandas as pd

from src import db

logger = logging.getLogger(__name__)


def filter_by_liquidity(prices, conn, min_adv_usd, *, exclude_foreign=True,
                        protect=(), adv_window=126):
    """Return *prices* with foreign listings and sub-ADV US ETFs removed.

    :param prices: wide price DataFrame (dates x symbols).
    :param conn: open SQLite connection (for the ADV lookup).
    :param min_adv_usd: minimum average daily dollar volume; ``0``/``None``
        disables the ADV stage, leaving only the foreign-listing filter.
    :param exclude_foreign: drop dot-suffix (non-US) symbols (default True).
    :param protect: iterable of symbols always kept (e.g. the must-haves),
        even if foreign or below the ADV floor.
    :param adv_window: trailing days used to compute ADV (default ~6 months).
    :return: a new DataFrame with the surviving columns (original order).
    """
    protect = {str(t).upper() for t in protect}
    cols = list(prices.columns)

    # Stage 1 — foreign (dot-suffix) listings.
    if exclude_foreign:
        keep = [c for c in cols if '.' not in c or c.upper() in protect]
    else:
        keep = list(cols)
    n_foreign = len(cols) - len(keep)

    # Stage 2 — ADV floor.
    n_below = n_unknown = 0
    if min_adv_usd and min_adv_usd > 0:
        adv = db.load_avg_dollar_volume(conn, exchange='US', asset_type='etf',
                                        tickers=keep, window=adv_window)
        kept = []
        for c in keep:
            if c.upper() in protect:
                kept.append(c)
                continue
            v = adv.get(c)
            if v is None or pd.isna(v):
                n_unknown += 1
            elif v >= min_adv_usd:
                kept.append(c)
            else:
                n_below += 1
        keep = kept

    logger.info(
        "Liquidity filter: %d -> %d tickers (dropped foreign=%d, below $%.0f "
        "ADV=%d, unknown-ADV=%d; protected=%d)",
        len(cols), len(keep), n_foreign, (min_adv_usd or 0), n_below,
        n_unknown, len(protect),
    )
    return prices[keep]
