"""One-time (re-runnable) share-volume backfill for liquidity-aware curation.

The price pipeline stores only adjusted close. Average dollar volume (ADV) is the
liquidity score used to pick the most-liquid representative per cluster in
``curate_universe.py``. This module fetches a recent window of daily volume from
Yahoo for the US-listed ETF candidate set and writes it onto existing price rows
via :func:`src.db.prices.update_volumes` (volume-only UPDATE — never touches close).

Re-running it simply refreshes the volume window, so it doubles as the ongoing
volume-maintenance path (the heavy proxy/circuit-breaker download pipeline is left
untouched).

Usage:
    python -m src.db backfill-volume [--period 9mo] [--batch-size 60]
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Optional

import pandas as pd

from src.db.connection import _get_exchange_id
from src.db.prices import update_volumes

logger = logging.getLogger(__name__)


def candidate_symbols(conn: sqlite3.Connection, exchange: str = 'US',
                      asset_type: str = 'etf') -> list[str]:
    """US-listed (no '.') ETF symbols eligible for volume/ADV backfill.

    min_history:* flags are advisory here (the production rebalance admits at
    ~2y and needs ADV for those tickers); hard flags still exclude.
    """
    exchange_id = _get_exchange_id(conn, exchange)
    rows = conn.execute(
        "SELECT symbol FROM tickers WHERE exchange_id = ? AND asset_type = ? "
        "AND (excluded IS NULL OR excluded LIKE 'min_history:%') "
        "AND symbol NOT LIKE '%.%' ORDER BY symbol",
        (exchange_id, asset_type)).fetchall()
    return [r['symbol'] for r in rows]


def _fetch_volume_batch(batch: list[str], period: str) -> Optional[pd.DataFrame]:
    """Fetch a wide volume DataFrame (dates x symbols) for a batch via yfinance."""
    import yfinance as yf
    data = yf.download(" ".join(batch), period=period, interval="1d",
                       group_by="ticker", auto_adjust=False, threads=True,
                       progress=False)
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    out = {}
    for t in batch:
        try:
            vol = data[t]["Volume"]
        except (KeyError, TypeError):
            continue
        if vol.notna().any():
            out[t] = vol
    if not out:
        return None
    return pd.DataFrame(out)


def backfill_volume(conn: sqlite3.Connection, exchange: str = 'US',
                    asset_type: str = 'etf', period: str = '9mo',
                    batch_size: int = 60, sleep: float = 0.5,
                    symbols: Optional[list[str]] = None) -> dict:
    """Fetch recent daily volume for the candidate set and store it.

    period: yfinance period string (~9mo safely covers the 126-day ADV window).
    Returns stats: {candidates, batches, symbols_with_volume, cells_written}.
    """
    syms = symbols if symbols is not None else candidate_symbols(conn, exchange, asset_type)
    batches = [syms[i:i + batch_size] for i in range(0, len(syms), batch_size)]
    logger.info("Volume backfill: %d candidate %s symbols, %d batches (period=%s)",
                len(syms), asset_type, len(batches), period)

    total_cells = 0
    symbols_seen = 0
    for i, batch in enumerate(batches, 1):
        try:
            vol_df = _fetch_volume_batch(batch, period)
        except Exception as e:  # network/transport — log and continue
            logger.warning("Batch %d/%d failed: %s", i, len(batches), e)
            vol_df = None
        if vol_df is not None and not vol_df.empty:
            symbols_seen += vol_df.shape[1]
            total_cells += update_volumes(conn, vol_df, exchange, asset_type)
        logger.info("  batch %d/%d done (symbols w/ volume so far: %d, cells: %d)",
                    i, len(batches), symbols_seen, total_cells)
        if sleep and i < len(batches):
            time.sleep(sleep)

    stats = {'candidates': len(syms), 'batches': len(batches),
             'symbols_with_volume': symbols_seen, 'cells_written': total_cells}
    logger.info("Volume backfill complete: %s", stats)
    return stats
