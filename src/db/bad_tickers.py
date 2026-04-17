"""Known-bad ticker cache for download retry logic."""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from src.db.connection import _get_exchange_id, _now

logger = logging.getLogger(__name__)


def save_known_bad_tickers(conn: sqlite3.Connection, symbols: list[str],
                           exchange: str = 'US') -> None:
    """Record failed tickers. Increments failure_count on repeated failures."""
    exchange_id = _get_exchange_id(conn, exchange)
    now = _now()
    # INSERT on first failure, increment count on subsequent failures
    conn.executemany(
        "INSERT INTO known_bad_tickers "
        "(symbol, exchange_id, failure_count, first_failed, last_failed) "
        "VALUES (?, ?, 1, ?, ?) "
        "ON CONFLICT(symbol, exchange_id) DO UPDATE SET "
        "failure_count = failure_count + 1, last_failed = ?",
        [(s, exchange_id, now, now, now) for s in symbols],
    )
    conn.commit()
    logger.info("Recorded %d failed tickers (exchange=%s)", len(symbols), exchange)


def load_known_bad_tickers(conn: sqlite3.Connection, exchange: str = 'US',
                           min_failures: int = 2) -> set[str]:
    """Return the set of ticker symbols that have failed >= min_failures times."""
    exchange_id = _get_exchange_id(conn, exchange)
    rows = conn.execute(
        "SELECT symbol FROM known_bad_tickers "
        "WHERE exchange_id = ? AND failure_count >= ?",
        (exchange_id, min_failures),
    ).fetchall()
    return {r[0] for r in rows}


def clear_known_bad_tickers(conn: sqlite3.Connection, exchange: Optional[str] = None) -> None:
    """Remove known-bad tickers. If exchange is None, clear all."""
    if exchange is not None:
        exchange_id = _get_exchange_id(conn, exchange)
        conn.execute(
            "DELETE FROM known_bad_tickers WHERE exchange_id = ?",
            (exchange_id,),
        )
    else:
        conn.execute("DELETE FROM known_bad_tickers")
    conn.commit()
    logger.info("Cleared known-bad ticker cache (exchange=%s)",
                exchange or 'all')
