"""Known-bad ticker cache for download retry logic.

Hardened (June 2026) after an incident where a single transient Yahoo failure
on 2026-04-16 blacklisted SPY/VOO/AGG (failure_count=1, no expiry) and the
pipeline silently skipped them on every run for ~2 months:

  * **Protected watchlist** — the liquid core (config.PIPELINE_PROTECTED_TICKERS_CSV,
    i.e. data/core_etfs.csv) is never cached or skipped.
  * **TTL self-heal** — entries carry ``expires_at``; once past it they are purged
    and re-attempted, so transient failures don't blacklist forever. Legacy rows
    (NULL expires_at) heal off ``last_failed`` + TTL.
  * **Higher threshold** — default ``min_failures`` is config-driven (3), so one
    blip no longer blacklists.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.db.connection import _get_exchange_id, _now

logger = logging.getLogger(__name__)

_protected_cache: Optional[set[str]] = None


def load_protected_tickers() -> set[str]:
    """Tickers that must never be cached as bad or skipped (the liquid core).

    Read (uppercased) from the 'Tickers' column of
    ``config.PIPELINE_PROTECTED_TICKERS_CSV``; cached after first read. Returns
    an empty set if the file is missing.
    """
    global _protected_cache
    if _protected_cache is not None:
        return _protected_cache
    from src import config
    path = config.PIPELINE_PROTECTED_TICKERS_CSV
    symbols: set[str] = set()
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            col = 'Tickers' if 'Tickers' in cols else (cols[0] if cols else None)
            if col:
                for row in reader:
                    val = (row.get(col) or '').strip()
                    if val:
                        symbols.add(val.upper())
    except FileNotFoundError:
        logger.warning("Protected-ticker file not found: %s", path)
    _protected_cache = symbols
    return symbols


def save_known_bad_tickers(conn: sqlite3.Connection, symbols: list[str],
                           exchange: str = 'US', ttl_days: Optional[int] = None) -> None:
    """Record failed tickers (skipping the protected watchlist), with a TTL.

    Increments failure_count and refreshes ``expires_at`` on repeat failures, so a
    persistently-failing ticker stays cached while a one-off failure ages out.
    """
    from src import config
    if ttl_days is None:
        ttl_days = config.PIPELINE_BAD_TICKER_TTL_DAYS
    protected = load_protected_tickers()
    symbols = [s for s in symbols if s.upper() not in protected]
    if not symbols:
        return
    exchange_id = _get_exchange_id(conn, exchange)
    now = _now()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
    conn.executemany(
        "INSERT INTO known_bad_tickers "
        "(symbol, exchange_id, failure_count, first_failed, last_failed, expires_at) "
        "VALUES (?, ?, 1, ?, ?, ?) "
        "ON CONFLICT(symbol, exchange_id) DO UPDATE SET "
        "failure_count = failure_count + 1, last_failed = ?, expires_at = ?",
        [(s, exchange_id, now, now, expires_at, now, expires_at) for s in symbols],
    )
    conn.commit()
    logger.info("Recorded %d failed tickers (exchange=%s, ttl=%dd)",
                len(symbols), exchange, ttl_days)


def load_known_bad_tickers(conn: sqlite3.Connection, exchange: str = 'US',
                           min_failures: Optional[int] = None) -> set[str]:
    """Return symbols failed >= min_failures, not expired, never protected.

    Purges expired entries first (TTL self-heal): a row is expired when its
    ``expires_at`` is past, or — for legacy rows with NULL expires_at — when
    ``last_failed`` is older than the TTL.
    """
    from src import config
    if min_failures is None:
        min_failures = config.PIPELINE_BAD_CACHE_MIN_FAILURES
    ttl_days = config.PIPELINE_BAD_TICKER_TTL_DAYS
    exchange_id = _get_exchange_id(conn, exchange)
    now = datetime.now(timezone.utc).isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()
    conn.execute(
        "DELETE FROM known_bad_tickers WHERE exchange_id = ? AND ("
        "  (expires_at IS NOT NULL AND expires_at <= ?) OR "
        "  (expires_at IS NULL AND last_failed <= ?))",
        (exchange_id, now, cutoff),
    )
    conn.commit()
    rows = conn.execute(
        "SELECT symbol FROM known_bad_tickers "
        "WHERE exchange_id = ? AND failure_count >= ?",
        (exchange_id, min_failures),
    ).fetchall()
    protected = load_protected_tickers()
    return {r[0] for r in rows if r[0].upper() not in protected}


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
