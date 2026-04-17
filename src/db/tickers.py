"""Ticker management functions."""

import logging

from src.db.connection import _get_exchange_id, _now

logger = logging.getLogger(__name__)


def _ensure_tickers(conn, symbols, exchange_id, asset_type='etf', names=None,
                    countries=None):
    """Ensure all symbols exist in tickers table. Returns {symbol: ticker_id}.

    asset_type: one of 'etf', 'stock', 'fund', 'managed_fund'.
    names: optional dict {symbol: name_string} to populate the name column.
    countries: optional dict {symbol: country_string} to populate the country column.
    """
    now = _now()
    # Fetch existing
    placeholders = ','.join('?' for _ in symbols)
    rows = conn.execute(
        f"SELECT id, symbol FROM tickers WHERE exchange_id = ? AND symbol IN ({placeholders})",
        [exchange_id] + list(symbols),
    ).fetchall()
    existing = {r['symbol']: r['id'] for r in rows}

    # Insert missing
    missing = [s for s in symbols if s not in existing]
    if missing:
        conn.executemany(
            "INSERT OR IGNORE INTO tickers "
            "(symbol, name, country, exchange_id, asset_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(s, names.get(s) if names else None,
              countries.get(s) if countries else None,
              exchange_id, asset_type, now, now)
             for s in missing],
        )
        # Re-fetch to get IDs for newly inserted
        rows = conn.execute(
            f"SELECT id, symbol FROM tickers WHERE exchange_id = ? AND symbol IN ({placeholders})",
            [exchange_id] + list(symbols),
        ).fetchall()
        existing = {r['symbol']: r['id'] for r in rows}

    # Backfill names for existing tickers that don't have one yet
    if names:
        conn.executemany(
            "UPDATE tickers SET name = ?, updated_at = ? WHERE id = ? AND name IS NULL",
            [(names[s], now, existing[s]) for s in existing if s in names and names[s]],
        )

    # Backfill countries for existing tickers that don't have one yet
    if countries:
        conn.executemany(
            "UPDATE tickers SET country = ?, updated_at = ? WHERE id = ? AND country IS NULL",
            [(countries[s], now, existing[s])
             for s in existing if s in countries and countries[s]],
        )

    return existing


def set_ticker_excluded(conn, ticker_id, reason):
    """Flag a ticker as excluded with a reason string."""
    conn.execute(
        "UPDATE tickers SET excluded = ?, updated_at = ? WHERE id = ?",
        (reason, _now(), ticker_id),
    )


def clear_ticker_excluded(conn, ticker_id):
    """Remove the exclusion flag from a ticker."""
    conn.execute(
        "UPDATE tickers SET excluded = NULL, updated_at = ? WHERE id = ?",
        (_now(), ticker_id),
    )


def get_excluded_tickers(conn, exchange=None):
    """Get all excluded tickers with their reasons."""
    query = "SELECT t.id, t.symbol, t.excluded FROM tickers t WHERE t.excluded IS NOT NULL"
    params = []
    if exchange is not None:
        exchange_id = _get_exchange_id(conn, exchange)
        query += " AND t.exchange_id = ?"
        params.append(exchange_id)
    query += " ORDER BY t.excluded, t.symbol"
    return conn.execute(query, params).fetchall()
