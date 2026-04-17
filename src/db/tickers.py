"""Ticker management functions."""

import logging

from src.db.connection import _get_exchange_id, _now

logger = logging.getLogger(__name__)


def _ensure_tickers(conn, symbols, exchange_id, asset_type='etf', names=None,
                    countries=None, sectors=None, industries=None,
                    category_groups=None, categories=None):
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

    # Backfill sector/industry/category_group/category
    for col, mapping in [('sector', sectors), ('industry', industries),
                         ('category_group', category_groups),
                         ('category', categories)]:
        if mapping:
            conn.executemany(
                f"UPDATE tickers SET {col} = ?, updated_at = ? "
                f"WHERE id = ? AND {col} IS NULL",
                [(mapping[s], now, existing[s])
                 for s in existing if s in mapping and mapping[s]],
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


def load_ticker_metadata(conn, symbols, exchange='US'):
    """Load metadata for a list of ticker symbols.

    Returns {symbol: {'country': ..., 'asset_type': ..., 'sector': ...,
                      'category_group': ...}}.
    Missing fields are None.
    """
    exchange_id = _get_exchange_id(conn, exchange)
    placeholders = ','.join('?' for _ in symbols)
    rows = conn.execute(
        f"SELECT symbol, country, asset_type, sector, category_group "
        f"FROM tickers WHERE symbol IN ({placeholders}) AND exchange_id = ?",
        list(symbols) + [exchange_id],
    ).fetchall()
    return {
        r['symbol']: {
            'country': r['country'],
            'asset_type': r['asset_type'],
            'sector': r['sector'],
            'category_group': r['category_group'],
        }
        for r in rows
    }


def backfill_metadata(conn, exchange='US'):
    """Backfill sector/category_group from FinanceDatabase for tickers missing them.

    Updates tickers in-place. Can be run standalone: ``python -m src.db backfill``.
    """
    import financedatabase as fd

    exchange_id = _get_exchange_id(conn, exchange)
    now = _now()

    # Find tickers needing backfill
    rows = conn.execute(
        "SELECT id, symbol, asset_type FROM tickers "
        "WHERE exchange_id = ? AND (sector IS NULL OR category_group IS NULL)",
        (exchange_id,),
    ).fetchall()
    if not rows:
        logger.info("backfill_metadata: nothing to backfill")
        return 0

    symbols_by_type = {}
    for r in rows:
        symbols_by_type.setdefault(r['asset_type'], []).append(
            (r['id'], r['symbol'])
        )

    updated = 0

    # Equities: sector from FinanceDatabase
    if 'stock' in symbols_by_type:
        try:
            eq_df = fd.Equities().select()
            for tid, sym in symbols_by_type['stock']:
                if sym in eq_df.index:
                    row = eq_df.loc[sym]
                    sector = row.get('sector') if hasattr(row, 'get') else None
                    industry = row.get('industry') if hasattr(row, 'get') else None
                    if sector:
                        conn.execute(
                            "UPDATE tickers SET sector = ?, updated_at = ? "
                            "WHERE id = ? AND sector IS NULL",
                            (sector, now, tid),
                        )
                        updated += 1
                    if industry:
                        conn.execute(
                            "UPDATE tickers SET industry = ?, updated_at = ? "
                            "WHERE id = ? AND industry IS NULL",
                            (industry, now, tid),
                        )
        except (KeyError, AttributeError, ValueError) as e:
            logger.warning("backfill_metadata: equities lookup failed: %s", e)

    # ETFs: category_group/category from FinanceDatabase
    if 'etf' in symbols_by_type:
        try:
            etf_df = fd.ETFs().select()
            for tid, sym in symbols_by_type['etf']:
                if sym in etf_df.index:
                    row = etf_df.loc[sym]
                    cg = row.get('category_group') if hasattr(row, 'get') else None
                    cat = row.get('category') if hasattr(row, 'get') else None
                    if cg:
                        conn.execute(
                            "UPDATE tickers SET category_group = ?, updated_at = ? "
                            "WHERE id = ? AND category_group IS NULL",
                            (cg, now, tid),
                        )
                        updated += 1
                    if cat:
                        conn.execute(
                            "UPDATE tickers SET category = ?, updated_at = ? "
                            "WHERE id = ? AND category IS NULL",
                            (cat, now, tid),
                        )
        except (KeyError, AttributeError, ValueError) as e:
            logger.warning("backfill_metadata: ETFs lookup failed: %s", e)

    conn.commit()
    logger.info("backfill_metadata: updated %d tickers", updated)
    return updated
