"""Price data storage and retrieval."""

import logging

import pandas as pd

from src.db.connection import _get_exchange_id, _now
from src.db.tickers import _ensure_tickers
from src.db.metadata import _save_data_source_no_commit

logger = logging.getLogger(__name__)


def save_prices(conn, prices_df, exchange, asset_type='etf', source=None,
                names=None, countries=None, sectors=None, industries=None,
                category_groups=None, categories=None):
    """
    Save a wide-format DataFrame of prices to the database.

    prices_df: index = dates (or integer index), columns = ticker symbols, values = close prices.
    exchange: 'US', 'NZX', 'ASX'
    names: optional dict {symbol: name_string} to populate ticker names.
    countries: optional dict {symbol: country_string} to populate ticker countries.
    Returns data_source id.
    """
    import time as _time
    t0 = _time.time()
    exchange_id = _get_exchange_id(conn, exchange)
    symbols = list(prices_df.columns)
    dupes = [s for s in set(symbols) if symbols.count(s) > 1]
    if dupes:
        raise ValueError(f"DataFrame has duplicate column names: {dupes}")
    ticker_map = _ensure_tickers(conn, symbols, exchange_id, asset_type,
                                 names=names, countries=countries,
                                 sectors=sectors, industries=industries,
                                 category_groups=category_groups,
                                 categories=categories)

    # Normalise index to date strings
    df = prices_df.copy()
    if hasattr(df.index, 'date'):
        # datetime index — convert to YYYY-MM-DD strings
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    else:
        # integer index — convert to string as-is
        df.index = df.index.astype(str)

    # Build rows for bulk insert
    rows = []
    for date_str in df.index:
        for symbol in symbols:
            val = df.at[date_str, symbol]
            if pd.notna(val):
                rows.append((ticker_map[symbol], date_str, float(val)))

    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO prices (ticker_id, date, close) VALUES (?, ?, ?)",
            rows,
        )

        # Record data source
        dates = [d for d in df.index]
        ds_id = _save_data_source_no_commit(
            conn,
            source=source or ('yahoo_finance' if exchange == 'US' else 'investnow'),
            exchange_id=exchange_id,
            date_range_start=min(dates) if dates else None,
            date_range_end=max(dates) if dates else None,
            num_tickers=len(symbols),
            num_rows=len(rows),
        )
    logger.info("save_prices: %d tickers, %d rows in %.1fs",
                len(symbols), len(rows), _time.time() - t0)
    return ds_id


def load_prices(conn, exchange=None, asset_type=None, start=None, end=None,
                tickers=None, exclude_countries=None, exclude_flagged=True,
                min_coverage=0.95, ffill_limit=5):
    """
    Load prices as a wide-format DataFrame (dates as index, tickers as columns).
    Matches the format returned by existing load_data() functions.

    asset_type: optional filter, e.g. 'etf', 'stock', 'fund'.
    exclude_countries: optional list of country strings to exclude.
    exclude_flagged: if True (default), skip tickers with non-NULL excluded column.
    ffill_limit: max consecutive NaN rows to forward-fill (default 5).
    """
    query = """
        SELECT t.symbol, p.date, p.close
        FROM prices p
        JOIN tickers t ON p.ticker_id = t.id
    """
    conditions = []
    params = []

    if exchange is not None:
        exchange_id = _get_exchange_id(conn, exchange)
        conditions.append("t.exchange_id = ?")
        params.append(exchange_id)
    if asset_type is not None:
        conditions.append("t.asset_type = ?")
        params.append(asset_type)
    if start is not None:
        conditions.append("p.date >= ?")
        params.append(start)
    if end is not None:
        conditions.append("p.date <= ?")
        params.append(end)
    if tickers is not None:
        placeholders = ','.join('?' for _ in tickers)
        conditions.append(f"t.symbol IN ({placeholders})")
        params.extend(tickers)
    if exclude_countries is not None:
        placeholders = ','.join('?' for _ in exclude_countries)
        conditions.append(f"(t.country IS NULL OR t.country NOT IN ({placeholders}))")
        params.extend(exclude_countries)
    if exclude_flagged:
        conditions.append("t.excluded IS NULL")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY p.date, t.symbol"

    import time as _time
    t0 = _time.time()
    rows = conn.execute(query, params).fetchall()
    if not rows:
        logger.info("load_prices: no rows found (%.1fs)", _time.time() - t0)
        return pd.DataFrame()

    # Pivot to wide format
    data = [(r['date'], r['symbol'], r['close']) for r in rows]
    df = pd.DataFrame(data, columns=['date', 'symbol', 'close'])
    df = df.pivot(index='date', columns='symbol', values='close')
    df.index.name = None
    df.columns.name = None

    # Apply min_coverage filter
    if min_coverage is not None and min_coverage > 0:
        threshold = int(min_coverage * len(df))
        df = df.dropna(axis=1, thresh=threshold)

    # Forward-fill NaN (capped to avoid propagating stale prices)
    df = df.ffill(limit=ffill_limit)

    logger.info("load_prices: %d rows x %d tickers in %.1fs",
                len(df), df.shape[1], _time.time() - t0)
    return df


def get_latest_prices_date(conn, exchange=None, asset_type=None):
    """Return the most recent date string in the prices table, or None.

    Useful for incremental downloads: start from the day after this date.
    """
    query = "SELECT MAX(p.date) FROM prices p"
    joins = []
    conditions = []
    params = []

    if exchange is not None or asset_type is not None:
        joins.append("JOIN tickers t ON p.ticker_id = t.id")
    if exchange is not None:
        exchange_id = _get_exchange_id(conn, exchange)
        conditions.append("t.exchange_id = ?")
        params.append(exchange_id)
    if asset_type is not None:
        conditions.append("t.asset_type = ?")
        params.append(asset_type)

    if joins:
        query += " " + " ".join(joins)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    row = conn.execute(query, params).fetchone()
    return row[0] if row else None


def get_tickers_with_prices(conn, exchange=None):
    """Return the set of ticker symbols that have at least one price row."""
    query = ("SELECT DISTINCT t.symbol FROM tickers t "
             "JOIN prices p ON t.id = p.ticker_id")
    params = []
    if exchange:
        exchange_id = _get_exchange_id(conn, exchange)
        query += " WHERE t.exchange_id = ?"
        params.append(exchange_id)
    rows = conn.execute(query, params).fetchall()
    return {r[0] for r in rows}
