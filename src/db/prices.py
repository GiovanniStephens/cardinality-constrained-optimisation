"""Price data storage and retrieval."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional

import pandas as pd

from src.db.connection import _get_exchange_id, _now
from src.db.tickers import _ensure_tickers
from src.db.metadata import _save_data_source_no_commit

logger = logging.getLogger(__name__)


def save_prices(conn: sqlite3.Connection, prices_df: pd.DataFrame,
                exchange: str, asset_type: str = 'etf',
                source: Optional[str] = None,
                names: Optional[dict[str, str]] = None,
                countries: Optional[dict[str, str]] = None,
                sectors: Optional[dict[str, str]] = None,
                industries: Optional[dict[str, str]] = None,
                category_groups: Optional[dict[str, str]] = None,
                categories: Optional[dict[str, str]] = None,
                volumes_df: Optional[pd.DataFrame] = None) -> int:
    """
    Save a wide-format DataFrame of prices to the database.

    prices_df: index = dates (or integer index), columns = ticker symbols, values = close prices.
    exchange: 'US', 'NZX', 'ASX'
    names: optional dict {symbol: name_string} to populate ticker names.
    countries: optional dict {symbol: country_string} to populate ticker countries.
    volumes_df: optional wide DataFrame (same shape/labels as prices_df) of share
        volumes. When given, the matching volume is stored alongside each close;
        otherwise volume is left NULL (legacy behaviour).
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

    # Align an optional volume frame to the same string index for lookup.
    vol = None
    if volumes_df is not None:
        vol = volumes_df.copy()
        if hasattr(vol.index, 'date'):
            vol.index = pd.to_datetime(vol.index).strftime('%Y-%m-%d')
        else:
            vol.index = vol.index.astype(str)

    def _volume_at(date_str: str, symbol: str):
        if vol is None or symbol not in vol.columns or date_str not in vol.index:
            return None
        v = vol.at[date_str, symbol]
        return int(v) if pd.notna(v) else None

    # Phantom-date guard (July 2026): no US listing trades on a weekend, so a
    # weekend-dated row for a dot-less symbol is always junk (the historical
    # source was promotion's unlimited ffill across mixed-calendar union
    # indexes — see purge_phantom_rows). Foreign dot-suffix listings keep
    # their legitimate weekend sessions (Tel Aviv trades Sundays); other
    # exchanges are untouched. Holidays are deliberately not guarded here —
    # data_quality detection covers them without calendar risk at write time.
    weekend_dates = set()
    if exchange == 'US':
        idx_dt = pd.to_datetime(pd.Index(df.index), errors='coerce')
        weekend_dates = {s for s, d in zip(df.index, idx_dt)
                         if pd.notna(d) and d.dayofweek >= 5}

    # Build rows for bulk insert
    rows = []
    n_weekend_skipped = 0
    for date_str in df.index:
        weekend = date_str in weekend_dates
        for symbol in symbols:
            if weekend and '.' not in symbol:
                if pd.notna(df.at[date_str, symbol]):
                    n_weekend_skipped += 1
                continue
            val = df.at[date_str, symbol]
            if pd.notna(val):
                rows.append((ticker_map[symbol], date_str, float(val),
                             _volume_at(date_str, symbol)))
    if n_weekend_skipped:
        logger.warning(
            "save_prices: skipped %d weekend-dated rows for US-listed "
            "symbols (phantom-date guard)", n_weekend_skipped)

    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO prices (ticker_id, date, close, volume) "
            "VALUES (?, ?, ?, ?)",
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


def load_prices(conn: sqlite3.Connection, exchange: Optional[str] = None,
                asset_type: Optional[str] = None,
                start: Optional[str] = None, end: Optional[str] = None,
                tickers: Optional[list[str]] = None,
                exclude_countries: Optional[list[str]] = None,
                exclude_flagged: bool = True,
                allow_min_history_flags: bool = False,
                min_coverage: Optional[float] = 0.95,
                ffill_limit: Optional[int] = 5) -> pd.DataFrame:
    """
    Load prices as a wide-format DataFrame (dates as index, tickers as columns).
    Matches the format returned by existing load_data() functions.

    asset_type: optional filter, e.g. 'etf', 'stock', 'fund'.
    exclude_countries: optional list of country strings to exclude.
    exclude_flagged: if True (default), skip tickers with non-NULL excluded column.
    allow_min_history_flags: if True (with exclude_flagged), treat
        ``min_history:*`` flags as ADVISORY — keep those tickers while still
        excluding hard flags (stale, frozen_price, suspect returns). Used by
        the production rebalance, whose admission bar is shorter than the 5y
        research standard; the caller's coverage window does the real gating.
    ffill_limit: max consecutive NaN rows to forward-fill (default 5).
        ``None``/``0`` disables filling entirely — ``promote_staging`` relies
        on this for verbatim copies. (July 2026: this used to be passed
        straight to pandas, where ``ffill(limit=None)`` means UNLIMITED fill;
        on the union date index of a mixed-calendar chunk, promotion
        materialised ~170k phantom weekend/holiday rows into prod — see the
        purge_phantom_rows docstring and CLAUDE.md data-refresh gotcha 4.)
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
        if allow_min_history_flags:
            conditions.append(
                "(t.excluded IS NULL OR t.excluded LIKE 'min_history:%')")
        else:
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

    # Forward-fill NaN (capped to avoid propagating stale prices). Falsy
    # ffill_limit means NO fill — never hand None to pandas (unlimited fill).
    if ffill_limit:
        df = df.ffill(limit=ffill_limit)

    logger.info("load_prices: %d rows x %d tickers in %.1fs",
                len(df), df.shape[1], _time.time() - t0)
    return df


def update_volumes(conn: sqlite3.Connection, volumes_df: pd.DataFrame,
                   exchange: str, asset_type: str = 'etf') -> int:
    """Write share volumes onto EXISTING price rows (UPDATE only).

    volumes_df: wide DataFrame (index = dates, columns = symbols, values = volume).
    Only (ticker_id, date) pairs that already have a close row are updated, so the
    carefully-maintained adjusted-close series is never disturbed and no synthetic
    close is fabricated. Symbols/dates without an existing row are skipped.
    Returns the number of rows updated.
    """
    exchange_id = _get_exchange_id(conn, exchange)
    symbols = list(volumes_df.columns)
    # Map symbols to ticker ids (only those already present).
    placeholders = ','.join('?' for _ in symbols) if symbols else "''"
    id_map = {r['symbol']: r['id'] for r in conn.execute(
        f"SELECT id, symbol FROM tickers WHERE exchange_id = ? "
        f"AND asset_type = ? AND symbol IN ({placeholders})",
        [exchange_id, asset_type, *symbols]).fetchall()}

    df = volumes_df.copy()
    if hasattr(df.index, 'date'):
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    else:
        df.index = df.index.astype(str)

    rows = []
    for symbol in symbols:
        tid = id_map.get(symbol)
        if tid is None:
            continue
        col = df[symbol]
        for date_str, val in col.items():
            if pd.notna(val):
                rows.append((int(val), tid, date_str))
    if not rows:
        return 0
    with conn:
        cur = conn.executemany(
            "UPDATE prices SET volume = ? WHERE ticker_id = ? AND date = ?", rows)
    updated = cur.rowcount if cur.rowcount is not None else 0
    logger.info("update_volumes: %d (ticker,date) volume cells written", updated)
    return updated


def load_avg_dollar_volume(conn: sqlite3.Connection,
                           exchange: Optional[str] = None,
                           asset_type: Optional[str] = None,
                           tickers: Optional[list[str]] = None,
                           window: Optional[int] = 126) -> pd.Series:
    """Return average dollar volume (close x volume) per ticker as a Series.

    Averages over each ticker's most recent `window` rows that carry a non-NULL
    volume (default ~126 trading days / 6 months). Tickers with no stored volume
    are omitted. Used as the liquidity score for universe curation.
    """
    query = """
        SELECT t.symbol AS symbol, p.date AS date, p.close AS close, p.volume AS volume
        FROM prices p
        JOIN tickers t ON p.ticker_id = t.id
        WHERE p.volume IS NOT NULL
    """
    params: list[Any] = []
    if exchange is not None:
        query += " AND t.exchange_id = ?"
        params.append(_get_exchange_id(conn, exchange))
    if asset_type is not None:
        query += " AND t.asset_type = ?"
        params.append(asset_type)
    if tickers:
        placeholders = ','.join('?' for _ in tickers)
        query += f" AND t.symbol IN ({placeholders})"
        params.extend(tickers)
    query += " ORDER BY t.symbol, p.date"

    rows = conn.execute(query, params).fetchall()
    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame([(r['symbol'], r['close'], r['volume']) for r in rows],
                      columns=['symbol', 'close', 'volume'])
    df['dollar'] = df['close'].astype(float) * df['volume'].astype(float)

    # rows arrive ordered by (symbol, date); keep each symbol's most recent window.
    recent = df.groupby('symbol', sort=False).tail(window) if window else df
    return recent.groupby('symbol', sort=True)['dollar'].mean()


def get_latest_prices_date(conn: sqlite3.Connection, exchange: Optional[str] = None,
                           asset_type: Optional[str] = None) -> Optional[str]:
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


def get_tickers_with_prices(conn: sqlite3.Connection, exchange: Optional[str] = None) -> set[str]:
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


# ---- Phantom-date purge (July 2026 incident) ----------------------------------


#: NYSE special full-day closures within the data range (not derivable from
#: recurring holiday rules): national days of mourning for G.H.W. Bush and
#: Jimmy Carter.
NYSE_SPECIAL_CLOSURES = ('2018-12-05', '2025-01-09')


def _nyse_closure_dates(start: str, end: str) -> set[str]:
    """NYSE full-closure weekday dates between start and end (YYYY-MM-DD).

    Built from pandas' holiday primitives (no extra dependency): the
    recurring NYSE holiday rules plus the special closures above. Known
    approximation: ``nearest_workday`` observes a Saturday Jan 1 on the
    preceding Dec 31, which the NYSE does not — callers must therefore treat
    membership as necessary-but-not-sufficient and cross-check against an
    anchor ticker before deleting anything (purge_phantom_rows does).
    """
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar,
        GoodFriday,
        Holiday,
        USLaborDay,
        USMartinLutherKingJr,
        USMemorialDay,
        USPresidentsDay,
        USThanksgivingDay,
        nearest_workday,
    )

    class _NYSEClosures(AbstractHolidayCalendar):
        rules = [
            Holiday('New Year', month=1, day=1, observance=nearest_workday),
            USMartinLutherKingJr,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            Holiday('Juneteenth', month=6, day=19,
                    start_date='2022-01-01', observance=nearest_workday),
            Holiday('Independence Day', month=7, day=4,
                    observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday('Christmas', month=12, day=25,
                    observance=nearest_workday),
        ]

    dates = _NYSEClosures().holidays(pd.Timestamp(start), pd.Timestamp(end))
    closures = {d.strftime('%Y-%m-%d') for d in dates}
    closures.update(d for d in NYSE_SPECIAL_CLOSURES if start <= d <= end)
    return closures


def purge_phantom_rows(conn: sqlite3.Connection, exchange: str = 'US',
                       dry_run: bool = False,
                       anchor: str = 'SPY') -> dict[str, Any]:
    """Delete phantom non-trading-day price rows for US-listed symbols.

    July 2026 incident: ``promote_staging`` passed ``ffill_limit=None`` to
    ``load_prices``, which pandas took as UNLIMITED forward fill — every
    promotion materialised each US ticker's previous close onto the other
    calendars in its 200-ticker chunk (Tel Aviv Sundays, Asian/European
    sessions on NYSE holidays): ~170k junk rows. Values were verified to be
    exact previous-close duplicates, so deletion loses no information.

    Scope: dot-less symbols on the given exchange only — foreign dot-suffix
    listings keep their legitimate weekend/holiday sessions.

    * Weekend rows are deleted unconditionally (never a US session).
    * NYSE-holiday weekday rows are deleted only when the date is BOTH in the
      computed closure calendar AND absent from the ``anchor`` ticker's dates
      (SPY trades every real session) — either signal alone is not trusted.
    * Computed closures where the anchor HAS a row are reported as calendar
      discrepancies; anchor-missing weekdays NOT in the calendar are reported
      as suspect anchor gaps. Neither is deleted.

    :param dry_run: report what would be deleted without touching the DB.
    :return: dict with counts and the report lists.
    """
    exchange_id = _get_exchange_id(conn, exchange)
    dotless = ("SELECT id FROM tickers "
               "WHERE exchange_id = ? AND symbol NOT LIKE '%.%'")

    # -- Category A: weekend rows ------------------------------------------
    weekend_pred = (f"ticker_id IN ({dotless}) "
                    "AND strftime('%w', date) IN ('0','6')")
    n_weekend = conn.execute(
        f"SELECT COUNT(*) FROM prices WHERE {weekend_pred}",
        (exchange_id,)).fetchone()[0]

    # -- Category B: NYSE-closure weekday rows ------------------------------
    anchor_dates = {r[0] for r in conn.execute(
        "SELECT DISTINCT p.date FROM prices p JOIN tickers t "
        "ON p.ticker_id = t.id WHERE t.symbol = ? AND t.exchange_id = ?",
        (anchor, exchange_id)).fetchall()}
    holiday_dates: list[str] = []
    discrepancies: list[str] = []
    suspect_gaps: list[str] = []
    n_holiday = 0
    if anchor_dates:
        lo, hi = min(anchor_dates), max(anchor_dates)
        closures = _nyse_closure_dates(lo, hi)
        holiday_dates = sorted(closures - anchor_dates)
        discrepancies = sorted(closures & anchor_dates)
        # Weekday dates carried by dot-less tickers that the anchor lacks and
        # the calendar cannot explain: suspect anchor data gaps. Report only.
        candidate_rows = conn.execute(
            f"SELECT DISTINCT date FROM prices WHERE ticker_id IN ({dotless}) "
            "AND strftime('%w', date) NOT IN ('0','6') "
            "AND date BETWEEN ? AND ?",
            (exchange_id, lo, hi)).fetchall()
        suspect_gaps = sorted(
            d[0] for d in candidate_rows
            if d[0] not in anchor_dates and d[0] not in closures)
        if holiday_dates:
            ph = ','.join('?' for _ in holiday_dates)
            n_holiday = conn.execute(
                f"SELECT COUNT(*) FROM prices WHERE ticker_id IN ({dotless}) "
                f"AND date IN ({ph})",
                (exchange_id, *holiday_dates)).fetchone()[0]
    else:
        logger.warning(
            "purge_phantom_rows: anchor %s has no rows on %s — holiday "
            "purge skipped, weekend purge still applies", anchor, exchange)

    logger.info(
        "purge_phantom_rows%s: %d weekend rows; %d rows on %d NYSE-closure "
        "dates; %d calendar discrepancies%s; %d suspect anchor-gap dates%s",
        " (dry run)" if dry_run else "", n_weekend, n_holiday,
        len(holiday_dates), len(discrepancies),
        f" {discrepancies[:5]}" if discrepancies else "",
        len(suspect_gaps), f" {suspect_gaps[:5]}" if suspect_gaps else "")

    if not dry_run:
        with conn:
            conn.execute(
                f"DELETE FROM prices WHERE {weekend_pred}", (exchange_id,))
            if n_holiday:
                ph = ','.join('?' for _ in holiday_dates)
                conn.execute(
                    f"DELETE FROM prices WHERE ticker_id IN ({dotless}) "
                    f"AND date IN ({ph})",
                    (exchange_id, *holiday_dates))
        logger.info("purge_phantom_rows: deleted %d rows",
                    n_weekend + n_holiday)

    return {
        'weekend_rows': n_weekend,
        'holiday_rows': n_holiday,
        'holiday_dates': holiday_dates,
        'calendar_discrepancies': discrepancies,
        'suspect_gaps': suspect_gaps,
        'deleted': not dry_run,
    }
