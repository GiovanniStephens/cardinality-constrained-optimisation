"""Price data loading: DB-first with CSV fallback, filtering, and cleaning."""

import logging

import pandas as pd

from src.config import (
    DATA_MIN_COVERAGE,
    DATA_FFILL_LIMIT,
    DATA_LOOKBACK_DAYS,
    DATA_MIN_COVERAGE_PERMISSIVE,
)

logger = logging.getLogger(__name__)


def load_prices(exchange='US', csv_fallback=None, conn=None,
                last_n_days=None, min_coverage=None, ffill_limit=None):
    """Load price data from DB with CSV fallback, applying standard filters.

    :param exchange: exchange code for DB query (default 'US').
    :param csv_fallback: CSV path to use if DB is empty.
    :param conn: sqlite3 connection. If None, opens and closes one automatically.
    :param last_n_days: if set, keep only the most recent N calendar days.
    :param min_coverage: minimum non-null fraction to keep a column (default from config).
    :param ffill_limit: max consecutive NaN to forward-fill (default from config).
    :return: cleaned DataFrame with dates as index, tickers as columns.
    """
    if min_coverage is None:
        min_coverage = DATA_MIN_COVERAGE
    if ffill_limit is None:
        ffill_limit = DATA_FFILL_LIMIT

    own_conn = False
    if conn is None:
        from src import db
        conn = db.get_connection()
        own_conn = True

    try:
        from src import db as _db
        data = _db.load_prices(conn, exchange=exchange)
    finally:
        if own_conn:
            conn.close()

    if data.empty and csv_fallback:
        logger.info("No data in DB, falling back to CSV: %s", csv_fallback)
        return load_prices_csv(csv_fallback, min_coverage=min_coverage,
                               last_n_days=last_n_days)
    if data.empty:
        return data

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    if last_n_days is not None:
        cutoff = data.index[-1] - pd.Timedelta(days=last_n_days)
        data = data[data.index >= cutoff]
    data = data.dropna(axis=1, thresh=int(min_coverage * len(data)))
    data = data.ffill(limit=ffill_limit)
    return data


def load_training_data(exchange='US', csv_fallback=None, lookback_days=None,
                       min_coverage=None):
    """Load and filter price data for training, with DB-first-CSV-fallback.

    Convenience wrapper for entry-point scripts that consolidates the common
    pattern of loading from DB, falling back to CSV, and applying standard
    filters (lookback, coverage, forward-fill).

    :param exchange: exchange code for DB query (default 'US').
    :param csv_fallback: CSV path to use if DB is empty.
    :param lookback_days: restrict to last N calendar days (default from config).
    :param min_coverage: minimum non-null fraction to keep a column (default from config).
    :return: cleaned DataFrame.
    :raises ValueError: if the resulting DataFrame is empty.
    """
    if lookback_days is None:
        lookback_days = DATA_LOOKBACK_DAYS
    if min_coverage is None:
        min_coverage = DATA_MIN_COVERAGE

    data = load_prices(exchange=exchange, csv_fallback=csv_fallback,
                       last_n_days=lookback_days, min_coverage=min_coverage)
    if data.empty:
        raise ValueError("No price data available from DB or CSV fallback.")
    logger.info("Loaded price data: %d rows x %d columns", *data.shape)
    return data


def load_prices_csv(filename, min_coverage=0.95, last_n_days=None):
    """Load price data from CSV with coverage filtering and forward-fill.

    :param filename: path to CSV file (index_col=0 assumed).
    :param min_coverage: minimum fraction of non-null rows to keep a column.
    :param last_n_days: if set, keep only the most recent N calendar days.
    :return: cleaned DataFrame with dates as index, tickers as columns.
    """
    prices_df = pd.read_csv(filename, index_col=0)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()
    if last_n_days is not None:
        cutoff = prices_df.index[-1] - pd.Timedelta(days=last_n_days)
        prices_df = prices_df[prices_df.index >= cutoff]
    thresh = int(min_coverage * len(prices_df))
    prices_df = prices_df.loc[:, prices_df.notna().sum() >= thresh]
    prices_df = prices_df.ffill(limit=DATA_FFILL_LIMIT)
    return prices_df


def load_data(filename, min_coverage=DATA_MIN_COVERAGE_PERMISSIVE, last_n_days=None):
    """Load price data with permissive coverage filtering.

    Convenience wrapper around load_prices_csv with a lower default
    min_coverage (0.10 vs 0.95) — used by entry points and scripts.
    Raises ValueError if the resulting DataFrame is empty.
    """
    prices_df = load_prices_csv(filename, min_coverage=min_coverage,
                                last_n_days=last_n_days)
    if prices_df.empty:
        raise ValueError(f"Loaded CSV '{filename}' is empty.")
    return prices_df
