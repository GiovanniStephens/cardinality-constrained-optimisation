"""CSV migration functions for importing legacy data."""

import logging
import os

import pandas as pd

from src.db.connection import _get_exchange_id
from src.db.tickers import _ensure_tickers
from src.db.prices import save_prices
from src.db.forecasts import save_forecast_results

logger = logging.getLogger(__name__)


def migrate_csvs(conn, data_dir=None):
    """One-time import of existing CSV data into the database."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')

    # 1. Import price CSVs
    _migrate_price_csv(conn, data_dir, 'ETF_Prices.csv', 'US', 'etf')
    _migrate_price_csv(conn, data_dir, 'time_series_20251016_113257.csv', 'NZX', 'managed_fund')
    _migrate_price_csv(conn, data_dir, 'leveraged_ETF_Prices.csv', 'US', 'etf')
    _migrate_price_csv(conn, data_dir, '2x_leveraged_ETF_Prices.csv', 'US', 'etf')
    _migrate_price_csv(conn, data_dir, 'NZ_ETF_Prices.csv', 'NZX', 'etf')

    # 2. Import forecast CSVs
    _migrate_forecasts(conn, data_dir, 'expected_returns.csv', 'variances.csv', 'US')
    _migrate_forecasts(conn, data_dir, 'NZ_expected_returns.csv', 'NZ_variances.csv', 'NZX')

    # 3. Import ticker lists for completeness
    _migrate_ticker_list(conn, data_dir, 'ETFs_Full.csv', 'US', 'etf')
    _migrate_ticker_list(conn, data_dir, 'US_Stocks.csv', 'US', 'stock')
    _migrate_ticker_list(conn, data_dir, 'NZ_ETFs.csv', 'NZX', 'etf')
    _migrate_ticker_list(conn, data_dir, '2x_leveraged_ETFs.csv', 'US', 'etf')
    _migrate_ticker_list(conn, data_dir, '3x_leveraged_ETFs.csv', 'US', 'etf')

    logger.info("Migration summary:")
    for table in ['tickers', 'prices', 'forecast_runs', 'expected_returns',
                   'variances', 'data_sources']:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info("  %s: %d rows", table, count)


def _migrate_price_csv(conn, data_dir, filename, exchange, asset_type):
    """Import a single price CSV file."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        logger.info("  Skipping %s (not found)", filename)
        return

    logger.info("Importing %s...", filename)
    df = pd.read_csv(filepath, index_col=0)

    # Handle the InvestNow time series with datetime+timezone index
    if 'time_series' in filename:
        df.index = pd.to_datetime(df.index, utc=True).strftime('%Y-%m-%d')
    else:
        # Try parsing index as dates (strings like '2024-01-01')
        try:
            parsed = pd.to_datetime(df.index, format='%Y-%m-%d')
            df.index = parsed.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            # Integer indices (0, 1, 2, ...) — cannot store without dates.
            logger.warning("  Skipping %s: index is not date-formatted. "
                           "Re-download with download_and_save() to get proper dates.",
                           filename)
            logger.debug("Date parsing traceback for %s:", filename, exc_info=True)
            return

    save_prices(conn, df, exchange=exchange, asset_type=asset_type,
                source='csv_migration')


def _migrate_forecasts(conn, data_dir, er_filename, var_filename, exchange):
    """Import paired expected returns and variances CSVs."""
    er_path = os.path.join(data_dir, er_filename)
    var_path = os.path.join(data_dir, var_filename)
    if not os.path.exists(er_path) or not os.path.exists(var_path):
        logger.info("  Skipping %s/%s (not found)", er_filename, var_filename)
        return

    logger.info("Importing %s + %s...", er_filename, var_filename)
    er = pd.read_csv(er_path, index_col=0)
    var = pd.read_csv(var_path, index_col=0)

    # These CSVs have a single column named '0'
    er_series = er['0'] if '0' in er.columns else er.iloc[:, 0]
    var_series = var['0'] if '0' in var.columns else var.iloc[:, 0]

    from src.config import TRADING_DAYS_PER_YEAR
    save_forecast_results(conn, er_series, var_series,
                          n_periods=TRADING_DAYS_PER_YEAR, exchange=exchange,
                          notes=f'Migrated from {er_filename} + {var_filename}')


def _migrate_ticker_list(conn, data_dir, filename, exchange, asset_type):
    """Import a ticker list CSV (single column of symbols)."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        logger.info("  Skipping %s (not found)", filename)
        return

    logger.info("Importing ticker list %s...", filename)
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    col = df.columns[0]  # Usually 'Tickers'
    symbols = df[col].dropna().tolist()
    exchange_id = _get_exchange_id(conn, exchange)
    _ensure_tickers(conn, symbols, exchange_id, asset_type)
    conn.commit()
