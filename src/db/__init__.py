"""
SQLite database module for portfolio optimisation.

Stores prices, forecasts, optimisation runs, and backtest results.
Uses sqlite3 (stdlib) + pandas. All timestamps are ISO 8601 UTC.
"""

from src.db.schema import SCHEMA_SQL, DEFAULT_EXCHANGES
from src.db.connection import DB_PATH, _now, get_connection, _get_exchange_id
from src.db.tickers import (
    _ensure_tickers,
    set_ticker_excluded,
    clear_ticker_excluded,
    get_excluded_tickers,
)
from src.db.prices import save_prices, load_prices, get_latest_prices_date
from src.db.forecasts import (
    save_forecast_results,
    load_expected_returns,
    load_variances,
)
from src.db.optimisation import save_optimisation_run, get_recent_runs, get_run_holdings
from src.db.backtest import (
    save_backtest_session,
    save_backtest_result,
    get_recent_backtests,
    get_backtest_results,
)
from src.db.metadata import (
    _save_data_source_no_commit,
    save_data_source,
    get_latest_data_source,
    get_latest_forecast,
)
from src.db.migrations import (
    migrate_csvs,
    _migrate_price_csv,
    _migrate_forecasts,
    _migrate_ticker_list,
)
