"""Database schema definition, migrations, and default seed data."""

import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- Broad market groupings (US, NZX, ASX, etc.)
CREATE TABLE IF NOT EXISTS exchanges (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    code     TEXT NOT NULL UNIQUE,
    name     TEXT,
    country  TEXT
);

-- Master list of instruments (unique per exchange)
CREATE TABLE IF NOT EXISTS tickers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    name        TEXT,
    country     TEXT,
    excluded    TEXT,
    exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
    asset_type  TEXT NOT NULL DEFAULT 'etf'
        CHECK(asset_type IN ('etf', 'stock', 'fund', 'managed_fund')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(symbol, exchange_id)
);

-- Daily close prices (normalised: one row per ticker per date)
CREATE TABLE IF NOT EXISTS prices (
    ticker_id  INTEGER NOT NULL REFERENCES tickers(id),
    date       TEXT NOT NULL
        CHECK(date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    close      REAL NOT NULL,
    PRIMARY KEY (ticker_id, date)
);

-- Tracks each ARIMA/GARCH forecast generation
CREATE TABLE IF NOT EXISTS forecast_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id     INTEGER REFERENCES exchanges(id),
    created_at      TEXT NOT NULL,
    num_tickers     INTEGER,
    n_periods       INTEGER,
    elapsed_seconds REAL,
    notes           TEXT
);

-- Calculated expected returns (linked to a forecast run)
CREATE TABLE IF NOT EXISTS expected_returns (
    ticker_id       INTEGER NOT NULL REFERENCES tickers(id),
    forecast_run_id INTEGER NOT NULL REFERENCES forecast_runs(id) ON DELETE CASCADE,
    value           REAL NOT NULL,
    PRIMARY KEY (ticker_id, forecast_run_id)
);

-- Calculated variances (linked to a forecast run)
CREATE TABLE IF NOT EXISTS variances (
    ticker_id       INTEGER NOT NULL REFERENCES tickers(id),
    forecast_run_id INTEGER NOT NULL REFERENCES forecast_runs(id) ON DELETE CASCADE,
    value           REAL NOT NULL,
    PRIMARY KEY (ticker_id, forecast_run_id)
);

-- Tracks each data download event
CREATE TABLE IF NOT EXISTS data_sources (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id      INTEGER REFERENCES exchanges(id),
    source           TEXT NOT NULL,
    downloaded_at    TEXT NOT NULL,
    date_range_start TEXT,
    date_range_end   TEXT,
    num_tickers      INTEGER,
    num_rows         INTEGER,
    notes            TEXT
);

-- One row per GA / MIP / Monte Carlo optimisation run
CREATE TABLE IF NOT EXISTS optimisation_runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,
    script                TEXT NOT NULL,
    data_source           TEXT,
    data_source_id        INTEGER REFERENCES data_sources(id),

    -- Shared constraint parameters
    min_securities        INTEGER,
    max_securities        INTEGER,
    min_weight            REAL,
    max_weight            REAL,
    target_return         REAL,
    target_risk           REAL,

    -- Feature flags
    use_forecasts         INTEGER DEFAULT 0,
    use_copulae           INTEGER DEFAULT 0,
    risk_parity           INTEGER DEFAULT 0,

    -- Algorithm-specific parameters (JSON)
    params_json           TEXT,

    -- Results
    best_sharpe           REAL,
    portfolio_return      REAL,
    portfolio_volatility  REAL,
    num_selected          INTEGER,
    elapsed_seconds       REAL,
    notes                 TEXT
);

-- Security selections + weights for each optimisation run
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL REFERENCES optimisation_runs(id) ON DELETE CASCADE,
    ticker_id INTEGER NOT NULL REFERENCES tickers(id),
    weight    REAL NOT NULL,
    UNIQUE(run_id, ticker_id)
);

-- One row per backtest session
CREATE TABLE IF NOT EXISTS backtest_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    data_source     TEXT,
    data_source_id  INTEGER REFERENCES data_sources(id),
    num_portfolios  INTEGER NOT NULL,
    num_days_oos    INTEGER NOT NULL,
    use_forecast    INTEGER DEFAULT 0,
    optimiser_params_json TEXT,
    elapsed_seconds REAL,
    notes           TEXT,
    window_train_start TEXT,
    window_train_end   TEXT,
    window_test_start  TEXT,
    window_test_end    TEXT,
    window_label       TEXT,
    run_group_id       TEXT
);

-- Individual portfolio results within a backtest session
CREATE TABLE IF NOT EXISTS backtest_results (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            INTEGER NOT NULL REFERENCES backtest_sessions(id) ON DELETE CASCADE,
    category              TEXT NOT NULL,
    portfolio_index       INTEGER NOT NULL,
    annualised_return     REAL,
    annualised_volatility REAL,
    sharpe_ratio          REAL,
    downside_deviation    REAL,
    max_drawdown          REAL,
    calmar_ratio          REAL,
    sortino_ratio         REAL,
    holdings_json         TEXT,
    UNIQUE(session_id, category, portfolio_index)
);

-- Normalised backtest holdings (source of truth alongside holdings_json cache)
CREATE TABLE IF NOT EXISTS backtest_holdings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL REFERENCES backtest_results(id) ON DELETE CASCADE,
    ticker_id INTEGER NOT NULL REFERENCES tickers(id),
    weight    REAL NOT NULL,
    UNIQUE(result_id, ticker_id)
);

CREATE TABLE IF NOT EXISTS known_bad_tickers (
    symbol         TEXT NOT NULL,
    exchange_id    INTEGER NOT NULL REFERENCES exchanges(id),
    failure_count  INTEGER NOT NULL DEFAULT 1,
    first_failed   TEXT NOT NULL,
    last_failed    TEXT NOT NULL,
    PRIMARY KEY (symbol, exchange_id)
);

CREATE INDEX IF NOT EXISTS idx_tickers_exchange ON tickers(exchange_id);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker_id);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
CREATE INDEX IF NOT EXISTS idx_expected_returns_forecast ON expected_returns(forecast_run_id);
CREATE INDEX IF NOT EXISTS idx_variances_forecast ON variances(forecast_run_id);
CREATE INDEX IF NOT EXISTS idx_optimisation_runs_created ON optimisation_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_backtest_results_session ON backtest_results(session_id);
CREATE INDEX IF NOT EXISTS idx_backtest_holdings_result ON backtest_holdings(result_id);
"""

DEFAULT_EXCHANGES = [
    ('US', 'United States', 'US'),
    ('NZX', 'New Zealand Exchange', 'NZ'),
    ('ASX', 'Australian Securities Exchange', 'AU'),
]

SCHEMA_VERSION = 2


def _migrate_to_1(conn):
    """Initial schema — applied by SCHEMA_SQL; this is a no-op sentinel."""
    pass


def _migrate_to_2(conn):
    """Add sector/industry/category columns to tickers for group constraints."""
    for col in ('sector', 'industry', 'category_group', 'category'):
        try:
            conn.execute(f"ALTER TABLE tickers ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists


MIGRATIONS = {
    1: _migrate_to_1,
    2: _migrate_to_2,
}


def _get_schema_version(conn):
    """Return the current schema version, or 0 if the table doesn't exist."""
    try:
        row = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        logger.debug("schema_version table not found, assuming version 0")
        return 0


def _apply_migrations(conn):
    """Apply any pending migrations to bring the DB up to SCHEMA_VERSION."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "  version    INTEGER PRIMARY KEY,"
        "  applied_at TEXT NOT NULL"
        ")"
    )
    current = _get_schema_version(conn)
    if current >= SCHEMA_VERSION:
        return

    for version in range(current + 1, SCHEMA_VERSION + 1):
        migrate_fn = MIGRATIONS.get(version)
        if migrate_fn is None:
            raise RuntimeError(
                f"Missing migration function for version {version}"
            )
        logger.info("Applying migration %d...", version)
        migrate_fn(conn)
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version, applied_at) "
            "VALUES (?, ?)",
            (version, datetime.now(timezone.utc).isoformat()),
        )
    conn.commit()
    logger.info("Schema at version %d", SCHEMA_VERSION)
