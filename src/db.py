"""
SQLite database module for portfolio optimisation.

Stores prices, forecasts, optimisation runs, and backtest results.
Uses sqlite3 (stdlib) + pandas. All timestamps are ISO 8601 UTC.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'portfolio.db')

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
    exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
    asset_type  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(symbol, exchange_id)
);

-- Daily close prices (normalised: one row per ticker per date)
CREATE TABLE IF NOT EXISTS prices (
    ticker_id  INTEGER NOT NULL REFERENCES tickers(id),
    date       TEXT NOT NULL,
    close      REAL NOT NULL,
    PRIMARY KEY (ticker_id, date)
);

-- Tracks each ARIMA/GARCH forecast generation
CREATE TABLE IF NOT EXISTS forecast_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
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
    num_generations       INTEGER,
    total_population_size INTEGER,
    mutation_rate         REAL,
    num_elites            INTEGER,
    migration_interval    INTEGER,
    migration_rate        REAL,
    num_islands           INTEGER,
    min_etfs              INTEGER,
    max_etfs              INTEGER,
    min_weight            REAL,
    max_weight            REAL,
    target_return         REAL,
    target_risk           REAL,
    use_forecasts         INTEGER DEFAULT 0,
    use_copulae           INTEGER DEFAULT 0,
    risk_parity           INTEGER DEFAULT 0,
    best_sharpe           REAL,
    portfolio_return      REAL,
    portfolio_volatility  REAL,
    num_selected          INTEGER,
    elapsed_seconds       REAL,
    notes                 TEXT
);

-- ETF selections + weights for each optimisation run
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER NOT NULL REFERENCES optimisation_runs(id) ON DELETE CASCADE,
    ticker  TEXT NOT NULL,
    weight  REAL NOT NULL,
    UNIQUE(run_id, ticker)
);

-- One row per backtest session
CREATE TABLE IF NOT EXISTS backtest_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    data_source     TEXT,
    num_portfolios  INTEGER NOT NULL,
    num_children    INTEGER NOT NULL,
    num_days_oos    INTEGER NOT NULL,
    use_forecast    INTEGER DEFAULT 0,
    elapsed_seconds REAL,
    notes           TEXT
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
    holdings_json         TEXT
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker_id);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
CREATE INDEX IF NOT EXISTS idx_expected_returns_forecast ON expected_returns(forecast_run_id);
CREATE INDEX IF NOT EXISTS idx_variances_forecast ON variances(forecast_run_id);
CREATE INDEX IF NOT EXISTS idx_optimisation_runs_created ON optimisation_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_backtest_results_session ON backtest_results(session_id);
"""

DEFAULT_EXCHANGES = [
    ('US', 'United States', 'US'),
    ('NZX', 'New Zealand Exchange', 'NZ'),
    ('ASX', 'Australian Securities Exchange', 'AU'),
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path=None):
    """Open a database connection, create tables if needed, seed exchanges."""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    # Seed exchanges if empty
    count = conn.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0]
    if count == 0:
        conn.executemany(
            "INSERT INTO exchanges (code, name, country) VALUES (?, ?, ?)",
            DEFAULT_EXCHANGES,
        )
        conn.commit()
    return conn


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_exchange_id(conn, code):
    """Look up exchange id by code. Raises ValueError if not found."""
    row = conn.execute(
        "SELECT id FROM exchanges WHERE code = ?", (code,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown exchange code: {code!r}")
    return row[0]


def _ensure_tickers(conn, symbols, exchange_id, asset_type=None):
    """Ensure all symbols exist in tickers table. Returns {symbol: ticker_id}."""
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
            "INSERT OR IGNORE INTO tickers (symbol, exchange_id, asset_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(s, exchange_id, asset_type, now, now) for s in missing],
        )
        # Re-fetch to get IDs for newly inserted
        rows = conn.execute(
            f"SELECT id, symbol FROM tickers WHERE exchange_id = ? AND symbol IN ({placeholders})",
            [exchange_id] + list(symbols),
        ).fetchall()
        existing = {r['symbol']: r['id'] for r in rows}

    return existing


# ─── Data storage ─────────────────────────────────────────────────────────────

def save_prices(conn, prices_df, exchange, asset_type='etf', source=None):
    """
    Save a wide-format DataFrame of prices to the database.

    prices_df: index = dates (or integer index), columns = ticker symbols, values = close prices.
    exchange: 'US', 'NZX', 'ASX'
    Returns data_source id.
    """
    exchange_id = _get_exchange_id(conn, exchange)
    symbols = list(prices_df.columns)
    ticker_map = _ensure_tickers(conn, symbols, exchange_id, asset_type)

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

    conn.executemany(
        "INSERT OR REPLACE INTO prices (ticker_id, date, close) VALUES (?, ?, ?)",
        rows,
    )

    # Record data source
    dates = [d for d in df.index]
    ds_id = save_data_source(
        conn,
        source=source or ('yahoo_finance' if exchange == 'US' else 'investnow'),
        exchange_id=exchange_id,
        date_range_start=min(dates) if dates else None,
        date_range_end=max(dates) if dates else None,
        num_tickers=len(symbols),
        num_rows=len(rows),
    )
    conn.commit()
    return ds_id


def load_prices(conn, exchange=None, start=None, end=None,
                tickers=None, min_coverage=0.95):
    """
    Load prices as a wide-format DataFrame (dates as index, tickers as columns).
    Matches the format returned by existing load_data() functions.
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

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY p.date, t.symbol"

    rows = conn.execute(query, params).fetchall()
    if not rows:
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

    # Forward-fill NaN
    df = df.ffill()

    return df


def save_forecast_results(conn, expected_returns_s, variances_s,
                          n_periods=None, elapsed_seconds=None,
                          notes=None, exchange='US'):
    """
    Save expected returns and variances from a forecast run.

    expected_returns_s: pd.Series indexed by ticker symbol
    variances_s: pd.Series indexed by ticker symbol
    Returns forecast_run_id.
    """
    now = _now()
    exchange_id = _get_exchange_id(conn, exchange)

    # All symbols from both series
    all_symbols = list(set(expected_returns_s.index) | set(variances_s.index))
    ticker_map = _ensure_tickers(conn, all_symbols, exchange_id)

    cur = conn.execute(
        "INSERT INTO forecast_runs (created_at, num_tickers, n_periods, elapsed_seconds, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (now, len(all_symbols), n_periods, elapsed_seconds, notes),
    )
    run_id = cur.lastrowid

    # Insert expected returns
    er_rows = []
    for symbol, value in expected_returns_s.items():
        if symbol in ticker_map and pd.notna(value):
            er_rows.append((ticker_map[symbol], run_id, float(value)))
    conn.executemany(
        "INSERT OR REPLACE INTO expected_returns (ticker_id, forecast_run_id, value) "
        "VALUES (?, ?, ?)",
        er_rows,
    )

    # Insert variances
    var_rows = []
    for symbol, value in variances_s.items():
        if symbol in ticker_map and pd.notna(value):
            var_rows.append((ticker_map[symbol], run_id, float(value)))
    conn.executemany(
        "INSERT OR REPLACE INTO variances (ticker_id, forecast_run_id, value) "
        "VALUES (?, ?, ?)",
        var_rows,
    )

    conn.commit()
    return run_id


def load_expected_returns(conn, forecast_run_id=None):
    """Load expected returns as a Series indexed by ticker symbol."""
    if forecast_run_id is None:
        row = conn.execute(
            "SELECT id FROM forecast_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return pd.Series(dtype=float)
        forecast_run_id = row[0]

    rows = conn.execute(
        "SELECT t.symbol, er.value FROM expected_returns er "
        "JOIN tickers t ON er.ticker_id = t.id "
        "WHERE er.forecast_run_id = ?",
        (forecast_run_id,),
    ).fetchall()
    return pd.Series(
        {r['symbol']: r['value'] for r in rows}, dtype=float
    )


def load_variances(conn, forecast_run_id=None):
    """Load variances as a Series indexed by ticker symbol."""
    if forecast_run_id is None:
        row = conn.execute(
            "SELECT id FROM forecast_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return pd.Series(dtype=float)
        forecast_run_id = row[0]

    rows = conn.execute(
        "SELECT t.symbol, v.value FROM variances v "
        "JOIN tickers t ON v.ticker_id = t.id "
        "WHERE v.forecast_run_id = ?",
        (forecast_run_id,),
    ).fetchall()
    return pd.Series(
        {r['symbol']: r['value'] for r in rows}, dtype=float
    )


# ─── Run history ──────────────────────────────────────────────────────────────

def save_optimisation_run(conn, params, results, holdings):
    """
    Save an optimisation run and its portfolio holdings.

    params: dict with keys like 'script', 'data_source', 'num_generations', etc.
    results: dict with keys like 'best_sharpe', 'portfolio_return', etc.
    holdings: list of (ticker, weight) tuples
    Returns run_id.
    """
    now = _now()
    all_fields = {**params, **results}
    cur = conn.execute(
        """INSERT INTO optimisation_runs (
            created_at, script, data_source,
            num_generations, total_population_size, mutation_rate,
            num_elites, migration_interval, migration_rate, num_islands,
            min_etfs, max_etfs, min_weight, max_weight,
            target_return, target_risk,
            use_forecasts, use_copulae, risk_parity,
            best_sharpe, portfolio_return, portfolio_volatility,
            num_selected, elapsed_seconds, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now,
            all_fields.get('script'),
            all_fields.get('data_source'),
            all_fields.get('num_generations'),
            all_fields.get('total_population_size'),
            all_fields.get('mutation_rate'),
            all_fields.get('num_elites'),
            all_fields.get('migration_interval'),
            all_fields.get('migration_rate'),
            all_fields.get('num_islands'),
            all_fields.get('min_etfs'),
            all_fields.get('max_etfs'),
            all_fields.get('min_weight'),
            all_fields.get('max_weight'),
            all_fields.get('target_return'),
            all_fields.get('target_risk'),
            int(all_fields.get('use_forecasts', 0)),
            int(all_fields.get('use_copulae', 0)),
            int(all_fields.get('risk_parity', 0)),
            all_fields.get('best_sharpe'),
            all_fields.get('portfolio_return'),
            all_fields.get('portfolio_volatility'),
            all_fields.get('num_selected'),
            all_fields.get('elapsed_seconds'),
            all_fields.get('notes'),
        ),
    )
    run_id = cur.lastrowid

    if holdings:
        conn.executemany(
            "INSERT INTO portfolio_holdings (run_id, ticker, weight) VALUES (?, ?, ?)",
            [(run_id, ticker, float(weight)) for ticker, weight in holdings],
        )

    conn.commit()
    return run_id


def get_recent_runs(conn, n=10, script=None):
    """Get the most recent optimisation runs."""
    query = "SELECT * FROM optimisation_runs"
    params = []
    if script is not None:
        query += " WHERE script = ?"
        params.append(script)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(n)
    return conn.execute(query, params).fetchall()


def get_run_holdings(conn, run_id):
    """Get portfolio holdings for a given run."""
    return conn.execute(
        "SELECT ticker, weight FROM portfolio_holdings WHERE run_id = ? ORDER BY weight DESC",
        (run_id,),
    ).fetchall()


# ─── Backtest ─────────────────────────────────────────────────────────────────

def save_backtest_session(conn, params):
    """
    Save a backtest session.

    params: dict with keys 'data_source', 'num_portfolios', 'num_children',
            'num_days_oos', 'use_forecast', 'elapsed_seconds', 'notes'
    Returns session_id.
    """
    now = _now()
    cur = conn.execute(
        """INSERT INTO backtest_sessions (
            created_at, data_source, num_portfolios, num_children,
            num_days_oos, use_forecast, elapsed_seconds, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now,
            params.get('data_source'),
            params['num_portfolios'],
            params['num_children'],
            params['num_days_oos'],
            int(params.get('use_forecast', 0)),
            params.get('elapsed_seconds'),
            params.get('notes'),
        ),
    )
    conn.commit()
    return cur.lastrowid


def save_backtest_result(conn, session_id, category, index, metrics, holdings=None):
    """
    Save a single portfolio result within a backtest session.

    metrics: dict with keys 'annualised_return', 'annualised_volatility',
             'sharpe_ratio', 'downside_deviation', 'max_drawdown',
             'calmar_ratio', 'sortino_ratio'
    holdings: optional list of (ticker, weight) tuples
    """
    holdings_json = None
    if holdings:
        holdings_json = json.dumps(
            [{'ticker': t, 'weight': float(w)} for t, w in holdings]
        )
    conn.execute(
        """INSERT INTO backtest_results (
            session_id, category, portfolio_index,
            annualised_return, annualised_volatility, sharpe_ratio,
            downside_deviation, max_drawdown, calmar_ratio, sortino_ratio,
            holdings_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id, category, index,
            metrics.get('annualised_return'),
            metrics.get('annualised_volatility'),
            metrics.get('sharpe_ratio'),
            metrics.get('downside_deviation'),
            metrics.get('max_drawdown'),
            metrics.get('calmar_ratio'),
            metrics.get('sortino_ratio'),
            holdings_json,
        ),
    )
    conn.commit()


def get_recent_backtests(conn, n=5):
    """Get the most recent backtest sessions."""
    return conn.execute(
        "SELECT * FROM backtest_sessions ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()


def get_backtest_results(conn, session_id):
    """Get all results for a backtest session."""
    return conn.execute(
        "SELECT * FROM backtest_results WHERE session_id = ? ORDER BY category, portfolio_index",
        (session_id,),
    ).fetchall()


# ─── Metadata ─────────────────────────────────────────────────────────────────

def save_data_source(conn, source, exchange_id=None, date_range_start=None,
                     date_range_end=None, num_tickers=None, num_rows=None,
                     notes=None):
    """Record a data download event. Returns data_source id."""
    now = _now()
    cur = conn.execute(
        """INSERT INTO data_sources (
            exchange_id, source, downloaded_at,
            date_range_start, date_range_end,
            num_tickers, num_rows, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (exchange_id, source, now, date_range_start, date_range_end,
         num_tickers, num_rows, notes),
    )
    conn.commit()
    return cur.lastrowid


def get_latest_data_source(conn, source=None):
    """Get the most recent data source entry."""
    if source:
        return conn.execute(
            "SELECT * FROM data_sources WHERE source = ? ORDER BY id DESC LIMIT 1",
            (source,),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM data_sources ORDER BY id DESC LIMIT 1"
    ).fetchone()


def get_latest_forecast(conn):
    """Get the most recent forecast run."""
    return conn.execute(
        "SELECT * FROM forecast_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()


# ─── CSV Migration ────────────────────────────────────────────────────────────

def migrate_csvs(conn, data_dir=None):
    """One-time import of existing CSV data into the database."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')

    # 1. Import price CSVs
    _migrate_price_csv(conn, data_dir, 'ETF_Prices.csv', 'US', 'etf')
    _migrate_price_csv(conn, data_dir, 'time_series_20251016_113257.csv', 'NZX', 'managed_fund')
    _migrate_price_csv(conn, data_dir, 'leveraged_ETF_Prices.csv', 'US', 'etf')
    _migrate_price_csv(conn, data_dir, '2x_leveraged_ETF_Prices.csv', 'US', 'etf')
    _migrate_price_csv(conn, data_dir, 'NZ_ETF_Prices.csv', 'NZX', 'etf')

    # 2. Import forecast CSVs
    _migrate_forecasts(conn, data_dir, 'expected_returns.csv', 'variances.csv', 'US')
    _migrate_forecasts(conn, data_dir, 'NZ_expected_returns.csv', 'NZ_variances.csv', 'US')

    # 3. Import ticker lists for completeness
    _migrate_ticker_list(conn, data_dir, 'ETFs_Full.csv', 'US', 'etf')
    _migrate_ticker_list(conn, data_dir, 'US_Stocks.csv', 'US', 'stock')
    _migrate_ticker_list(conn, data_dir, 'NZ_ETFs.csv', 'NZX', 'etf')
    _migrate_ticker_list(conn, data_dir, '2x_leveraged_ETFs.csv', 'US', 'etf')
    _migrate_ticker_list(conn, data_dir, '3x_leveraged_ETFs.csv', 'US', 'etf')

    # Print summary
    print("\nMigration summary:")
    for table in ['tickers', 'prices', 'forecast_runs', 'expected_returns',
                   'variances', 'data_sources']:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")


def _migrate_price_csv(conn, data_dir, filename, exchange, asset_type):
    """Import a single price CSV file."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"  Skipping {filename} (not found)")
        return

    print(f"Importing {filename}...")
    df = pd.read_csv(filepath, index_col=0)

    # Handle the InvestNow time series with datetime+timezone index
    if 'time_series' in filename:
        df.index = pd.to_datetime(df.index, utc=True).strftime('%Y-%m-%d')
    elif df.index.dtype == object:
        # Try parsing as dates
        try:
            parsed = pd.to_datetime(df.index)
            df.index = parsed.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            pass  # Keep as-is (integer indices)

    save_prices(conn, df, exchange=exchange, asset_type=asset_type,
                source='csv_migration')


def _migrate_forecasts(conn, data_dir, er_filename, var_filename, exchange):
    """Import paired expected returns and variances CSVs."""
    er_path = os.path.join(data_dir, er_filename)
    var_path = os.path.join(data_dir, var_filename)
    if not os.path.exists(er_path) or not os.path.exists(var_path):
        print(f"  Skipping {er_filename}/{var_filename} (not found)")
        return

    print(f"Importing {er_filename} + {var_filename}...")
    er = pd.read_csv(er_path, index_col=0)
    var = pd.read_csv(var_path, index_col=0)

    # These CSVs have a single column named '0'
    er_series = er['0'] if '0' in er.columns else er.iloc[:, 0]
    var_series = var['0'] if '0' in var.columns else var.iloc[:, 0]

    save_forecast_results(conn, er_series, var_series,
                          n_periods=252, exchange=exchange,
                          notes=f'Migrated from {er_filename} + {var_filename}')


def _migrate_ticker_list(conn, data_dir, filename, exchange, asset_type):
    """Import a ticker list CSV (single column of symbols)."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"  Skipping {filename} (not found)")
        return

    print(f"Importing ticker list {filename}...")
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    col = df.columns[0]  # Usually 'Tickers'
    symbols = df[col].dropna().tolist()
    exchange_id = _get_exchange_id(conn, exchange)
    _ensure_tickers(conn, symbols, exchange_id, asset_type)
    conn.commit()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'migrate':
        conn = get_connection()
        migrate_csvs(conn)
        conn.close()
    else:
        # Create empty database with schema
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        print(f"Database created at: {DB_PATH}")
        print(f"Tables ({len(tables)}):")
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t['name']}").fetchone()[0]
            print(f"  {t['name']}: {count} rows")
        conn.close()
