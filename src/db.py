"""
SQLite database module for portfolio optimisation.

Stores prices, forecasts, optimisation runs, and backtest results.
Uses sqlite3 (stdlib) + pandas. All timestamps are ISO 8601 UTC.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

from src.config import DB_PATH

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
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

def _get_exchange_id(conn: sqlite3.Connection, code: str) -> int:
    """Look up exchange id by code. Raises ValueError if not found."""
    row = conn.execute(
        "SELECT id FROM exchanges WHERE code = ?", (code,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown exchange code: {code!r}")
    return row[0]


def _ensure_tickers(conn: sqlite3.Connection, symbols: list[str],
                    exchange_id: int, asset_type: str = 'etf',
                    names: Optional[dict[str, str]] = None,
                    countries: Optional[dict[str, str]] = None) -> dict[str, int]:
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


# ─── Data storage ─────────────────────────────────────────────────────────────

def save_prices(conn: sqlite3.Connection, prices_df: pd.DataFrame,
                exchange: str, asset_type: str = 'etf',
                source: Optional[str] = None,
                names: Optional[dict[str, str]] = None,
                countries: Optional[dict[str, str]] = None) -> int:
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
                                 names=names, countries=countries)

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


def load_prices(conn: sqlite3.Connection, exchange: Optional[str] = None,
                asset_type: Optional[str] = None, start: Optional[str] = None,
                end: Optional[str] = None, tickers: Optional[list[str]] = None,
                exclude_countries: Optional[list[str]] = None,
                exclude_flagged: bool = True,
                min_coverage: Optional[float] = 0.95,
                ffill_limit: Optional[int] = 5) -> pd.DataFrame:
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
    conditions: list[str] = []
    params: list[Any] = []

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


def set_ticker_excluded(conn: sqlite3.Connection, ticker_id: int,
                        reason: str) -> None:
    """Flag a ticker as excluded with a reason string."""
    conn.execute(
        "UPDATE tickers SET excluded = ?, updated_at = ? WHERE id = ?",
        (reason, _now(), ticker_id),
    )


def clear_ticker_excluded(conn: sqlite3.Connection, ticker_id: int) -> None:
    """Remove the exclusion flag from a ticker."""
    conn.execute(
        "UPDATE tickers SET excluded = NULL, updated_at = ? WHERE id = ?",
        (_now(), ticker_id),
    )


def get_excluded_tickers(conn: sqlite3.Connection,
                         exchange: Optional[str] = None) -> list[sqlite3.Row]:
    """Get all excluded tickers with their reasons."""
    query = "SELECT t.id, t.symbol, t.excluded FROM tickers t WHERE t.excluded IS NOT NULL"
    params: list[Any] = []
    if exchange is not None:
        exchange_id = _get_exchange_id(conn, exchange)
        query += " AND t.exchange_id = ?"
        params.append(exchange_id)
    query += " ORDER BY t.excluded, t.symbol"
    return conn.execute(query, params).fetchall()


def get_latest_prices_date(conn: sqlite3.Connection, exchange: Optional[str] = None,
                           asset_type: Optional[str] = None) -> Optional[str]:
    """Return the most recent date string in the prices table, or None.

    Useful for incremental downloads: start from the day after this date.
    """
    query = "SELECT MAX(p.date) FROM prices p"
    joins: list[str] = []
    conditions: list[str] = []
    params: list[Any] = []

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


def save_forecast_results(conn: sqlite3.Connection,
                          expected_returns_s: pd.Series,
                          variances_s: pd.Series,
                          n_periods: Optional[int] = None,
                          elapsed_seconds: Optional[float] = None,
                          notes: Optional[str] = None,
                          exchange: str = 'US') -> int:
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

    with conn:
        cur = conn.execute(
            "INSERT INTO forecast_runs (exchange_id, created_at, num_tickers, n_periods, "
            "elapsed_seconds, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (exchange_id, now, len(all_symbols), n_periods, elapsed_seconds, notes),
        )
        run_id: int = cur.lastrowid  # type: ignore[assignment]

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

    return run_id


def load_expected_returns(conn: sqlite3.Connection,
                          forecast_run_id: Optional[int] = None) -> pd.Series:
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


def load_variances(conn: sqlite3.Connection,
                    forecast_run_id: Optional[int] = None) -> pd.Series:
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

def save_optimisation_run(conn: sqlite3.Connection, params: dict[str, Any],
                          results: dict[str, Any],
                          holdings: list[tuple[str, float]],
                          exchange: str = 'US') -> int:
    """
    Save an optimisation run and its portfolio holdings.

    params: dict with keys like 'script', 'data_source', 'min_securities', etc.
        Algorithm-specific keys (e.g. 'num_generations', 'mutation_rate' for GA;
        'risk_aversion' for MIP; 'num_trials' for Monte Carlo) are stored in
        params_json automatically.
    results: dict with keys like 'best_sharpe', 'portfolio_return', etc.
    holdings: list of (ticker_symbol, weight) tuples — symbols are resolved to
        ticker_ids via the tickers table.
    exchange: exchange code for resolving ticker symbols (default 'US').
    Returns run_id.
    """
    now = _now()
    all_fields = {**params, **results}

    # Shared columns that have dedicated DB columns
    _shared_keys = {
        'script', 'data_source', 'data_source_id',
        'min_securities', 'max_securities', 'min_weight', 'max_weight',
        'target_return', 'target_risk',
        'use_forecasts', 'use_copulae', 'risk_parity',
        'best_sharpe', 'portfolio_return', 'portfolio_volatility',
        'num_selected', 'elapsed_seconds', 'notes',
    }
    # Everything else goes into params_json
    algo_params = {k: v for k, v in all_fields.items() if k not in _shared_keys}
    params_json = json.dumps(algo_params) if algo_params else None

    exchange_id = _get_exchange_id(conn, exchange)

    with conn:
        cur = conn.execute(
            """INSERT INTO optimisation_runs (
                created_at, script, data_source, data_source_id,
                min_securities, max_securities, min_weight, max_weight,
                target_return, target_risk,
                use_forecasts, use_copulae, risk_parity,
                params_json,
                best_sharpe, portfolio_return, portfolio_volatility,
                num_selected, elapsed_seconds, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                all_fields.get('script'),
                all_fields.get('data_source'),
                all_fields.get('data_source_id'),
                all_fields.get('min_securities'),
                all_fields.get('max_securities'),
                all_fields.get('min_weight'),
                all_fields.get('max_weight'),
                all_fields.get('target_return'),
                all_fields.get('target_risk'),
                int(all_fields.get('use_forecasts', 0)),
                int(all_fields.get('use_copulae', 0)),
                int(all_fields.get('risk_parity', 0)),
                params_json,
                all_fields.get('best_sharpe'),
                all_fields.get('portfolio_return'),
                all_fields.get('portfolio_volatility'),
                all_fields.get('num_selected'),
                all_fields.get('elapsed_seconds'),
                all_fields.get('notes'),
            ),
        )
        run_id: int = cur.lastrowid  # type: ignore[assignment]

        if holdings:
            symbols = [ticker for ticker, _ in holdings]
            ticker_map = _ensure_tickers(conn, symbols, exchange_id)
            conn.executemany(
                "INSERT INTO portfolio_holdings (run_id, ticker_id, weight) VALUES (?, ?, ?)",
                [(run_id, ticker_map[ticker], float(weight))
                 for ticker, weight in holdings],
            )

    return run_id


def get_recent_runs(conn: sqlite3.Connection, n: int = 10,
                    script: Optional[str] = None) -> list[sqlite3.Row]:
    """Get the most recent optimisation runs."""
    query = "SELECT * FROM optimisation_runs"
    params: list[Any] = []
    if script is not None:
        query += " WHERE script = ?"
        params.append(script)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(n)
    return conn.execute(query, params).fetchall()


def get_run_holdings(conn: sqlite3.Connection,
                     run_id: int) -> list[sqlite3.Row]:
    """Get portfolio holdings for a given run. Returns rows with ticker (symbol) and weight."""
    return conn.execute(
        "SELECT t.symbol AS ticker, ph.weight "
        "FROM portfolio_holdings ph "
        "JOIN tickers t ON ph.ticker_id = t.id "
        "WHERE ph.run_id = ? ORDER BY ph.weight DESC",
        (run_id,),
    ).fetchall()


# ─── Backtest ─────────────────────────────────────────────────────────────────

def save_backtest_session(conn: sqlite3.Connection,
                          params: dict[str, Any]) -> int:
    """
    Save a backtest session.

    params: dict with keys 'data_source', 'data_source_id', 'num_portfolios',
            'num_days_oos', 'use_forecast', 'optimiser_params' (dict),
            'elapsed_seconds', 'notes', and optional rolling-window fields:
            'window_train_start', 'window_train_end', 'window_test_start',
            'window_test_end', 'window_label', 'run_group_id'.
    Returns session_id.
    """
    now = _now()
    opt_params = params.get('optimiser_params')
    opt_params_json = json.dumps(opt_params) if opt_params else None
    with conn:
        cur = conn.execute(
            """INSERT INTO backtest_sessions (
                created_at, data_source, data_source_id, num_portfolios,
                num_days_oos, use_forecast, optimiser_params_json,
                elapsed_seconds, notes,
                window_train_start, window_train_end,
                window_test_start, window_test_end,
                window_label, run_group_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                params.get('data_source'),
                params.get('data_source_id'),
                params['num_portfolios'],
                params['num_days_oos'],
                int(params.get('use_forecast', 0)),
                opt_params_json,
                params.get('elapsed_seconds'),
                params.get('notes'),
                params.get('window_train_start'),
                params.get('window_train_end'),
                params.get('window_test_start'),
                params.get('window_test_end'),
                params.get('window_label'),
                params.get('run_group_id'),
            ),
        )
    return cur.lastrowid  # type: ignore[return-value]


def save_backtest_result(conn: sqlite3.Connection, session_id: int,
                         category: str, index: int,
                         metrics: dict[str, Any],
                         holdings: Optional[list[tuple[str, float]]] = None,
                         exchange: str = 'US') -> None:
    """
    Save a single portfolio result within a backtest session.

    metrics: dict with keys 'annualised_return', 'annualised_volatility',
             'sharpe_ratio', 'downside_deviation', 'max_drawdown',
             'calmar_ratio', 'sortino_ratio'
    holdings: optional list of (ticker_symbol, weight) tuples
    exchange: exchange code for resolving ticker symbols (default 'US').
    """
    holdings_json = None
    if holdings:
        holdings_json = json.dumps(
            [{'ticker': t, 'weight': float(w)} for t, w in holdings]
        )
    with conn:
        cur = conn.execute(
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
        result_id = cur.lastrowid

        # Save normalised holdings
        if holdings:
            exchange_id = _get_exchange_id(conn, exchange)
            symbols = [t for t, _ in holdings]
            ticker_map = _ensure_tickers(conn, symbols, exchange_id)
            conn.executemany(
                "INSERT INTO backtest_holdings (result_id, ticker_id, weight) "
                "VALUES (?, ?, ?)",
                [(result_id, ticker_map[t], float(w)) for t, w in holdings],
            )


def get_recent_backtests(conn: sqlite3.Connection,
                         n: int = 5) -> list[sqlite3.Row]:
    """Get the most recent backtest sessions."""
    return conn.execute(
        "SELECT * FROM backtest_sessions ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()


def get_backtest_results(conn: sqlite3.Connection,
                         session_id: int) -> list[sqlite3.Row]:
    """Get all results for a backtest session."""
    return conn.execute(
        "SELECT * FROM backtest_results WHERE session_id = ? ORDER BY category, portfolio_index",
        (session_id,),
    ).fetchall()


# ─── Metadata ─────────────────────────────────────────────────────────────────

def _save_data_source_no_commit(conn: sqlite3.Connection, source: str,
                                exchange_id: Optional[int] = None,
                                date_range_start: Optional[str] = None,
                                date_range_end: Optional[str] = None,
                                num_tickers: Optional[int] = None,
                                num_rows: Optional[int] = None,
                                notes: Optional[str] = None) -> int:
    """Insert a data_source row without committing (for use inside transactions)."""
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
    return cur.lastrowid  # type: ignore[return-value]


def save_data_source(conn: sqlite3.Connection, source: str,
                     exchange_id: Optional[int] = None,
                     date_range_start: Optional[str] = None,
                     date_range_end: Optional[str] = None,
                     num_tickers: Optional[int] = None,
                     num_rows: Optional[int] = None,
                     notes: Optional[str] = None) -> int:
    """Record a data download event. Returns data_source id."""
    with conn:
        return _save_data_source_no_commit(
            conn, source, exchange_id, date_range_start, date_range_end,
            num_tickers, num_rows, notes,
        )


def get_latest_data_source(conn: sqlite3.Connection,
                           source: Optional[str] = None) -> Optional[sqlite3.Row]:
    """Get the most recent data source entry."""
    if source:
        return conn.execute(
            "SELECT * FROM data_sources WHERE source = ? ORDER BY id DESC LIMIT 1",
            (source,),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM data_sources ORDER BY id DESC LIMIT 1"
    ).fetchone()


def get_latest_forecast(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    """Get the most recent forecast run."""
    return conn.execute(
        "SELECT * FROM forecast_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()


# ─── CSV Migration ────────────────────────────────────────────────────────────

def migrate_csvs(conn: sqlite3.Connection,
                  data_dir: Optional[str] = None) -> None:
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


def _migrate_price_csv(conn: sqlite3.Connection, data_dir: str,
                       filename: str, exchange: str,
                       asset_type: str) -> None:
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


def _migrate_forecasts(conn: sqlite3.Connection, data_dir: str,
                       er_filename: str, var_filename: str,
                       exchange: str) -> None:
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


def _migrate_ticker_list(conn: sqlite3.Connection, data_dir: str,
                         filename: str, exchange: str,
                         asset_type: str) -> None:
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


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from src.logging_config import setup_logging
    setup_logging()

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
        logger.info("Database created at: %s", DB_PATH)
        logger.info("Tables (%d):", len(tables))
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t['name']}").fetchone()[0]
            logger.info("  %s: %d rows", t['name'], count)
        conn.close()
