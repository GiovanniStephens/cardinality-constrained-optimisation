"""Forecast data storage and retrieval."""

import logging

import pandas as pd

from src.db.connection import _get_exchange_id, _now
from src.db.tickers import _ensure_tickers

logger = logging.getLogger(__name__)


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

    with conn:
        cur = conn.execute(
            "INSERT INTO forecast_runs (exchange_id, created_at, num_tickers, n_periods, "
            "elapsed_seconds, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (exchange_id, now, len(all_symbols), n_periods, elapsed_seconds, notes),
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
