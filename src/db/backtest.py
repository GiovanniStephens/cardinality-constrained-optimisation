"""Backtest session and result storage."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from typing import Any, Optional

from src.db.connection import _get_exchange_id, _now
from src.db.tickers import _ensure_tickers

logger = logging.getLogger(__name__)


def save_backtest_session(conn: sqlite3.Connection, params: dict[str, Any]) -> int:
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
    return cur.lastrowid


def save_backtest_result(conn: sqlite3.Connection, session_id: int,
                         category: str, index: int,
                         metrics: dict[str, Any],
                         holdings: Optional[list[tuple[str, float]]] = None,
                         exchange: str = 'US') -> None:
    """
    Save a single portfolio result within a backtest session.

    metrics: dict with keys 'annualised_return', 'annualised_volatility',
             'sharpe_ratio', 'downside_deviation', 'max_drawdown',
             'calmar_ratio', 'sortino_ratio', 'information_ratio'
    holdings: optional list of (ticker_symbol, weight) tuples
    exchange: exchange code for resolving ticker symbols (default 'US').
    """
    holdings_json = None
    if holdings:
        holdings_json = json.dumps(
            [{'ticker': t, 'weight': float(w)} for t, w in holdings]
        )
    # NaN means "benchmark unavailable for this window" — store as NULL so
    # SQL aggregates skip it (sqlite would coerce NaN to NULL anyway; be
    # explicit).
    ir = metrics.get('information_ratio')
    if ir is not None and math.isnan(ir):
        ir = None
    with conn:
        cur = conn.execute(
            """INSERT INTO backtest_results (
                session_id, category, portfolio_index,
                annualised_return, annualised_volatility, sharpe_ratio,
                downside_deviation, max_drawdown, calmar_ratio, sortino_ratio,
                information_ratio, holdings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, category, index,
                metrics.get('annualised_return'),
                metrics.get('annualised_volatility'),
                metrics.get('sharpe_ratio'),
                metrics.get('downside_deviation'),
                metrics.get('max_drawdown'),
                metrics.get('calmar_ratio'),
                metrics.get('sortino_ratio'),
                ir,
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


def get_recent_backtests(conn: sqlite3.Connection, n: int = 5) -> list[sqlite3.Row]:
    """Get the most recent backtest sessions."""
    return conn.execute(
        "SELECT * FROM backtest_sessions ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()


def get_backtest_results(conn: sqlite3.Connection, session_id: int) -> list[sqlite3.Row]:
    """Get all results for a backtest session."""
    return conn.execute(
        "SELECT * FROM backtest_results WHERE session_id = ? ORDER BY category, portfolio_index",
        (session_id,),
    ).fetchall()
