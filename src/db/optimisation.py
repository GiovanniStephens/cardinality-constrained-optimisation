"""Optimisation run storage and retrieval."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Optional

from src.db.connection import _get_exchange_id, _now
from src.db.tickers import _ensure_tickers

logger = logging.getLogger(__name__)


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
        run_id = cur.lastrowid

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
    params = []
    if script is not None:
        query += " WHERE script = ?"
        params.append(script)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(n)
    return conn.execute(query, params).fetchall()


def get_run_holdings(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """Get portfolio holdings for a given run. Returns rows with ticker (symbol) and weight."""
    return conn.execute(
        "SELECT t.symbol AS ticker, ph.weight "
        "FROM portfolio_holdings ph "
        "JOIN tickers t ON ph.ticker_id = t.id "
        "WHERE ph.run_id = ? ORDER BY ph.weight DESC",
        (run_id,),
    ).fetchall()
