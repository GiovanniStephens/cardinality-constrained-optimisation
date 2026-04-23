"""DB persistence helper for optimisation results.

For portfolio utilities, import directly from the source submodule:
    src.returns, src.covariance, src.metrics, src.weights,
    src.data_loading, src.binary_io.
"""

import logging

import numpy as np

# Canonical definition now lives in src.optimisers.base; re-export for
# backward compatibility so ``from src.portfolio_utils import OptimisationResult``
# continues to work.
from src.optimisers.base import OptimisationResult  # noqa: F401

# ─── DB Persistence (thin bridge between submodules and db) ──────────────────

logger = logging.getLogger(__name__)


def save_optimisation_result(conn, selected_tickers, weights, prices,
                             script_name, params=None, exchange='US',
                             elapsed_seconds=None):
    """Compute portfolio metrics and persist an optimisation run to DB.

    Consolidates the common pattern: compute log returns for selected
    tickers → expected returns and covariance → portfolio return/vol →
    save to ``optimisation_runs`` and ``portfolio_holdings``.

    :param conn: sqlite3 connection.
    :param selected_tickers: list of ticker symbols in the portfolio.
    :param weights: array of portfolio weights (same order as tickers).
    :param prices: full price DataFrame (will be sliced to selected_tickers).
    :param script_name: identifier for the optimisation script.
    :param params: extra parameter dict merged into the DB params column.
    :param exchange: exchange code (default 'US').
    :param elapsed_seconds: optional wall-clock time for the run.
    :return: run_id (int) from the database.
    """
    from src import db
    from src.returns import calculate_log_returns, calculate_expected_returns
    from src.covariance import calculate_covariance_matrix
    from src.weights import calculate_portfolio_variance

    log_returns = calculate_log_returns(prices[selected_tickers])
    er = calculate_expected_returns(log_returns)
    cov = calculate_covariance_matrix(log_returns)
    w = np.asarray(weights)
    portfolio_return = float(np.dot(w, er))
    portfolio_vol = float(
        np.sqrt(calculate_portfolio_variance(w, cov)))
    sr = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0

    all_params = {'script': script_name}
    if params:
        all_params.update(params)

    results = {
        'best_sharpe': sr,
        'portfolio_return': portfolio_return,
        'portfolio_volatility': portfolio_vol,
        'num_selected': len(selected_tickers),
    }
    if elapsed_seconds is not None:
        results['elapsed_seconds'] = elapsed_seconds

    run_id = db.save_optimisation_run(
        conn, params=all_params, results=results,
        holdings=list(zip(selected_tickers, weights)),
        exchange=exchange,
    )
    return run_id
