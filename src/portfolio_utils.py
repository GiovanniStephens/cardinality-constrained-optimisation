"""OptimisationResult dataclass and DB persistence helper.

For portfolio utilities, import directly from the source submodule:
    src.returns, src.covariance, src.metrics, src.weights,
    src.data_loading, src.binary_io.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.config import NUMERICAL_TOLERANCE

# ─── Common interface ────────────────────────────────────────────────────────


@dataclass
class OptimisationResult:
    """Standard output from any optimiser.

    WARNING: sharpe_ratio is an in-sample value computed on the training data.
    It is biased upward by selection bias — typical IS-to-OOS degradation is
    30-50%. See CLAUDE.md "Sharpe Ratio Overfitting" for details.
    """
    selected_tickers: List[str]
    weights: np.ndarray
    sharpe_ratio: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate result integrity (skip for empty-result sentinels)."""
        if len(self.weights) == 0:
            return
        if len(self.weights) != len(self.selected_tickers):
            raise ValueError(
                f"weights length ({len(self.weights)}) != "
                f"selected_tickers length ({len(self.selected_tickers)})"
            )
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contain NaN or inf values")
        if np.any(self.weights < -NUMERICAL_TOLERANCE):
            raise ValueError("weights contain negative values")
        if not np.isfinite(self.sharpe_ratio):
            raise ValueError(f"sharpe_ratio is not finite: {self.sharpe_ratio}")


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
