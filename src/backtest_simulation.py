"""Portfolio simulation and OOS evaluation for backtesting."""

import logging

import numpy as np

from src.metrics import (
    maximum_drawdown,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
)
from src.config import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

METRIC_NAMES = [
    'annualised_return', 'annualised_volatility', 'sharpe_ratio',
    'downside_deviation', 'max_drawdown', 'calmar_ratio', 'sortino_ratio',
]


def get_random_weights(portfolio):
    """
    Creates a set of random weighting that sum to 1.

    :portfolio: Input portfolio to get the length.
    :return: A set of random weights equal in length to the portfolio.
    """
    if not portfolio:
        raise ValueError("Cannot generate weights for an empty portfolio.")
    random_weights = np.random.random(len(portfolio))
    random_weights /= np.sum(random_weights)
    return random_weights


def run_portfolio(portfolio, weights, oos_log_returns):
    """
    Buy-and-hold simulation over the OOS period with natural weight drift.

    :param portfolio: list of ticker strings.
    :param weights: initial weight allocations (same order as portfolio).
    :param oos_log_returns: DataFrame with dates as rows, tickers as columns.
                            Only the OOS period — no offset needed.
    :return: list of daily portfolio log returns.
    """
    subset = oos_log_returns[portfolio]
    portfolio_returns = []
    w = weights.copy()
    for i in range(len(subset)):
        step_returns = subset.iloc[i].values
        weighted_return = float(np.sum(step_returns * w))
        portfolio_returns.append(weighted_return)
        w = w * np.exp(step_returns) / (1 + weighted_return)
    return portfolio_returns


def get_statistics(portfolio, weights, oos_log_returns):
    """
    Compute all performance metrics for one portfolio on an OOS window.

    :param portfolio: list of ticker strings.
    :param weights: weight allocations.
    :param oos_log_returns: OOS log returns DataFrame (dates x tickers).
    :return: dict keyed by metric name.
    """
    portfolio_returns = run_portfolio(portfolio, weights, oos_log_returns)
    max_dd = maximum_drawdown(portfolio_returns)
    dd = downside_deviation(portfolio_returns)
    r = np.mean(portfolio_returns) * TRADING_DAYS_PER_YEAR
    std = np.std(portfolio_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = r / std if std != 0 else 0.0
    return dict(zip(METRIC_NAMES, [
        r, std, sharpe, dd, max_dd,
        calmar_ratio(r, max_dd),
        sortino_ratio(r, dd),
    ]))
