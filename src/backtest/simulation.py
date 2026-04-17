"""Portfolio simulation: weight generation, OOS evaluation, and metrics."""

import logging
from typing import List

import numpy as np

from src.config import (
    BACKTEST_MAX_WEIGHT_FLOOR,
    TRADING_DAYS_PER_YEAR,
    GA_MIN_SECURITIES,
    GA_MAX_SECURITIES,
)
from src.returns import calculate_log_returns, calculate_expected_returns
from src.covariance import calculate_covariance_matrix
from src.metrics import (
    maximum_drawdown,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
)

from .types import MethodResults, PortfolioResult

logger = logging.getLogger(__name__)

METRIC_NAMES = [
    'annualised_return', 'annualised_volatility', 'sharpe_ratio',
    'downside_deviation', 'max_drawdown', 'calmar_ratio', 'sortino_ratio',
]

# Module-level state set before spawning worker pools.
_backtest_data = None
_use_forecast = False
# Log returns (transposed: tickers x dates) and expected returns for weight optimisation.
_backtest_log_returns = None
_backtest_expected_returns = None


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


def optimal_weights(portfolio, use_copulae=False):
    """
    Finds the optimal weights (allocations) for the
    input portfolio.

    :portfolio: The input portfolio. List of ticker strings.
    :use_copulae: Whether to use copulae or not.
    :return: A list of weights for the input portfolio.
    """
    if len(portfolio) < 2:
        raise ValueError("Portfolio must contain at least 2 assets.")
    missing = set(portfolio) - set(_backtest_log_returns.index)
    if missing:
        raise KeyError(f"Tickers not found in data: {missing}")
    random_weights = get_random_weights(portfolio)
    subset = _backtest_log_returns.loc[portfolio, :].transpose()
    er = _backtest_expected_returns.loc[subset.columns].values
    max_weight = max(1 / (len(portfolio) - 1), BACKTEST_MAX_WEIGHT_FLOOR)

    if use_copulae:
        from src.covariance import estimate_corr_using_copulas
        corr = estimate_corr_using_copulas(subset)
        D = np.diag(subset.std().values * np.sqrt(TRADING_DAYS_PER_YEAR))
        cov = np.matmul(np.matmul(D, corr), D)
    else:
        cov = calculate_covariance_matrix(subset).values

    from src.weights import optimise_weights
    result = optimise_weights(
        expected_returns=er, cov_matrix=cov,
        max_weight=max_weight,
        initial_weights=get_random_weights(portfolio),
    )
    if not result.success:
        logger.warning("Weight optimization did not converge: %s", result.message)
    return result['x']


def run_portfolio(portfolio, weights, oos_log_returns):
    """
    Buy-and-hold simulation over the OOS period with natural weight drift.

    :param portfolio: list of ticker strings.
    :param weights: initial weight allocations (same order as portfolio).
    :param oos_log_returns: DataFrame with dates as rows, tickers as columns.
                            Only the OOS period -- no offset needed.
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


def fitness(portfolio_returns):
    """
    Calculates the portfolio Sharpe Ratio.

    :portfolio_returns: The input portfolio returns. List of floats.
    :return: The fitness of the portfolio.
    """
    portfolio_std = np.std(portfolio_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if portfolio_std == 0:
        return 0.0
    return (np.mean(portfolio_returns) * TRADING_DAYS_PER_YEAR) / portfolio_std


def create_random_portfolios(columns, num_portfolios, min_securities=None,
                             max_securities=None):
    """
    Create a list of random portfolios by selecting random subsets of tickers.

    :param columns: available ticker names (Index or list).
    :param num_portfolios: number of portfolios to generate.
    :param min_securities: minimum cardinality (default: GA_MIN_SECURITIES).
    :param max_securities: maximum cardinality (default: GA_MAX_SECURITIES).
    :return: list of lists of ticker strings.
    """
    if min_securities is None:
        min_securities = GA_MIN_SECURITIES
    if max_securities is None:
        max_securities = GA_MAX_SECURITIES
    num_tickers = len(columns)
    ticker_names = list(columns)
    portfolios = []
    for _ in range(num_portfolios):
        portfolios.append(list(_random_selection(
            num_tickers, min_securities, max_securities, ticker_names)))
    return portfolios


def evaluate_portfolios(portfolios, weights_list, oos_log_returns,
                        train_log_returns, category):
    """
    Evaluate a set of portfolios and return a MethodResults object.

    :param portfolios: list of portfolios (each a list of ticker strings).
    :param weights_list: list of weight arrays (same length as portfolios).
    :param oos_log_returns: OOS log returns DataFrame (dates x tickers).
    :param train_log_returns: training-period log returns for IS Sharpe.
    :param category: category name for the MethodResults.
    :return: MethodResults object.
    """
    prs = []
    for p, w in zip(portfolios, weights_list):
        metrics = get_statistics(p, w, oos_log_returns)
        try:
            is_stats = get_statistics(p, w, train_log_returns)
            is_sr = is_stats['sharpe_ratio']
        except Exception:
            is_sr = None
        prs.append(PortfolioResult(
            portfolio=p, weights=w, metrics=metrics, is_sharpe=is_sr,
        ))
    return MethodResults(category=category, portfolios=prs)


# ---- GA Worker Helpers -------------------------------------------------------


def _random_selection(num_tickers, min_k, max_k, ticker_names):
    """Generate a random portfolio by selecting min_k..max_k tickers."""
    k = np.random.randint(min_k, max_k + 1)
    chosen = np.random.choice(num_tickers, k, replace=False)
    return [ticker_names[i] for i in chosen]


def _init_worker(training_data, use_forecast):
    """Pool initializer -- sets module globals in each worker process."""
    global _backtest_data, _use_forecast
    _backtest_data = training_data
    _use_forecast = use_forecast


def _init_weight_worker(log_returns_T, expected_returns):
    """Pool initializer for weight computation workers."""
    global _backtest_log_returns, _backtest_expected_returns
    _backtest_log_returns = log_returns_T
    _backtest_expected_returns = expected_returns


def _compute_weights_for_portfolio(args):
    """Top-level function for Pool.map -- computes weights for a single portfolio.

    Must be top-level (not a lambda/closure) for macOS spawn-based multiprocessing.
    """
    portfolio, mode = args
    if mode == 'random':
        return get_random_weights(portfolio)
    elif mode == 'copulae':
        return optimal_weights(portfolio, use_copulae=True)
    else:  # 'optimal'
        return optimal_weights(portfolio, use_copulae=False)


def create_portfolio(num_children):
    """
    Creates a cardinality-constrained portfolio with the
    training data.

    :num_children: The number of children in the GA to create.
    :return: A list of tickers.
    """
    from src.config import (
        GA_NUM_GENERATIONS,
        GA_MIN_WEIGHT,
        GA_MAX_WEIGHT,
    )
    from src.optimisers.pygad_ga import PygadOptimiser
    opt = PygadOptimiser(
        num_children=num_children,
        num_generations=GA_NUM_GENERATIONS,
        min_securities=GA_MIN_SECURITIES,
        max_securities=GA_MAX_SECURITIES,
        min_weight=GA_MIN_WEIGHT,
        max_weight=GA_MAX_WEIGHT,
        target_return=None,
        use_forecasts=_use_forecast,
    )
    result = opt.optimise(_backtest_data)
    return result.selected_tickers
