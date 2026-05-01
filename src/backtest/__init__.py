"""
Forward-walk backtesting package for portfolio optimisation.

Evaluates portfolios across rolling train/test windows using GA, Monte Carlo,
and random selection. Computes OOS performance metrics and runs statistical
tests to compare methods.
"""

from src.backtest.types import WindowSpec, PortfolioResult, MethodResults, WindowResult
from src.backtest.windows import generate_windows, slice_window_data
from src.backtest.simulation import (
    METRIC_NAMES,
    get_random_weights,
    optimal_weights,
    benchmark_portfolio,
    run_portfolio,
    get_statistics,
    fitness,
    create_random_portfolios,
    evaluate_portfolios,
    create_portfolio,
    _backtest_data,
    _backtest_log_returns,
    _backtest_expected_returns,
    _init_worker,
    _init_weight_worker,
    _compute_weights_for_portfolio,
    _random_selection,
)
from src.backtest.statistics import (
    difference_of_means_hypothesis_test,
    paired_t_test,
    friedman_test,
    aggregate_cross_window,
)
from src.backtest.runner import evaluate_window, main

__all__ = [
    # types
    'WindowSpec', 'PortfolioResult', 'MethodResults', 'WindowResult',
    # windows
    'generate_windows', 'slice_window_data',
    # simulation
    'METRIC_NAMES', 'get_random_weights', 'optimal_weights',
    'benchmark_portfolio', 'run_portfolio', 'get_statistics', 'fitness',
    'create_random_portfolios', 'evaluate_portfolios', 'create_portfolio',
    # statistics
    'difference_of_means_hypothesis_test', 'paired_t_test',
    'friedman_test', 'aggregate_cross_window',
    # runner
    'evaluate_window', 'main',
]
