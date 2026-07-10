"""Return calculations: log returns, expected returns, variances."""

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR


def calculate_log_returns(prices):
    """Calculate log returns, replacing NaN and inf with 0.

    :param prices: DataFrame of prices.
    :return: DataFrame of log returns.
    :raises ValueError: if the index is not monotonically increasing.
    """
    if not prices.index.is_monotonic_increasing:
        raise ValueError(
            "prices index must be sorted in ascending order. "
            "Unsorted data causes .shift(1) to compute returns against wrong date pairs."
        )
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.fillna(0)
    log_returns = log_returns.replace([np.inf, -np.inf], 0)
    return log_returns


def calculate_expected_returns(log_returns, annualise=True):
    """Mean log returns per asset.

    :param log_returns: DataFrame of log returns.
    :param annualise: multiply by 252 trading days (default True).
    :return: Series of expected returns.
    """
    er = log_returns.mean()
    if annualise:
        er = er * TRADING_DAYS_PER_YEAR
    return er


def calculate_variances(log_returns, annualise=True):
    """Variance of log returns per asset.

    :param log_returns: DataFrame of log returns.
    :param annualise: multiply by 252 trading days (default True).
    :return: Series of variances.
    """
    var = log_returns.var()
    if annualise:
        var = var * TRADING_DAYS_PER_YEAR
    return var


def calculate_asset_betas(log_returns, benchmark_log_returns):
    """Per-asset OLS beta to a benchmark: beta_i = cov(r_i, r_b) / var(r_b).

    Pure computation — callers align the two inputs (pandas pairwise ``cov``
    handles residual NaN overlap). Shared by the production rebalance's beta
    floor and the backtest's beta-1 experiment arm.

    :param log_returns: DataFrame of daily log returns (dates x tickers).
    :param benchmark_log_returns: Series of benchmark daily log returns.
    :return: Series of betas aligned to log_returns.columns; NaN-safe
        (no overlap / zero benchmark variance -> beta 0).
    """
    var_b = benchmark_log_returns.var()
    betas = log_returns.apply(lambda col: col.cov(benchmark_log_returns)) / var_b
    return betas.fillna(0.0)


def prepare_portfolio_inputs(prices):
    """Compute standard portfolio inputs from a price DataFrame.

    :param prices: DataFrame with dates as index, tickers as columns.
    :return: (log_returns, expected_returns, cov_matrix) as numpy arrays.
    """
    from src.covariance import calculate_covariance_matrix
    log_returns = calculate_log_returns(prices)
    expected_returns = calculate_expected_returns(log_returns).values
    cov_matrix = calculate_covariance_matrix(log_returns).values
    return log_returns, expected_returns, cov_matrix
