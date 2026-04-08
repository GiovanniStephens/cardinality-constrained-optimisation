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
