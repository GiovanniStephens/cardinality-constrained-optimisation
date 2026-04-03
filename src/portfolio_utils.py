"""Shared portfolio utility functions used across all optimisation methods."""

import numpy as np
import pandas as pd


def load_prices_csv(filename, min_coverage=0.95, last_n_days=None):
    """Load price data from CSV with coverage filtering and forward-fill.

    :param filename: path to CSV file (index_col=0 assumed).
    :param min_coverage: minimum fraction of non-null rows to keep a column.
    :param last_n_days: if set, keep only the most recent N calendar days.
    :return: cleaned DataFrame with dates as index, tickers as columns.
    """
    prices_df = pd.read_csv(filename, index_col=0)
    if last_n_days is not None:
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index()
        cutoff = prices_df.index[-1] - pd.Timedelta(days=last_n_days)
        prices_df = prices_df[prices_df.index >= cutoff]
    thresh = int(min_coverage * len(prices_df))
    prices_df = prices_df.loc[:, prices_df.notna().sum() >= thresh]
    prices_df = prices_df.ffill()
    return prices_df


def calculate_log_returns(prices):
    """Calculate log returns, replacing NaN and inf with 0.

    :param prices: DataFrame of prices.
    :return: DataFrame of log returns.
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.fillna(0)
    log_returns = log_returns.replace([np.inf, -np.inf], 0)
    return log_returns


def calculate_covariance_matrix(log_returns, annualise=True):
    """Sample covariance matrix of log returns.

    :param log_returns: DataFrame of log returns.
    :param annualise: multiply by 252 trading days (default True).
    :return: DataFrame covariance matrix.
    """
    cov = log_returns.cov()
    if annualise:
        cov = cov * 252
    return cov


def calculate_expected_returns(log_returns, annualise=True):
    """Mean log returns per asset.

    :param log_returns: DataFrame of log returns.
    :param annualise: multiply by 252 trading days (default True).
    :return: Series of expected returns.
    """
    er = log_returns.mean()
    if annualise:
        er = er * 252
    return er


def calculate_variances(log_returns, annualise=True):
    """Variance of log returns per asset.

    :param log_returns: DataFrame of log returns.
    :param annualise: multiply by 252 trading days (default True).
    :return: Series of variances.
    """
    var = log_returns.var()
    if annualise:
        var = var * 252
    return var


def sharpe_ratio(weights, expected_returns, cov_matrix):
    """Portfolio Sharpe ratio (positive).

    :param weights: array of portfolio weights.
    :param expected_returns: array/Series of expected returns.
    :param cov_matrix: covariance matrix (array or DataFrame).
    :return: Sharpe ratio as a float.
    """
    if len(weights) != len(expected_returns) or len(weights) != cov_matrix.shape[0]:
        raise ValueError(
            "weights, expected_returns, and cov_matrix dimensions must match"
        )
    p_return = np.sum(weights * expected_returns)
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if p_volatility == 0:
        return 0.0
    return p_return / p_volatility


def negative_sharpe_ratio(weights, expected_returns, cov_matrix):
    """Negated Sharpe ratio for use as a minimisation objective (e.g. SLSQP).

    :param weights: array of portfolio weights.
    :param expected_returns: array/Series of expected returns.
    :param cov_matrix: covariance matrix (array or DataFrame).
    :return: negative Sharpe ratio as a float.
    """
    return -sharpe_ratio(weights, expected_returns, cov_matrix)


def optimise_weights(selection_vector, data, min_weight=0.0, max_weight=1.0,
                     min_return=None):
    """SLSQP weight optimisation for a selected subset of securities.

    :param selection_vector: binary array (1 = selected, 0 = not).
    :param data: DataFrame of prices (index=dates, columns=tickers).
    :param min_weight: lower bound per position weight.
    :param max_weight: upper bound per position weight.
    :param min_return: if set, adds an inequality constraint for minimum
        annualised portfolio return.
    :return: scipy.optimize.OptimizeResult with optimised weights in .x
    """
    from scipy.optimize import minimize

    selected = data.columns[selection_vector == 1]
    log_returns = calculate_log_returns(data[selected])
    expected_returns = calculate_expected_returns(log_returns)
    cov_matrix = calculate_covariance_matrix(log_returns)
    n = len(selected)

    bounds = [(min_weight, max_weight) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if min_return is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: np.dot(expected_returns, x) - min_return,
        })

    def objective(x):
        return negative_sharpe_ratio(x, expected_returns, cov_matrix)

    return minimize(objective, x0=np.ones(n) / n, method='SLSQP',
                    bounds=bounds, constraints=constraints)
