"""Portfolio performance metrics: Sharpe, Sortino, Calmar, drawdown."""

import logging

import numpy as np
from scipy.stats import norm

from src.config import (
    NUMERICAL_TOLERANCE,
    SHARPE_WARN_THRESHOLD,
    SHARPE_CRITICAL_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Fitness value returned when cardinality or return constraints are violated.
CONSTRAINT_VIOLATION_FITNESS = -1e4


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


def sharpe_loss(weights, expected_returns, cov_matrix):
    """Negated Sharpe ratio for use as a minimisation objective (e.g. SLSQP).

    :param weights: array of portfolio weights.
    :param expected_returns: array/Series of expected returns.
    :param cov_matrix: covariance matrix (array or DataFrame).
    :return: negative Sharpe ratio as a float.
    """
    return -sharpe_ratio(weights, expected_returns, cov_matrix)




def equal_weight_fitness(selection_mask, expected_returns, cov_matrix,
                         min_count, max_count, min_return=None):
    """Equal-weight Sharpe ratio for a binary selection vector with cardinality constraints.

    :param selection_mask: binary array (1 = selected).
    :param expected_returns: array of annualised expected returns per asset.
    :param cov_matrix: annualised covariance matrix (array).
    :param min_count: minimum number of selected assets.
    :param max_count: maximum number of selected assets.
    :param min_return: optional minimum portfolio return threshold.
    :return: Sharpe ratio (float), or CONSTRAINT_VIOLATION_FITNESS if
        cardinality or minimum-return constraints are violated, or 0.0 if
        the portfolio is empty or has zero variance.
    """
    selected = selection_mask == 1
    n = np.sum(selected)
    if n < min_count or n > max_count:
        return CONSTRAINT_VIOLATION_FITNESS
    if not np.any(selected):
        return 0.0
    filtered_returns = expected_returns[selected]
    filtered_cov = cov_matrix[np.ix_(selected, selected)]
    weights = np.ones(n) / n
    port_return = np.dot(weights, filtered_returns)
    if min_return is not None and port_return < min_return:
        return CONSTRAINT_VIOLATION_FITNESS
    port_variance = np.dot(weights, np.dot(filtered_cov, weights))
    return port_return / np.sqrt(port_variance) if port_variance > 0 else 0.0


def maximum_drawdown(portfolio_returns):
    """Maximum peak-to-trough decline from cumulative log returns.

    Computes the largest percentage drop from a running peak of the
    compounded equity curve: MDD = min_t (V_t / max_{s<=t} V_s - 1),
    where V_t = exp(sum of log returns up to t).

    :param portfolio_returns: list of per-period log returns.
    :return: maximum drawdown as a negative float (e.g. -0.25 = 25% drawdown).
        Returns 0 if the equity curve never declines.
    """
    if not portfolio_returns:
        raise ValueError("portfolio_returns cannot be empty")
    cummax = []
    cum_return = []
    drawdowns = []

    cummax.append(np.exp(portfolio_returns[0]))
    cum_return.append(np.exp(portfolio_returns[0]))
    drawdowns.append(0)
    for i in range(1, len(portfolio_returns)):
        cummax.append(max(np.exp(portfolio_returns[i]) * cum_return[i-1], cum_return[i-1]))
        cum_return.append(np.exp(portfolio_returns[i]) * cum_return[i-1])
        drawdowns.append(cum_return[i] / cummax[i] - 1)
    return min(drawdowns)


def downside_deviation(portfolio_returns, mar=0):
    """Semi-deviation of returns below a minimum acceptable return (MAR).

    DD = sqrt( (1/N) * sum_{r_i < MAR} (r_i - MAR)^2 )

    Only periods where the return falls below *mar* contribute to the
    sum; all N periods are used in the denominator (full-length
    normalisation, consistent with Sortino & Price 1994).

    :param portfolio_returns: list of per-period log returns.
    :param mar: minimum acceptable return threshold (default 0).
    :return: downside deviation as a non-negative float. Returns 0.0 if
        *portfolio_returns* is empty.
    """
    if not portfolio_returns:
        return 0.0
    squared_dev = 0
    for i in portfolio_returns:
        if i < mar:
            squared_dev += (i - mar)**2
    return (squared_dev / len(portfolio_returns))**0.5


def sortino_ratio(r, downside_deviation, MAR=0):
    """Sortino ratio: excess return per unit of downside risk.

    Sortino = (R - MAR) / DD

    Unlike the Sharpe ratio, only downside volatility is penalised,
    making this more appropriate when return distributions are skewed.

    :param r: annualised portfolio return.
    :param downside_deviation: downside deviation (see :func:`downside_deviation`).
    :param MAR: minimum acceptable return threshold (default 0).
    :return: Sortino ratio as a float. Returns 0.0 if downside_deviation is 0.
    """
    if downside_deviation == 0:
        return 0.0
    return (r - MAR) / downside_deviation


def calmar_ratio(r, downside_drawdown):
    """Calmar ratio: annualised return divided by maximum drawdown.

    Calmar = R / |MDD|

    Measures return per unit of tail risk. Higher values indicate better
    risk-adjusted performance relative to the worst historical decline.

    :param r: annualised portfolio return.
    :param downside_drawdown: maximum drawdown (negative float from
        :func:`maximum_drawdown`). The absolute value is used.
    :return: Calmar ratio as a non-negative float. Returns 0.0 if
        drawdown is 0.
    """
    if downside_drawdown == 0:
        return 0.0
    return r / abs(downside_drawdown)


def sharpe_ratio_variance(sr, n, skewness=0.0, excess_kurtosis=0.0):
    """Variance of the Sharpe ratio estimator (Lo 2002, Bailey & López de Prado 2014).

    Accounts for non-normality of returns via skewness and excess kurtosis.
    For normal returns (skewness=0, excess_kurtosis=0), simplifies to
    (1 + sr^2/2) / n.

    :param sr: observed Sharpe ratio.
    :param n: number of return observations.
    :param skewness: sample skewness of returns (default 0).
    :param excess_kurtosis: sample excess kurtosis of returns (default 0).
    :return: variance of the SR estimator.
    """
    return (1 - skewness * sr + (excess_kurtosis / 4) * sr ** 2) / n


def deflated_sharpe_ratio(observed_sr, n, num_trials, skewness=0.0,
                          excess_kurtosis=0.0, sr_benchmark=0.0):
    """Probability that observed Sharpe is genuine after multiple testing correction.

    Implements the Deflated Sharpe Ratio of Bailey & López de Prado (2014).
    Returns P(SR > E[max SR under null]) — values near 1.0 indicate the
    observed Sharpe likely reflects genuine skill, values near 0 indicate
    it is likely due to overfitting.

    :param observed_sr: the best observed Sharpe ratio.
    :param n: number of return observations used to compute the SR.
    :param num_trials: number of independent strategy variations tested
        (for GA: ~population_size * generations).
    :param skewness: sample skewness of returns.
    :param excess_kurtosis: sample excess kurtosis of returns.
    :param sr_benchmark: Sharpe ratio of the null hypothesis (default 0).
    :return: DSR probability in [0, 1].
    """
    sr_var = sharpe_ratio_variance(observed_sr, n, skewness, excess_kurtosis)
    sr_std = np.sqrt(max(sr_var, NUMERICAL_TOLERANCE))

    # Expected maximum SR under the null (Euler-Mascheroni approximation)
    euler_gamma = 0.5772156649
    if num_trials <= 1:
        expected_max_sr = sr_benchmark
    else:
        z = norm.ppf(1 - 1 / num_trials)
        expected_max_sr = sr_std * (
            (1 - euler_gamma) * z
            + euler_gamma * norm.ppf(1 - 1 / (num_trials * np.e))
        ) + sr_benchmark

    # DSR = P(observed SR > expected max SR under null)
    return float(norm.cdf((observed_sr - expected_max_sr) / sr_std))


def warn_if_sharpe_suspicious(sr, context, logger=None):
    """Log warnings if a Sharpe ratio is suspiciously high (likely overfit).

    :param sr: observed Sharpe ratio.
    :param context: descriptive label for log messages (e.g. "GA in-sample").
    :param logger: logger instance (defaults to module logger).
    """
    log = logger or logging.getLogger(__name__)
    if sr > SHARPE_CRITICAL_THRESHOLD:
        log.warning(
            "%s: Sharpe=%.2f exceeds %.1f — almost certainly overfit on annual "
            "data (Harvey et al. 2016). Apply heavy OOS discount. "
            "See CLAUDE.md 'Sharpe Ratio Overfitting'.",
            context, sr, SHARPE_CRITICAL_THRESHOLD,
        )
    elif sr > SHARPE_WARN_THRESHOLD:
        log.warning(
            "%s: Sharpe=%.2f exceeds %.1f — likely inflated by in-sample "
            "optimisation bias. Expect 30-50%% OOS degradation. "
            "See CLAUDE.md 'Sharpe Ratio Overfitting'.",
            context, sr, SHARPE_WARN_THRESHOLD,
        )
