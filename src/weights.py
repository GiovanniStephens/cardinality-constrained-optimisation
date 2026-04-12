"""Portfolio weight optimisation: SLSQP, risk-parity, and portfolio variance."""

import logging

import numpy as np

from src.metrics import sharpe_loss

logger = logging.getLogger(__name__)
from src.returns import calculate_log_returns, calculate_expected_returns
from src.covariance import calculate_covariance_matrix


def calculate_portfolio_variance(weights, cov_matrix):
    """Portfolio variance: w^T @ Cov @ w.

    Accepts numpy arrays or pandas DataFrames for *cov_matrix*.

    :param weights: array of portfolio weights.
    :param cov_matrix: covariance matrix (array or DataFrame).
    :return: portfolio variance as a float.
    """
    cov = cov_matrix.values if hasattr(cov_matrix, 'values') else np.asarray(cov_matrix)
    w = np.asarray(weights).flatten()
    return float(np.dot(w, np.dot(cov, w)))


def calculate_portfolio_return(weights, expected_returns):
    """Portfolio expected return: w^T @ er.

    :param weights: array of portfolio weights.
    :param expected_returns: array of expected returns.
    :return: portfolio return as a float.
    """
    return float(np.dot(np.asarray(weights), np.asarray(expected_returns)))


def _risk_parity_portfolio_var(w, V):
    """Portfolio variance via np.matrix (internal, used by risk-parity)."""
    w = np.matrix(w)
    return (w * V * w.T)[0, 0]


def calculate_risk_contribution(w, V):
    """Asset contribution to total risk for risk-parity objective.

    :param w: weight vector.
    :param V: covariance matrix (np.matrix).
    :return: column vector of risk contributions.
    """
    w = np.matrix(w)
    portfolio_var = _risk_parity_portfolio_var(w, V)
    if portfolio_var <= 0:
        return np.zeros_like(w.T)
    sigma = np.sqrt(portfolio_var)
    MRC = V * w.T
    RC = np.multiply(MRC, w.T) / sigma
    return RC


def risk_budget_objective(x, pars):
    """Risk-parity objective: minimise deviation from equal risk contribution.

    :param x: weight vector.
    :param pars: [covariance_matrix, risk_target_proportions].
    :return: sum of squared deviations from target risk contributions.
    """
    V = pars[0]     # covariance table
    x_t = pars[1]   # risk target in percent of portfolio risk
    sig_p = np.sqrt(_risk_parity_portfolio_var(x, V))
    risk_target = np.asmatrix(np.multiply(sig_p, x_t))
    asset_RC = calculate_risk_contribution(x, V)
    J = sum(np.square(asset_RC - risk_target.T))[0, 0]
    return J


def optimise_weights(selection_vector=None, data=None, min_weight=0.0,
                     max_weight=1.0, min_return=None, *,
                     expected_returns=None, cov_matrix=None,
                     target_return=None, target_risk=None,
                     initial_weights=None, risk_parity=False,
                     group_constraints=None, group_membership=None,
                     selected_tickers=None):
    """SLSQP weight optimisation for a portfolio.

    Can be called in two modes:

    1. **Selection-vector mode** (original API): pass ``selection_vector``
       and ``data`` (prices DataFrame).  Log returns, expected returns, and
       covariance are computed internally.

    2. **Pre-computed mode**: pass ``expected_returns`` and ``cov_matrix``
       directly (as numpy arrays).  Useful when the caller has already
       computed these (e.g. backtest weight workers, GA fitness closures).

    :param selection_vector: binary array (1 = selected, 0 = not).
        Required in mode 1, ignored in mode 2.
    :param data: DataFrame of prices (index=dates, columns=tickers).
        Required in mode 1, ignored in mode 2.
    :param min_weight: lower bound per position weight.
    :param max_weight: upper bound per position weight.
    :param min_return: if set, adds an inequality constraint for minimum
        annualised portfolio return.
    :param expected_returns: pre-computed expected returns (array).
        If provided, ``cov_matrix`` must also be provided.
    :param cov_matrix: pre-computed covariance matrix (array).
        If provided, ``expected_returns`` must also be provided.
    :param target_return: if set, adds an equality constraint for the
        annualised portfolio return.
    :param target_risk: if set, adds an equality constraint for the
        annualised portfolio standard deviation.
    :param initial_weights: starting point for the optimiser.  Defaults to
        equal weights.
    :param risk_parity: if True, minimise risk-budget deviation instead of
        negative Sharpe ratio.
    :param group_constraints: optional GROUP_CONSTRAINTS dict from config.
    :param group_membership: optional membership dict from load_membership().
    :param selected_tickers: optional list of ticker symbols (required when
        group_constraints is non-empty).
    :return: scipy.optimize.OptimizeResult with optimised weights in .x
    """
    from scipy.optimize import minimize

    # ── Resolve inputs ────────────────────────────────────────────────────
    if expected_returns is not None and cov_matrix is not None:
        # Pre-computed mode
        er = np.asarray(expected_returns)
        cov = np.asarray(cov_matrix)
        n = len(er)
    elif selection_vector is not None and data is not None:
        # Selection-vector mode (original API)
        selected = data.columns[selection_vector == 1]
        log_returns = calculate_log_returns(data[selected])
        er = calculate_expected_returns(log_returns).values
        cov = calculate_covariance_matrix(log_returns).values
        n = len(selected)
    else:
        raise ValueError(
            "Either (selection_vector, data) or "
            "(expected_returns, cov_matrix) must be provided."
        )

    if initial_weights is not None:
        x0 = np.asarray(initial_weights, dtype=float)
    else:
        x0 = np.ones(n) / n

    # ── Constraints ───────────────────────────────────────────────────────
    bounds = [(min_weight, max_weight) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    if min_return is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, _er=er: np.dot(_er, x) - min_return,
        })
    if target_return is not None and target_risk is None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x, _er=er: np.sum(_er * x) - target_return,
        })
    if target_risk is not None and target_return is None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x, _cov=cov: target_risk -
            np.sqrt(np.dot(x.T, np.dot(_cov, x))),
        })

    # ── Group allocation constraints ──────────────────────────────────────
    if group_constraints and group_membership and selected_tickers:
        from src.group_constraints import build_slsqp_constraints
        constraints.extend(
            build_slsqp_constraints(selected_tickers, group_membership,
                                    group_constraints)
        )

    # ── Objective ─────────────────────────────────────────────────────────
    if risk_parity:
        risk_proportion = [1 / n] * n
        return minimize(risk_budget_objective, x0,
                        args=([np.matrix(cov), risk_proportion]),
                        method='SLSQP', bounds=bounds,
                        constraints=constraints)

    def objective(x):
        return sharpe_loss(x, er, cov)

    result = minimize(objective, x0=x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    if not result.success:
        logger.warning("SLSQP did not converge: %s. Falling back to equal weights.",
                        result.message)
        result.x = np.ones(n) / n
        result.fun = sharpe_loss(result.x, er, cov)
    return result
