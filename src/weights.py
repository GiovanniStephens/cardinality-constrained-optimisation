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
    """Portfolio variance w^T V w (internal, used by risk-parity)."""
    w = np.asarray(w).ravel()
    V = np.asarray(V)
    return float(w @ V @ w)


def calculate_risk_contribution(w, V):
    """Asset contribution to total risk for risk-parity objective.

    :param w: weight vector.
    :param V: covariance matrix (any array-like, np.matrix accepted).
    :return: (n, 1) column vector of risk contributions.
    """
    w = np.asarray(w).ravel()
    V = np.asarray(V)
    portfolio_var = _risk_parity_portfolio_var(w, V)
    if portfolio_var <= 0:
        return np.zeros((w.size, 1))
    sigma = np.sqrt(portfolio_var)
    MRC = V @ w                       # marginal risk contributions, shape (n,)
    RC = (MRC * w) / sigma
    return RC.reshape(-1, 1)


def risk_budget_objective(x, pars):
    """Risk-parity objective: minimise deviation from equal risk contribution.

    :param x: weight vector.
    :param pars: [covariance_matrix, risk_target_proportions].
    :return: sum of squared deviations from target risk contributions.
    """
    V = pars[0]     # covariance table
    x_t = pars[1]   # risk target in percent of portfolio risk
    sig_p = np.sqrt(_risk_parity_portfolio_var(x, V))
    risk_target = sig_p * np.asarray(x_t, dtype=float).ravel()
    asset_RC = calculate_risk_contribution(x, V).ravel()
    return float(np.sum(np.square(asset_RC - risk_target)))


def optimise_weights(selection_vector=None, data=None, min_weight=0.0,
                     max_weight=1.0, min_return=None, *,
                     expected_returns=None, cov_matrix=None,
                     target_return=None, target_risk=None,
                     initial_weights=None, risk_parity=False,
                     minimize_variance=False,
                     group_constraints=None, group_membership=None,
                     selected_tickers=None,
                     asset_betas=None, min_beta=None, target_beta=None):
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
    :param minimize_variance: if True, minimise portfolio variance (wᵀΣw)
        instead of negative Sharpe ratio. Expected returns are not used.
        Mutually exclusive with risk_parity.
    :param group_constraints: optional GROUP_CONSTRAINTS dict from config.
    :param group_membership: optional membership dict from load_membership().
    :param selected_tickers: optional list of ticker symbols (required when
        group_constraints is non-empty).
    :param asset_betas: per-asset betas to a benchmark (array aligned with the
        selection). Portfolio beta is linear in weights, so together with
        ``min_beta`` this adds the inequality ``sum(w * beta) >= min_beta`` —
        a market-participation floor the Sharpe objective cannot satisfy with
        low-covariance look-alikes (unlike name-based category minimums).
    :param min_beta: if set (with ``asset_betas``), minimum portfolio beta.
    :param target_beta: if set (with ``asset_betas``), pins the portfolio beta
        exactly: equality constraint ``sum(w * beta) == target_beta``. Since
        beta is linear in weights this is just a second linear equality
        alongside the budget constraint. Callers should verify the target is
        reachable (see :func:`reachable_beta_interval`) and start from a
        feasible point on the hyperplane, or SLSQP may fail into the
        equal-weight fallback.
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

    # ── Adaptive lower-bound floor ────────────────────────────────────────
    # If the per-position floors consume (nearly) the whole budget, the
    # feasible region collapses: with n positions each >= min_weight and
    # sum(w) = 1, an n*min_weight of 1.0 forces every weight to exactly 1/n,
    # so SLSQP cannot move and every objective returns the same 1/N portfolio
    # (and n*min_weight > 1 is infeasible outright). Relax the floor so a
    # meaningful share of the budget stays free to optimise.
    FLOOR_BUDGET = 0.70
    if min_weight > 0 and n * min_weight > FLOOR_BUDGET:
        relaxed = FLOOR_BUDGET / n
        logger.debug("Relaxing min_weight %.3f -> %.3f for n=%d (floor budget "
                     "%.2f) to preserve optimisation freedom.",
                     min_weight, relaxed, n, FLOOR_BUDGET)
        min_weight = relaxed

    # ── Constraints ───────────────────────────────────────────────────────
    bounds = [(min_weight, max_weight) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    if min_return is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, _er=er: np.dot(_er, x) - min_return,
        })
    if min_beta is not None and asset_betas is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, _b=np.asarray(asset_betas, dtype=float):
                np.dot(_b, x) - min_beta,
        })
    if target_beta is not None and asset_betas is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x, _b=np.asarray(asset_betas, dtype=float):
                np.dot(_b, x) - target_beta,
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
        from src.group_constraints import (build_slsqp_constraints,
                                           check_floor_feasibility)
        # Uses the post-relaxation (effective) min_weight: if a capped group
        # holds k members, k * floor > cap means no feasible point exists and
        # SLSQP will fail into the equal-weight fallback — name the culprit
        # instead of failing silently (July 2026 Unknown-cap incident).
        for msg in check_floor_feasibility(selected_tickers, group_membership,
                                           group_constraints, min_weight):
            logger.error("Group cap infeasible vs weight floor — SLSQP cannot "
                         "converge: %s", msg)
        constraints.extend(
            build_slsqp_constraints(selected_tickers, group_membership,
                                    group_constraints)
        )

    # ── Objective ─────────────────────────────────────────────────────────
    if risk_parity and minimize_variance:
        raise ValueError("risk_parity and minimize_variance are mutually exclusive.")

    if risk_parity:
        risk_proportion = [1 / n] * n
        return minimize(risk_budget_objective, x0,
                        args=([np.asarray(cov), risk_proportion]),
                        method='SLSQP', bounds=bounds,
                        constraints=constraints)

    if minimize_variance:
        def objective(x, _cov=cov):
            return float(x @ _cov @ x)
    else:
        def objective(x):
            return sharpe_loss(x, er, cov)

    result = minimize(objective, x0=x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    if not result.success:
        logger.warning("SLSQP did not converge: %s. Falling back to equal weights.",
                        result.message)
        result.x = np.ones(n) / n
        result.fun = objective(result.x)
    return result


def greedy_beta_vertex(asset_betas, max_weight, min_weight=0.0, maximise=True):
    """Extreme point of portfolio beta over the box + simplex.

    Portfolio beta ``b·w`` is linear in the weights, so its maximum (minimum)
    over ``{w : min_weight <= w_i <= max_weight, sum(w) = 1}`` sits at an LP
    vertex: give every asset the floor, then pour the remaining budget into
    assets in descending (ascending) beta order up to the cap.

    :param asset_betas: per-asset betas (array-like, aligned with the weights).
    :param max_weight: per-asset weight cap.
    :param min_weight: per-asset weight floor (0 on the backtest path).
    :param maximise: True for the highest reachable beta, False for the lowest.
    :return: tuple ``(portfolio_beta, weights)`` at the vertex.
    """
    b = np.asarray(asset_betas, dtype=float)
    n = b.size
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible bounds: {n} assets * max_weight {max_weight} < 1.")
    w = np.full(n, min_weight, dtype=float)
    budget = 1.0 - n * min_weight
    if budget < -1e-12:
        raise ValueError(
            f"Infeasible bounds: {n} assets * min_weight {min_weight} > 1.")
    order = np.argsort(b)
    if maximise:
        order = order[::-1]
    headroom = max_weight - min_weight
    for i in order:
        add = min(headroom, budget)
        w[i] += add
        budget -= add
        if budget <= 1e-15:
            break
    return float(np.dot(b, w)), w


def reachable_beta_interval(asset_betas, max_weight, min_weight=0.0):
    """Range of portfolio betas achievable under the box + simplex constraints.

    :return: tuple ``(lowest, highest)`` reachable portfolio beta.
    """
    lo, _ = greedy_beta_vertex(asset_betas, max_weight, min_weight,
                               maximise=False)
    hi, _ = greedy_beta_vertex(asset_betas, max_weight, min_weight,
                               maximise=True)
    return lo, hi


def beta_target_start(asset_betas, target_beta, max_weight, min_weight=0.0):
    """Feasible SLSQP starting point exactly on the ``b·w = target`` hyperplane.

    Blends the two beta-extreme vertices: beta is linear in w, so the convex
    combination ``(1-λ)·w_lo + λ·w_hi`` stays inside the box + simplex and hits
    any target inside the reachable interval exactly. Targets outside the
    interval land on the nearest vertex (callers should clamp the target
    first — an unreachable equality cannot converge).

    :return: weight vector on (or nearest to) the target hyperplane.
    """
    b_lo, w_lo = greedy_beta_vertex(asset_betas, max_weight, min_weight,
                                    maximise=False)
    b_hi, w_hi = greedy_beta_vertex(asset_betas, max_weight, min_weight,
                                    maximise=True)
    if b_hi - b_lo <= 1e-12:
        return w_lo
    lam = float(np.clip((target_beta - b_lo) / (b_hi - b_lo), 0.0, 1.0))
    return (1.0 - lam) * w_lo + lam * w_hi
