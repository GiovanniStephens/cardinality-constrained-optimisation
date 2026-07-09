"""Group allocation constraints for SLSQP portfolio optimisation.

Supports country, sector, asset_type, and category_group constraints.
Each constraint specifies a (min_weight, max_weight) fraction of the
total portfolio that may be allocated to a given group.
"""

import logging

import numpy as np

from src import db

logger = logging.getLogger(__name__)


def load_membership(conn, symbols, exchange='US'):
    """Load group membership for a list of ticker symbols.

    Returns {symbol: {'country': ..., 'asset_type': ..., 'sector': ...,
                      'category_group': ...}}.
    """
    return db.load_ticker_metadata(conn, symbols, exchange=exchange)


def build_slsqp_constraints(selected_tickers, membership, constraints):
    """Build scipy constraint dicts for SLSQP from group constraints.

    :param selected_tickers: list of ticker symbols in the portfolio.
    :param membership: dict from load_membership().
    :param constraints: GROUP_CONSTRAINTS dict from config.
    :returns: list of scipy constraint dicts.
    """
    scipy_constraints = []

    for dimension, groups in constraints.items():
        for group_name, (min_frac, max_frac) in groups.items():
            # Find indices of selected tickers belonging to this group
            idx = []
            for i, ticker in enumerate(selected_tickers):
                meta = membership.get(ticker, {})
                if meta.get(dimension) == group_name:
                    idx.append(i)

            if not idx:
                continue

            idx = np.array(idx)

            # Lower bound: sum(w[idx]) >= min_frac
            if min_frac > 0.0:
                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, _idx=idx, _min=min_frac: (
                        np.sum(w[_idx]) - _min
                    ),
                })

            # Upper bound: max_frac >= sum(w[idx])
            if max_frac < 1.0:
                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, _idx=idx, _max=max_frac: (
                        _max - np.sum(w[_idx])
                    ),
                })

    return scipy_constraints


def check_floor_feasibility(selected_tickers, membership, constraints, min_weight):
    """Detect group caps that are mathematically infeasible against the
    per-holding weight floor.

    If a capped group contains k selected members, every feasible point gives
    it at least k * min_weight, so k * min_weight > max_frac means SLSQP can
    never converge — the July 2026 incident shape (>=4 unnamed holdings x 5%
    floor > the 10% Unknown cap -> silent equal-weight fallback). This turns
    that silent failure class into an immediate, named diagnosis.

    :param min_weight: the *effective* per-holding floor (post any relaxation).
    :returns: list of human-readable descriptions, empty when feasible.
    """
    problems = []
    if not min_weight or min_weight <= 0:
        return problems
    for dimension, groups in (constraints or {}).items():
        for group_name, (_min_frac, max_frac) in groups.items():
            if max_frac >= 1.0:
                continue
            k = sum(1 for t in selected_tickers
                    if membership.get(t, {}).get(dimension) == group_name)
            if k and k * min_weight > max_frac + 1e-9:
                problems.append(
                    f"{dimension}/{group_name}: {k} members x {min_weight:.1%} "
                    f"floor = {k * min_weight:.1%} > {max_frac:.1%} cap")
    return problems


def check_constraints(selected_tickers, weights, membership, constraints):
    """Check whether a portfolio satisfies group constraints.

    :returns: (is_valid, violations) where violations is a list of strings.
    """
    violations = []

    for dimension, groups in constraints.items():
        for group_name, (min_frac, max_frac) in groups.items():
            group_weight = 0.0
            for i, ticker in enumerate(selected_tickers):
                meta = membership.get(ticker, {})
                if meta.get(dimension) == group_name:
                    group_weight += weights[i]

            if group_weight < min_frac - 1e-6:
                violations.append(
                    f"{dimension}/{group_name}: {group_weight:.1%} < min {min_frac:.1%}"
                )
            if group_weight > max_frac + 1e-6:
                violations.append(
                    f"{dimension}/{group_name}: {group_weight:.1%} > max {max_frac:.1%}"
                )

    return (len(violations) == 0, violations)
