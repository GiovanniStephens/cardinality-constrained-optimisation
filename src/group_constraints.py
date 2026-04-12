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
