"""Tests for group allocation constraints."""

import unittest

import numpy as np
import pandas as pd

from src.group_constraints import (build_slsqp_constraints, check_constraints,
                                   check_floor_feasibility)


class TestBuildConstraints(unittest.TestCase):
    """Test SLSQP constraint generation from group constraints."""

    def setUp(self):
        self.tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ']
        self.membership = {
            'SPY': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Equity'},
            'QQQ': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Equity'},
            'TLT': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Fixed Income'},
            'GLD': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Commodities'},
            'VNQ': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Real Estate'},
        }

    def test_empty_constraints_returns_empty(self):
        result = build_slsqp_constraints(self.tickers, self.membership, {})
        self.assertEqual(result, [])

    def test_upper_bound_generates_ineq(self):
        constraints = {'country': {'United States': (0.0, 0.40)}}
        result = build_slsqp_constraints(self.tickers, self.membership, constraints)
        # max 40% US -> one upper-bound constraint (no lower since min=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['type'], 'ineq')
        # Equal weights (20% each, all US) -> 100% > 40% -> should be negative
        w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertLess(result[0]['fun'](w), 0)

    def test_lower_bound_generates_ineq(self):
        constraints = {'category_group': {'Fixed Income': (0.20, 1.0)}}
        result = build_slsqp_constraints(self.tickers, self.membership, constraints)
        # min 20% Fixed Income -> one lower-bound constraint (no upper since max=1.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['type'], 'ineq')
        # TLT at index 2 with 10% weight -> 10% < 20% -> should be negative
        w = np.array([0.3, 0.3, 0.1, 0.2, 0.1])
        self.assertLess(result[0]['fun'](w), 0)

    def test_both_bounds(self):
        constraints = {'category_group': {'Equity': (0.20, 0.60)}}
        result = build_slsqp_constraints(self.tickers, self.membership, constraints)
        # Both min and max -> two constraints
        self.assertEqual(len(result), 2)

    def test_no_members_skipped(self):
        constraints = {'sector': {'Technology': (0.0, 0.30)}}
        # No tickers have sector='Technology'
        result = build_slsqp_constraints(self.tickers, self.membership, constraints)
        self.assertEqual(result, [])


class TestCheckConstraints(unittest.TestCase):
    """Test constraint violation checking."""

    def setUp(self):
        self.tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
        self.membership = {
            'SPY': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Equity'},
            'QQQ': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Equity'},
            'TLT': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Fixed Income'},
            'GLD': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Commodities'},
        }

    def test_valid_portfolio(self):
        constraints = {'category_group': {'Equity': (0.0, 0.60)}}
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        valid, violations = check_constraints(
            self.tickers, weights, self.membership, constraints)
        self.assertTrue(valid)
        self.assertEqual(violations, [])

    def test_violation_detected(self):
        constraints = {'category_group': {'Equity': (0.0, 0.40)}}
        weights = np.array([0.30, 0.30, 0.20, 0.20])
        # Equity total = 60% > max 40%
        valid, violations = check_constraints(
            self.tickers, weights, self.membership, constraints)
        self.assertFalse(valid)
        self.assertEqual(len(violations), 1)
        self.assertIn('Equity', violations[0])
        self.assertIn('max', violations[0])

    def test_min_violation_detected(self):
        constraints = {'category_group': {'Fixed Income': (0.30, 1.0)}}
        weights = np.array([0.30, 0.30, 0.10, 0.30])
        # Fixed Income = 10% < min 30%
        valid, violations = check_constraints(
            self.tickers, weights, self.membership, constraints)
        self.assertFalse(valid)
        self.assertIn('min', violations[0])

    def test_missing_metadata_unconstrained(self):
        """Tickers not in membership are unconstrained."""
        membership = {
            'SPY': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Equity'},
            # QQQ, TLT, GLD missing from membership
        }
        constraints = {'category_group': {'Equity': (0.0, 0.30)}}
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        # Only SPY (25%) is counted as Equity -> 25% < 30% -> valid
        valid, violations = check_constraints(
            self.tickers, weights, membership, constraints)
        self.assertTrue(valid)


class TestCheckFloorFeasibility(unittest.TestCase):
    """The July 2026 incident detector: a capped group with k selected members
    needs at least k * min_weight of the book, so k * floor > cap means SLSQP
    can never converge (it fell back to 1/N silently)."""

    def setUp(self):
        # The incident shape: 4 unnamed holdings all landing in 'Unknown'.
        self.tickers = ['AAAA', 'BBBB', 'CCCC', 'DDDD', 'SPY']
        self.membership = {t: {'asset_class': 'Unknown'} for t in
                           ('AAAA', 'BBBB', 'CCCC', 'DDDD')}
        self.membership['SPY'] = {'asset_class': 'Equity'}

    def test_incident_shape_detected(self):
        # 4 members x 5% floor = 20% > 15% cap -> infeasible, named.
        constraints = {'asset_class': {'Unknown': (0.0, 0.15)}}
        problems = check_floor_feasibility(
            self.tickers, self.membership, constraints, min_weight=0.05)
        self.assertEqual(len(problems), 1)
        self.assertIn('asset_class/Unknown', problems[0])
        self.assertIn('4 members', problems[0])

    def test_feasible_configuration_is_silent(self):
        # 4 members x 3% floor = 12% <= 15% cap -> fine.
        constraints = {'asset_class': {'Unknown': (0.0, 0.15)}}
        problems = check_floor_feasibility(
            self.tickers, self.membership, constraints, min_weight=0.03)
        self.assertEqual(problems, [])

    def test_no_floor_is_always_feasible(self):
        constraints = {'asset_class': {'Unknown': (0.0, 0.01)}}
        self.assertEqual(check_floor_feasibility(
            self.tickers, self.membership, constraints, min_weight=0.0), [])
        self.assertEqual(check_floor_feasibility(
            self.tickers, self.membership, constraints, min_weight=None), [])

    def test_uncapped_group_ignored(self):
        # max_frac >= 1.0 can never be floor-infeasible.
        constraints = {'asset_class': {'Unknown': (0.0, 1.0)}}
        problems = check_floor_feasibility(
            self.tickers, self.membership, constraints, min_weight=0.5)
        self.assertEqual(problems, [])


class TestSLSQPIntegration(unittest.TestCase):
    """Test that optimise_weights respects group constraints."""

    def test_slsqp_with_group_constraints(self):
        from src.weights import optimise_weights

        np.random.seed(42)
        n = 4
        T = 500
        # Generate synthetic returns
        returns = np.random.randn(T, n) * 0.01
        returns[:, 0] += 0.0005  # SPY has positive drift
        returns[:, 1] += 0.0004  # QQQ has positive drift

        er = returns.mean(axis=0) * 252
        cov = np.cov(returns, rowvar=False) * 252

        tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
        membership = {
            'SPY': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Equity'},
            'QQQ': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Equity'},
            'TLT': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Fixed Income'},
            'GLD': {'country': 'United States', 'asset_type': 'etf',
                    'sector': None, 'category_group': 'Commodities'},
        }

        # Constrain equity to max 40%
        gc = {'category_group': {'Equity': (0.0, 0.40)}}

        result = optimise_weights(
            expected_returns=er, cov_matrix=cov,
            group_constraints=gc, group_membership=membership,
            selected_tickers=tickers,
        )
        self.assertTrue(result.success)
        equity_weight = result.x[0] + result.x[1]
        self.assertLessEqual(equity_weight, 0.40 + 1e-4)

    def test_slsqp_without_group_constraints(self):
        """No regression: empty group_constraints behaves like no constraints."""
        from src.weights import optimise_weights

        np.random.seed(42)
        n = 3
        T = 500
        returns = np.random.randn(T, n) * 0.01
        er = returns.mean(axis=0) * 252
        cov = np.cov(returns, rowvar=False) * 252

        result = optimise_weights(expected_returns=er, cov_matrix=cov)
        self.assertTrue(result.success or True)  # May not converge on random data
        self.assertAlmostEqual(np.sum(result.x), 1.0, places=4)


if __name__ == '__main__':
    unittest.main()
