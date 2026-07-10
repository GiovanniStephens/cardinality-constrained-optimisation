"""Tests for the per-mode weight dispatch in src/backtest/simulation.py.

Covers the new modes added in PR 2: equal weights, minimum-variance, the
CCC-baseline path that uses historical std as a pseudo-forecast variance,
and benchmark-portfolio construction.
"""

import unittest

import numpy as np
import pandas as pd

from tests.helpers import make_synthetic_prices
from src.returns import calculate_log_returns, calculate_expected_returns
from src.backtest import simulation
from src.backtest.simulation import (
    benchmark_portfolio,
    _compute_weights_for_portfolio,
    _equal_weights,
    _max_sharpe_weights,
    _min_variance_weights,
)


class TestWeightModes(unittest.TestCase):
    """Direct unit tests for the new weight helpers."""

    @classmethod
    def setUpClass(cls):
        cls.prices = make_synthetic_prices(n_days=400, n_tickers=8, seed=7)
        log_returns = calculate_log_returns(cls.prices)
        # The simulation helpers read from these module globals.
        simulation._backtest_log_returns = log_returns.transpose()
        simulation._backtest_expected_returns = calculate_expected_returns(
            log_returns)
        cls.portfolio = list(cls.prices.columns[:5])

    def test_equal_weights_sums_to_one_and_uniform(self):
        w = _equal_weights(self.portfolio)
        self.assertEqual(len(w), len(self.portfolio))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
        self.assertTrue(np.allclose(w, 1.0 / len(self.portfolio)))

    def test_min_variance_weights_sum_to_one(self):
        w = _min_variance_weights(self.portfolio)
        self.assertEqual(len(w), len(self.portfolio))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=4)
        self.assertTrue(np.all(w >= -1e-6))

    def test_max_sharpe_with_er_override_changes_weights(self):
        """An ER override should move weight off a penalised ticker.

        Uses a bearish override (very negative ER on ticker 0) rather than a
        bullish one: a bullish override is brittle because the baseline may
        already saturate the per-holding weight cap (so the favoured ticker
        can't rise — this differs by scipy/SLSQP version). A penalty can always
        push a non-trivial weight down, so the assertion is version-robust.
        """
        baseline = _max_sharpe_weights(self.portfolio)
        override = np.zeros(len(self.portfolio))
        override[0] = -10.0  # heavily penalise ticker 0
        biased = _max_sharpe_weights(
            self.portfolio, expected_returns_override=override)
        self.assertLess(biased[0], baseline[0])
        self.assertAlmostEqual(float(biased.sum()), 1.0, places=4)

    def test_optimal_ccc_path_uses_forecast_variances(self):
        """`optimal_ccc` mode should differ from `optimal` when variances differ."""
        var_series = pd.Series(
            np.linspace(0.01, 0.5, len(self.portfolio)),
            index=self.portfolio,
        )
        ccc = _max_sharpe_weights(
            self.portfolio, forecast_variances=var_series)
        opt = _max_sharpe_weights(self.portfolio)
        # With wildly different variances we expect different weights.
        self.assertFalse(np.allclose(ccc, opt, atol=1e-4))
        self.assertAlmostEqual(float(ccc.sum()), 1.0, places=4)


class TestDispatchRoutesModes(unittest.TestCase):
    """`_compute_weights_for_portfolio` should accept all new modes."""

    @classmethod
    def setUpClass(cls):
        prices = make_synthetic_prices(n_days=400, n_tickers=8, seed=11)
        log_returns = calculate_log_returns(prices)
        simulation._backtest_log_returns = log_returns.transpose()
        simulation._backtest_expected_returns = calculate_expected_returns(
            log_returns)
        cls.portfolio = list(prices.columns[:5])
        cls.var_series = log_returns.var().loc[cls.portfolio] * 252

    def _check(self, mode, kwargs=None):
        if kwargs is None:
            w = _compute_weights_for_portfolio((self.portfolio, mode))
        else:
            w = _compute_weights_for_portfolio((self.portfolio, mode, kwargs))
        self.assertEqual(len(w), len(self.portfolio))
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=3)

    def test_random_mode_legacy_tuple(self):
        self._check('random')

    def test_optimal_mode_legacy_tuple(self):
        self._check('optimal')

    def test_copulae_mode_legacy_tuple(self):
        self._check('copulae')

    def test_optimal_ccc_mode(self):
        self._check('optimal_ccc', {'var': self.var_series})

    def test_optimal_beta1_mode(self):
        """Beta-pinned mode: dispatch works AND the pin binds exactly."""
        from src.weights import reachable_beta_interval
        lr = simulation._backtest_log_returns.transpose()
        betas_all = (lr.apply(lambda col: col.cov(lr[self.portfolio[0]]))
                     / lr[self.portfolio[0]].var()).fillna(0.0)
        b = betas_all.loc[self.portfolio].values.astype(float)
        max_w = max(1 / (len(self.portfolio) - 1), 0.3)
        lo, hi = reachable_beta_interval(b, max_w)
        target = float(np.clip(1.0, lo, hi))  # clamp as the runner does
        w = _compute_weights_for_portfolio(
            (self.portfolio, 'optimal_beta1',
             {'betas': b, 'target_beta': target}))
        self.assertEqual(len(w), len(self.portfolio))
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=3)
        self.assertAlmostEqual(float(np.dot(b, w)), target, places=4)

    def test_min_variance_mode(self):
        self._check('min_variance', {})

    def test_equal_mode(self):
        self._check('equal', {})

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            _compute_weights_for_portfolio(
                (self.portfolio, 'mystery_mode', {}))


class TestBenchmarkPortfolio(unittest.TestCase):
    """Construction of fixed-weight market benchmarks."""

    def _frame(self, *cols):
        idx = pd.bdate_range('2020-01-01', periods=10)
        return pd.DataFrame(
            np.random.randn(len(idx), len(cols)),
            index=idx, columns=list(cols))

    def test_spy_present(self):
        train = self._frame('SPY', 'X')
        oos = self._frame('SPY', 'X')
        tickers, weights = benchmark_portfolio('bench_spy', train, oos)
        self.assertEqual(tickers, ['SPY'])
        self.assertTrue(np.allclose(weights, [1.0]))

    def test_6040_present(self):
        train = self._frame('SPY', 'AGG', 'X')
        oos = self._frame('SPY', 'AGG', 'X')
        tickers, weights = benchmark_portfolio('bench_6040', train, oos)
        self.assertEqual(tickers, ['SPY', 'AGG'])
        self.assertTrue(np.allclose(weights, [0.6, 0.4]))

    def test_missing_in_train_returns_none(self):
        train = self._frame('X', 'Y')
        oos = self._frame('SPY', 'AGG')
        self.assertEqual(
            benchmark_portfolio('bench_spy', train, oos), (None, None))

    def test_missing_in_oos_returns_none(self):
        train = self._frame('SPY', 'AGG')
        oos = self._frame('X', 'Y')
        self.assertEqual(
            benchmark_portfolio('bench_6040', train, oos), (None, None))

    def test_partial_missing_for_6040(self):
        """6040 needs both SPY and AGG; missing one ⇒ None."""
        train = self._frame('SPY', 'X')
        oos = self._frame('SPY', 'X')
        self.assertEqual(
            benchmark_portfolio('bench_6040', train, oos), (None, None))

    def test_unknown_category_raises(self):
        train = self._frame('SPY')
        oos = self._frame('SPY')
        with self.assertRaises(ValueError):
            benchmark_portfolio('bench_qqq', train, oos)


if __name__ == '__main__':
    unittest.main()
