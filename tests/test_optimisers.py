"""Tests for the Optimiser classes (MIP, Monte Carlo, Island GA, Pygad)."""

import unittest

import numpy as np
import pandas as pd

from src.portfolio_utils import OptimisationResult


def _make_synthetic_prices(n_days=200, n_tickers=10, seed=42):
    """Small synthetic price matrix for fast tests."""
    np.random.seed(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days, freq='B')
    tickers = [f'T{i}' for i in range(n_tickers)]
    log_rets = np.random.randn(n_days, n_tickers) * 0.01
    prices = 100 * np.exp(log_rets.cumsum(axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


class TestMIPOptimiser(unittest.TestCase):
    """Tests for MIPOptimiser."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices()

    def _run(self, **kwargs):
        from src.optimisers.mip import MIPOptimiser
        opt = MIPOptimiser(max_securities=5, **kwargs)
        return opt.optimise(self.prices)

    def test_returns_optimisation_result(self):
        result = self._run()
        self.assertIsInstance(result, OptimisationResult)

    def test_selected_tickers_subset(self):
        result = self._run()
        for t in result.selected_tickers:
            self.assertIn(t, self.prices.columns)

    def test_weights_length_matches_tickers(self):
        result = self._run()
        self.assertEqual(len(result.weights), len(result.selected_tickers))

    def test_weights_sum_to_one(self):
        result = self._run()
        self.assertGreaterEqual(len(result.selected_tickers), 1)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=4)

    def test_sharpe_is_finite(self):
        result = self._run()
        self.assertTrue(np.isfinite(result.sharpe_ratio))

    def test_respects_max_securities(self):
        result = self._run()
        self.assertLessEqual(len(result.selected_tickers), 5)

    def test_deterministic(self):
        r1 = self._run()
        r2 = self._run()
        self.assertEqual(r1.selected_tickers, r2.selected_tickers)

    def test_metadata_has_solver_status(self):
        result = self._run()
        self.assertIn('solver_status', result.metadata)


class TestMonteCarloOptimiser(unittest.TestCase):
    """Tests for MonteCarloOptimiser."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices()

    def _run(self, **kwargs):
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        opt = MonteCarloOptimiser(
            n_trials=500, min_securities=2, max_securities=5,
            num_processes=1, **kwargs,
        )
        return opt.optimise(self.prices)

    def test_returns_optimisation_result(self):
        result = self._run()
        self.assertIsInstance(result, OptimisationResult)

    def test_selected_tickers_subset(self):
        result = self._run()
        for t in result.selected_tickers:
            self.assertIn(t, self.prices.columns)

    def test_weights_length_matches_tickers(self):
        result = self._run()
        self.assertEqual(len(result.weights), len(result.selected_tickers))

    def test_weights_sum_to_one(self):
        result = self._run()
        self.assertGreaterEqual(len(result.selected_tickers), 2)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=4)

    def test_sharpe_is_finite(self):
        result = self._run()
        self.assertTrue(np.isfinite(result.sharpe_ratio))

    def test_respects_cardinality(self):
        result = self._run()
        n = len(result.selected_tickers)
        self.assertGreaterEqual(n, 2)
        self.assertLessEqual(n, 5)

    def test_metadata_has_elapsed(self):
        result = self._run()
        self.assertIn('elapsed_seconds', result.metadata)


class TestIslandGAOptimiser(unittest.TestCase):
    """Tests for IslandGAOptimiser."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices()

    def _run(self, **kwargs):
        from src.optimisers.island_ga import IslandGAOptimiser
        opt = IslandGAOptimiser(
            num_generations=3, population_size=100, num_elites=10,
            min_securities=2, max_securities=5, min_return=None,
            **kwargs,
        )
        return opt.optimise(self.prices)

    def test_returns_optimisation_result(self):
        result = self._run()
        self.assertIsInstance(result, OptimisationResult)

    def test_selected_tickers_subset(self):
        result = self._run()
        for t in result.selected_tickers:
            self.assertIn(t, self.prices.columns)

    def test_weights_length_matches_tickers(self):
        result = self._run()
        self.assertEqual(len(result.weights), len(result.selected_tickers))

    def test_sharpe_is_finite(self):
        result = self._run()
        self.assertTrue(np.isfinite(result.sharpe_ratio))

    def test_metadata_has_elapsed(self):
        result = self._run()
        self.assertIn('elapsed_seconds', result.metadata)


class TestPygadOptimiser(unittest.TestCase):
    """Tests for PygadOptimiser."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices()

    def _run(self, **kwargs):
        from src.optimisers.pygad_ga import PygadOptimiser
        opt = PygadOptimiser(
            num_children=30, num_generations=2,
            min_securities=2, max_securities=5,
            min_weight=0.0, max_weight=1.0,
            target_return=None, use_forecasts=False,
            **kwargs,
        )
        return opt.optimise(self.prices)

    def test_returns_optimisation_result(self):
        result = self._run()
        self.assertIsInstance(result, OptimisationResult)

    def test_selected_tickers_subset(self):
        result = self._run()
        for t in result.selected_tickers:
            self.assertIn(t, self.prices.columns)

    def test_weights_length_matches_tickers(self):
        result = self._run()
        self.assertEqual(len(result.weights), len(result.selected_tickers))

    def test_sharpe_is_finite(self):
        result = self._run()
        self.assertTrue(np.isfinite(result.sharpe_ratio))

    def test_metadata_has_elapsed(self):
        result = self._run()
        self.assertIn('elapsed_seconds', result.metadata)


class TestSeedDeterminism(unittest.TestCase):
    """Verify that passing a seed produces reproducible results."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices(n_days=200, n_tickers=20)

    def test_monte_carlo_same_seed_same_result(self):
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        r1 = MonteCarloOptimiser(
            n_trials=500, min_securities=2, max_securities=5,
            num_processes=1, seed=42).optimise(self.prices)
        r2 = MonteCarloOptimiser(
            n_trials=500, min_securities=2, max_securities=5,
            num_processes=1, seed=42).optimise(self.prices)
        self.assertEqual(r1.selected_tickers, r2.selected_tickers)
        self.assertAlmostEqual(r1.sharpe_ratio, r2.sharpe_ratio)

    def test_monte_carlo_different_seed_different_result(self):
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        r1 = MonteCarloOptimiser(
            n_trials=500, min_securities=2, max_securities=5,
            num_processes=1, seed=42).optimise(self.prices)
        r2 = MonteCarloOptimiser(
            n_trials=500, min_securities=2, max_securities=5,
            num_processes=1, seed=99).optimise(self.prices)
        # With different seeds on 20 tickers, selections should differ
        self.assertNotEqual(r1.selected_tickers, r2.selected_tickers)

    def test_pygad_same_seed_same_result(self):
        from src.optimisers.pygad_ga import PygadOptimiser
        kwargs = dict(num_children=20, num_generations=2,
                      min_securities=2, max_securities=5,
                      min_weight=0.0, max_weight=1.0,
                      target_return=None, use_forecasts=False)
        r1 = PygadOptimiser(seed=42, **kwargs).optimise(self.prices)
        r2 = PygadOptimiser(seed=42, **kwargs).optimise(self.prices)
        self.assertEqual(r1.selected_tickers, r2.selected_tickers)
        self.assertAlmostEqual(r1.sharpe_ratio, r2.sharpe_ratio, places=4)


class TestMIPMaxSecurities(unittest.TestCase):
    """Verify MIP respects different max_securities values."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices()

    def test_max_securities_3(self):
        from src.optimisers.mip import MIPOptimiser
        result = MIPOptimiser(max_securities=3).optimise(self.prices)
        self.assertLessEqual(len(result.selected_tickers), 3)

    def test_max_securities_8(self):
        from src.optimisers.mip import MIPOptimiser
        result = MIPOptimiser(max_securities=8).optimise(self.prices)
        self.assertLessEqual(len(result.selected_tickers), 8)


if __name__ == '__main__':
    unittest.main()
