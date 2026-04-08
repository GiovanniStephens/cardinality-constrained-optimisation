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


class TestOptimiserDataBoundary(unittest.TestCase):
    """Tests that optimisers only use the data passed to optimise(), not external data."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices(n_days=200, n_tickers=10, seed=42)
        # Training subset: first 150 days
        cls.train_prices = cls.prices.iloc[:150]
        # Future data that must not leak in
        cls.full_prices = cls.prices

    def test_pygad_only_sees_training_prices(self):
        """PygadOptimiser should only use prices passed to optimise().

        Pass 150 training days. All selected tickers should exist in
        training data and the optimiser should not have seen the full 200 days.
        """
        from src.optimisers.pygad_ga import PygadOptimiser
        opt = PygadOptimiser(
            num_children=20, num_generations=2,
            min_securities=2, max_securities=5,
            min_weight=0.0, max_weight=1.0,
            target_return=None, use_forecasts=False,
        )
        result = opt.optimise(self.train_prices)
        # All selected tickers must be in training data columns
        for t in result.selected_tickers:
            self.assertIn(t, self.train_prices.columns)
        # Internal data should have 150-day shape (after log returns)
        self.assertEqual(opt._data.shape[1], len(self.train_prices),
                         "PygadOptimiser._data should have training-period columns only")

    def test_monte_carlo_only_sees_training_prices(self):
        """MonteCarloOptimiser should only use prices passed to optimise()."""
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        opt = MonteCarloOptimiser(
            n_trials=100, min_securities=2, max_securities=5,
            num_processes=1,
        )
        result = opt.optimise(self.train_prices)
        for t in result.selected_tickers:
            self.assertIn(t, self.train_prices.columns)

    def test_island_ga_only_sees_training_prices(self):
        """IslandGAOptimiser should only use prices passed to optimise()."""
        import multiprocessing
        from src.optimisers.island_ga import IslandGAOptimiser
        # Skip on macOS spawn — local functions in IslandGA can't be pickled
        if multiprocessing.get_start_method() == 'spawn':
            self.skipTest("IslandGA multiprocessing incompatible with 'spawn' start method")
        opt = IslandGAOptimiser(
            num_generations=2, population_size=50, num_elites=5,
            min_securities=2, max_securities=5, min_return=None,
        )
        result = opt.optimise(self.train_prices)
        for t in result.selected_tickers:
            self.assertIn(t, self.train_prices.columns)

    def test_optimiser_excludes_insufficient_data_tickers(self):
        """An all-NaN ticker should be excluded by upstream coverage filtering.

        Optimisers rely on load_prices_csv (min_coverage) to drop tickers
        with insufficient data. This test verifies that pipeline-level
        filtering removes the all-NaN ticker before the optimiser sees it.
        """
        import tempfile, os
        from src.portfolio_utils import load_prices_csv

        # Extend training prices with a late-entry ticker (all NaN)
        late_ticker = pd.DataFrame(
            np.nan, index=self.train_prices.index,
            columns=['LATE'],
        )
        prices_with_late = pd.concat([self.train_prices, late_ticker], axis=1)

        # Save to CSV and reload with coverage filtering (the standard pipeline)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            prices_with_late.to_csv(f.name)
            path = f.name
        try:
            filtered = load_prices_csv(path, min_coverage=0.5)
            self.assertNotIn('LATE', filtered.columns,
                             "All-NaN ticker should be excluded by coverage filtering")

            from src.optimisers.monte_carlo import MonteCarloOptimiser
            opt = MonteCarloOptimiser(
                n_trials=200, min_securities=2, max_securities=5,
                num_processes=1,
            )
            result = opt.optimise(filtered)
            self.assertNotIn('LATE', result.selected_tickers,
                             "All-NaN ticker should not appear after filtering")
        finally:
            os.unlink(path)


class TestPygadFunctions(unittest.TestCase):
    """Tests for pygad_ga module-level functions (fitness, covariance, copulas).

    These tests require data/ETF_Prices.csv to be present.
    """

    def _load(self):
        from src.optimisers import pygad_ga as op
        data = op.load_data('data/ETF_Prices.csv')
        return op, data

    def test_get_cov_matrix(self):
        op, data = self._load()
        log_returns = op.calculate_returns(data)
        cov = log_returns.iloc[:, :2].cov() * 252
        cov_matrix = op.get_cov_matrix(log_returns.iloc[:, :2])
        np.testing.assert_array_almost_equal(cov_matrix, cov.values)

    def test_optimisation_max_weight(self):
        op, data = self._load()
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        num_stocks = 5
        max_weight = 0.3
        initial_weights = [1 / num_stocks] * num_stocks
        sol = op.optimize(log_returns.iloc[:, :num_stocks],
                          initial_weights, max_weight=max_weight)
        self.assertLessEqual(max(sol['x']), max_weight)

    def test_optimisation_min_weight(self):
        op, data = self._load()
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        num_stocks = 5
        initial_weights = [1 / num_stocks] * num_stocks
        sol = op.optimize(log_returns.iloc[:, :num_stocks], initial_weights)
        self.assertGreaterEqual(min(sol['x']), 0)

    def test_fitness_too_many_ETFs(self):
        op, data = self._load()
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        individual = [1] * log_returns.shape[1]
        fitness = op.fitness(individual, log_returns.T)
        self.assertLess(fitness, 0)

    def test_fitness_too_few_ETFs(self):
        op, data = self._load()
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        individual = [0] * log_returns.shape[1]
        fitness = op.fitness(individual, log_returns.T)
        self.assertLessEqual(fitness, 0)

    def test_fitness_normal(self):
        op, data = self._load()
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        num_stocks = 8
        individual = [1] * num_stocks + [0] * (log_returns.shape[1] - num_stocks)
        fitness = op.fitness(individual, log_returns.T)
        self.assertGreater(fitness, 0)
        self.assertLess(fitness, 5)

    def test_prepare_opt_inputs(self):
        op, data = self._load()
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=True)
        self.assertEqual(len(op.data), len(log_returns.T))

    def test_prepare_opt_inputs_variances_null(self):
        op, data = self._load()
        op.prepare_opt_inputs(data, use_forecasts=False)
        self.assertIsNone(op.variances)

    def test_estimate_corr_using_copulas_is_correlation_matrix(self):
        op, data = self._load()
        log_returns = op.calculate_returns(data)
        subset = log_returns.iloc[:, :5]
        corr = op.estimate_corr_using_copulas(subset)
        self.assertEqual(corr.shape, (5, 5))
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(5))
        np.testing.assert_array_almost_equal(corr, corr.T)
        eigenvalues = np.linalg.eigvalsh(corr)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_get_cov_matrix_with_copulae_no_forecasts(self):
        op, data = self._load()
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        subset = log_returns.iloc[:, :5]
        cov_matrix = op.get_cov_matrix(subset, use_copulae=True)
        self.assertEqual(cov_matrix.shape, (5, 5))
        np.testing.assert_array_almost_equal(cov_matrix, cov_matrix.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        self.assertTrue(np.all(eigenvalues >= -1e-10))
        self.assertTrue(np.all(np.diag(cov_matrix) > 0))


class TestCopulaTemporalIntegrity(unittest.TestCase):
    """Tests that copula estimation uses only the provided data, not stale globals."""

    def _make_correlated_returns(self, n=300, rho=0.8, seed=42):
        np.random.seed(seed)
        L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
        z = np.random.randn(n, 2)
        returns = (z @ L.T) * 0.02
        dates = pd.bdate_range('2018-01-01', periods=n, freq='B')
        return pd.DataFrame(returns, index=dates, columns=['X', 'Y'])

    def test_copula_uses_only_provided_data(self):
        from src.optimisers.pygad_ga import estimate_corr_using_copulas
        pos_data = self._make_correlated_returns(n=500, rho=0.8, seed=42)
        corr_pos = estimate_corr_using_copulas(pos_data)
        off_diag_pos = corr_pos[0, 1] if isinstance(corr_pos, np.ndarray) else corr_pos.iloc[0, 1]
        self.assertGreater(off_diag_pos, 0.3)

        neg_data = self._make_correlated_returns(n=500, rho=-0.7, seed=99)
        corr_neg = estimate_corr_using_copulas(neg_data)
        off_diag_neg = corr_neg[0, 1] if isinstance(corr_neg, np.ndarray) else corr_neg.iloc[0, 1]
        self.assertLess(off_diag_neg, 0.0)

    def test_copula_fallback_preserves_correlation_sign(self):
        from src.optimisers.pygad_ga import estimate_corr_using_copulas
        pos_data = self._make_correlated_returns(n=15, rho=0.9, seed=42)
        corr = estimate_corr_using_copulas(pos_data)
        off_diag = corr[0, 1] if isinstance(corr, np.ndarray) else corr.iloc[0, 1]
        self.assertGreater(off_diag, 0.0)


if __name__ == '__main__':
    unittest.main()
