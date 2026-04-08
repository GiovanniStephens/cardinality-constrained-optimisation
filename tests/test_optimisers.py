"""Tests for the Optimiser classes (MIP, Monte Carlo, Island GA, Pygad)."""

import multiprocessing
import unittest

import numpy as np
import pandas as pd

from src.portfolio_utils import OptimisationResult, calculate_log_returns
from tests.helpers import (
    OptimiserTestMixin,
    make_small_divergent_prices,
    brute_force_optimal,
    assert_result_integrity,
)


def _make_synthetic_prices(n_days=200, n_tickers=10, seed=42):
    """Small synthetic price matrix for fast tests."""
    np.random.seed(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days, freq='B')
    tickers = [f'T{i}' for i in range(n_tickers)]
    log_rets = np.random.randn(n_days, n_tickers) * 0.01
    prices = 100 * np.exp(log_rets.cumsum(axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


class TestMIPOptimiser(OptimiserTestMixin, unittest.TestCase):
    """Tests for MIPOptimiser."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices()

    def _run(self, **kwargs):
        from src.optimisers.mip import MIPOptimiser
        opt = MIPOptimiser(max_securities=5, **kwargs)
        return opt.optimise(self.prices)

    def test_weights_sum_to_one(self):
        result = self._run()
        self.assertGreaterEqual(len(result.selected_tickers), 1)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=4)

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


class TestMonteCarloOptimiser(OptimiserTestMixin, unittest.TestCase):
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

    def test_weights_sum_to_one(self):
        result = self._run()
        self.assertGreaterEqual(len(result.selected_tickers), 2)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=4)

    def test_respects_cardinality(self):
        result = self._run()
        n = len(result.selected_tickers)
        self.assertGreaterEqual(n, 2)
        self.assertLessEqual(n, 5)

    def test_metadata_has_elapsed(self):
        result = self._run()
        self.assertIn('elapsed_seconds', result.metadata)


class TestIslandGAOptimiser(OptimiserTestMixin, unittest.TestCase):
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

    def test_metadata_has_elapsed(self):
        result = self._run()
        self.assertIn('elapsed_seconds', result.metadata)


class TestPygadOptimiser(OptimiserTestMixin, unittest.TestCase):
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
        self.assertEqual(opt._data.shape[0], len(self.train_prices),
                         "PygadOptimiser._data should have training-period rows only")

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
    """Legacy module-level function tests removed in Phase 5 cleanup.

    The module-level functions (prepare_opt_inputs, fitness, optimize,
    get_cov_matrix, create_individual, etc.) have been removed.
    Equivalent functionality is tested via TestPygadOptimiser and
    TestCopulaTemporalIntegrity.
    """
    pass


class TestBatchFitnessSubsetCov(unittest.TestCase):
    """Verify that batch_fitness using centered returns matches per-subset covariance."""

    def test_matches_manual_computation(self):
        from src.optimisers.island_ga import batch_fitness
        from src.portfolio_utils import (
            calculate_log_returns, calculate_expected_returns,
        )
        from src.config import TRADING_DAYS_PER_YEAR

        prices = _make_synthetic_prices(n_days=200, n_tickers=10, seed=42)
        log_returns = calculate_log_returns(prices)
        expected_returns = calculate_expected_returns(log_returns).values
        centered = (log_returns - log_returns.mean(axis=0)).values
        T_obs = centered.shape[0]

        # Create a few test individuals
        np.random.seed(123)
        population = np.zeros((5, 10), dtype=int)
        for i in range(5):
            idx = np.random.choice(10, np.random.randint(3, 6), replace=False)
            population[i, idx] = 1

        fitness_vec = batch_fitness(population, expected_returns, centered,
                                    T_obs, min_etfs=2, max_etfs=8,
                                    min_return=None)

        # Manually compute Sharpe for each individual
        for i in range(5):
            sel = population[i] == 1
            n = np.sum(sel)
            if n < 2:
                continue
            sub_returns = log_returns.iloc[:, sel]
            # Raw sample covariance (no shrinkage) for comparison
            sub_cov = sub_returns.cov().values * TRADING_DAYS_PER_YEAR
            w = np.ones(n) / n
            port_ret = np.dot(w, expected_returns[sel])
            port_var = np.dot(w, np.dot(sub_cov, w))
            expected_sharpe = port_ret / np.sqrt(port_var) if port_var > 0 else -1e4
            self.assertAlmostEqual(fitness_vec[i], expected_sharpe, places=6,
                                   msg=f"Individual {i} fitness mismatch")


class TestRepairCardinality(unittest.TestCase):
    """Tests for the repair_cardinality function."""

    def test_repairs_too_many(self):
        from src.optimisers.island_ga import repair_cardinality
        np.random.seed(42)
        offspring = np.ones((5, 20), dtype=int)  # all 20 selected
        repaired = repair_cardinality(offspring, min_etfs=3, max_etfs=10)
        counts = repaired.sum(axis=1)
        for c in counts:
            self.assertLessEqual(c, 10)
            self.assertGreaterEqual(c, 3)

    def test_repairs_too_few(self):
        from src.optimisers.island_ga import repair_cardinality
        np.random.seed(42)
        offspring = np.zeros((5, 20), dtype=int)  # none selected
        repaired = repair_cardinality(offspring, min_etfs=3, max_etfs=10)
        counts = repaired.sum(axis=1)
        for c in counts:
            self.assertGreaterEqual(c, 3)

    def test_no_change_when_valid(self):
        from src.optimisers.island_ga import repair_cardinality
        np.random.seed(42)
        offspring = np.zeros((3, 20), dtype=int)
        for i in range(3):
            offspring[i, :5] = 1  # exactly 5 selected
        original = offspring.copy()
        repaired = repair_cardinality(offspring, min_etfs=3, max_etfs=10)
        np.testing.assert_array_equal(repaired, original)


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


class TestCrossOptimiserConvergence(unittest.TestCase):
    """On a tiny 5-ticker problem, all optimisers should achieve near-optimal Sharpe."""

    @classmethod
    def setUpClass(cls):
        cls.prices = make_small_divergent_prices(n_days=300, seed=42)
        cls.k = 3
        cls.ref = brute_force_optimal(cls.prices, cls.k)

    def test_brute_force_reference_valid(self):
        self.assertEqual(len(self.ref.selected_tickers), self.k)
        self.assertTrue(np.isfinite(self.ref.sharpe_ratio))

    def test_monte_carlo_converges(self):
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        opt = MonteCarloOptimiser(
            n_trials=200, min_securities=self.k, max_securities=self.k,
            num_processes=1, seed=42,
        )
        result = opt.optimise(self.prices)
        self.assertEqual(len(result.selected_tickers), self.k)
        # MC uses equal-weight search so may pick different tickers,
        # but SLSQP-refined Sharpe should be close to brute-force optimum
        self.assertGreater(result.sharpe_ratio, self.ref.sharpe_ratio * 0.80)

    def test_pygad_converges(self):
        from src.optimisers.pygad_ga import PygadOptimiser
        opt = PygadOptimiser(
            num_children=50, num_generations=10,
            min_securities=self.k, max_securities=self.k,
            min_weight=0.0, max_weight=1.0,
            target_return=None, use_forecasts=False, seed=42,
        )
        result = opt.optimise(self.prices)
        self.assertEqual(len(result.selected_tickers), self.k)
        self.assertGreater(result.sharpe_ratio, self.ref.sharpe_ratio * 0.80)

    def test_island_ga_converges(self):
        if multiprocessing.get_start_method() == 'spawn':
            self.skipTest("IslandGA incompatible with 'spawn' start method")
        from src.optimisers.island_ga import IslandGAOptimiser
        opt = IslandGAOptimiser(
            num_generations=10, population_size=200, num_elites=20,
            min_securities=self.k, max_securities=self.k, min_return=None,
        )
        result = opt.optimise(self.prices)
        self.assertEqual(len(result.selected_tickers), self.k)
        self.assertGreater(result.sharpe_ratio, self.ref.sharpe_ratio * 0.80)


class TestWeightBoundCompliance(unittest.TestCase):
    """Verify SLSQP-refined weights respect [min_weight, max_weight] bounds."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices(n_days=300, n_tickers=15, seed=42)

    def test_pygad_custom_bounds(self):
        from src.optimisers.pygad_ga import PygadOptimiser
        opt = PygadOptimiser(
            num_children=30, num_generations=3,
            min_securities=3, max_securities=8,
            min_weight=0.05, max_weight=0.45,
            target_return=None, use_forecasts=False, seed=42,
        )
        result = opt.optimise(self.prices)
        tol = 1e-6
        for i, w in enumerate(result.weights):
            self.assertGreaterEqual(w, 0.05 - tol,
                                    f"weight[{i}]={w} below min_weight=0.05")
            self.assertLessEqual(w, 0.45 + tol,
                                 f"weight[{i}]={w} above max_weight=0.45")

    def test_monte_carlo_default_bounds(self):
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        opt = MonteCarloOptimiser(
            n_trials=200, min_securities=3, max_securities=8,
            num_processes=1, seed=42,
        )
        result = opt.optimise(self.prices)
        tol = 1e-6
        for i, w in enumerate(result.weights):
            self.assertGreaterEqual(w, -tol, f"weight[{i}]={w} is negative")
            self.assertLessEqual(w, 1.0 + tol, f"weight[{i}]={w} above 1.0")


class TestDateAlignmentValidation(unittest.TestCase):
    """Verify date alignment assumptions hold."""

    def test_synthetic_prices_have_uniform_dates(self):
        prices = _make_synthetic_prices(n_days=200, n_tickers=10)
        for col in prices.columns:
            self.assertEqual(prices[col].dropna().shape[0], 200)

    def test_misaligned_concat_produces_nan(self):
        p1 = _make_synthetic_prices(n_days=100, n_tickers=3, seed=1)
        p2 = _make_synthetic_prices(n_days=80, n_tickers=3, seed=2)
        p2.columns = ['X0', 'X1', 'X2']
        # Shift p2 dates forward so they only partially overlap
        p2.index = pd.bdate_range(p1.index[50], periods=80, freq='B')
        merged = pd.concat([p1, p2], axis=1)
        # Merged should have NaN in the non-overlapping regions
        self.assertTrue(merged.isna().any().any(),
                        "Misaligned concat should produce NaN")

    def test_unsorted_index_raises(self):
        prices = _make_synthetic_prices(n_days=100, n_tickers=5)
        shuffled = prices.sample(frac=1)  # shuffle rows
        with self.assertRaises(ValueError):
            calculate_log_returns(shuffled)

    def test_aligned_data_works(self):
        prices = _make_synthetic_prices(n_days=200, n_tickers=10, seed=42)
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        opt = MonteCarloOptimiser(
            n_trials=100, min_securities=2, max_securities=5,
            num_processes=1, seed=42,
        )
        result = opt.optimise(prices)
        self.assertIsInstance(result, OptimisationResult)
        self.assertTrue(np.isfinite(result.sharpe_ratio))


class TestNaNPropagation(unittest.TestCase):
    """Verify NaN/inf handling in the pipeline."""

    def test_nan_in_prices_gives_zero_log_return(self):
        prices = _make_synthetic_prices(n_days=50, n_tickers=3, seed=42)
        prices.iloc[10, 0] = np.nan
        lr = calculate_log_returns(prices)
        # The NaN should become 0 in log returns
        self.assertEqual(lr.iloc[10, 0], 0.0)

    def test_zero_price_gives_zero_log_return(self):
        prices = _make_synthetic_prices(n_days=50, n_tickers=3, seed=42)
        prices.iloc[10, 1] = 0.0  # will cause -inf in log
        lr = calculate_log_returns(prices)
        self.assertEqual(lr.iloc[10, 1], 0.0)

    def test_partial_nan_column_doesnt_crash(self):
        prices = _make_synthetic_prices(n_days=200, n_tickers=10, seed=42)
        # Inject NaN in 10% of one column
        prices.iloc[5:25, 3] = np.nan
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        opt = MonteCarloOptimiser(
            n_trials=100, min_securities=2, max_securities=5,
            num_processes=1, seed=42,
        )
        result = opt.optimise(prices)
        self.assertIsInstance(result, OptimisationResult)

    def test_no_nan_inf_in_output_weights(self):
        prices = _make_synthetic_prices(n_days=200, n_tickers=10, seed=42)
        prices.iloc[5:15, 2] = np.nan
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        opt = MonteCarloOptimiser(
            n_trials=100, min_securities=2, max_securities=5,
            num_processes=1, seed=42,
        )
        result = opt.optimise(prices)
        self.assertTrue(np.all(np.isfinite(result.weights)),
                        "output weights contain NaN or inf")


class TestResultIntegrity(unittest.TestCase):
    """Run assert_result_integrity on all 4 optimisers."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _make_synthetic_prices(n_days=200, n_tickers=10, seed=42)

    def test_mip_integrity(self):
        from src.optimisers.mip import MIPOptimiser
        result = MIPOptimiser(max_securities=5).optimise(self.prices)
        assert_result_integrity(self, result, self.prices)

    def test_monte_carlo_integrity(self):
        from src.optimisers.monte_carlo import MonteCarloOptimiser
        result = MonteCarloOptimiser(
            n_trials=200, min_securities=2, max_securities=5,
            num_processes=1,
        ).optimise(self.prices)
        assert_result_integrity(self, result, self.prices)

    def test_pygad_integrity(self):
        from src.optimisers.pygad_ga import PygadOptimiser
        result = PygadOptimiser(
            num_children=30, num_generations=2,
            min_securities=2, max_securities=5,
            min_weight=0.0, max_weight=1.0,
            target_return=None, use_forecasts=False,
        ).optimise(self.prices)
        assert_result_integrity(self, result, self.prices)

    def test_island_ga_integrity(self):
        if multiprocessing.get_start_method() == 'spawn':
            self.skipTest("IslandGA incompatible with 'spawn' start method")
        from src.optimisers.island_ga import IslandGAOptimiser
        result = IslandGAOptimiser(
            num_generations=3, population_size=100, num_elites=10,
            min_securities=2, max_securities=5, min_return=None,
        ).optimise(self.prices)
        assert_result_integrity(self, result, self.prices)


if __name__ == '__main__':
    unittest.main()
