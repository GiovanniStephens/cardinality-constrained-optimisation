import unittest
import os
import tempfile

import numpy as np
import pandas as pd

from src.optimisers.pygad_ga import PygadOptimiser, estimate_corr_using_copulas
from src.portfolio_utils import (
    load_prices_csv, calculate_log_returns, negative_sharpe_ratio,
)
from src.config import (
    GA_MIN_SECURITIES, GA_MAX_SECURITIES, TRADING_DAYS_PER_YEAR,
)
from tests import requires_integration


class TestDataLoading(unittest.TestCase):
    """Unit tests for data loading with synthetic data."""

    def test_load_data_no_file(self):
        with self.assertRaises(FileNotFoundError):
            load_prices_csv('data/ETF_Prices_missing.csv', min_coverage=0.10)

    def test_load_data_empty_csv(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                          delete=False) as f:
            f.write('')
            temp_path = f.name
        try:
            with self.assertRaises((ValueError, pd.errors.EmptyDataError)):
                data = load_prices_csv(temp_path, min_coverage=0.10)
                if data.empty:
                    raise ValueError("Empty CSV")
        finally:
            os.unlink(temp_path)

    def test_calculate_returns_dummy_data(self):
        data = pd.DataFrame([100, 150, 100], columns=['TEST'])
        log_returns = calculate_log_returns(data)
        self.assertAlmostEqual(sum(log_returns['TEST']), 0)

    def test_calculate_returns_dummy_data_first_value(self):
        data = pd.DataFrame([100, 150, 100], columns=['TEST'])
        log_returns = calculate_log_returns(data)
        self.assertEqual(log_returns.iloc[0, 0], 0)

    def test_calculate_returns_handles_zeros(self):
        data = pd.DataFrame({'A': [100, 0, 50]})
        log_returns = calculate_log_returns(data)
        self.assertFalse(np.any(np.isinf(log_returns.values)))


@requires_integration
class TestDataLoadingCSV(unittest.TestCase):
    """Integration tests that load real CSV data."""

    def test_load_data(self):
        data = load_prices_csv('data/ETF_Prices.csv', min_coverage=0.10)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(data.shape[0], 0)
        self.assertGreater(data.shape[1], 0)

    def test_calculate_returns(self):
        data = load_prices_csv('data/ETF_Prices.csv', min_coverage=0.10)
        log_returns = calculate_log_returns(data)
        self.assertEqual(log_returns.shape, data.shape)

    def test_calculate_returns_first_value(self):
        data = load_prices_csv('data/ETF_Prices.csv', min_coverage=0.10)
        log_returns = calculate_log_returns(data)
        self.assertAlmostEqual(log_returns.iloc[0, 0], 0.0)


class TestSharpeRatio(unittest.TestCase):
    """Tests for Sharpe ratio computation."""

    def test_sharpe_ratio_known_value(self):
        weights = np.array([0.5, 0.5])
        returns = np.array([0.2, 0.3])
        corr = 0.5
        std_devs = [0.1, 0.2]
        cov = corr * std_devs[0] * std_devs[1]
        cov_matrix = np.array([
            [std_devs[0]**2, cov],
            [cov, std_devs[1]**2],
        ])
        # Expected: port_return=0.25, port_var=0.0175, sharpe≈1.8898
        expected_sharpe = 0.25 / np.sqrt(0.0175)
        model_sharpe = negative_sharpe_ratio(weights, returns, cov_matrix)
        self.assertAlmostEqual(-model_sharpe, expected_sharpe, places=4)

    def test_sharpe_ratio_zero_volatility(self):
        weights = np.array([0.0, 0.0])
        returns = np.array([0.2, 0.3])
        cov_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = negative_sharpe_ratio(weights, returns, cov_matrix)
        self.assertEqual(result, 0.0)


@requires_integration
class TestPygadOptimiser(unittest.TestCase):
    """Tests for PygadOptimiser methods using real data."""

    @classmethod
    def setUpClass(cls):
        cls.prices = load_prices_csv('data/ETF_Prices.csv', min_coverage=0.10)
        cls.log_returns = calculate_log_returns(cls.prices)
        cls.optimiser = PygadOptimiser(
            target_return=None, use_forecasts=False,
        )
        cls.optimiser._prepare_inputs(cls.prices)

    def test_prepare_inputs_data_length(self):
        self.assertEqual(len(self.optimiser._data), len(self.log_returns.T))

    def test_prepare_inputs_no_forecasts_variances_none(self):
        self.assertIsNone(self.optimiser._variances)

    def test_prepare_inputs_expected_returns_present(self):
        self.assertIsNotNone(self.optimiser._expected_returns)
        self.assertGreater(len(self.optimiser._expected_returns), 0)

    def test_prepare_inputs_with_forecasts(self):
        opt = PygadOptimiser(target_return=None, use_forecasts=True)
        opt._prepare_inputs(self.prices)
        self.assertIsNotNone(opt._expected_returns)
        self.assertGreater(len(opt._expected_returns), 0)

    def test_prepare_inputs_none_prices(self):
        opt = PygadOptimiser()
        with self.assertRaises(ValueError):
            opt._prepare_inputs(None)

    def test_prepare_inputs_empty_prices(self):
        opt = PygadOptimiser()
        with self.assertRaises(ValueError):
            opt._prepare_inputs(pd.DataFrame())

    def test_optimize_weights_max_weight(self):
        num_stocks = 5
        max_weight = 0.3
        initial_weights = [1 / num_stocks] * num_stocks
        ret_data = self.log_returns.iloc[:, :num_stocks]
        sol = self.optimiser._optimize_weights(
            ret_data, initial_weights, max_weight=max_weight)
        self.assertLessEqual(max(sol['x']), max_weight + 1e-10)

    def test_optimize_weights_min_weight(self):
        num_stocks = 5
        initial_weights = [1 / num_stocks] * num_stocks
        ret_data = self.log_returns.iloc[:, :num_stocks]
        sol = self.optimiser._optimize_weights(ret_data, initial_weights)
        self.assertGreaterEqual(min(sol['x']), -1e-10)

    def test_optimize_weights_mismatched_length(self):
        ret_data = self.log_returns.iloc[:, :5]
        with self.assertRaises((ValueError, KeyError)):
            self.optimiser._optimize_weights(ret_data, [0.5, 0.5])

    def test_optimize_weights_risk_constraint(self):
        num_stocks = 30
        target_risk = 0.15
        opt = PygadOptimiser(
            target_return=None, target_risk=target_risk,
            max_weight=2, min_weight=-2, use_forecasts=False,
        )
        opt._prepare_inputs(self.prices)
        initial_weights = [1 / num_stocks] * num_stocks
        ret_data = self.log_returns.iloc[:, :num_stocks]
        sol = opt._optimize_weights(ret_data, initial_weights)
        weights = sol['x']
        cov = opt._get_cov_matrix(ret_data)
        risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        self.assertAlmostEqual(risk, target_risk, places=5)

    def test_optimize_weights_return_constraint(self):
        num_stocks = 30
        target_return = 0.15
        opt = PygadOptimiser(
            target_return=target_return, target_risk=None,
            max_weight=2, min_weight=-2, use_forecasts=False,
        )
        opt._prepare_inputs(self.prices)
        initial_weights = [1 / num_stocks] * num_stocks
        ret_data = self.log_returns.iloc[:, :num_stocks]
        sol = opt._optimize_weights(ret_data, initial_weights)
        weights = sol['x']
        returns = self.log_returns.iloc[:, :num_stocks].mean() * TRADING_DAYS_PER_YEAR
        portfolio_return = np.dot(weights, returns)
        self.assertAlmostEqual(portfolio_return, target_return)

    def test_create_individual(self):
        individual = self.optimiser._create_individual()
        num_ones = np.count_nonzero(individual)
        self.assertGreaterEqual(num_ones, 1)

    def test_get_cov_matrix(self):
        subset = self.log_returns.iloc[:, :2]
        expected = subset.cov() * TRADING_DAYS_PER_YEAR
        cov_matrix = self.optimiser._get_cov_matrix(subset)
        np.testing.assert_array_almost_equal(cov_matrix, expected.values)

    def test_fitness_too_many_etfs(self):
        fitness_fn = self.optimiser._make_fitness_fn()
        individual = [1] * len(self.optimiser._data)
        result = fitness_fn(None, individual, 0)
        self.assertLess(result, 0)

    def test_fitness_too_few_etfs(self):
        fitness_fn = self.optimiser._make_fitness_fn()
        individual = [0] * len(self.optimiser._data)
        result = fitness_fn(None, individual, 0)
        self.assertLessEqual(result, 0)

    def test_fitness_normal(self):
        fitness_fn = self.optimiser._make_fitness_fn()
        num_stocks = 8
        individual = [1] * num_stocks + [0] * (len(self.optimiser._data) - num_stocks)
        result = fitness_fn(None, individual, 0)
        self.assertGreater(result, 0)
        self.assertLess(result, 5)


@requires_integration
class TestCopulaCorrelation(unittest.TestCase):
    """Tests for copula-based correlation estimation."""

    @classmethod
    def setUpClass(cls):
        prices = load_prices_csv('data/ETF_Prices.csv', min_coverage=0.10)
        cls.log_returns = calculate_log_returns(prices)

    def test_estimate_corr_is_correlation_matrix(self):
        subset = self.log_returns.iloc[:, :5]
        corr = estimate_corr_using_copulas(subset)
        self.assertEqual(corr.shape, (5, 5))
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(5))
        np.testing.assert_array_almost_equal(corr, corr.T)
        eigenvalues = np.linalg.eigvalsh(corr)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_get_cov_matrix_with_copulae(self):
        optimiser = PygadOptimiser(
            target_return=None, use_forecasts=False,
        )
        optimiser._prepare_inputs(
            load_prices_csv('data/ETF_Prices.csv', min_coverage=0.10))
        subset = self.log_returns.iloc[:, :5]
        cov_matrix = optimiser._get_cov_matrix(subset, use_copulae=True)
        self.assertEqual(cov_matrix.shape, (5, 5))
        np.testing.assert_array_almost_equal(cov_matrix, cov_matrix.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        self.assertTrue(np.all(eigenvalues >= -1e-10))
        self.assertTrue(np.all(np.diag(cov_matrix) > 0))

    def test_get_cov_matrix_returns_numpy(self):
        optimiser = PygadOptimiser(
            target_return=None, use_forecasts=False,
        )
        optimiser._prepare_inputs(
            load_prices_csv('data/ETF_Prices.csv', min_coverage=0.10))
        subset = self.log_returns.iloc[:, :3]
        cov_matrix = optimiser._get_cov_matrix(subset, use_copulae=True)
        self.assertIsInstance(cov_matrix, np.ndarray)


if __name__ == '__main__':
    unittest.main()
