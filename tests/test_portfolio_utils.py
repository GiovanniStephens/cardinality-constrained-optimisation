"""Tests for portfolio_utils shared functions."""

import unittest

import numpy as np
import pandas as pd

from src.portfolio_utils import (
    load_prices_csv,
    calculate_log_returns,
    calculate_covariance_matrix,
    calculate_expected_returns,
    calculate_variances,
    sharpe_ratio,
    negative_sharpe_ratio,
)


class TestLoadPricesCsv(unittest.TestCase):
    def test_returns_dataframe(self):
        df = load_prices_csv('Data/ETF_Prices.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.shape[0], 0)
        self.assertGreater(df.shape[1], 0)

    def test_min_coverage_filters_columns(self):
        strict = load_prices_csv('Data/ETF_Prices.csv', min_coverage=0.99)
        lenient = load_prices_csv('Data/ETF_Prices.csv', min_coverage=0.50)
        self.assertGreaterEqual(lenient.shape[1], strict.shape[1])

    def test_last_n_days(self):
        full = load_prices_csv('Data/time_series_20251016_113257.csv')
        recent = load_prices_csv('Data/time_series_20251016_113257.csv', last_n_days=365)
        self.assertLess(recent.shape[0], full.shape[0])


class TestCalculateLogReturns(unittest.TestCase):
    def test_shape_preserved(self):
        prices = pd.DataFrame({'A': [100, 110, 121], 'B': [50, 55, 60]})
        returns = calculate_log_returns(prices)
        self.assertEqual(returns.shape, prices.shape)

    def test_first_row_zero(self):
        prices = pd.DataFrame({'A': [100, 110, 121]})
        returns = calculate_log_returns(prices)
        self.assertEqual(returns.iloc[0, 0], 0.0)

    def test_no_nans_or_infs(self):
        prices = pd.DataFrame({'A': [100, 0, 110], 'B': [50, 50, 0]})
        returns = calculate_log_returns(prices)
        self.assertFalse(returns.isna().any().any())
        self.assertFalse(np.isinf(returns.values).any())

    def test_known_values(self):
        prices = pd.DataFrame({'A': [100.0, 200.0]})
        returns = calculate_log_returns(prices)
        self.assertAlmostEqual(returns.iloc[1, 0], np.log(2), places=10)


class TestCovarianceMatrix(unittest.TestCase):
    def test_square_symmetric(self):
        returns = pd.DataFrame(np.random.randn(100, 5))
        cov = calculate_covariance_matrix(returns)
        self.assertEqual(cov.shape, (5, 5))
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_annualisation(self):
        returns = pd.DataFrame(np.random.randn(100, 3))
        ann = calculate_covariance_matrix(returns, annualise=True)
        raw = calculate_covariance_matrix(returns, annualise=False)
        np.testing.assert_array_almost_equal(ann.values, raw.values * 252)


class TestExpectedReturns(unittest.TestCase):
    def test_returns_series(self):
        returns = pd.DataFrame(np.random.randn(100, 4))
        er = calculate_expected_returns(returns)
        self.assertIsInstance(er, pd.Series)
        self.assertEqual(len(er), 4)

    def test_annualisation(self):
        returns = pd.DataFrame(np.random.randn(100, 3))
        ann = calculate_expected_returns(returns, annualise=True)
        raw = calculate_expected_returns(returns, annualise=False)
        np.testing.assert_array_almost_equal(ann.values, raw.values * 252)


class TestVariances(unittest.TestCase):
    def test_positive(self):
        returns = pd.DataFrame(np.random.randn(100, 3))
        var = calculate_variances(returns)
        self.assertTrue((var > 0).all())

    def test_annualisation(self):
        returns = pd.DataFrame(np.random.randn(100, 3))
        ann = calculate_variances(returns, annualise=True)
        raw = calculate_variances(returns, annualise=False)
        np.testing.assert_array_almost_equal(ann.values, raw.values * 252)


class TestSharpeRatio(unittest.TestCase):
    def test_positive_return_positive_sharpe(self):
        weights = np.array([0.5, 0.5])
        er = np.array([0.10, 0.12])
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        sr = sharpe_ratio(weights, er, cov)
        self.assertGreater(sr, 0)

    def test_negative_sharpe_negates(self):
        weights = np.array([0.5, 0.5])
        er = np.array([0.10, 0.12])
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        sr = sharpe_ratio(weights, er, cov)
        nsr = negative_sharpe_ratio(weights, er, cov)
        self.assertAlmostEqual(sr, -nsr)

    def test_zero_volatility(self):
        weights = np.array([1.0])
        er = np.array([0.10])
        cov = np.array([[0.0]])
        sr = sharpe_ratio(weights, er, cov)
        self.assertEqual(sr, 0.0)


if __name__ == '__main__':
    unittest.main()
