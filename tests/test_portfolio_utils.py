"""Tests for portfolio_utils shared functions."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from unittest.mock import MagicMock

from tests import requires_integration
from src.portfolio_utils import (
    load_prices_csv,
    calculate_log_returns,
    calculate_covariance_matrix,
    calculate_expected_returns,
    calculate_variances,
    sharpe_ratio,
    negative_sharpe_ratio,
    maximum_drawdown,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
    optimise_weights,
    sharpe_ratio_variance,
    deflated_sharpe_ratio,
    warn_if_sharpe_suspicious,
    SHARPE_WARN_THRESHOLD,
    SHARPE_CRITICAL_THRESHOLD,
)


@requires_integration
class TestLoadPricesCsvIntegration(unittest.TestCase):
    """Integration tests that load real CSV price data."""

    def test_returns_dataframe(self):
        df = load_prices_csv('data/ETF_Prices.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.shape[0], 0)
        self.assertGreater(df.shape[1], 0)

    def test_min_coverage_filters_columns(self):
        strict = load_prices_csv('data/ETF_Prices.csv', min_coverage=0.99)
        lenient = load_prices_csv('data/ETF_Prices.csv', min_coverage=0.50)
        self.assertGreaterEqual(lenient.shape[1], strict.shape[1])

    def test_last_n_days(self):
        full = load_prices_csv('data/time_series_20251016_113257.csv')
        recent = load_prices_csv('data/time_series_20251016_113257.csv', last_n_days=365)
        self.assertLess(recent.shape[0], full.shape[0])


class TestLoadPricesCsv(unittest.TestCase):
    def test_headers_only_returns_empty_dataframe(self):
        """
        A CSV with column headers but no data rows returns an empty DataFrame
        without raising.
        """
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f:
            f.write('Date,SPY,QQQ\n')
            path = f.name
        try:
            result = load_prices_csv(path)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 0)
        finally:
            os.unlink(path)


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

    def test_all_nan_column_becomes_zero(self):
        """
        A column that is entirely NaN is silently converted to all-zero returns
        by the fillna(0) guard. No exception should be raised.
        """
        prices = pd.DataFrame({
            'A': [100.0, 110.0, 121.0],
            'B': [float('nan'), float('nan'), float('nan')],
        })
        returns = calculate_log_returns(prices)
        self.assertFalse(returns.isna().any().any())
        self.assertTrue((returns['B'] == 0.0).all())


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

    def test_mismatched_dimensions_raises(self):
        """
        weights (2,) vs expected_returns (3,) causes a numpy broadcast error.
        No guard exists in sharpe_ratio(), so this crashes.
        """
        weights = np.array([0.5, 0.5])
        er = np.array([0.10, 0.12, 0.08])
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        with self.assertRaises((ValueError, IndexError)):
            sharpe_ratio(weights, er, cov)


class TestMaximumDrawdown(unittest.TestCase):
    def test_known_values(self):
        # Up 1%, down 5%, up 2%: peak at day 1, trough at day 2
        returns = [0.01, -0.05, 0.02]
        dd = maximum_drawdown(returns)
        self.assertLess(dd, 0)

    def test_all_positive_returns(self):
        returns = [0.01, 0.02, 0.01]
        dd = maximum_drawdown(returns)
        self.assertEqual(dd, 0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            maximum_drawdown([])


class TestDownsideDeviation(unittest.TestCase):
    def test_no_downside(self):
        returns = [0.01, 0.02, 0.03]
        dd = downside_deviation(returns, mar=0)
        self.assertEqual(dd, 0.0)

    def test_known_values(self):
        returns = [-0.02, -0.01, 0.01]
        dd = downside_deviation(returns, mar=0)
        self.assertGreater(dd, 0)

    def test_empty_returns_zero(self):
        self.assertEqual(downside_deviation([]), 0.0)


class TestSortinoRatio(unittest.TestCase):
    def test_known_values(self):
        result = sortino_ratio(0.10, 0.05)
        self.assertAlmostEqual(result, 2.0)

    def test_zero_deviation_returns_zero(self):
        self.assertEqual(sortino_ratio(0.10, 0), 0.0)


class TestCalmarRatio(unittest.TestCase):
    def test_known_values(self):
        result = calmar_ratio(0.10, -0.20)
        self.assertAlmostEqual(result, 0.5)

    def test_zero_drawdown_returns_zero(self):
        self.assertEqual(calmar_ratio(0.10, 0), 0.0)


class TestOptimiseWeights(unittest.TestCase):
    def test_returns_optimize_result(self):
        from scipy.optimize import OptimizeResult
        prices = pd.DataFrame(
            np.random.RandomState(42).randn(100, 4).cumsum(axis=0) + 100,
            columns=['A', 'B', 'C', 'D'],
        )
        selection = np.array([1, 1, 0, 1])
        result = optimise_weights(selection, prices)
        self.assertIsInstance(result, OptimizeResult)

    def test_weights_sum_to_one(self):
        prices = pd.DataFrame(
            np.random.RandomState(42).randn(100, 3).cumsum(axis=0) + 100,
            columns=['X', 'Y', 'Z'],
        )
        selection = np.array([1, 1, 1])
        result = optimise_weights(selection, prices)
        self.assertTrue(result.success, f"Optimisation failed: {result.message}")
        self.assertAlmostEqual(sum(result.x), 1.0, places=4)


class TestSharpeRatioVariance(unittest.TestCase):
    def test_normal_returns_sr_zero(self):
        """For normal returns and SR=0, Var(SR) = 1/n."""
        var = sharpe_ratio_variance(sr=0.0, n=252)
        self.assertAlmostEqual(var, 1 / 252)

    def test_normal_returns_sr_nonzero(self):
        """For normal returns, Var(SR) = (1 + sr^2/2) / n via the general formula
        with skewness=0 and excess_kurtosis=0 the formula gives 1/n (no sr^2 term
        because excess_kurtosis=0). The (1+sr^2/2)/n form requires kurtosis=3."""
        var = sharpe_ratio_variance(sr=1.0, n=252, skewness=0.0, excess_kurtosis=0.0)
        self.assertAlmostEqual(var, 1 / 252)

    def test_skewness_changes_result(self):
        """Negative skewness with positive SR should increase variance."""
        var_normal = sharpe_ratio_variance(sr=1.0, n=252, skewness=0.0)
        var_skewed = sharpe_ratio_variance(sr=1.0, n=252, skewness=-1.0)
        self.assertGreater(var_skewed, var_normal)

    def test_excess_kurtosis_changes_result(self):
        """Positive excess kurtosis should increase variance for nonzero SR."""
        var_normal = sharpe_ratio_variance(sr=1.0, n=252, excess_kurtosis=0.0)
        var_fat = sharpe_ratio_variance(sr=1.0, n=252, excess_kurtosis=3.0)
        self.assertGreater(var_fat, var_normal)

    def test_more_observations_reduces_variance(self):
        var_short = sharpe_ratio_variance(sr=1.0, n=100)
        var_long = sharpe_ratio_variance(sr=1.0, n=1000)
        self.assertGreater(var_short, var_long)


class TestDeflatedSharpeRatio(unittest.TestCase):
    def test_range_zero_to_one(self):
        dsr = deflated_sharpe_ratio(observed_sr=1.5, n=252, num_trials=100)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)

    def test_more_trials_lowers_dsr(self):
        """More trials increases expected max SR under null, lowering DSR.
        Uses few observations (n=50) so SR_std is large enough for the
        multiple testing penalty to bite."""
        dsr_few = deflated_sharpe_ratio(observed_sr=0.5, n=50, num_trials=10)
        dsr_many = deflated_sharpe_ratio(observed_sr=0.5, n=50, num_trials=100000)
        self.assertGreater(dsr_few, dsr_many)

    def test_higher_sr_raises_dsr(self):
        dsr_low = deflated_sharpe_ratio(observed_sr=0.5, n=252, num_trials=100)
        dsr_high = deflated_sharpe_ratio(observed_sr=3.0, n=252, num_trials=100)
        self.assertGreater(dsr_high, dsr_low)

    def test_single_trial_returns_high_dsr(self):
        """With only 1 trial, there is no multiple testing penalty."""
        dsr = deflated_sharpe_ratio(observed_sr=1.0, n=252, num_trials=1)
        self.assertGreater(dsr, 0.5)

    def test_massive_trials_low_sr_gives_low_dsr(self):
        """A modest SR with huge trial count and few observations should
        produce a low DSR since expected max under null exceeds observed."""
        dsr = deflated_sharpe_ratio(observed_sr=0.3, n=50, num_trials=100000)
        self.assertLess(dsr, 0.5)


class TestEqualWeightSharpe(unittest.TestCase):
    """Tests for the shared equal_weight_sharpe() fitness function."""

    def setUp(self):
        np.random.seed(42)
        n = 10
        # Synthetic: positive expected returns, diagonal covariance
        self.expected_returns = np.array([0.05 + 0.02 * i for i in range(n)])
        self.cov_matrix = np.diag([0.04] * n)  # 20% vol each, uncorrelated

    def test_normal_selection(self):
        from src.portfolio_utils import equal_weight_sharpe
        sel = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        result = equal_weight_sharpe(sel, self.expected_returns,
                                     self.cov_matrix, 3, 8)
        self.assertGreater(result, 0)
        self.assertTrue(np.isfinite(result))

    def test_too_few_selected(self):
        from src.portfolio_utils import equal_weight_sharpe
        sel = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        result = equal_weight_sharpe(sel, self.expected_returns,
                                     self.cov_matrix, 3, 8)
        self.assertEqual(result, -1e4)

    def test_too_many_selected(self):
        from src.portfolio_utils import equal_weight_sharpe
        sel = np.ones(10, dtype=bool)
        result = equal_weight_sharpe(sel, self.expected_returns,
                                     self.cov_matrix, 3, 8)
        self.assertEqual(result, -1e4)

    def test_none_selected(self):
        from src.portfolio_utils import equal_weight_sharpe
        sel = np.zeros(10, dtype=bool)
        result = equal_weight_sharpe(sel, self.expected_returns,
                                     self.cov_matrix, 0, 8)
        self.assertEqual(result, 0.0)

    def test_zero_variance_returns_zero(self):
        from src.portfolio_utils import equal_weight_sharpe
        zero_cov = np.zeros((10, 10))
        sel = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        result = equal_weight_sharpe(sel, self.expected_returns,
                                     zero_cov, 3, 8)
        self.assertEqual(result, 0.0)

    def test_matches_manual_sharpe(self):
        from src.portfolio_utils import equal_weight_sharpe
        sel = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        result = equal_weight_sharpe(sel, self.expected_returns,
                                     self.cov_matrix, 2, 8)
        # Manual: weights=[0.5, 0.5], returns=[0.05, 0.07], cov=diag(0.04)
        w = np.array([0.5, 0.5])
        ret = np.dot(w, [0.05, 0.07])  # 0.06
        var = np.dot(w, np.dot(np.diag([0.04, 0.04]), w))  # 0.02
        expected = ret / np.sqrt(var)
        self.assertAlmostEqual(result, expected, places=10)


class TestWarnIfSharpeSuspicious(unittest.TestCase):
    def test_no_warning_below_threshold(self):
        mock_logger = MagicMock()
        warn_if_sharpe_suspicious(1.5, "test", logger=mock_logger)
        mock_logger.warning.assert_not_called()

    def test_warn_above_warn_threshold(self):
        mock_logger = MagicMock()
        warn_if_sharpe_suspicious(2.5, "test", logger=mock_logger)
        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        self.assertIn("30-50%", msg)

    def test_critical_above_critical_threshold(self):
        mock_logger = MagicMock()
        warn_if_sharpe_suspicious(4.0, "test", logger=mock_logger)
        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        self.assertIn("Harvey", msg)

    def test_uses_module_logger_by_default(self):
        """Should not raise when no logger is passed."""
        warn_if_sharpe_suspicious(1.0, "test")


if __name__ == '__main__':
    unittest.main()
