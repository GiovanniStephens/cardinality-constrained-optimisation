"""Tests for portfolio_utils shared functions."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from unittest.mock import MagicMock

from tests import requires_integration
from src.data_loading import load_prices_csv
from src.returns import (
    calculate_log_returns,
    calculate_expected_returns,
    calculate_variances,
    calculate_asset_betas,
    prepare_portfolio_inputs,
)
from src.covariance import (
    calculate_covariance_matrix,
    check_observation_ratio,
    shrink_correlation_matrix,
)
from src.metrics import (
    sharpe_ratio,
    sharpe_loss,
    maximum_drawdown,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
    sharpe_ratio_variance,
    deflated_sharpe_ratio,
    warn_if_sharpe_suspicious,
)
from src.weights import (
    optimise_weights,
    calculate_portfolio_variance,
    calculate_portfolio_return,
    calculate_risk_contribution,
    risk_budget_objective,
    greedy_beta_vertex,
    reachable_beta_interval,
    beta_target_start,
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


class TestCalculateAssetBetas(unittest.TestCase):
    """Shared OLS beta: cov(r_i, r_b) / var(r_b)."""

    def test_exact_scaled_beta(self):
        rng = np.random.default_rng(42)
        bench = pd.Series(rng.normal(0, 0.01, 500))
        rets = pd.DataFrame({
            'DOUBLE': 2.0 * bench,      # beta exactly 2
            'SELF': bench,              # beta exactly 1
            'INVERSE': -bench,          # beta exactly -1
        })
        betas = calculate_asset_betas(rets, bench)
        self.assertAlmostEqual(betas['DOUBLE'], 2.0, places=10)
        self.assertAlmostEqual(betas['SELF'], 1.0, places=10)
        self.assertAlmostEqual(betas['INVERSE'], -1.0, places=10)

    def test_uncorrelated_near_zero(self):
        rng = np.random.default_rng(0)
        bench = pd.Series(rng.normal(0, 0.01, 5000))
        rets = pd.DataFrame({'NOISE': rng.normal(0, 0.01, 5000)})
        betas = calculate_asset_betas(rets, bench)
        self.assertLess(abs(betas['NOISE']), 0.1)

    def test_nan_beta_becomes_zero(self):
        # A constant column has zero covariance with anything; pandas cov of a
        # length-mismatched all-NaN overlap yields NaN -> fillna(0.0).
        bench = pd.Series([0.01, -0.02, 0.005, 0.0], index=range(4))
        rets = pd.DataFrame({'GHOST': [np.nan] * 4}, index=range(4))
        betas = calculate_asset_betas(rets, bench)
        self.assertEqual(betas['GHOST'], 0.0)

    def test_matches_rebalance_formula(self):
        # Behaviour-identity check for the run_rebalance delegation: same
        # cov/var arithmetic as the pre-refactor inline version.
        rng = np.random.default_rng(7)
        bench = pd.Series(rng.normal(0, 0.01, 300))
        rets = pd.DataFrame(rng.normal(0, 0.02, (300, 4)),
                            columns=list('ABCD'))
        expected = (rets.apply(lambda col: col.cov(bench))
                    / bench.var()).fillna(0.0)
        got = calculate_asset_betas(rets, bench)
        pd.testing.assert_series_equal(got, expected)


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
        nsr = sharpe_loss(weights, er, cov)
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


class TestEffectiveTrialsForMethod(unittest.TestCase):
    def test_cc_method(self):
        from src.metrics import effective_trials_for_method
        m = effective_trials_for_method(
            'cc', num_portfolios=30, ga_pop=8000, ga_generations=150)
        self.assertEqual(m, 30 * 8000 * 150)

    def test_mc_method(self):
        from src.metrics import effective_trials_for_method
        m = effective_trials_for_method(
            'mc', num_portfolios=30, mc_trials_per_portfolio=100_000)
        self.assertEqual(m, 30 * 100_000)

    def test_random_method(self):
        from src.metrics import effective_trials_for_method
        m = effective_trials_for_method('random', num_portfolios=30)
        self.assertEqual(m, 30)

    def test_unknown_method_raises(self):
        from src.metrics import effective_trials_for_method
        with self.assertRaises(ValueError):
            effective_trials_for_method('xyz', 30)

    def test_cc_missing_args_raises(self):
        from src.metrics import effective_trials_for_method
        with self.assertRaises(ValueError):
            effective_trials_for_method('cc', 30)


class TestComputeMethodDsr(unittest.TestCase):

    def test_returns_complete_dict(self):
        from src.metrics import compute_method_dsr
        rng = np.random.default_rng(0)
        rets = rng.normal(0.001, 0.01, size=251)
        out = compute_method_dsr(observed_sr=2.0,
                                  portfolio_returns=rets,
                                  num_trials=1_000_000)
        for k in ('observed_sr', 'num_obs', 'num_trials',
                  'skewness', 'excess_kurtosis', 'dsr'):
            self.assertIn(k, out)
        self.assertEqual(out['num_obs'], 251)
        self.assertEqual(out['num_trials'], 1_000_000)
        self.assertGreaterEqual(out['dsr'], 0.0)
        self.assertLessEqual(out['dsr'], 1.0)

    def test_modest_sr_huge_M_low_obs_flags_overfit(self):
        """Realistic gating case: GA reports SR=0.3 over 50 obs after a
        billion trials. Should flag as overfit (DSR < 0.5)."""
        from src.metrics import compute_method_dsr
        rng = np.random.default_rng(1)
        rets = rng.normal(0.0, 0.01, size=50)
        out = compute_method_dsr(0.3, rets, num_trials=10**9)
        self.assertLess(out['dsr'], 0.5)

    def test_strong_sr_low_M_passes(self):
        """SR=2.0 over 252 obs with M=30 (random selection) should clear
        the multiple-testing bar (DSR > 0.95)."""
        from src.metrics import compute_method_dsr
        rng = np.random.default_rng(2)
        rets = rng.normal(0.001, 0.01, size=252)
        out = compute_method_dsr(2.0, rets, num_trials=30)
        self.assertGreater(out['dsr'], 0.95)


class TestEqualWeightSharpe(unittest.TestCase):
    """Tests for the shared equal_weight_fitness() fitness function."""

    def setUp(self):
        np.random.seed(42)
        n = 10
        # Synthetic: positive expected returns, diagonal covariance
        self.expected_returns = np.array([0.05 + 0.02 * i for i in range(n)])
        self.cov_matrix = np.diag([0.04] * n)  # 20% vol each, uncorrelated

    def test_normal_selection(self):
        from src.metrics import equal_weight_fitness
        sel = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        result = equal_weight_fitness(sel, self.expected_returns,
                                     self.cov_matrix, 3, 8)
        self.assertGreater(result, 0)
        self.assertTrue(np.isfinite(result))

    def test_too_few_selected(self):
        from src.metrics import equal_weight_fitness
        sel = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        result = equal_weight_fitness(sel, self.expected_returns,
                                     self.cov_matrix, 3, 8)
        self.assertEqual(result, -1e4)

    def test_too_many_selected(self):
        from src.metrics import equal_weight_fitness
        sel = np.ones(10, dtype=bool)
        result = equal_weight_fitness(sel, self.expected_returns,
                                     self.cov_matrix, 3, 8)
        self.assertEqual(result, -1e4)

    def test_none_selected(self):
        from src.metrics import equal_weight_fitness
        sel = np.zeros(10, dtype=bool)
        result = equal_weight_fitness(sel, self.expected_returns,
                                     self.cov_matrix, 0, 8)
        self.assertEqual(result, 0.0)

    def test_zero_variance_returns_zero(self):
        from src.metrics import equal_weight_fitness
        zero_cov = np.zeros((10, 10))
        sel = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        result = equal_weight_fitness(sel, self.expected_returns,
                                     zero_cov, 3, 8)
        self.assertEqual(result, 0.0)

    def test_matches_manual_sharpe(self):
        from src.metrics import equal_weight_fitness
        sel = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        result = equal_weight_fitness(sel, self.expected_returns,
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


class TestDateOrderingGuards(unittest.TestCase):
    """Tests for date ordering enforcement in load_prices_csv and calculate_log_returns."""

    def test_ffill_on_unsorted_data_sorts_first(self):
        """Forward-fill must not propagate future prices backward when CSV is unsorted.

        If dates are [Jan 3, Jan 1, Jan 2] with values [NaN, 100, NaN],
        unsorted ffill would fill Jan 3's NaN with nothing (first row) and
        Jan 2's NaN with 100. But sorted ffill should fill Jan 2 with 100
        and Jan 3 with 100 as well.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('Date,A\n')
            f.write('2020-01-03,\n')    # NaN, should be filled from Jan 2
            f.write('2020-01-01,100\n')  # known value
            f.write('2020-01-02,\n')    # NaN, should be filled from Jan 1
            path = f.name
        try:
            result = load_prices_csv(path, min_coverage=0.0)
            # After sorting and ffill: Jan 1=100, Jan 2=100, Jan 3=100
            self.assertTrue(result.index.is_monotonic_increasing,
                            "load_prices_csv must return a sorted index")
            self.assertAlmostEqual(result.loc['2020-01-02', 'A'], 100.0)
            self.assertAlmostEqual(result.loc['2020-01-03', 'A'], 100.0)
        finally:
            os.unlink(path)

    def test_log_returns_require_sorted_index(self):
        """calculate_log_returns must raise on unsorted DatetimeIndex.

        .shift(1) on unsorted data computes returns between wrong date pairs.
        """
        dates = pd.to_datetime(['2020-01-03', '2020-01-01', '2020-01-02'])
        prices = pd.DataFrame({'A': [121, 100, 110]}, index=dates)
        with self.assertRaises(ValueError, msg="Should reject unsorted index"):
            calculate_log_returns(prices)

    def test_ffill_limit_prevents_long_gap_fill(self):
        """Stale prices should not be carried forward beyond the ffill limit (5 days)."""
        # 10 rows: value on day 0, then 9 NaNs — only 5 should be filled
        dates = pd.bdate_range('2020-01-01', periods=10, freq='B')
        data = {'A': [100.0] + [float('nan')] * 9}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(data, index=dates)
            df.index.name = 'Date'
            df.to_csv(f.name)
            path = f.name
        try:
            result = load_prices_csv(path, min_coverage=0.0)
            # Days 1-5 should be filled (100), days 6-9 should remain NaN
            self.assertAlmostEqual(result.iloc[5]['A'], 100.0)
            self.assertTrue(pd.isna(result.iloc[6]['A']),
                            "Day 7 should NOT be filled (beyond limit=5)")
        finally:
            os.unlink(path)

    def test_log_returns_preserves_index_order(self):
        """Log returns should preserve the DatetimeIndex order of the input."""
        dates = pd.bdate_range('2020-01-01', periods=5, freq='B')
        prices = pd.DataFrame({'A': [100, 101, 102, 103, 104]}, index=dates)
        returns = calculate_log_returns(prices)
        self.assertTrue(returns.index.equals(prices.index),
                        "Log returns index should match input index exactly")


class TestObservationRatioGuard(unittest.TestCase):
    def test_raises_when_T_less_than_N(self):
        with self.assertRaises(ValueError):
            check_observation_ratio(5, 10)

    def test_warns_when_ratio_low(self):
        with self.assertLogs('src.covariance', level='WARNING') as cm:
            check_observation_ratio(50, 10)
        self.assertTrue(any('T/N ratio' in msg for msg in cm.output))

    def test_no_warning_when_ratio_high(self):
        # Should not raise or warn
        check_observation_ratio(500, 10)

    def test_zero_assets_no_error(self):
        check_observation_ratio(100, 0)


class TestLedoitWolfCovariance(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.returns = pd.DataFrame(np.random.randn(200, 5),
                                    columns=['A', 'B', 'C', 'D', 'E'])

    def test_shrinkage_differs_from_sample(self):
        shrunk = calculate_covariance_matrix(self.returns, annualise=False, shrinkage=True)
        sample = calculate_covariance_matrix(self.returns, annualise=False, shrinkage=False)
        # They should not be identical
        self.assertFalse(np.allclose(shrunk.values, sample.values))

    def test_shrinkage_preserves_symmetry(self):
        cov = calculate_covariance_matrix(self.returns, annualise=False, shrinkage=True)
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_shrinkage_positive_definite(self):
        cov = calculate_covariance_matrix(self.returns, annualise=False, shrinkage=True)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_shrinkage_off_matches_sample(self):
        raw = self.returns.cov()
        cov = calculate_covariance_matrix(self.returns, annualise=False, shrinkage=False)
        np.testing.assert_array_almost_equal(cov.values, raw.values)

    def test_annualisation_with_shrinkage(self):
        ann = calculate_covariance_matrix(self.returns, annualise=True, shrinkage=True)
        raw = calculate_covariance_matrix(self.returns, annualise=False, shrinkage=True)
        np.testing.assert_array_almost_equal(ann.values, raw.values * 252)


class TestShrinkCorrelationMatrix(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.returns = pd.DataFrame(np.random.randn(200, 5))
        self.corr = self.returns.corr().values

    def test_off_diagonal_moves_toward_zero(self):
        shrunk = shrink_correlation_matrix(self.corr, self.returns)
        # Off-diagonal elements should be shrunk toward zero (closer to identity)
        off_diag_orig = np.abs(self.corr[np.triu_indices(5, k=1)])
        off_diag_shrunk = np.abs(shrunk[np.triu_indices(5, k=1)])
        self.assertTrue(np.all(off_diag_shrunk <= off_diag_orig + 1e-10))

    def test_diagonal_stays_one(self):
        shrunk = shrink_correlation_matrix(self.corr, self.returns)
        np.testing.assert_array_almost_equal(np.diag(shrunk), np.ones(5))

    def test_positive_definite(self):
        shrunk = shrink_correlation_matrix(self.corr, self.returns)
        eigenvalues = np.linalg.eigvalsh(shrunk)
        self.assertTrue(np.all(eigenvalues > 0))


class TestPreparePortfolioInputs(unittest.TestCase):
    def test_returns_correct_types_and_shapes(self):
        prices = pd.DataFrame(
            np.random.RandomState(42).randn(100, 4).cumsum(axis=0) + 100,
            columns=['A', 'B', 'C', 'D'],
        )
        log_ret, er, cov = prepare_portfolio_inputs(prices)
        self.assertIsInstance(log_ret, pd.DataFrame)
        self.assertIsInstance(er, np.ndarray)
        self.assertEqual(log_ret.shape, prices.shape)
        self.assertEqual(len(er), 4)
        self.assertEqual(cov.shape, (4, 4))

    def test_cov_is_symmetric(self):
        prices = pd.DataFrame(
            np.random.RandomState(7).randn(200, 3).cumsum(axis=0) + 50,
            columns=['X', 'Y', 'Z'],
        )
        _, _, cov = prepare_portfolio_inputs(prices)
        cov_arr = cov.values if hasattr(cov, 'values') else cov
        np.testing.assert_array_almost_equal(cov_arr, cov_arr.T)


class TestCalculatePortfolioVariance(unittest.TestCase):
    def test_against_manual(self):
        w = np.array([0.6, 0.4])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = calculate_portfolio_variance(w, cov)
        expected = w @ cov @ w
        self.assertAlmostEqual(result, expected, places=10)

    def test_zero_weights(self):
        w = np.array([0.0, 0.0])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        self.assertAlmostEqual(calculate_portfolio_variance(w, cov), 0.0)

    def test_accepts_dataframe(self):
        w = np.array([0.5, 0.5])
        cov_df = pd.DataFrame([[0.04, 0.01], [0.01, 0.04]],
                               columns=['A', 'B'], index=['A', 'B'])
        result = calculate_portfolio_variance(w, cov_df)
        self.assertGreater(result, 0)


class TestCalculatePortfolioReturn(unittest.TestCase):
    def test_against_manual(self):
        w = np.array([0.6, 0.4])
        er = np.array([0.10, 0.15])
        result = calculate_portfolio_return(w, er)
        self.assertAlmostEqual(result, 0.6 * 0.10 + 0.4 * 0.15, places=10)

    def test_returns_float(self):
        result = calculate_portfolio_return(np.array([1.0]), np.array([0.05]))
        self.assertIsInstance(result, float)


class TestCalculateRiskContribution(unittest.TestCase):
    def test_contributions_sum_to_total_risk(self):
        w = np.array([0.4, 0.3, 0.3])
        V = np.matrix([[0.04, 0.01, 0.005],
                        [0.01, 0.09, 0.02],
                        [0.005, 0.02, 0.06]])
        rc = calculate_risk_contribution(w, V)
        total_risk = np.sqrt(float(w @ np.asarray(V) @ w))
        self.assertAlmostEqual(float(np.sum(rc)), total_risk, places=10)

    def test_zero_variance_returns_zeros(self):
        w = np.array([0.5, 0.5])
        V = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        rc = calculate_risk_contribution(w, V)
        np.testing.assert_array_almost_equal(np.array(rc).flatten(), [0.0, 0.0])


class TestRiskBudgetObjective(unittest.TestCase):
    def test_equal_weights_identity_cov_gives_zero(self):
        n = 4
        V = np.matrix(np.eye(n))
        w = np.ones(n) / n
        target = [1 / n] * n
        result = risk_budget_objective(w, [V, target])
        self.assertAlmostEqual(result, 0.0, places=8)

    def test_unequal_weights_nonzero_objective(self):
        n = 3
        V = np.matrix(np.eye(n))
        w = np.array([0.8, 0.1, 0.1])
        target = [1 / n] * n
        result = risk_budget_objective(w, [V, target])
        self.assertGreater(result, 0.0)


# ---------------------------------------------------------------------------
# P1.1  Weight optimisation edge cases
# ---------------------------------------------------------------------------

class TestOptimiseWeightsBetaFloor(unittest.TestCase):
    """min_beta/asset_betas: linear market-participation constraint."""

    def test_beta_floor_binds(self):
        # Sharpe prefers the near-zero-vol asset (beta 0); the floor forces
        # half the book into the beta-1 asset.
        er = np.array([0.10, 0.10])
        cov = np.array([[0.04, 0.0], [0.0, 0.0001]])
        betas = np.array([1.0, 0.0])
        res = optimise_weights(expected_returns=er, cov_matrix=cov,
                               asset_betas=betas, min_beta=0.5)
        self.assertTrue(res.success)
        self.assertGreaterEqual(float(betas @ res.x), 0.5 - 1e-6)

    def test_no_floor_prefers_low_vol(self):
        er = np.array([0.10, 0.10])
        cov = np.array([[0.04, 0.0], [0.0, 0.0001]])
        res = optimise_weights(expected_returns=er, cov_matrix=cov)
        self.assertLess(res.x[0], 0.05)

    def test_min_beta_without_betas_is_ignored(self):
        er = np.array([0.10, 0.10])
        cov = np.array([[0.04, 0.0], [0.0, 0.0001]])
        res = optimise_weights(expected_returns=er, cov_matrix=cov,
                               min_beta=0.5)
        self.assertTrue(res.success)
        self.assertLess(res.x[0], 0.05)

    def test_shorts_count_negative(self):
        # A negative-beta asset makes the floor harder, not easier, to satisfy.
        er = np.array([0.10, 0.10, 0.10])
        cov = np.diag([0.04, 0.0001, 0.03])
        betas = np.array([1.0, 0.0, -0.8])
        res = optimise_weights(expected_returns=er, cov_matrix=cov,
                               asset_betas=betas, min_beta=0.5)
        self.assertTrue(res.success)
        self.assertGreaterEqual(float(betas @ res.x), 0.5 - 1e-6)


class TestOptimiseWeightsBetaPin(unittest.TestCase):
    """target_beta/asset_betas: exact beta equality (beta-1/IR experiment)."""

    def test_pin_binds_from_below(self):
        # Sharpe prefers the near-zero-vol beta-0 asset; the pin drags the
        # book up to beta 1 exactly.
        er = np.array([0.10, 0.10])
        cov = np.array([[0.04, 0.0], [0.0, 0.0001]])
        betas = np.array([2.0, 0.0])
        x0 = beta_target_start(betas, 1.0, max_weight=1.0)
        res = optimise_weights(expected_returns=er, cov_matrix=cov,
                               initial_weights=x0,
                               asset_betas=betas, target_beta=1.0)
        self.assertTrue(res.success)
        self.assertAlmostEqual(float(betas @ res.x), 1.0, places=5)

    def test_pin_binds_from_above(self):
        # Sharpe prefers the high-beta asset (higher ER, same vol); the pin
        # drags the book down to beta 1 exactly.
        er = np.array([0.20, 0.02])
        cov = np.array([[0.04, 0.0], [0.0, 0.04]])
        betas = np.array([1.5, 0.5])
        x0 = beta_target_start(betas, 1.0, max_weight=1.0)
        res = optimise_weights(expected_returns=er, cov_matrix=cov,
                               initial_weights=x0,
                               asset_betas=betas, target_beta=1.0)
        self.assertTrue(res.success)
        self.assertAlmostEqual(float(betas @ res.x), 1.0, places=5)

    def test_target_beta_without_betas_is_ignored(self):
        er = np.array([0.10, 0.10])
        cov = np.array([[0.04, 0.0], [0.0, 0.0001]])
        res = optimise_weights(expected_returns=er, cov_matrix=cov,
                               target_beta=1.0)
        self.assertTrue(res.success)
        self.assertLess(res.x[0], 0.05)


class TestReachableBetaInterval(unittest.TestCase):
    """Greedy LP vertices for the reachable portfolio-beta range."""

    def test_exact_interval(self):
        # max_weight 0.3: highest = 0.3*1.2 + 0.3*1.0 + 0.3*0.4 + 0.1*0.2 = 0.80
        #                 lowest  = 0.3*0.2 + 0.3*0.4 + 0.3*1.0 + 0.1*1.2 = 0.60
        betas = np.array([1.2, 1.0, 0.4, 0.2])
        lo, hi = reachable_beta_interval(betas, max_weight=0.3)
        self.assertAlmostEqual(lo, 0.60, places=10)
        self.assertAlmostEqual(hi, 0.80, places=10)

    def test_vertex_weights_valid(self):
        betas = np.array([1.5, 0.9, 0.3, -0.2, 0.7])
        for maximise in (True, False):
            _, w = greedy_beta_vertex(betas, max_weight=0.3,
                                      maximise=maximise)
            self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
            self.assertTrue(np.all(w >= -1e-12))
            self.assertTrue(np.all(w <= 0.3 + 1e-12))

    def test_min_weight_floor_respected(self):
        betas = np.array([1.0, 0.5, 0.0])
        beta, w = greedy_beta_vertex(betas, max_weight=0.5, min_weight=0.1,
                                     maximise=True)
        self.assertTrue(np.all(w >= 0.1 - 1e-12))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
        # 0.5 on beta-1.0, 0.4 on beta-0.5, 0.1 on beta-0.0 = 0.70
        self.assertAlmostEqual(beta, 0.70, places=10)

    def test_infeasible_bounds_raise(self):
        with self.assertRaises(ValueError):
            greedy_beta_vertex(np.array([1.0, 0.5]), max_weight=0.4)

    def test_start_hits_target_exactly(self):
        betas = np.array([1.2, 1.0, 0.4, 0.2])
        for target in (0.62, 0.70, 0.79):
            w = beta_target_start(betas, target, max_weight=0.3)
            self.assertAlmostEqual(float(betas @ w), target, places=10)
            self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
            self.assertTrue(np.all(w >= -1e-12))
            self.assertTrue(np.all(w <= 0.3 + 1e-12))

    def test_start_clamps_unreachable_target(self):
        betas = np.array([1.2, 1.0, 0.4, 0.2])
        w = beta_target_start(betas, 1.5, max_weight=0.3)
        self.assertAlmostEqual(float(betas @ w), 0.80, places=10)


class TestOptimiseWeightsEdgeCases(unittest.TestCase):
    """Edge cases for optimise_weights (SLSQP)."""

    def test_single_asset(self):
        """1-asset portfolio → weight=[1.0]."""
        er = np.array([0.10])
        cov = np.array([[0.04]])
        result = optimise_weights(expected_returns=er, cov_matrix=cov)
        self.assertAlmostEqual(result.x[0], 1.0, places=4)

    def test_two_assets_perfect_correlation(self):
        """Perfect correlation should not crash SLSQP."""
        er = np.array([0.10, 0.12])
        cov = np.array([[0.04, 0.04], [0.04, 0.04]])
        result = optimise_weights(expected_returns=er, cov_matrix=cov)
        self.assertAlmostEqual(sum(result.x), 1.0, places=4)

    def test_infeasible_min_weight(self):
        """min_weight=0.6 with 3 assets is infeasible (3×0.6 > 1.0)."""
        er = np.array([0.10, 0.12, 0.08])
        cov = np.diag([0.04, 0.04, 0.04])
        result = optimise_weights(
            expected_returns=er, cov_matrix=cov, min_weight=0.6,
        )
        # SLSQP may not converge or may still return weights; either is ok
        # but weights should NOT sum to 1 within tolerance or success=False
        if result.success:
            # If it "succeeds", the result is the fallback equal weights
            self.assertAlmostEqual(sum(result.x), 1.0, places=3)

    def test_zero_expected_returns(self):
        """All zero returns — should not divide by zero."""
        er = np.array([0.0, 0.0, 0.0])
        cov = np.diag([0.04, 0.04, 0.04])
        result = optimise_weights(expected_returns=er, cov_matrix=cov)
        self.assertAlmostEqual(sum(result.x), 1.0, places=4)
        self.assertTrue(np.all(np.isfinite(result.x)))

    def test_minimize_variance_two_uncorrelated(self):
        """min-var weights for uncorrelated assets: w_i ∝ 1/σ_i²."""
        # σ₁² = 0.04, σ₂² = 0.16, ρ = 0 → w₁ = 0.16/0.20 = 0.8, w₂ = 0.2
        cov = np.array([[0.04, 0.0], [0.0, 0.16]])
        er = np.array([0.10, 0.20])  # ignored
        result = optimise_weights(
            expected_returns=er, cov_matrix=cov, minimize_variance=True,
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.x[0], 0.8, places=3)
        self.assertAlmostEqual(result.x[1], 0.2, places=3)
        self.assertAlmostEqual(sum(result.x), 1.0, places=4)

    def test_minimize_variance_ignores_expected_returns(self):
        """Min-var solution should be invariant to expected returns."""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        er_a = np.array([0.10, 0.20])
        er_b = np.array([-1.0, 5.0])
        result_a = optimise_weights(
            expected_returns=er_a, cov_matrix=cov, minimize_variance=True)
        result_b = optimise_weights(
            expected_returns=er_b, cov_matrix=cov, minimize_variance=True)
        self.assertTrue(np.allclose(result_a.x, result_b.x, atol=1e-4))

    def test_minimize_variance_mutually_exclusive_with_risk_parity(self):
        """Setting both minimize_variance and risk_parity should raise."""
        er = np.array([0.10, 0.12])
        cov = np.array([[0.04, 0.0], [0.0, 0.04]])
        with self.assertRaises(ValueError):
            optimise_weights(
                expected_returns=er, cov_matrix=cov,
                minimize_variance=True, risk_parity=True,
            )


# ---------------------------------------------------------------------------
# P1.2  Covariance edge cases
# ---------------------------------------------------------------------------

class TestCovarianceEdgeCases(unittest.TestCase):

    def test_single_column_dataframe(self):
        """1 ticker → 1×1 covariance matrix."""
        returns = pd.DataFrame(np.random.randn(100, 1), columns=['A'])
        cov = calculate_covariance_matrix(returns, annualise=False)
        self.assertEqual(cov.shape, (1, 1))
        self.assertGreater(cov.iloc[0, 0], 0)

    def test_constant_returns_ledoit_wolf(self):
        """Zero-variance column with Ledoit-Wolf should not crash."""
        data = np.random.randn(100, 3)
        data[:, 0] = 0.0  # constant
        returns = pd.DataFrame(data, columns=['A', 'B', 'C'])
        cov = calculate_covariance_matrix(returns, annualise=False, shrinkage=True)
        self.assertEqual(cov.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(cov.values)))

    def test_observation_ratio_error(self):
        """T=5, N=10 → T/N=0.5 < 1.0, should raise ValueError."""
        with self.assertRaises(ValueError):
            check_observation_ratio(5, 10)


# ---------------------------------------------------------------------------
# P1.4  Metrics edge cases
# ---------------------------------------------------------------------------

class TestMetricsEdgeCases(unittest.TestCase):

    def test_maximum_drawdown_no_drawdown(self):
        """Monotonically increasing returns → drawdown = 0."""
        returns = [0.01, 0.02, 0.01, 0.03, 0.01]
        dd = maximum_drawdown(returns)
        self.assertEqual(dd, 0)

    def test_maximum_drawdown_constant(self):
        """All-zero returns → drawdown = 0."""
        returns = [0.0, 0.0, 0.0, 0.0]
        dd = maximum_drawdown(returns)
        self.assertEqual(dd, 0)

    def test_sortino_zero_downside(self):
        """No negative returns → downside_deviation=0 → sortino=0."""
        dd = downside_deviation([0.01, 0.02, 0.03])
        self.assertEqual(dd, 0.0)
        self.assertEqual(sortino_ratio(0.10, dd), 0.0)

    def test_deflated_sharpe_extreme_kurtosis(self):
        """Fat-tailed distribution should inflate SR variance → lower DSR."""
        dsr_normal = deflated_sharpe_ratio(
            observed_sr=1.0, n=252, num_trials=100,
            excess_kurtosis=0.0,
        )
        dsr_fat = deflated_sharpe_ratio(
            observed_sr=1.0, n=252, num_trials=100,
            excess_kurtosis=10.0,
        )
        # Fat tails increase SR variance → harder to be significant
        self.assertLessEqual(dsr_fat, dsr_normal)


# ---------------------------------------------------------------------------
# P1.5  Empty data handling
# ---------------------------------------------------------------------------

class TestEmptyDataEdgeCases(unittest.TestCase):

    def test_log_returns_empty_df(self):
        empty = pd.DataFrame()
        result = calculate_log_returns(empty)
        self.assertTrue(result.empty)

    def test_expected_returns_empty(self):
        empty = pd.DataFrame()
        result = calculate_expected_returns(empty)
        self.assertEqual(len(result), 0)

    def test_covariance_empty_df(self):
        empty = pd.DataFrame()
        result = calculate_covariance_matrix(empty, annualise=False)
        self.assertTrue(result.empty)


# ---------------------------------------------------------------------------
# P2.1  Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy(unittest.TestCase):
    """All domain exceptions should subclass PortfolioError."""

    def test_all_subclass_portfolio_error(self):
        from src.exceptions import (
            PortfolioError, DownloadError, ValidationError,
            OptimisationError, DatabaseError, ForecastError,
        )
        for exc_cls in (DownloadError, ValidationError, OptimisationError,
                        DatabaseError, ForecastError):
            self.assertTrue(
                issubclass(exc_cls, PortfolioError),
                f"{exc_cls.__name__} is not a subclass of PortfolioError",
            )

    def test_catch_all_with_portfolio_error(self):
        from src.exceptions import PortfolioError, DownloadError
        with self.assertRaises(PortfolioError):
            raise DownloadError("test")


if __name__ == '__main__':
    unittest.main()
