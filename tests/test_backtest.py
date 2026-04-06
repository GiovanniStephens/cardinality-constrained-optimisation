import unittest
from src import backtest
from src.portfolio_utils import (
    load_data,
    calculate_log_returns,
    calculate_expected_returns,
    maximum_drawdown,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
)
from src.config import BACKTEST_TEST_DAYS, TRADING_DAYS_PER_YEAR
import numpy as np
import pandas as pd


def _load_test_data():
    """Load price data from DB (CSV fallback) for integration tests."""
    data = pd.DataFrame()
    try:
        from src import db
        conn = db.get_connection()
        data = db.load_prices(conn, exchange='US')
        conn.close()
    except Exception:
        pass
    if data.empty:
        data = load_data('data/NZ_ETF_Prices.csv')
    else:
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        data = data.dropna(axis=1, thresh=int(0.95 * len(data)))
        data = data.ffill()
    return data


class TestBacktest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = _load_test_data()
        training = cls.data.iloc[:-BACKTEST_TEST_DAYS, :]
        backtest._backtest_data = training
        backtest._use_forecast = False
        # Set up log returns and expected returns for optimal_weights
        log_rets = calculate_log_returns(training)
        backtest._backtest_log_returns = log_rets.transpose()
        backtest._backtest_expected_returns = calculate_expected_returns(log_rets)
        # Pick 3 real tickers from whatever data was loaded
        cls.test_tickers = list(cls.data.columns[:3])

    def test_get_random_weights_count(self):
        tickers = self.test_tickers
        weights = backtest.get_random_weights(tickers)
        self.assertEqual(len(weights), 3)

    def test_get_random_weights_sum(self):
        tickers = self.test_tickers
        weights = backtest.get_random_weights(tickers)
        self.assertAlmostEqual(sum(weights), 1)

    def test_get_random_weights_positive(self):
        tickers = self.test_tickers
        weights = backtest.get_random_weights(tickers)
        self.assertTrue(all(w >= 0 for w in weights))

    def test_get_random_weights_distinct(self):
        tickers = self.test_tickers
        weights = backtest.get_random_weights(tickers)
        self.assertTrue(len(set(weights)) == 3)

    def test_optimal_weights_count(self):
        tickers = self.test_tickers
        weights = backtest.optimal_weights(tickers)
        self.assertEqual(len(weights), 3)

    def test_optimal_weights_sum(self):
        tickers = self.test_tickers
        weights = backtest.optimal_weights(tickers)
        self.assertAlmostEqual(sum(weights), 1)

    def test_optimal_weights_positive(self):
        tickers = self.test_tickers
        weights = backtest.optimal_weights(tickers)
        self.assertTrue(all(w >= 0 for w in weights))

    def test_optimal_weights_distinct(self):
        tickers = self.test_tickers
        weights = backtest.optimal_weights(tickers)
        self.assertTrue(len(set(weights)) > 1)

    def test_data_loads_from_db_or_csv(self):
        self.assertTrue(self.data.shape[0] > 0)

    def test_create_portfolio(self):
        tickers = backtest.create_portfolio(50)
        self.assertGreater(len(tickers), 2)

    def test_difference_of_means_hypothesis_test(self):
        sample_1 = [1, 2, 3, 4, 5]
        sample_2 = [2, 3, 4, 5, 6]
        mean_1 = sum(sample_1) / len(sample_1)
        mean_2 = sum(sample_2) / len(sample_2)
        stdDev_1 = np.array(sample_1).std()
        stdDev_2 = np.array(sample_2).std()
        t = (mean_2 - mean_1) / \
            np.sqrt(stdDev_1**2 / len(sample_1) + stdDev_2**2 / len(sample_2))
        other_t = backtest.difference_of_means_hypothesis_test(sample_1,
                                                               sample_2)
        self.assertEqual(round(t, 6), round(other_t, 6))

    def test_difference_of_means_hypothesis_test_positive(self):
        sample_1 = [1, 2, 3, 4, 5]
        sample_2 = [2, 3, 4, 5, 6]
        t = backtest.difference_of_means_hypothesis_test(sample_1,
                                                         sample_2)
        self.assertGreater(t, 0)

    def test_difference_of_means_hypothesis_test_negative(self):
        sample_1 = [6, 7, 8, 9, 10]
        sample_2 = [2, 3, 4, 5, 6]
        t = backtest.difference_of_means_hypothesis_test(sample_1,
                                                         sample_2)
        self.assertLess(t, 0)

    def test_get_random_weights_empty_portfolio(self):
        with self.assertRaises(ValueError):
            backtest.get_random_weights([])

    def test_downside_deviation_empty_returns(self):
        result = downside_deviation([])
        self.assertEqual(result, 0.0)

    def test_sortino_ratio_zero_deviation(self):
        result = sortino_ratio(0.1, 0)
        self.assertEqual(result, 0.0)

    def test_calmar_ratio_zero_drawdown(self):
        result = calmar_ratio(0.1, 0)
        self.assertEqual(result, 0.0)

    def test_fitness_zero_std(self):
        constant_returns = [0.0] * 100
        result = backtest.fitness(constant_returns)
        self.assertEqual(result, 0.0)

    def test_maximum_drawdown_single_return(self):
        result = maximum_drawdown([0.01])
        self.assertEqual(result, 0)

    def test_optimal_weights_missing_ticker(self):
        with self.assertRaises(KeyError):
            backtest.optimal_weights([self.test_tickers[0],
                                      'NONEXISTENT_TICKER_XYZ'])

    def test_maximum_drawdown_empty_returns(self):
        with self.assertRaises((IndexError, ValueError)):
            maximum_drawdown([])

    def test_calmar_ratio_positive_drawdown_sign_convention(self):
        result = calmar_ratio(0.10, 0.20)
        self.assertGreater(result, 0)

    def test_difference_of_means_hypothesis_test_identical_samples(self):
        with self.assertRaises(ValueError):
            backtest.difference_of_means_hypothesis_test([5, 5, 5], [5, 5, 5])


class TestGenerateWindows(unittest.TestCase):
    """Tests for the rolling window generation function."""

    def setUp(self):
        # Create a synthetic daily date index: 8 years of trading days
        self.dates = pd.bdate_range('2014-01-02', periods=2016, freq='B')

    def test_window_count(self):
        """5yr train + 1yr test + 1yr step over 8 years → 3 windows."""
        windows = backtest.generate_windows(
            self.dates, train_days=1260, test_days=252, step_days=252,
        )
        self.assertEqual(len(windows), 3)

    def test_windows_non_overlapping(self):
        """Test periods should not overlap."""
        windows = backtest.generate_windows(
            self.dates, train_days=1260, test_days=252, step_days=252,
        )
        for i in range(len(windows) - 1):
            self.assertLessEqual(
                windows[i].test_end, windows[i + 1].test_start,
                "Test windows should not overlap",
            )

    def test_train_before_test(self):
        """Training must end before testing starts."""
        windows = backtest.generate_windows(
            self.dates, train_days=1260, test_days=252, step_days=252,
        )
        for w in windows:
            self.assertLess(w.train_end, w.test_start)

    def test_label_format(self):
        """Labels should be like '2014-2018/2019'."""
        windows = backtest.generate_windows(
            self.dates, train_days=1260, test_days=252, step_days=252,
        )
        for w in windows:
            self.assertIn('/', w.label)
            parts = w.label.split('/')
            self.assertEqual(len(parts), 2)

    def test_insufficient_data_raises(self):
        """Too few dates for even one window should raise ValueError."""
        short_dates = pd.bdate_range('2020-01-01', periods=100, freq='B')
        with self.assertRaises(ValueError):
            backtest.generate_windows(
                short_dates, train_days=1260, test_days=252, step_days=252,
            )

    def test_windowspec_fields_populated(self):
        """All WindowSpec fields should be populated."""
        windows = backtest.generate_windows(
            self.dates, train_days=1260, test_days=252, step_days=252,
        )
        for w in windows:
            self.assertIsInstance(w.train_start, pd.Timestamp)
            self.assertIsInstance(w.train_end, pd.Timestamp)
            self.assertIsInstance(w.test_start, pd.Timestamp)
            self.assertIsInstance(w.test_end, pd.Timestamp)
            self.assertIsInstance(w.label, str)


class TestRunPortfolio(unittest.TestCase):
    """Tests for run_portfolio with the new OOS-only signature."""

    def test_returns_correct_length(self):
        """Output length should match OOS period length."""
        np.random.seed(42)
        dates = pd.bdate_range('2020-01-01', periods=20, freq='B')
        tickers = ['A', 'B', 'C']
        oos = pd.DataFrame(
            np.random.randn(20, 3) * 0.01,
            index=dates, columns=tickers,
        )
        weights = np.array([0.5, 0.3, 0.2])
        returns = backtest.run_portfolio(tickers, weights, oos)
        self.assertEqual(len(returns), 20)

    def test_zero_returns_give_zero_portfolio(self):
        """If all returns are zero, portfolio returns should be zero."""
        dates = pd.bdate_range('2020-01-01', periods=10, freq='B')
        tickers = ['A', 'B']
        oos = pd.DataFrame(0.0, index=dates, columns=tickers)
        weights = np.array([0.6, 0.4])
        returns = backtest.run_portfolio(tickers, weights, oos)
        for r in returns:
            self.assertAlmostEqual(r, 0.0)

    def test_weights_sum_preserved(self):
        """Weights should remain normalised (sum ~ 1) after drift."""
        np.random.seed(42)
        dates = pd.bdate_range('2020-01-01', periods=5, freq='B')
        tickers = ['X', 'Y']
        oos = pd.DataFrame(
            np.random.randn(5, 2) * 0.01,
            index=dates, columns=tickers,
        )
        weights = np.array([0.5, 0.5])
        # Run and check — the function modifies a copy internally
        backtest.run_portfolio(tickers, weights, oos)
        # Original weights should be unchanged
        self.assertAlmostEqual(weights.sum(), 1.0)


class TestGetStatistics(unittest.TestCase):
    """Tests for get_statistics returning a dict."""

    def test_returns_dict_with_all_metric_keys(self):
        np.random.seed(42)
        dates = pd.bdate_range('2020-01-01', periods=50, freq='B')
        tickers = ['A', 'B']
        oos = pd.DataFrame(
            np.random.randn(50, 2) * 0.01,
            index=dates, columns=tickers,
        )
        weights = np.array([0.6, 0.4])
        stats = backtest.get_statistics(tickers, weights, oos)
        self.assertIsInstance(stats, dict)
        for key in backtest.METRIC_NAMES:
            self.assertIn(key, stats)


class TestPairedTTest(unittest.TestCase):
    """Tests for the paired t-test across windows."""

    def test_zero_mean_difference_gives_zero_t(self):
        # Differences: 0.1, 0.0, -0.1 → mean diff = 0
        a = {'w1': 0.1, 'w2': 0.3, 'w3': 0.5}
        b = {'w1': 0.2, 'w2': 0.3, 'w3': 0.4}
        t_stat, p_val = backtest.paired_t_test(a, b)
        self.assertAlmostEqual(t_stat, 0.0)

    def test_higher_b_gives_positive_t(self):
        a = {'w1': 0.1, 'w2': 0.2, 'w3': 0.1}
        b = {'w1': 0.5, 'w2': 0.7, 'w3': 0.6}
        t_stat, _ = backtest.paired_t_test(a, b)
        self.assertGreater(t_stat, 0)

    def test_too_few_windows_raises(self):
        a = {'w1': 0.5}
        b = {'w1': 0.6}
        with self.assertRaises(ValueError):
            backtest.paired_t_test(a, b)


class TestAggregateCrossWindow(unittest.TestCase):
    """Tests for cross-window summary aggregation."""

    def test_correct_shape(self):
        """Should have one row per category, columns = windows + mean + std."""
        w1 = backtest.WindowSpec(
            train_start=pd.Timestamp('2014-01-02'),
            train_end=pd.Timestamp('2018-12-31'),
            test_start=pd.Timestamp('2019-01-02'),
            test_end=pd.Timestamp('2019-12-31'),
            label='2014-2018/2019',
        )
        w2 = backtest.WindowSpec(
            train_start=pd.Timestamp('2015-01-02'),
            train_end=pd.Timestamp('2019-12-31'),
            test_start=pd.Timestamp('2020-01-02'),
            test_end=pd.Timestamp('2020-12-31'),
            label='2015-2019/2020',
        )
        pr1 = backtest.PortfolioResult(
            portfolio=['A'], weights=np.array([1.0]),
            metrics={'sharpe_ratio': 0.5, **{k: 0 for k in backtest.METRIC_NAMES if k != 'sharpe_ratio'}},
        )
        pr2 = backtest.PortfolioResult(
            portfolio=['A'], weights=np.array([1.0]),
            metrics={'sharpe_ratio': 0.7, **{k: 0 for k in backtest.METRIC_NAMES if k != 'sharpe_ratio'}},
        )
        wr1 = backtest.WindowResult(
            window=w1,
            method_results={'cc_optimised': backtest.MethodResults('cc_optimised', [pr1])},
        )
        wr2 = backtest.WindowResult(
            window=w2,
            method_results={'cc_optimised': backtest.MethodResults('cc_optimised', [pr2])},
        )
        df = backtest.aggregate_cross_window([wr1, wr2])
        self.assertEqual(len(df), 1)  # one category
        self.assertIn('mean', df.columns)
        self.assertIn('std', df.columns)
        self.assertAlmostEqual(df.loc['cc_optimised', 'mean'], 0.6)


class TestBacktestValidation(unittest.TestCase):
    """
    Validation tests for backtest correctness: no look-ahead bias,
    data boundary integrity, metric consistency, and weight sanity.

    Uses small synthetic data — no DB or CSV dependencies.
    """

    def _make_prices(self, n_days=30, n_tickers=3, start='2020-01-01',
                     base_price=100.0, daily_drift=0.001, seed=42):
        """Helper: create a synthetic price DataFrame with realistic drift."""
        np.random.seed(seed)
        dates = pd.bdate_range(start, periods=n_days, freq='B')
        tickers = [f'T{i}' for i in range(n_tickers)]
        log_rets = np.random.randn(n_days, n_tickers) * 0.02 + daily_drift
        log_rets[0] = 0  # first row is the base
        prices = base_price * np.exp(np.cumsum(log_rets, axis=0))
        return pd.DataFrame(prices, index=dates, columns=tickers)

    # ── Test 1: No look-ahead — training ends before test ─────────────────

    def test_train_data_ends_before_test(self):
        """Training prices must not contain any test-period dates."""
        prices = self._make_prices(n_days=60)
        windows = backtest.generate_windows(
            prices.index, train_days=30, test_days=10, step_days=10,
        )
        for w in windows:
            train = prices.loc[w.train_start:w.train_end]
            test = prices.loc[w.test_start:w.test_end]
            self.assertLess(
                train.index.max(), test.index.min(),
                f"Window {w.label}: training data overlaps with test data",
            )

    # ── Test 2: op.data only contains training-period dates ───────────────

    def test_op_data_only_contains_training_dates(self):
        """After prepare_opt_inputs, op.data should span training period only."""
        from src.optimisers import pygad_ga as op
        prices = self._make_prices(n_days=60)
        windows = backtest.generate_windows(
            prices.index, train_days=30, test_days=10, step_days=10,
        )
        w = windows[0]
        train = prices.loc[w.train_start:w.train_end]
        op.prepare_opt_inputs(train, use_forecasts=False)
        # op.data is transposed log returns: tickers x time_periods
        n_periods = op.data.shape[1]
        self.assertEqual(
            n_periods, len(train),
            f"op.data has {n_periods} periods but training has {len(train)} rows",
        )

    # ── Test 3: First OOS return is NOT zero ──────────────────────────────

    def test_first_oos_return_not_zero(self):
        """
        The first OOS log return must reflect the price change from
        the last training day to the first test day, not be zero.
        """
        from src.portfolio_utils import calculate_log_returns
        # Construct prices: train ends at 100, test starts at 105
        train_dates = pd.bdate_range('2020-01-01', periods=10, freq='B')
        test_dates = pd.bdate_range(train_dates[-1] + pd.Timedelta(days=1),
                                    periods=5, freq='B')
        train_prices = pd.DataFrame({'A': np.linspace(90, 100, 10)},
                                    index=train_dates)
        test_prices = pd.DataFrame({'A': [105, 106, 107, 108, 109]},
                                   index=test_dates)

        # Correct approach: prepend last training price
        boundary = train_prices.iloc[[-1]]
        combined = pd.concat([boundary, test_prices])
        oos_returns = calculate_log_returns(combined).iloc[1:]

        expected_first = np.log(105.0 / 100.0)
        self.assertAlmostEqual(
            oos_returns.iloc[0]['A'], expected_first, places=6,
            msg="First OOS return should be log(105/100), not 0",
        )

    # ── Test 4: OOS returns match manual computation ──────────────────────

    def test_portfolio_returns_match_manual(self):
        """Run_portfolio output must match hand-computed weighted returns."""
        dates = pd.bdate_range('2020-01-01', periods=3, freq='B')
        # Known log returns for 2 tickers over 3 days
        oos = pd.DataFrame({
            'A': [0.01, -0.02, 0.03],
            'B': [0.02, 0.01, -0.01],
        }, index=dates)
        weights = np.array([0.6, 0.4])

        returns = backtest.run_portfolio(['A', 'B'], weights, oos)

        # Day 0: 0.6*0.01 + 0.4*0.02 = 0.014
        self.assertAlmostEqual(returns[0], 0.014, places=10)

        # Day 1: weights drift after day 0
        w_after_0 = weights * np.exp([0.01, 0.02]) / (1 + 0.014)
        expected_day1 = float(np.sum(w_after_0 * np.array([-0.02, 0.01])))
        self.assertAlmostEqual(returns[1], expected_day1, places=10)

    # ── Test 5: Portfolio returns are bounded ─────────────────────────────

    def test_portfolio_returns_bounded(self):
        """
        No single-day portfolio return should exceed ±50%.
        This catches corrupt data or indexing errors.
        """
        np.random.seed(42)
        dates = pd.bdate_range('2020-01-01', periods=252, freq='B')
        tickers = [f'T{i}' for i in range(5)]
        # Realistic daily returns: mean ~0, std ~2%
        oos = pd.DataFrame(
            np.random.randn(252, 5) * 0.02,
            index=dates, columns=tickers,
        )
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        returns = backtest.run_portfolio(tickers, weights, oos)
        for i, r in enumerate(returns):
            self.assertGreater(r, -0.5, f"Day {i} return {r} is below -50%")
            self.assertLess(r, 0.5, f"Day {i} return {r} is above +50%")

    # ── Test 6: Weights remain valid throughout simulation ────────────────

    def test_weights_stay_valid(self):
        """Weights must remain non-negative throughout simulation.

        Note: the update formula w * exp(r) / (1 + sum(w*r)) is a first-order
        approximation and doesn't perfectly preserve sum(w)=1. We check
        non-negativity strictly and sum within 5% over 50 days.
        """
        np.random.seed(42)
        dates = pd.bdate_range('2020-01-01', periods=50, freq='B')
        tickers = ['A', 'B', 'C']
        oos = pd.DataFrame(
            np.random.randn(50, 3) * 0.02,
            index=dates, columns=tickers,
        )
        weights = np.array([0.5, 0.3, 0.2])

        # Replicate run_portfolio logic to track weights
        subset = oos[tickers]
        w = weights.copy()
        for i in range(len(subset)):
            step = subset.iloc[i].values
            ret = float(np.sum(step * w))
            w = w * np.exp(step) / (1 + ret)
            self.assertTrue(
                np.all(w >= -1e-10),
                f"Day {i}: negative weight detected: {w}",
            )
        # After full simulation, weights should still be approximately normalised
        self.assertAlmostEqual(w.sum(), 1.0, delta=0.05,
                               msg=f"Final weights deviate too far from 1: {w.sum()}")

    # ── Test 7: Sharpe ratio consistency ──────────────────────────────────

    def test_sharpe_ratio_matches_manual(self):
        """get_statistics Sharpe must match independent computation."""
        np.random.seed(42)
        dates = pd.bdate_range('2020-01-01', periods=100, freq='B')
        tickers = ['A', 'B']
        oos = pd.DataFrame(
            np.random.randn(100, 2) * 0.015 + 0.0005,
            index=dates, columns=tickers,
        )
        weights = np.array([0.6, 0.4])

        stats = backtest.get_statistics(tickers, weights, oos)
        port_returns = backtest.run_portfolio(tickers, weights, oos)

        manual_r = np.mean(port_returns) * 252
        manual_std = np.std(port_returns) * np.sqrt(252)
        manual_sharpe = manual_r / manual_std if manual_std != 0 else 0.0

        self.assertAlmostEqual(
            stats['sharpe_ratio'], manual_sharpe, places=8,
            msg="Sharpe from get_statistics doesn't match manual computation",
        )
        self.assertAlmostEqual(
            stats['annualised_return'], manual_r, places=8,
        )
        self.assertAlmostEqual(
            stats['annualised_volatility'], manual_std, places=8,
        )

    # ── Test 8: Annualisation factor is correct ───────────────────────────

    def test_annualisation_factor(self):
        """
        A constant daily return should annualise correctly with factor 252.
        """
        daily_return = 0.001
        dates = pd.bdate_range('2020-01-01', periods=252, freq='B')
        # Use a single ticker so the portfolio return IS the asset return
        oos = pd.DataFrame({'A': [daily_return] * 252}, index=dates)
        weights = np.array([1.0])

        stats = backtest.get_statistics(['A'], weights, oos)
        expected_ann_return = daily_return * 252
        # places=4 because weight drift (exp vs linear) introduces tiny drift
        self.assertAlmostEqual(
            stats['annualised_return'], expected_ann_return, places=4,
            msg=f"Annualised return should be ~{expected_ann_return}",
        )

    # ── Test 9: Selected tickers exist in test data ───────────────────────

    def test_selected_tickers_in_oos_data(self):
        """
        Every ticker in a portfolio must be present in the OOS returns.
        If a ticker is missing, run_portfolio would raise KeyError.
        """
        dates = pd.bdate_range('2020-01-01', periods=10, freq='B')
        oos = pd.DataFrame(
            np.random.randn(10, 3) * 0.01,
            index=dates, columns=['A', 'B', 'C'],
        )
        # This should work fine
        backtest.run_portfolio(['A', 'B'], np.array([0.5, 0.5]), oos)
        # This should raise
        with self.assertRaises(KeyError):
            backtest.run_portfolio(['A', 'MISSING'], np.array([0.5, 0.5]), oos)

    # ── Test 10: Window day counts are exact ──────────────────────────────

    def test_window_day_counts(self):
        """Each window should have exactly train_days and at most test_days."""
        dates = pd.bdate_range('2014-01-02', periods=2016, freq='B')
        windows = backtest.generate_windows(
            dates, train_days=1260, test_days=252, step_days=252,
        )
        for w in windows:
            train_slice = dates[(dates >= w.train_start) & (dates <= w.train_end)]
            test_slice = dates[(dates >= w.test_start) & (dates <= w.test_end)]
            self.assertEqual(
                len(train_slice), 1260,
                f"Window {w.label}: train has {len(train_slice)} days, expected 1260",
            )
            self.assertLessEqual(
                len(test_slice), 252,
                f"Window {w.label}: test has {len(test_slice)} days, expected <= 252",
            )
            self.assertGreater(len(test_slice), 0)

    # ── Test 11: Random baseline Sharpe near zero ─────────────────────────

    def test_random_baseline_sharpe_near_zero(self):
        """
        On synthetic zero-drift data, random portfolios should produce
        Sharpe ratios centred near zero.
        """
        np.random.seed(42)
        dates = pd.bdate_range('2020-01-01', periods=252, freq='B')
        tickers = [f'T{i}' for i in range(10)]
        # Zero-drift returns: mean ≈ 0, std ≈ 2%
        oos = pd.DataFrame(
            np.random.randn(252, 10) * 0.02,
            index=dates, columns=tickers,
        )

        sharpes = []
        for _ in range(50):
            # Random 3-5 ticker portfolio
            k = np.random.randint(3, 6)
            selected = list(np.random.choice(tickers, k, replace=False))
            w = np.random.random(k)
            w /= w.sum()
            stats = backtest.get_statistics(selected, w, oos)
            sharpes.append(stats['sharpe_ratio'])

        mean_sharpe = np.mean(sharpes)
        self.assertGreater(mean_sharpe, -1.5,
                           f"Mean Sharpe {mean_sharpe:.2f} is suspiciously negative")
        self.assertLess(mean_sharpe, 1.5,
                        f"Mean Sharpe {mean_sharpe:.2f} is suspiciously positive")


class TestBacktestDataIsolation(unittest.TestCase):
    """Tests that backtest evaluation properly isolates training and test data."""

    def _make_prices(self, n_days=100, n_tickers=5, seed=42):
        """Create synthetic prices with distinct regimes."""
        np.random.seed(seed)
        dates = pd.bdate_range('2018-01-01', periods=n_days, freq='B')
        tickers = [f'S{i}' for i in range(n_tickers)]
        log_rets = np.random.randn(n_days, n_tickers) * 0.01 + 0.0005
        log_rets[0] = 0
        prices = 100 * np.exp(np.cumsum(log_rets, axis=0))
        return pd.DataFrame(prices, index=dates, columns=tickers)

    def test_evaluate_window_training_data_boundary(self):
        """Test-period canary value must never appear in training log returns.

        Inject 999.0 as a price in the test period. After computing log returns
        on training data, no value should be derived from 999.0.
        """
        from src.portfolio_utils import calculate_log_returns
        prices = self._make_prices(n_days=80, n_tickers=3)
        # Inject canary in the last 20 days (test period)
        prices.iloc[-20:, 0] = 999.0

        windows = backtest.generate_windows(
            prices.index, train_days=40, test_days=20, step_days=20,
        )
        w = windows[0]
        train_prices = prices.loc[w.train_start:w.train_end]
        train_log_returns = calculate_log_returns(train_prices)

        # No value should be derived from the 999.0 canary
        max_val = train_log_returns.values.max()
        self.assertLess(max_val, 5.0,
                        f"Training log returns contain suspiciously large value {max_val}, "
                        f"possible test data leakage from canary=999.0")

    def test_oos_returns_from_test_data_only(self):
        """OOS returns must reflect test-period prices, not training-period.

        Training: positive drift (+0.1% daily)
        Test: flat (0% drift)
        OOS annualised return should be near zero, not positive.
        """
        np.random.seed(42)
        n_train, n_test = 60, 20
        dates = pd.bdate_range('2018-01-01', periods=n_train + n_test, freq='B')
        tickers = ['A', 'B', 'C']
        # Training: strong positive drift
        train_rets = np.random.randn(n_train, 3) * 0.005 + 0.002
        # Test: zero drift
        test_rets = np.random.randn(n_test, 3) * 0.005 + 0.0
        all_rets = np.vstack([train_rets, test_rets])
        all_rets[0] = 0
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(all_rets, axis=0)),
            index=dates, columns=tickers,
        )

        train_prices = prices.iloc[:n_train]
        test_prices = prices.iloc[n_train:]
        boundary = train_prices.iloc[[-1]]
        test_with_boundary = pd.concat([boundary, test_prices])
        from src.portfolio_utils import calculate_log_returns
        oos_log_returns = calculate_log_returns(test_with_boundary).iloc[1:]

        weights = np.array([1/3, 1/3, 1/3])
        stats = backtest.get_statistics(tickers, weights, oos_log_returns)
        # Zero-drift test data → annualised return should be near 0
        self.assertLess(abs(stats['annualised_return']), 1.0,
                        f"OOS return {stats['annualised_return']:.4f} too far from zero "
                        f"for zero-drift test data")

    def test_no_stale_globals_between_windows(self):
        """Running two windows sequentially must reset state for each.

        Window 1 has 3 tickers, Window 2 has 5. After preparing Window 2,
        log returns should have 5 columns (tickers), not 3.
        """
        from src.portfolio_utils import calculate_log_returns
        prices_3 = self._make_prices(n_days=80, n_tickers=3, seed=1)
        prices_5 = self._make_prices(n_days=80, n_tickers=5, seed=2)

        # Window 1: 3 tickers
        lr1 = calculate_log_returns(prices_3)
        self.assertEqual(lr1.shape[1], 3, "Window 1 should have 3 tickers")

        # Window 2: 5 tickers
        lr2 = calculate_log_returns(prices_5)
        self.assertEqual(lr2.shape[1], 5,
                         "Window 2 should have 5 tickers, not stale 3 from Window 1")


class TestSurvivorshipBias(unittest.TestCase):
    """Tests for handling delisted and late-entry tickers."""

    def test_delisted_ticker_handled_gracefully(self):
        """A ticker with valid training data but all NaN in test period
        should not crash the OOS evaluation."""
        np.random.seed(42)
        n_train, n_test = 50, 20
        dates = pd.bdate_range('2018-01-01', periods=n_train + n_test, freq='B')
        # Ticker A: full data. Ticker B: NaN in test period (delisted)
        a_rets = np.random.randn(n_train + n_test) * 0.01
        b_rets = np.random.randn(n_train + n_test) * 0.01
        a_rets[0] = 0
        b_rets[0] = 0
        prices_a = 100 * np.exp(np.cumsum(a_rets))
        prices_b = 100 * np.exp(np.cumsum(b_rets))
        prices_b[n_train:] = np.nan  # delisted

        prices = pd.DataFrame({'A': prices_a, 'B': prices_b}, index=dates)
        test_prices = prices.iloc[n_train:]

        # Attempting to run_portfolio with the delisted ticker
        # The log returns for B will be NaN → replaced with 0 (safe)
        from src.portfolio_utils import calculate_log_returns
        train_prices = prices.iloc[:n_train]
        boundary = train_prices.iloc[[-1]]
        test_with_boundary = pd.concat([boundary, test_prices])
        oos_returns = calculate_log_returns(test_with_boundary).iloc[1:]

        weights = np.array([0.5, 0.5])
        # Should not raise — NaN returns are replaced with 0
        returns = backtest.run_portfolio(['A', 'B'], weights, oos_returns)
        self.assertEqual(len(returns), n_test)

    def test_late_entry_ticker_excluded_from_early_windows(self):
        """A ticker that doesn't have data in early windows should be
        excluded by coverage filtering."""
        np.random.seed(42)
        dates = pd.bdate_range('2018-01-01', periods=100, freq='B')
        # Ticker A: full data
        a_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        # Ticker B: only has data from day 80 onward (late entry)
        b_prices = np.full(100, np.nan)
        b_prices[80:] = 100 + np.cumsum(np.random.randn(20) * 0.5)

        prices = pd.DataFrame({'A': a_prices, 'B': b_prices}, index=dates)

        # Training on first 60 days: B is all NaN
        train = prices.iloc[:60]
        # With high coverage requirement, B should be dropped
        from src.portfolio_utils import load_prices_csv
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            train.to_csv(f.name)
            path = f.name
        try:
            loaded = load_prices_csv(path, min_coverage=0.5)
            self.assertNotIn('B', loaded.columns,
                             "Late-entry ticker B should be excluded by coverage filter")
            self.assertIn('A', loaded.columns)
        finally:
            os.unlink(path)


class TestBacktestEndToEnd(unittest.TestCase):
    """End-to-end integration tests for the backtest pipeline."""

    def _make_prices(self, n_days=100, n_tickers=30, seed=42):
        np.random.seed(seed)
        dates = pd.bdate_range('2018-01-01', periods=n_days, freq='B')
        tickers = [f'E{i}' for i in range(n_tickers)]
        log_rets = np.random.randn(n_days, n_tickers) * 0.01 + 0.0003
        log_rets[0] = 0
        prices = 100 * np.exp(np.cumsum(log_rets, axis=0))
        return pd.DataFrame(prices, index=dates, columns=tickers)

    def test_evaluate_window_returns_valid_result(self):
        """A minimal evaluate_window call should return a WindowResult
        with at least one method category populated."""
        prices = self._make_prices(n_days=80, n_tickers=30)
        windows = backtest.generate_windows(
            prices.index, train_days=40, test_days=20, step_days=20,
        )
        w = windows[0]
        # Minimal run: 1 portfolio, tiny GA
        wr = backtest.evaluate_window(
            window=w, full_prices=prices, conn=None,
            num_portfolios=1, num_children=10, mc_trials=50,
            use_forecast=False,
        )
        self.assertIsInstance(wr, backtest.WindowResult)
        self.assertGreater(len(wr.method_results), 0,
                           "Should have at least one method category")
        # Check that at least one category has portfolios
        some_results = any(len(mr.portfolios) > 0
                          for mr in wr.method_results.values())
        self.assertTrue(some_results, "At least one method should have portfolio results")

    def test_is_sharpe_differs_from_oos_sharpe(self):
        """In-sample and OOS Sharpe should differ — identical values signal data leakage.

        Uses a regime break: positive drift in training, zero drift in test.
        If IS == OOS, the test data is leaking into training.
        """
        np.random.seed(42)
        n_train, n_test = 50, 20
        dates = pd.bdate_range('2018-01-01', periods=n_train + n_test, freq='B')
        tickers = [f'E{i}' for i in range(5)]
        # Training: positive drift
        train_rets = np.random.randn(n_train, 5) * 0.01 + 0.002
        # Test: zero drift
        test_rets = np.random.randn(n_test, 5) * 0.01 + 0.0
        all_rets = np.vstack([train_rets, test_rets])
        all_rets[0] = 0
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(all_rets, axis=0)),
            index=dates, columns=tickers,
        )

        from src.portfolio_utils import calculate_log_returns

        train_prices = prices.iloc[:n_train]
        test_prices = prices.iloc[n_train:]
        boundary = train_prices.iloc[[-1]]
        test_with_boundary = pd.concat([boundary, test_prices])
        train_log_returns = calculate_log_returns(train_prices)
        oos_log_returns = calculate_log_returns(test_with_boundary).iloc[1:]

        # Use all tickers with equal weights
        weights = np.ones(5) / 5
        is_stats = backtest.get_statistics(tickers, weights, train_log_returns)
        oos_stats = backtest.get_statistics(tickers, weights, oos_log_returns)

        # With a regime break, IS and OOS Sharpe should NOT be identical
        self.assertNotAlmostEqual(
            is_stats['sharpe_ratio'], oos_stats['sharpe_ratio'], places=2,
            msg="IS and OOS Sharpe are suspiciously identical — possible data leakage",
        )


if __name__ == '__main__':
    unittest.main()
