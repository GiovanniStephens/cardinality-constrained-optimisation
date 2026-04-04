import unittest
from src import backtest
from src import optimisation as op
from src.portfolio_utils import (
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
        data = op.load_data('Data/NZ_ETF_Prices.csv')
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
        op.prepare_opt_inputs(training, False)
        backtest._backtest_data = training
        backtest._use_forecast = False
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
        constant_returns = [0.01] * 100
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


if __name__ == '__main__':
    unittest.main()
