import unittest
import backtest
import numpy as np


class TestBacktest(unittest.TestCase):
    def test_get_random_weights_count(self):
        """
        Asserts that you get three random weights.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        weights = backtest.get_random_weights(tickers)
        self.assertEqual(len(weights), 3)

    def test_get_random_weights_sum(self):
        """
        Asserts that the weights all sum to 1.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        weights = backtest.get_random_weights(tickers)
        self.assertAlmostEqual(sum(weights), 1)

    def test_get_random_weights_positive(self):
        """
        Asserts that the weights are all positive.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        weights = backtest.get_random_weights(tickers)
        self.assertTrue(all(w >= 0 for w in weights))

    def test_get_random_weights_distinct(self):
        """
        Asserts that the weights are all distinct.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        weights = backtest.get_random_weights(tickers)
        self.assertTrue(len(set(weights)) == 3)

    def test_optimal_weights_count(self):
        """
        Asserts that you get three optimal weights.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        backtest.op.prepare_opt_inputs(backtest.data
                                       .iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE,
                                             :],
                                       backtest.USE_FORECAST)
        weights = backtest.optimal_weights(tickers)
        self.assertEqual(len(weights), 3)

    def test_optimal_weights_sum(self):
        """
        Asserts that the weights all sum to 1.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        backtest.op.prepare_opt_inputs(backtest.data
                                       .iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE,
                                             :],
                                       backtest.USE_FORECAST)
        weights = backtest.optimal_weights(tickers)
        self.assertAlmostEqual(sum(weights), 1)

    def test_optimal_weights_positive(self):
        """
        Asserts that the weights are all positive.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        backtest.op.prepare_opt_inputs(backtest.data
                                       .iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE,
                                             :],
                                       backtest.USE_FORECAST)
        weights = backtest.optimal_weights(tickers)
        self.assertTrue(all(w >= 0 for w in weights))

    def test_optimal_weights_distinct(self):
        """
        Asserts that the weights are all distinct.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        backtest.op.prepare_opt_inputs(backtest.data
                                       .iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE,
                                             :],
                                       backtest.USE_FORECAST)
        weights = backtest.optimal_weights(tickers)
        self.assertTrue(len(set(weights)) > 1)

    def test_data_gets_loaded(self):
        """
        Tests that the global variable data
        gets loaded.
        """
        self.assertTrue(backtest.data.shape[0] > 0)

    def test_create_portfolio(self):
        """
        Tests that a lsit of tickers is returned.
        """
        tickers = backtest.create_portfolio(50)
        self.assertGreater(len(tickers), 2)

    def test_difference_of_means_hypothesis_test(self):
        """
        Tests that the t value is being calculated correctly.
        """
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
        """
        Tests to see that the t-value is negative if the mean is
        smaller and positive if larger.

        Sample 2 mean is greater, so the t value should be positive.
        """
        sample_1 = [1, 2, 3, 4, 5]
        sample_2 = [2, 3, 4, 5, 6]
        t = backtest.difference_of_means_hypothesis_test(sample_1,
                                                         sample_2)
        self.assertGreater(t, 0)

    def test_difference_of_means_hypothesis_test_negative(self):
        """
        Tests to see that the t-value is negative if the mean is
        smaller and positive if larger.

        Sample 1 mean is greater, so the t value should be negative.
        """
        sample_1 = [6, 7, 8, 9, 10]
        sample_2 = [2, 3, 4, 5, 6]
        t = backtest.difference_of_means_hypothesis_test(sample_1,
                                                         sample_2)
        self.assertLess(t, 0)


    def test_get_random_weights_empty_portfolio(self):
        """
        An empty portfolio should raise ValueError.
        """
        with self.assertRaises(ValueError):
            backtest.get_random_weights([])

    def test_downside_deviation_empty_returns(self):
        """
        Empty returns list should return 0.
        """
        result = backtest.downside_deviation([])
        self.assertEqual(result, 0.0)

    def test_sortino_ratio_zero_deviation(self):
        """
        Zero downside deviation should return 0, not raise.
        """
        result = backtest.sortino_ratio(0.1, 0)
        self.assertEqual(result, 0.0)

    def test_calmar_ratio_zero_drawdown(self):
        """
        Zero max drawdown should return 0, not raise.
        """
        result = backtest.calmar_ratio(0.1, 0)
        self.assertEqual(result, 0.0)

    def test_fitness_zero_std(self):
        """
        Constant returns (zero std) should return 0, not raise.
        """
        constant_returns = [0.01] * 100
        result = backtest.fitness(constant_returns)
        self.assertEqual(result, 0.0)

    def test_maximum_drawdown_single_return(self):
        """
        A single return should produce a valid max drawdown.
        """
        result = backtest.maximum_drawdown([0.01])
        self.assertEqual(result, 0)

    def test_optimal_weights_missing_ticker(self):
        """
        If tickers are not in the data, should raise KeyError.
        """
        backtest.op.prepare_opt_inputs(
            backtest.data.iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE, :],
            backtest.USE_FORECAST
        )
        with self.assertRaises(KeyError):
            backtest.optimal_weights(['NONEXISTENT_TICKER_XYZ'])


if __name__ == '__main__':
    unittest.main()
