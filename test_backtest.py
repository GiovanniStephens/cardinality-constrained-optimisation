import unittest
import backtest

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
        backtest.op.prepare_opt_inputs(backtest.data.iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE, :],
                                       backtest.USE_FORECAST)
        weights = backtest.optimal_weights(tickers)
        self.assertEqual(len(weights), 3)

    def test_optimal_weights_sum(self):
        """
        Asserts that the weights all sum to 1.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        backtest.op.prepare_opt_inputs(backtest.data.iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE, :],
                                       backtest.USE_FORECAST)
        weights = backtest.optimal_weights(tickers)
        self.assertAlmostEqual(sum(weights), 1)

    def test_optimal_weights_positive(self):
        """
        Asserts that the weights are all positive.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        backtest.op.prepare_opt_inputs(backtest.data.iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE, :],
                                       backtest.USE_FORECAST)
        weights = backtest.optimal_weights(tickers)
        self.assertTrue(all(w >= 0 for w in weights))

    def test_optimal_weights_distinct(self):
        """
        Asserts that the weights are all distinct.
        """
        tickers = ['QQQ', 'SPY', 'JJT']
        backtest.op.prepare_opt_inputs(backtest.data.iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE, :],
                                       backtest.USE_FORECAST)
        weights = backtest.optimal_weights(tickers)
        self.assertTrue(len(set(weights)) == 3)

    def test_data_gets_loaded(self):
        """
        Tests that the global variable data
        gets loaded.
        """
        self.assertTrue(backtest.data.shape[0] > 0)


if __name__ == '__main__':
    unittest.main()