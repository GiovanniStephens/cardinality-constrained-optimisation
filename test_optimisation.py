from logging import log
import unittest
import optimisation as op
import pandas as pd
import numpy as np

class TestOptimisation(unittest.TestCase):
    def test_load_data(self):
        """
        Asserts that the load_data function returns a pandas DataFrame.
        """
        data = op.load_data('ETF_Prices.csv')
        self.assertEqual(data.shape, (756, 1792))


    def test_calculate_returns(self):
        """
        The dataframe of returns should be equal in
        length to the dataframe of prices.

        The returns should be N-1, but I fill the first value with a 0% return.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        self.assertEqual(log_returns.shape, data.shape)


    def test_calculate_returns_first_value(self):
        """
        Asserts that the calculate_returns function
        returns a pandas DataFrame.
        The dataframe of returns should be equal in
        length to the dataframe of prices.

        The returns should be N-1, but I fill the first value with a 0% return.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        self.assertEqual(log_returns.iloc[0,0], 0)

    
    def test_calculate_returns_dummy_data(self):
        """
        Asserts that the returns are correctly calculated.
        """
        data = pd.DataFrame([100,150,100], columns=['TEST'])
        log_returns = op.calculate_returns(data)
        self.assertAlmostEqual(sum(log_returns['TEST']), 0)


    def test_calculate_returns_dummy_data_first_value(self):
        """
        Asserts that the first value of the returns
        are correctly calculated.
        """
        data = pd.DataFrame([100,150,100], columns=['TEST'])
        log_returns = op.calculate_returns(data)
        self.assertEqual(log_returns.iloc[0,0], 0)

    
    def test_sharpe_ratio(self):
        """
        Asserts that the sharpe ratio is correctly calculated.
        This function manually calculates it with dummy data
        and then compares it to the optimiser function.

        The optimiser function calculates the sharpe ratio and returns
        the negative value (hence the negative in the assert).
        """
        weights = [0.5, 0.5]
        returns = [0.2, 0.3]
        corr = 0.5
        stdDevs = [0.1, 0.2]
        cov = corr*stdDevs[0]*stdDevs[1]
        port_var = (weights[0]**2)*(stdDevs[0]**2) + (weights[1]**2)*(stdDevs[1]**2) + 2*cov*weights[0]*weights[1]
        cov_matrix = [[stdDevs[0]**2, cov], [cov, stdDevs[1]**2]]
        sharpe_ratio = np.dot(weights, returns)/np.sqrt(port_var)
        model_sharpe = op.sharpe_ratio(np.array(weights), np.array(returns), cov_matrix)
        self.assertAlmostEqual(sharpe_ratio, -model_sharpe)

    
    def test_load_data_no_file(self):
        """
        Asserts that the load_data function raises an error
        when the file is not found.
        """
        with self.assertRaises(FileNotFoundError):
            data = op.load_data('ETF_Prices_missing.csv')

    
    def test_load_data_returns_df(self):
        """
        Asserts that the load_data function returns a pandas DataFrame.
        """
        data = op.load_data('ETF_Prices.csv')
        self.assertEqual(isinstance(data, pd.DataFrame), True)

    
    def test_get_cov_matrix(self):
        """
        Asserts that the get_cov_matrix function calculates
        the covariances correctly.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        cov = log_returns.iloc[:,:2].cov()*252
        cov_matrix = op.get_cov_matrix(log_returns.iloc[:, :2])
        self.assertEqual(cov_matrix.values.all(), cov.values.all())


    def test_optimisation_max_weight(self):
        """
        Asserts that the optimisation function returns
        weights under the maximum weight.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        num_stocks = 5
        max_weight = 0.3
        initial_weights = [1/num_stocks]*num_stocks
        sol = op.optimize(log_returns.iloc[:, :num_stocks],
                                  initial_weights,
                                  max_weight = max_weight)
        self.assertLessEqual(max(sol['x']), max_weight)

    
    def test_optimisation_min_weight(self):
        """
        Asserts that the optimisation function returns
        weights over the minimum weight.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        num_stocks = 5
        min_weight = 0
        initial_weights = [1/num_stocks]*num_stocks
        sol = op.optimize(log_returns.iloc[:, :num_stocks],
                                  initial_weights)
        self.assertGreaterEqual(min(sol['x']), min_weight)

    
    def test_optimisation_risk_constraint(self):
        """
        Tests that the risk constraint is indeed being
        applied in the optimisation.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        num_stocks = 30 # Need sufficient stocks to get the risk to 0.15
        max_weight = 2
        min_weight = -2 # Can short
        target_risk = 0.15
        initial_weights = [1/num_stocks]*num_stocks
        sol = op.optimize(log_returns.iloc[:, :num_stocks],
                                  initial_weights,
                                  target_risk = target_risk,
                                  target_return=None,
                                  max_weight = max_weight,
                                  min_weight = min_weight)
        weights = sol['x']
        cov = op.get_cov_matrix(log_returns.iloc[:, :num_stocks])
        risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        self.assertAlmostEqual(risk, target_risk)

    
    def test_optimisation_return_constraint(self):
        """
        Tests that the return constraint is indeed being
        applied in the optimisation.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        num_stocks = 30
        max_weight = 2
        min_weight = -2
        target_return = 0.15
        initial_weights = [1/num_stocks]*num_stocks
        sol = op.optimize(log_returns.iloc[:, :num_stocks],
                                    initial_weights,
                                    target_return = target_return,
                                    target_risk = None,
                                    max_weight = max_weight,
                                    min_weight = min_weight)
        weights = sol['x']
        returns = op.calculate_returns(data)
        returns = returns.iloc[:, :num_stocks].mean()*252
        returns = np.dot(weights, returns)
        self.assertAlmostEqual(returns, target_return)


if __name__ == '__main__':
    unittest.main()