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


if __name__ == '__main__':
    unittest.main()