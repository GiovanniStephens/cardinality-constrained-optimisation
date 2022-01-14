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
        self.assertEqual(log_returns.iloc[0, 0], 0)

    def test_calculate_returns_dummy_data(self):
        """
        Asserts that the returns are correctly calculated.
        """
        data = pd.DataFrame([100, 150, 100], columns=['TEST'])
        log_returns = op.calculate_returns(data)
        self.assertAlmostEqual(sum(log_returns['TEST']), 0)

    def test_calculate_returns_dummy_data_first_value(self):
        """
        Asserts that the first value of the returns
        are correctly calculated.
        """
        data = pd.DataFrame([100, 150, 100], columns=['TEST'])
        log_returns = op.calculate_returns(data)
        self.assertEqual(log_returns.iloc[0, 0], 0)

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
        port_var = (weights[0]**2)*(stdDevs[0]**2) + \
                   (weights[1]**2)*(stdDevs[1]**2) + \
            2*cov*weights[0]*weights[1]
        cov_matrix = [[stdDevs[0]**2, cov], [cov, stdDevs[1]**2]]
        sharpe_ratio = np.dot(weights, returns)/np.sqrt(port_var)
        model_sharpe = op.sharpe_ratio(np.array(weights),
                                       np.array(returns),
                                       cov_matrix)
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
        cov = log_returns.iloc[:, :2].cov()*252
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
                          max_weight=max_weight)
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
        num_stocks = 30  # Need sufficient stocks to get the risk to 0.15
        max_weight = 2
        min_weight = -2  # Can short
        target_risk = 0.15
        initial_weights = [1/num_stocks]*num_stocks
        sol = op.optimize(log_returns.iloc[:, :num_stocks],
                          initial_weights,
                          target_risk=target_risk,
                          target_return=None,
                          max_weight=max_weight,
                          min_weight=min_weight)
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
                          target_return=target_return,
                          target_risk=None,
                          max_weight=max_weight,
                          min_weight=min_weight)
        weights = sol['x']
        returns = op.calculate_returns(data)
        returns = returns.iloc[:, :num_stocks].mean()*252
        returns = np.dot(weights, returns)
        self.assertAlmostEqual(returns, target_return)

    def test_create_individual(self):
        """
        Tests that the create_individual function returns
        a chromosome where the number of stocks is between
        the minimum and maximum number of stocks.

        It is a random draw from a binomial distribution, so the
        probability of getting a number of stocks between the minimum
        and maximum is like 99%.
        """
        min_num = 1
        max_num = 20
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        individual = op.create_individual(log_returns)
        num_ones = np.count_nonzero(individual)
        self.assertGreaterEqual(num_ones, min_num)
        self.assertLessEqual(num_ones, max_num)

    def test_fitness_too_many_ETFs(self):
        """
        Tests the fitness function used in the
        GA. If there are too many 1's, the fitness
        should be the negative sum of the 1's.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        individual = [1]*log_returns.shape[1]
        fitness = op.fitness(individual, log_returns)
        self.assertEqual(fitness, -log_returns.shape[1])

    def test_fitness_too_few_ETFs(self):
        """
        Tests the fitness function used in the
        GA. If there are too few 1's, the fitness
        should be 0.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        individual = [0]*log_returns.shape[1]
        fitness = op.fitness(individual, log_returns)
        self.assertEqual(fitness, 0)

    def test_fitness_normal(self):
        """
        Tests the fitness function used in the
        GA. If there are the right number of 1's,
        the fitness should be between 1 and 5.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        num_stocks = 8
        individual = [1]*num_stocks + [0]*(log_returns.shape[1]-num_stocks)
        fitness = op.fitness(individual, log_returns.T)
        self.assertGreater(fitness, 1)
        self.assertLess(fitness, 5)

    def test_fitness_2(self):
        """
        Tests the fitness function that is
        required for pyGAD.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=False)
        num_stocks = 8
        individual = [1]*num_stocks + [0]*(log_returns.shape[1]-num_stocks)
        fitness = op.fitness_2(individual, 0)
        self.assertAlmostEqual(round(fitness, 4),
                               round(op.fitness(individual,
                                                log_returns.T), 4))

    def test_prepare_opt_inputs(self):
        """
        Tests that the prepare_opt_inputs function
        loads returns of the correct length.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=True)
        self.assertEqual(len(op.data), len(log_returns.T))

    def test_prepare_opt_inputs_variances(self):
        """
        Should have an equal number of variances to ETFs.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=True)
        self.assertEqual(len(op.variances), len(log_returns.T))

    def test_prepare_opt_inputs_forecasts(self):
        """
        Should have an equal number of forecasts to ETFs.
        """
        data = op.load_data('ETF_Prices.csv')
        log_returns = op.calculate_returns(data)
        op.prepare_opt_inputs(data, use_forecasts=True)
        self.assertEqual(len(op.expected_returns), len(log_returns.T))

    def test_prepare_opt_inputs_variances_null(self):
        """
        When not importing forecast variances, the variances
        variable should be None.
        """
        data = op.load_data('ETF_Prices.csv')
        op.prepare_opt_inputs(data, use_forecasts=False)
        self.assertEqual(op.variances, None)


if __name__ == '__main__':
    unittest.main()
