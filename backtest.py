import optimisation as op
import numpy as np
import multiprocessing as mp
from multiprocessing import cpu_count
from matplotlib import pyplot as plt
import seaborn as sns

# This is the backtest sample size.
# Ideally, it would be >= 30 to get a robust statistic.
# It is a bit slow creating the cardinality-constrained portfolios,
# even with parallelisation.
NUM_PORTFOLIOS = 30

# This is the number of children in the GA.
# This works best on my computer with a number b/w 1000-2000.
# Under 50, the GA converges too quickly and the results are
# equal to a random selection.
NUM_CHILDREN = 500

# This is the number of days out of sample for the backtest.
NUM_DAYS_OUT_OF_SAMPLE = 150

# Used in the multiprocessing to calculate NUM_PORTFOLIOS
# cardinality-constrained portfolios.
NUM_JOBS = cpu_count()

USE_FORECAST = False

# Load the price data.
data = op.load_data('ETF_Prices.csv')


def get_random_weights(portfolio):
    """
    Creates a set of random weighting that sum to 1.

    :portfolio: Input portfolio to get the length.
    :return: A set of random weights equal in length to the portfolio.
    """
    random_weights = np.random.random(len(portfolio))
    random_weights /= np.sum(random_weights)
    return random_weights


def optimal_weights(portfolio):
    """
    Finds the optimal weights (allocations) for the
    input portfolio.

    :portfolio: The input portfolio. List of ticker strings.
    :return: A list of weights for the input portfolio.
    """
    random_weights = get_random_weights(portfolio)
    return op.optimize(op.data.loc[portfolio, :].transpose(),
                       random_weights,
                       max_weight=max(1/(len(portfolio)-1), 0.2))['x']


def run_portfolio(portfolio, weights, log_returns):
    """
    This is the backtest part of the code.
    It takes in a portfolio and weights and returns the portfolio's
    out-of-sample returns performance. This assumes that there
    is not rebalancing in the whole out-of-sample period.

    :portfolio: The input portfolio. List of ticker strings.
    :weights: The input weights. List of floats.
    :log_returns: The input log returns. Dataframe.
    """
    portfolio_returns = []
    # Run the backtest from the first out-of-sample day
    start_i = op.data.shape[1] + 1
    for i in range(NUM_DAYS_OUT_OF_SAMPLE-1):
        subset_returns = log_returns.transpose().loc[portfolio]
        current_step_returns = subset_returns.iloc[:, start_i+i]
        weighted_returns = np.sum(current_step_returns*weights)
        portfolio_returns.append(weighted_returns)
        # update the weights
        weights = weights*np.exp(current_step_returns) / \
            (1+portfolio_returns[-1])
    return portfolio_returns


def fitness(portfolio_returns):
    """
    Calculates the portfolio Sharpe Ratio.

    :portfolio_returns: The input portfolio returns. List of floats.
    :return: The fitness of the portfolio.
    """
    return (np.mean(portfolio_returns) * 252) / \
           (np.std(portfolio_returns) * np.sqrt(252))


def create_portfolio(num_children):
    """
    Creates a cardinality-constrained portfolio with the
    training data.

    :num_children: The number of children in the GA to create.
    :return: A list of tickers.
    """
    op.prepare_opt_inputs(data.iloc[:-NUM_DAYS_OUT_OF_SAMPLE, :], USE_FORECAST)
    op.TARGET_RETURN = None
    portfolio = op.create_portfolio(num_children)
    return portfolio


def difference_of_means_hypothesis_test(sample_1, sample_2):
    """
    Calculates the t statistic for the difference of means.

    Second sample mean minus the first. (i.e. if positive,
    the second is greater than the first.)

    :sample_1: The first sample. List of floats.
    :sample_2: The second sample. List of floats.
    :return: The t statistic.
    """
    return (np.mean(sample_2) - np.mean(sample_1)) / \
        np.sqrt(np.var(sample_1) / len(sample_1) +
                np.var(sample_2) / len(sample_2))


def main():
    # Create a pool of workers
    pool = mp.Pool(processes=NUM_JOBS)
    # Create a list of cardinality-constrained portfolios
    portfolios = pool.map(create_portfolio,
                          [NUM_CHILDREN]*NUM_PORTFOLIOS)
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    global USE_FORECAST
    USE_FORECAST = True
    pool = mp.Pool(processes=NUM_JOBS)
    # Create a list of cardinality-constrained portfolios
    forecast_portfolios = pool.map(create_portfolio,
                          [NUM_CHILDREN]*NUM_PORTFOLIOS)
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Create a set of randomly selected portfolios
    op.prepare_opt_inputs(data.iloc[:-NUM_DAYS_OUT_OF_SAMPLE, :], USE_FORECAST)
    op.TARGET_RETURN = None
    log_returns = op.calculate_returns(data)
    random_portfolios = []
    for i in range(NUM_PORTFOLIOS):
        indices = op.create_individual(op.data).astype(bool)
        random_portfolios.append(list(log_returns.iloc[:, indices].columns))

    # Create starting allocations for each of the portfolios
    portfolios_weights = [optimal_weights(portfolio)
                          for portfolio in portfolios]
    forecast_portfolios_weights = [optimal_weights(portfolio)
                                   for portfolio in forecast_portfolios]
    portfolios_random_weights = [get_random_weights(portfolio)
                                 for portfolio in portfolios]
    random_portfolios_weights = [optimal_weights(portfolio)
                                 for portfolio in random_portfolios]
    random_portfolios_random_weights = [get_random_weights(portfolio)
                                        for portfolio in random_portfolios]

    # Run the backtests for each of the portfolios
    portfolios_fitness = [fitness(run_portfolio(portfolio,
                                                weights,
                                                log_returns))
                          for portfolio,
                          weights
                          in zip(portfolios,
                                 portfolios_weights)]
    forecast_portfolios_fitness = [fitness(run_portfolio(portfolio,
                                                         weights,
                                                         log_returns))
                                    for portfolio,
                                    weights
                                    in zip(forecast_portfolios,
                                           forecast_portfolios_weights)]
    portfolios_random_fitness = [fitness(run_portfolio(portfolio,
                                                       weights,
                                                       log_returns))
                                 for portfolio,
                                 weights
                                 in zip(portfolios,
                                        portfolios_random_weights)]
    random_portfolios_fitness = [fitness(run_portfolio(portfolio,
                                                       weights,
                                                       log_returns))
                                 for portfolio,
                                 weights
                                 in zip(random_portfolios,
                                        random_portfolios_weights)]
    random_portfolios_random_fitness = [fitness(run_portfolio(portfolio,
                                                              weights,
                                                              log_returns))
                                        for portfolio,
                                        weights
                                        in zip(random_portfolios,
                                               random_portfolios_random_weights)]

    print(f'Cardinality-constrained, optimised portfolio mean: \
          {np.array(portfolios_fitness).mean()}')
    print(f'Cardinality-constrained, optimised portfolio std: \
          {np.array(portfolios_fitness).std()}')
    print(f'Cardinality-constrained, optimised portfolio w/ forecasts mean: \
          {np.array(forecast_portfolios_fitness).mean()}')
    print(f'Cardinality-constrained, optimised portfolio w/ forecasts std: \
          {np.array(forecast_portfolios_fitness).std()}')
    print(f'Cardinality-constrained, random weightings portfolio mean: \
          {np.array(portfolios_random_fitness).mean()}')
    print(f'Cardinality-constrained, random weightings portfolio std: \
          {np.array(portfolios_random_fitness).std()}')
    print(f'\nRandom selections, optimised portfolio mean: \
          {np.array(random_portfolios_fitness).mean()}')
    print(f'Random selections, optimised portfolio std: \
          {np.array(random_portfolios_fitness).std()}')
    print(f'Random selections, random weightings portfolio mean: \
          {np.array(random_portfolios_random_fitness).mean()}')
    print(f'Random selections, random weightings portfolio std: \
          {np.array(random_portfolios_random_fitness).std()}')

    # Perform a hypothesis test to see if the difference in means is significant
    print(f'\nCardinality-constrained, optimised portfolio vs. random weightings t-statistic: \
        {difference_of_means_hypothesis_test(portfolios_fitness, portfolios_random_fitness)}')
    print(f'Random selections, optimised portfolio vs. random weightings t-statistic: \
        {difference_of_means_hypothesis_test(random_portfolios_fitness, random_portfolios_random_fitness)}')
    print(f'Cardinality-constrained, optimised portfolio vs. random selection, optimised t-statistic: \
        {difference_of_means_hypothesis_test(portfolios_fitness, random_portfolios_fitness)}')
    print(f'Cardinality-constrained, optimised portfolio vs. random selection, random weightings t-statistic: \
        {difference_of_means_hypothesis_test(portfolios_fitness, random_portfolios_random_fitness)}')
    print(f'Cardinality-constrained, optimised portfolio vs. cardinality-constrained w/ forecast values and optimal weightings t-statistic: \
        {difference_of_means_hypothesis_test(portfolios_fitness, forecast_portfolios_fitness)}')

    # Plot histograms of the fitnesses using seaborn
    sns.set(style="whitegrid")
    sns.histplot(portfolios_fitness,
                 kde=True,
                 color='orange',
                 label='Cardinality-constrained, optimised portfolios')
    sns.histplot(portfolios_random_fitness,
                 kde=True,
                 color='blue',
                 label='Cardinality-constrained, random weightings portfolios')
    sns.histplot(random_portfolios_fitness,
                 kde=True,
                 color='green',
                 label='Random selections, optimised portfolios')
    sns.histplot(random_portfolios_random_fitness,
                 kde=True,
                 color='red',
                 label='Random selections, random weightings portfolios')
    sns.histplot(forecast_portfolios_fitness,
                 kde=True,
                 color='purple',
                 label='Cardinality-constrained, optimised portfolios w/ forecasts')
    plt.legend()
    plt.title('Fitness Distribution')
    plt.xlabel('Fitness')
    # # Change the x-axis to 0.5 increments
    # plt.xticks(np.arange(-3, 2.6, 0.5))
    plt.ylabel('Density')
    plt.show()


if __name__ == '__main__':
    main()
