import numpy as np
import pandas as pd
import scipy.optimize as opt
from pyeasyga import pyeasyga

MAX_NUM_STOCKS = 10

def sharpe_ratio(weights: np.array, returns: pd.DataFrame) -> float:
    """
    Calculates the Sharpe ratio of a portfolio.

    :weights: numpy array of weights.
    :returns: numpy array of log returns.
    :return: float of the negative Sharpe ratio.
    """
    p_returns = np.sum((returns.mean()*weights*252))
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    return -p_returns/p_volatility


def load_data(filename: str) -> pd.DataFrame:
    """
    Loads the data from a CSV file.

    :filename: string of the filename.
    :return: pandas dataframe of the data.
    """
    return pd.read_csv(filename)


def optimize(data: pd.DataFrame, initial_weights: np.array,
             max_weight: float=0.5) -> float:
    """
    Optimizes the portfolio using the Sharpe ratio.

    :data: pandas dataframe of the data.
    :initial_weights: numpy array of initial weights.
    :return: numpy array of optimized weights.
    """
    cons = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})
    bounds = tuple((0, max_weight) for x in range(len(initial_weights)))
    sol = opt.minimize(sharpe_ratio, initial_weights, args=(data),
                       method='SLSQP', bounds=bounds, constraints=cons)
    return -sol['fun']


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the log returns of the data.

    :data: pandas dataframe of the data.
    :return: pandas dataframe of the log returns.
    """
    log_returns = np.log(data/data.shift(1))
    log_returns = log_returns.dropna()
    log_returns = log_returns.replace([np.inf, -np.inf], 0)
    return log_returns


def fitness(individual, data):
    """
    Fitness function for the genetic algorithm.

    :individual: binary array.
    :data: pandas dataframe of the data.
    :return: float of the fitness (i.e. Sharpe Ratio)
    """
    fitness = 0
    if individual.count(1) <= MAX_NUM_STOCKS:
        random_weights = np.random.random(individual.count(1))
        random_weights /= np.sum(random_weights)
        subset = data.iloc[np.array(individual).astype(bool),:]
        fitness = optimize(subset.transpose(), random_weights)
    print(fitness)
    return fitness


def mutate(individual):
    """
    Mutates a random bit in an individual.

    :individual: numpy array of weights.
    :return: numpy array of mutated weights.
    """
    mutate_index = np.random.randint(0, len(individual))
    if individual[mutate_index] == 0:
        individual[mutate_index] = np.random.binomial(1,0.5)
    else:
        individual[mutate_index] = 0


def create_individual(data):
    return [np.random.binomial(1,0.05) for _ in range(len(data))]


def cardinality_constrained_optimisation(data: pd.DataFrame):
    ga = pyeasyga.GeneticAlgorithm(data.transpose(),
                                   population_size=2000,
                                   generations=50,
                                   crossover_probability=0.85,
                                   mutation_probability=0.9,
                                   elitism=True,
                                   maximise_fitness=True)
    ga.fitness_function = fitness
    ga.mutate_function = mutate
    ga.create_individual = create_individual
    ga.run()
    return ga.best_individual()


if __name__ == '__main__':
    prices_df = load_data('ETF_Prices.csv')
    prices_df = prices_df.drop(prices_df.columns[0], axis=1)
    log_returns = calculate_returns(prices_df)
    best_individual = cardinality_constrained_optimisation(log_returns)
    print(best_individual)