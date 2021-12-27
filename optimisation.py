import numpy as np
import pandas as pd
import scipy.optimize as opt
from pyeasyga import pyeasyga
import pygad


MAX_NUM_STOCKS = 10
data = None


def sharpe_ratio(weights: np.array, returns: list, cov: list) -> float:
    """
    Calculates the Sharpe ratio of a portfolio.

    :weights: numpy array of weights.
    :p_returns: list of the portfolio's expected return.
    :return: float of the negative Sharpe ratio.
    """
    p_returns = np.sum(weights*returns)
    p_volatility = np.sqrt(np.dot(weights.T,
                                  np.dot(cov,
                                         weights)))
    return -p_returns/p_volatility


def load_data(filename: str) -> pd.DataFrame:
    """
    Loads the data from a CSV file in the local directory.

    :filename: string of the filename.
    :return: pandas dataframe of the data.
    """
    return pd.read_csv(filename)


def optimize(data: pd.DataFrame, initial_weights: np.array,
             target_risk: float = 0.15,
             max_weight: float = 0.3333) -> float:
    """
    Optimizes the portfolio using the Sharpe ratio.

    :data: pandas dataframe of the log returns data.
    :initial_weights: numpy array of initial weights.
    :max_weight: float of the maximum weight of any single stock.
    :return: numpy array of optimized weights.
    """
    cov = data.cov()*252
    expected_returns = data.mean()*252
    cons = ({'type': 'eq',
             'fun': lambda x: 1 - np.sum(x)},
            {'type': 'eq',
             'fun': lambda W: target_risk -
                            np.sqrt(np.dot(W.T,
                                           np.dot(cov,
                                                  W)))})
    bounds = tuple((0, max_weight) for _ in range(len(initial_weights)))
    sol = opt.minimize(sharpe_ratio,
                       initial_weights,
                       args=(expected_returns, cov),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=cons)
    return -sol['fun']


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the log returns of the data.
    (Note that it replaces inf returns with 0.)

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
    :data: pandas dataframe of the returns data.
    :return: float of the fitness (i.e. Sharpe Ratio)
    """
    fitness = 0
    if np.count_nonzero(individual) <= MAX_NUM_STOCKS \
       and np.count_nonzero(individual) > 1:
        random_weights = np.random.random(np.count_nonzero(individual))
        random_weights /= np.sum(random_weights)
        subset = data.iloc[np.array(individual).astype(bool), :]
        fitness = optimize(subset.transpose(), random_weights, max_weight=0.2)
    else:
        fitness = -np.count_nonzero(individual)
    return fitness


def fitness_2(solution: np.array, solution_idx: int) -> float:
    """
    Fitness function for the pygad genetic algorithm.

    :solution: binary array.
    :solution_idx: int of the solution index.
    """
    fit = fitness(solution, data)
    return fit


def mutate(individual):
    """
    Mutates a random bit in an individual.

    :individual: numpy array of weights.
    :return: numpy array of mutated weights.
    """
    mutate_index = np.random.randint(0, len(individual))
    if individual[mutate_index] == 0:
        individual[mutate_index] = np.random.binomial(1, 0.8)
    else:
        individual[mutate_index] = 0


def create_individual(data):
    """
    Creates an individual.

    :data: pandas dataframe of the returns data.
    :return: a binary array of the individual.
    """
    individual = np.zeros(len(data))
    for i in range(len(individual)):
        individual[i] = np.random.binomial(1, MAX_NUM_STOCKS/len(individual))
    return individual


def crossover(parent_1, parent_2):
    """
    Crossover function for the genetic algorithm.

    :parent_1: binary array.
    :parent_2: binary array.
    :return: a binary array of the offspring.
    """
    crossover_index = np.random.randint(0, len(parent_1))
    child_1 = np.append(parent_1[:crossover_index], parent_2[crossover_index:])
    child_2 = np.append(parent_2[crossover_index:], parent_1[:crossover_index])
    return child_1, child_2


def cardinality_constrained_optimisation(data: pd.DataFrame) -> list:
    """
    Performs the cardinality constrained optimisation.

    :data: pandas dataframe of the returns data.
    :return: the best Sharpe Ratio and the individual (portfolio).
    """
    ga = pyeasyga.GeneticAlgorithm(data.transpose(),
                                   population_size=1000,
                                   generations=3,
                                   crossover_probability=0.85,
                                   mutation_probability=0.01,
                                   elitism=True,
                                   maximise_fitness=True)
    ga.fitness_function = fitness
    ga.mutate_function = mutate
    ga.create_individual = create_individual
    ga.crossover_function = crossover
    ga.run()
    return ga.best_individual()


last_fitness = 0
def on_generation(ga_instance: pygad.GA) -> None:
    """
    On each generation in the GA, this function is called.

    :ga_instance: the GA instance.
    """
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


def cardinality_constrained_optimisation_2():
    """
    Performs the cardinality constrained optimisation.

    :data: pandas dataframe of the returns data.
    :return: the best Sharpe Ratio and the individual (portfolio).
    """
    ga_instance = pygad.GA(num_generations=25,
                  initial_population=np.array([create_individual(data) for _ in range(500)]),
                  num_parents_mating=100,
                #   sol_per_pop=3000,
                #   num_genes=data.shape[0],
                  gene_type=int,
                  init_range_low=0,
                  init_range_high=2,
                  mutation_probability=[0.9, 0.7],
                  parent_selection_type='rank',
                #   mutation_probability=0.1,
                  random_mutation_min_val=-1,
                  random_mutation_max_val=1,
                  mutation_type="adaptive",
                  crossover_type="single_point",
                  crossover_probability=0.85,
                  fitness_func=fitness_2,
                  on_generation=on_generation,
                  stop_criteria='saturate_3')
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    return solution


if __name__ == '__main__':
    prices_df = load_data('ETF_Prices.csv')
    prices_df = prices_df.drop(prices_df.columns[0], axis=1)
    # Remove columns with 50% or more null values
    prices_df = prices_df.dropna(axis=1, thresh=int(len(prices_df)/2))
    # Fill the null values with the previous day's close price
    prices_df = prices_df.fillna(method='ffill')
    log_returns = calculate_returns(prices_df)
    data = log_returns.transpose()
    # best_individual = cardinality_constrained_optimisation(log_returns)
    best_individual = cardinality_constrained_optimisation_2()
    # print(best_individual[0])
    print(fitness(best_individual, log_returns.T))
    print(prices_df.iloc[:, np.array(best_individual).astype(bool)].columns)
    # print(best_individual)
