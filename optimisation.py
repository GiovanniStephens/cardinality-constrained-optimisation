import numpy as np
import pandas as pd
import scipy.optimize as opt
import pygad


MAX_NUM_STOCKS = 10
TARGET_RETURN = None
TARGET_RISK = 0.15
MAX_WEIGHT = 0.2
last_fitness = 0
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
    prices_df = pd.read_csv(filename, index_col=0)
    # Remove columns with 50% or more null values
    prices_df = prices_df.dropna(axis=1, thresh=int(len(prices_df)/2))
    # Fill the null values with the previous day's close price
    prices_df = prices_df.fillna(method='ffill')
    return prices_df


def optimize(data: pd.DataFrame, initial_weights: np.array,
             target_risk: float = None,
             target_return: float = None,
             max_weight: float = 0.3333) -> float:
    """
    Optimizes the portfolio using the Sharpe ratio.

    :data: pandas dataframe of the log returns data.
    :initial_weights: numpy array of initial weights.
    :target_risk: float of the target risk
                  (annualised portfolio standard deviation).
    :target_return: float of the target return
                    (annualised portfolio mean return).
    :max_weight: float of the maximum weight of any single stock.
    :return: pcipy optimization result.
    """
    cov = data.cov()*252
    expected_returns = data.mean()*252
    cons = [{'type': 'eq',
             'fun': lambda x: 1 - np.sum(x)}]
    if target_risk is not None and target_return is None:
        cons.append(
            {'type': 'eq',
             'fun': lambda W: target_risk -
             np.sqrt(np.dot(W.T,
                            np.dot(cov,
                                   W)))})
    if target_return is not None and target_risk is None:
        cons.append(
            {'type': 'eq',
             'fun': lambda W: target_return -
             np.sum(expected_returns*W)})
    bounds = tuple((0, max_weight) for _ in range(len(initial_weights)))
    sol = opt.minimize(sharpe_ratio,
                       initial_weights,
                       args=(expected_returns, cov),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=cons)
    return sol


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the log returns of the data.
    (Note that it replaces inf returns with 0.)

    :data: pandas dataframe of the data.
    :return: pandas dataframe of the log returns.
    """
    log_returns = np.log(data/data.shift(1))
    log_returns = log_returns.fillna(0)
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
        fitness = -optimize(subset.transpose(),
                            random_weights,
                            target_return=TARGET_RETURN,
                            target_risk=TARGET_RISK,
                            max_weight=MAX_WEIGHT)['fun']
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


def on_generation(ga_instance: pygad.GA) -> None:
    """
    On each generation in the GA, this function is called.

    :ga_instance: the GA instance.
    """
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


def cardinality_constrained_optimisation(num_children: int=1000, verbose: bool=False):
    """
    Performs the cardinality constrained optimisation.

    :num_children: int of the number of children to create.
    :return: the best Sharpe Ratio and the individual (portfolio).
    """
    if verbose:
        on_gen = on_generation
    else:
        on_gen = None
    ga_instance = pygad.GA(num_generations=25,
                           initial_population=np.array([create_individual(data)
                                                        for _ in range(num_children)]),
                           num_parents_mating=num_children//10,
                           gene_type=int,
                           init_range_low=0,
                           init_range_high=2,
                           mutation_probability=[0.9, 0.7],
                           parent_selection_type='rank',
                           random_mutation_min_val=-1,
                           random_mutation_max_val=1,
                           mutation_type="adaptive",
                           crossover_type="single_point",
                           crossover_probability=0.85,
                           fitness_func=fitness_2,
                           on_generation=on_gen,
                           stop_criteria='saturate_3')
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
    return solution


def create_portfolio(num_children=100) -> list:
    """
    Creates a cardinality constrained portfolio.

    :return: pandas dataframe of the portfolio.
    """
    individual = cardinality_constrained_optimisation(num_children=num_children)
    indices = np.array(individual).astype(bool)
    portfolio = data.transpose().iloc[:, indices].columns
    return list(portfolio)


if __name__ == '__main__':
    # Load the data
    prices_df = load_data('ETF_Prices.csv')
    # Calculate the returns
    log_returns = calculate_returns(prices_df)
    # Set the global data variable
    data = log_returns.transpose()
    # Run the cardinality constrained optimisation
    best_individual = cardinality_constrained_optimisation(num_children=100,
                                                           verbose=True)
    indeces = np.array(best_individual).astype(bool)
    # Print the portfolio metrics for the best portfolio we could find.
    best_portfolio_returns = log_returns.iloc[:, indeces]
    random_weights = np.random.random(np.count_nonzero(best_individual))
    random_weights /= np.sum(random_weights)
    sol = optimize(best_portfolio_returns,
                   random_weights,
                   target_return=TARGET_RETURN,
                   target_risk=TARGET_RISK,
                   max_weight=MAX_WEIGHT)
    # Print the optimal weights
    print(sol.x)
    best_weights = sol['x']
    # Print the portfolio return
    print(np.sum(best_weights*(best_portfolio_returns.mean()*252)))
    cov = best_portfolio_returns.cov()*252
    risk = np.sqrt(np.dot(best_weights.T, np.dot(cov, best_weights)))
    # Print the portfolio standard deviation
    print(risk)
    # Print the Sharpe Ratio
    print(fitness(best_individual, log_returns.T))
    # Print the portfolio constituents with their optimal allocations
    stock_allocations = {ticker: weight for ticker, weight in
                         zip(prices_df.iloc[:, indeces].columns,
                             sol.x)}
    print(stock_allocations)
