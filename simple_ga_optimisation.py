import numpy as np
import pandas as pd
import scipy.optimize as opt
from multiprocessing import Pool, Manager
import os


def load_data(filename: str) -> pd.DataFrame:
    prices_df = pd.read_csv(filename, index_col=0)
    prices_df = prices_df.dropna(axis=1, thresh=0.95*len(prices_df))
    prices_df = prices_df.fillna(method='ffill')
    return prices_df


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(data / data.shift(1))
    log_returns = log_returns.fillna(0)
    log_returns = log_returns.replace([np.inf, -np.inf], 0)
    return log_returns


def calculate_variances(log_returns: pd.DataFrame) -> pd.Series:
    return log_returns.var() * 252


def calculate_expected_returns(log_returns: pd.DataFrame, use_forecasts=False) -> pd.Series:
    if use_forecasts:
        expected_returns = pd.read_csv('Data/expected_returns.csv', index_col=0)
        expected_returns = expected_returns[expected_returns.index.isin(log_returns.columns)]
        for ticker in expected_returns.index:
            if expected_returns.loc[ticker, '0'] > 0.5 or np.isnan(expected_returns.loc[ticker, '0']):
                expected_returns.loc[ticker, '0'] = log_returns[ticker].mean() * 252
        return expected_returns['0']
    else:
        return log_returns.mean() * 252


def calculate_covariance_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    return log_returns.cov() * 252


def initialise_population(size, num_etfs, max_num_etfs):
    if max_num_etfs > num_etfs:
        raise ValueError("max_num_etfs cannot be greater than num_etfs")
    if max_num_etfs <= 0:
        raise ValueError("max_num_etfs must be a positive integer")
    p = max_num_etfs / num_etfs
    if not (0 <= p <= 1):
        raise ValueError("Calculated probability is out of valid range [0,1]")
    return np.random.binomial(1, p, size=(size, num_etfs))


def calculate_fitness(individual, expected_returns, cov_matrix_subset):
    selected_indices = individual == 1
    num_selected_etfs = np.sum(selected_indices)
    if num_selected_etfs < 8 or num_selected_etfs > 20:
        return -1e4
    if not selected_indices.any():
        return 0
    filtered_returns = expected_returns[selected_indices]
    weights = np.ones(num_selected_etfs) / num_selected_etfs
    portfolio_return = np.dot(weights, filtered_returns)
    if portfolio_return < 0.10:
        return -1e4
    portfolio_variance = np.dot(weights, np.dot(cov_matrix_subset, weights))
    return portfolio_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0


def select_parents(population, fitness, num_parents):
    ranked_indices = np.argsort(fitness)[::-1]
    top_indices = ranked_indices[:num_parents]
    return population[top_indices]


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    num_genes = offspring_size[1]
    for k in range(offspring_size[0]):
        parent1_idx = np.random.randint(0, len(parents))
        parent2_idx = np.random.randint(0, len(parents))
        for gene in range(num_genes):
            if np.random.rand() > 0.5:
                offspring[k, gene] = parents[parent1_idx, gene]
            else:
                offspring[k, gene] = parents[parent2_idx, gene]
    return offspring


def mutate(offspring, mutation_rate):
    for idx in range(offspring.shape[0]):
        for gene in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[idx, gene] = 1 - offspring[idx, gene]
    return offspring


def elitism(population, fitness, num_elites):
    elite_indices = np.argsort(-fitness)[:num_elites]
    return population[elite_indices], elite_indices


def genetic_algorithm(island_id, num_islands, data, num_generations,
                      population_size, mutation_rate, num_elites,
                      migration_interval, migration_rate, return_dict):
    num_etfs = data.shape[1]
    population = initialise_population(population_size, num_etfs, max_num_etfs=20)
    log_returns = calculate_returns(data)
    expected_returns = calculate_expected_returns(log_returns)
    cov_matrix = calculate_covariance_matrix(log_returns)
    best_overall_fitness = float('-inf')
    best_overall_individual = None
    for generation in range(num_generations):
        if generation % migration_interval == 0 and generation != 0:
            source_island = (island_id - 1 + num_islands) % num_islands
            migrants = return_dict.pop(source_island, None)
            if migrants is not None:
                num_received = len(migrants)
                if num_received > 0:
                    replace_indices = np.random.choice(
                        population_size, size=num_received, replace=False
                    )
                    population[replace_indices] = migrants
        fitness = []
        for ind in population:
            selected_indices = ind == 1
            if np.sum(selected_indices) > 0:
                cov_matrix_subset = cov_matrix.loc[selected_indices, selected_indices]
            else:
                cov_matrix_subset = np.array([[]])
            fitness.append(calculate_fitness(ind, expected_returns, cov_matrix_subset))
        fitness = np.array(fitness)
        elites, elite_indices = elitism(population, fitness, num_elites)
        current_best_fitness = np.max(fitness)
        print(f"Island {island_id}, Generation {generation}, Best Fitness: {current_best_fitness}")
        parents = select_parents(population, fitness, num_elites)
        offspring = crossover(parents, (population_size - num_elites, num_etfs))
        offspring = mutate(offspring, mutation_rate)
        population[:num_elites] = elites
        population[num_elites:] = offspring
        if (generation + 1) % migration_interval == 0:
            num_migrants = int(migration_rate * population_size)
            if num_migrants > 0:
                migrant_indices = np.argsort(-fitness)[:num_migrants]
                return_dict[island_id] = population[migrant_indices].copy()
        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_individual = population[np.argmax(fitness)].copy()
    return best_overall_individual, best_overall_fitness


def run_parallel_ga(data, num_generations, total_population_size,
                    mutation_rate, num_elites, migration_interval,
                    migration_rate):
    num_islands = os.cpu_count()
    manager = Manager()
    return_dict = manager.dict()

    def init_random_state():
        np.random.seed(None)

    with Pool(num_islands, initializer=init_random_state) as pool:
        island_pop_size = total_population_size // num_islands
        args = [(i, num_islands, data, num_generations, island_pop_size,
                 mutation_rate, num_elites, migration_interval,
                 migration_rate, return_dict) for i in range(num_islands)]
        results = pool.starmap(genetic_algorithm, args)
    best_fitness = float('-inf')
    best_solution = None
    for result in results:
        if result is not None:
            solution, fitness = result
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution
    return best_solution, best_fitness


def optimise_weights(best_solution, data):
    selected_etfs = data.columns[best_solution == 1]
    data = data[selected_etfs]
    log_returns = calculate_returns(data)
    expected_returns = calculate_expected_returns(log_returns)
    cov_matrix = calculate_covariance_matrix(log_returns)
    num_etfs = len(selected_etfs)
    bounds = [(0, 1) for _ in range(num_etfs)]
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    constraints.append({'type': 'ineq', 'fun': lambda x: np.dot(expected_returns, x) - 0.12})

    def objective(x):
        portfolio_return = np.dot(expected_returns, x)
        portfolio_volatility = np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))
        if portfolio_volatility == 0:
            return 0
        return -(portfolio_return / portfolio_volatility)

    result = opt.minimize(objective, x0=np.ones(num_etfs) / num_etfs, method='SLSQP', bounds=bounds,
                          constraints=constraints)
    return result


def print_results(tickers, optimal_weights, amount_to_allocate=5000):
    print("\nOptimised Portfolio Allocation:")
    for ticker, weight in zip(tickers, optimal_weights):
        if weight > 1e-4:
            print(f"{ticker}: {weight*100:.1f}% (${weight*amount_to_allocate:.2f})")


if __name__ == '__main__':
    data = load_data('Data/ETF_Prices.csv')
    etfs = pd.read_csv('Data/ETFs.csv')
    etfs_in_data_but_not_in_etfs = set(data.columns) - set(etfs['Tickers'])
    data = data.drop(columns=list(etfs_in_data_but_not_in_etfs))
    mutation_rate = 1 / data.shape[1]
    best_solution, best_fitness = run_parallel_ga(data,
                                                  num_generations=100,
                                                  total_population_size=8000,
                                                  mutation_rate=mutation_rate,
                                                  num_elites=100,
                                                  migration_interval=10,
                                                  migration_rate=0.1)
    if best_solution is not None:
        print("Best Solution (ETF Selection Vector):", best_solution.astype(int))
        print("Best Fitness (Sharpe Ratio from GA):", best_fitness)
        selected_etfs = data.columns[best_solution == 1]
        print(f"\nSelected {len(selected_etfs)} ETFs:")
        print(list(selected_etfs))
        optimised_result = optimise_weights(best_solution, data)
        if optimised_result.success:
            print_results(selected_etfs, optimised_result.x)
            final_sharpe = -optimised_result.fun
            print(f"\nFinal Optimised Sharpe Ratio: {final_sharpe:.4f}")
        else:
            print("\nWeight optimisation failed:", optimised_result.message)
    else:
        print("Genetic algorithm did not find a valid solution.")
