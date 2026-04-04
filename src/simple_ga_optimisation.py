import logging
import os
import time

import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager

from src.portfolio_utils import (
    load_prices_csv,
    calculate_log_returns,
    calculate_expected_returns,
    calculate_covariance_matrix,
    OptimisationResult,
)
from src.config import (
    DATA_LOOKBACK_DAYS, DATA_MIN_COVERAGE,
    ISLAND_GA_NUM_GENERATIONS, ISLAND_GA_POPULATION_SIZE,
    ISLAND_GA_NUM_ELITES, ISLAND_GA_MIGRATION_INTERVAL,
    ISLAND_GA_MIGRATION_RATE,
)

logger = logging.getLogger(__name__)


def initialise_population(size, num_etfs, max_num_etfs):
    if max_num_etfs > num_etfs:
        raise ValueError("max_num_etfs cannot be greater than num_etfs")
    if max_num_etfs <= 0:
        raise ValueError("max_num_etfs must be a positive integer")
    p = max_num_etfs / num_etfs
    if not (0 <= p <= 1):
        raise ValueError("Calculated probability is out of valid range [0,1]")
    return np.random.binomial(1, p, size=(size, num_etfs))


def calculate_fitness(individual, expected_returns, log_returns, min_etfs=8, max_etfs=20,
                      min_return=0.12):
    selected_indices = individual == 1
    num_selected_etfs = np.sum(selected_indices)
    if num_selected_etfs < min_etfs or num_selected_etfs > max_etfs:
        return -1e4
    if not selected_indices.any():
        return 0
    filtered_log_returns = log_returns.loc[:, selected_indices]
    cov_matrix_subset = calculate_covariance_matrix(filtered_log_returns)
    filtered_returns = expected_returns[selected_indices]
    weights = np.ones(num_selected_etfs) / num_selected_etfs
    portfolio_return = np.dot(weights, filtered_returns)
    if min_return is not None and portfolio_return < min_return:
        return -1e4
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_subset, weights))
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
                      migration_interval, migration_rate, return_dict,
                      convergence_log=None, start_time=None, time_budget=None,
                      min_etfs=8, max_etfs=20, min_return=0.12):
    num_etfs = data.shape[1]
    population = initialise_population(population_size, num_etfs, max_num_etfs=20)
    log_returns = calculate_log_returns(data)
    expected_returns = calculate_expected_returns(log_returns)
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
            fitness.append(calculate_fitness(ind, expected_returns, log_returns,
                                            min_etfs=min_etfs, max_etfs=max_etfs,
                                            min_return=min_return))
        fitness = np.array(fitness)
        elites, elite_indices = elitism(population, fitness, num_elites)
        current_best_fitness = np.max(fitness)
        logger.debug("Island %d, Generation %d, Best Fitness: %.6f", island_id, generation, current_best_fitness)
        if convergence_log is not None:
            convergence_log.append((time.time() - start_time, generation,
                                    current_best_fitness, float(np.mean(fitness)), island_id))
        if time_budget is not None and start_time is not None:
            if (time.time() - start_time) > time_budget:
                break
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

    logger.info(
        "Starting island GA: %d islands, %d generations, population=%d",
        num_islands, num_generations, total_population_size,
    )
    start = time.time()
    with Pool(num_islands, initializer=init_random_state) as pool:
        island_pop_size = total_population_size // num_islands
        args = [(i, num_islands, data, num_generations, island_pop_size,
                 mutation_rate, num_elites, migration_interval,
                 migration_rate, return_dict) for i in range(num_islands)]
        results = pool.starmap(genetic_algorithm, args)
    elapsed = time.time() - start
    best_fitness = float('-inf')
    best_solution = None
    for result in results:
        if result is not None:
            solution, fitness = result
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution
    logger.info("Island GA completed in %.1fs, best fitness=%.6f", elapsed, best_fitness)
    return best_solution, best_fitness


def optimise_weights(best_solution, data, min_return=0.12):
    from src.portfolio_utils import optimise_weights as _optimise_weights
    return _optimise_weights(best_solution, data, min_return=min_return)


class IslandGAOptimiser:
    """Parallel island-model genetic algorithm for portfolio selection."""

    def __init__(self, num_generations=ISLAND_GA_NUM_GENERATIONS,
                 population_size=ISLAND_GA_POPULATION_SIZE,
                 num_elites=ISLAND_GA_NUM_ELITES,
                 migration_interval=ISLAND_GA_MIGRATION_INTERVAL,
                 migration_rate=ISLAND_GA_MIGRATION_RATE,
                 min_securities=8, max_securities=20,
                 min_return=0.12):
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_elites = num_elites
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.min_securities = min_securities
        self.max_securities = max_securities
        self.min_return = min_return

    def optimise(self, prices: pd.DataFrame) -> OptimisationResult:
        mutation_rate = 1 / prices.shape[1]
        start = time.time()
        best_solution, best_fitness = run_parallel_ga(
            prices,
            num_generations=self.num_generations,
            total_population_size=self.population_size,
            mutation_rate=mutation_rate,
            num_elites=self.num_elites,
            migration_interval=self.migration_interval,
            migration_rate=self.migration_rate,
        )
        elapsed = time.time() - start

        if best_solution is None:
            return OptimisationResult(
                selected_tickers=[], weights=np.array([]),
                sharpe_ratio=float('-inf'),
                metadata={'error': 'No valid solution found'},
            )

        selected = list(prices.columns[best_solution == 1])
        weights = np.ones(len(selected)) / len(selected)

        result = optimise_weights(best_solution, prices,
                                  min_return=self.min_return)
        if result.success:
            weights = result.x
            best_fitness = -result.fun

        return OptimisationResult(
            selected_tickers=selected,
            weights=weights,
            sharpe_ratio=best_fitness,
            metadata={
                'num_generations': self.num_generations,
                'population_size': self.population_size,
                'elapsed_seconds': elapsed,
            },
        )


def print_results(tickers, optimal_weights, amount_to_allocate=5000):
    logger.info("Optimised Portfolio Allocation:")
    for ticker, weight in zip(tickers, optimal_weights):
        if weight > 1e-4:
            logger.info("  %s: %.1f%% ($%.2f)", ticker, weight*100, weight*amount_to_allocate)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Load from database (falls back to CSV if DB is empty)
    from src import db
    conn = db.get_connection()
    data = db.load_prices(conn, exchange='US')
    conn.close()
    if data.empty:
        logger.info("No data in DB, falling back to CSV")
        data = load_prices_csv('data/time_series_20251016_113257.csv', last_n_days=730)
    else:
        # Apply same filters: last 2 years, 95% coverage, ffill
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        cutoff = data.index[-1] - pd.Timedelta(days=DATA_LOOKBACK_DAYS)
        data = data[data.index >= cutoff]
        data = data.dropna(axis=1, thresh=int(DATA_MIN_COVERAGE * len(data)))
        data = data.ffill()
    logger.info("Loaded price data: %d rows x %d columns", *data.shape)
    num_generations = ISLAND_GA_NUM_GENERATIONS
    total_population_size = ISLAND_GA_POPULATION_SIZE
    mutation_rate = 1 / data.shape[1]
    num_elites = ISLAND_GA_NUM_ELITES
    migration_interval = ISLAND_GA_MIGRATION_INTERVAL
    migration_rate_val = ISLAND_GA_MIGRATION_RATE

    ga_start = time.time()
    best_solution, best_fitness = run_parallel_ga(data,
                                                  num_generations=num_generations,
                                                  total_population_size=total_population_size,
                                                  mutation_rate=mutation_rate,
                                                  num_elites=num_elites,
                                                  migration_interval=migration_interval,
                                                  migration_rate=migration_rate_val)
    ga_elapsed = time.time() - ga_start

    if best_solution is not None:
        logger.info("Best Solution (ETF Selection Vector): %s", best_solution.astype(int))
        logger.info("Best Fitness (Sharpe Ratio from GA): %.6f", best_fitness)
        selected_etfs = data.columns[best_solution == 1]
        logger.info("Selected %d ETFs: %s", len(selected_etfs), list(selected_etfs))
        optimised_result = optimise_weights(best_solution, data)
        if optimised_result.success:
            print_results(selected_etfs, optimised_result.x, amount_to_allocate=20000)
            final_sharpe = -optimised_result.fun
            logger.info("Final Optimised Sharpe Ratio: %.4f", final_sharpe)

            # Save to database
            log_returns = calculate_log_returns(data[selected_etfs])
            expected_rets = calculate_expected_returns(log_returns)
            portfolio_return = float(np.dot(optimised_result.x, expected_rets))
            cov_matrix = calculate_covariance_matrix(log_returns)
            portfolio_vol = float(np.sqrt(np.dot(optimised_result.x.T,
                                                 np.dot(cov_matrix, optimised_result.x))))

            from src import db
            conn = db.get_connection()
            run_id = db.save_optimisation_run(conn,
                params={
                    'script': 'simple_ga_optimisation',
                    'data_source': 'investnow',
                    'num_generations': num_generations,
                    'total_population_size': total_population_size,
                    'mutation_rate': mutation_rate,
                    'num_elites': num_elites,
                    'migration_interval': migration_interval,
                    'migration_rate': migration_rate_val,
                    'num_islands': os.cpu_count(),
                    'min_securities': 8,
                    'max_securities': 20,
                },
                results={
                    'best_sharpe': final_sharpe,
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_vol,
                    'num_selected': len(selected_etfs),
                    'elapsed_seconds': ga_elapsed,
                },
                holdings=list(zip(selected_etfs, optimised_result.x)),
                exchange='NZX')
            logger.info("Run saved to database (id=%d)", run_id)
            conn.close()
        else:
            logger.warning("Weight optimisation failed: %s", optimised_result.message)
    else:
        logger.error("Genetic algorithm did not find a valid solution.")
