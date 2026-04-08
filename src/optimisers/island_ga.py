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
    equal_weight_fitness,
    OptimisationResult,
)
from src.config import TRADING_DAYS_PER_YEAR as _TRADING_DAYS
from src.optimisers.base import BaseOptimiser
from src.config import (
    DATA_LOOKBACK_DAYS, DATA_MIN_COVERAGE, DATA_FFILL_LIMIT,
    ISLAND_GA_NUM_GENERATIONS, ISLAND_GA_POPULATION_SIZE,
    ISLAND_GA_NUM_ELITES, ISLAND_GA_MIGRATION_INTERVAL,
    ISLAND_GA_MIGRATION_RATE,
    ISLAND_GA_MIN_SECURITIES, ISLAND_GA_MAX_SECURITIES, ISLAND_GA_MIN_RETURN,
)

logger = logging.getLogger(__name__)


def initialise_population(size, num_etfs, max_num_etfs):
    """Create a random binary population matrix with expected cardinality ~ max_num_etfs."""
    if max_num_etfs > num_etfs:
        raise ValueError("max_num_etfs cannot be greater than num_etfs")
    if max_num_etfs <= 0:
        raise ValueError("max_num_etfs must be a positive integer")
    p = max_num_etfs / num_etfs
    if not (0 <= p <= 1):
        raise ValueError("Calculated probability is out of valid range [0,1]")
    return np.random.binomial(1, p, size=(size, num_etfs))


def calculate_fitness(individual, expected_returns, centered_returns, T_obs,
                      min_etfs=ISLAND_GA_MIN_SECURITIES,
                      max_etfs=ISLAND_GA_MAX_SECURITIES, min_return=ISLAND_GA_MIN_RETURN):
    """Equal-weight Sharpe ratio with cardinality and return constraints."""
    return batch_fitness(individual[np.newaxis, :], expected_returns,
                         centered_returns, T_obs,
                         min_etfs=min_etfs, max_etfs=max_etfs,
                         min_return=min_return)[0]


def batch_fitness(population, expected_returns, centered_returns, T_obs,
                  min_etfs=ISLAND_GA_MIN_SECURITIES,
                  max_etfs=ISLAND_GA_MAX_SECURITIES, min_return=ISLAND_GA_MIN_RETURN):
    """Vectorised equal-weight Sharpe ratio for the entire population at once.

    Computes portfolio variance directly from centered returns without
    forming an N×N covariance matrix. For equal weights on a binary mask:
      port_var = (1/(n²(T-1))) * Σ_t (Σ_{j∈selected} centered_tj)²

    Vectorized across the population via a single matrix multiply:
      S = centered_returns @ pop.T   (T × pop_size)
      port_var = sum(S², axis=0) / ((T-1) * n²) * 252
    """
    pop = population.astype(np.float64)
    counts = pop.sum(axis=1)

    # Cardinality constraint mask
    valid = (counts >= min_etfs) & (counts <= max_etfs)

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        # Portfolio return: (pop @ expected_returns) / counts
        raw_ret = pop @ expected_returns

        # Portfolio variance via centered returns (no N×N intermediate)
        S = centered_returns @ pop.T             # T × pop_size
        raw_var = np.sum(S ** 2, axis=0)         # unnormalized sum of squared sums

        # Avoid division by zero
        safe_counts = np.where(counts > 0, counts, 1.0)
        port_ret = raw_ret / safe_counts
        port_var = raw_var / ((T_obs - 1) * safe_counts ** 2) * _TRADING_DAYS

        # Min return constraint
        if min_return is not None:
            valid = valid & (port_ret >= min_return)

        # Sharpe ratio
        fitness = np.where(
            valid & (port_var > 0),
            port_ret / np.sqrt(port_var),
            -1e4
        )

    return fitness


def select_parents(population, fitness, num_parents):
    """Select the top num_parents individuals by fitness (rank selection)."""
    ranked_indices = np.argsort(fitness)[::-1]
    top_indices = ranked_indices[:num_parents]
    return population[top_indices]


def crossover(parents, offspring_size):
    """Uniform crossover: each gene randomly inherited from one of two parents."""
    num_offspring, num_genes = offspring_size
    num_parents = len(parents)
    parent1_indices = np.random.randint(0, num_parents, size=num_offspring)
    parent2_indices = np.random.randint(0, num_parents, size=num_offspring)
    mask = np.random.random((num_offspring, num_genes)) > 0.5
    offspring = np.where(mask, parents[parent1_indices], parents[parent2_indices])
    return offspring


def mutate(offspring, mutation_rate):
    """Flip each gene with probability mutation_rate."""
    mutation_mask = np.random.random(offspring.shape) < mutation_rate
    offspring[mutation_mask] = 1 - offspring[mutation_mask]
    return offspring


def elitism(population, fitness, num_elites):
    """Return the top num_elites individuals and their indices."""
    elite_indices = np.argsort(-fitness)[:num_elites]
    return population[elite_indices], elite_indices


def genetic_algorithm(island_id, num_islands, data, num_generations,
                      population_size, mutation_rate, num_elites,
                      migration_interval, migration_rate, return_dict,
                      convergence_log=None, start_time=None, time_budget=None,
                      min_etfs=ISLAND_GA_MIN_SECURITIES, max_etfs=ISLAND_GA_MAX_SECURITIES,
                      min_return=ISLAND_GA_MIN_RETURN):
    """Run a single island GA: evolve population with migration between islands."""
    num_etfs = data.shape[1]
    population = initialise_population(population_size, num_etfs, max_num_etfs=max_etfs)
    log_returns = calculate_log_returns(data)
    expected_returns = calculate_expected_returns(log_returns).values
    centered_returns = (log_returns - log_returns.mean(axis=0)).values
    T_obs = centered_returns.shape[0]
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
        fitness = batch_fitness(population, expected_returns, centered_returns,
                                T_obs, min_etfs=min_etfs, max_etfs=max_etfs,
                                min_return=min_return)
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


def _init_random_state():
    """Pool initializer — reseed numpy RNG in each worker (top-level for spawn)."""
    np.random.seed(None)


def run_parallel_ga(data, num_generations, total_population_size,
                    mutation_rate, num_elites, migration_interval,
                    migration_rate, min_etfs=ISLAND_GA_MIN_SECURITIES,
                    max_etfs=ISLAND_GA_MAX_SECURITIES, min_return=ISLAND_GA_MIN_RETURN):
    """Distribute island GAs across CPU cores and return the best solution."""
    num_islands = os.cpu_count()
    manager = Manager()
    return_dict = manager.dict()

    logger.info(
        "Starting island GA: %d islands, %d generations, population=%d",
        num_islands, num_generations, total_population_size,
    )
    start = time.time()
    with Pool(num_islands, initializer=_init_random_state) as pool:
        island_pop_size = total_population_size // num_islands
        args = [(i, num_islands, data, num_generations, island_pop_size,
                 mutation_rate, num_elites, migration_interval,
                 migration_rate, return_dict,
                 None, None, None,
                 min_etfs, max_etfs, min_return) for i in range(num_islands)]
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


def optimise_weights(best_solution, data, min_return=ISLAND_GA_MIN_RETURN):
    """SLSQP weight refinement for a binary selection vector."""
    from src.portfolio_utils import optimise_weights as _optimise_weights
    return _optimise_weights(best_solution, data, min_return=min_return)


class IslandGAOptimiser(BaseOptimiser):
    """Parallel island-model genetic algorithm for portfolio selection."""

    def __init__(self, num_generations=ISLAND_GA_NUM_GENERATIONS,
                 population_size=ISLAND_GA_POPULATION_SIZE,
                 num_elites=ISLAND_GA_NUM_ELITES,
                 migration_interval=ISLAND_GA_MIGRATION_INTERVAL,
                 migration_rate=ISLAND_GA_MIGRATION_RATE,
                 min_securities=ISLAND_GA_MIN_SECURITIES,
                 max_securities=ISLAND_GA_MAX_SECURITIES,
                 min_return=ISLAND_GA_MIN_RETURN):
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
            min_etfs=self.min_securities,
            max_etfs=self.max_securities,
            min_return=self.min_return,
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
    """Log the portfolio allocation with dollar amounts."""
    logger.info("Optimised Portfolio Allocation:")
    for ticker, weight in zip(tickers, optimal_weights):
        if weight > 1e-4:
            logger.info("  %s: %.1f%% ($%.2f)", ticker, weight*100, weight*amount_to_allocate)


if __name__ == '__main__':
    from src.logging_config import setup_logging
    setup_logging(
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
        data = data.ffill(limit=DATA_FFILL_LIMIT)
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
                    'script': 'island_ga',
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
