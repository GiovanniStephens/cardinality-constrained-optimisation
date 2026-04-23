import logging
import os
import time

import numpy as np
import pandas as pd
from multiprocessing import Pool

from src.returns import calculate_log_returns, calculate_expected_returns
from src.metrics import equal_weight_fitness
from src.weights import optimise_weights
from src.optimisers.base import BaseOptimiser, OptimisationResult
from src.config import TRADING_DAYS_PER_YEAR as _TRADING_DAYS
from src.config import (
    GA_MIN_SECURITIES, GA_MAX_SECURITIES,
    ISLAND_GA_MIN_SECURITIES, ISLAND_GA_MAX_SECURITIES,
    MC_NUM_TRIALS,
    ETF_PRICES_CSV,
)

logger = logging.getLogger(__name__)


def random_portfolio(num_etfs, min_num_etfs, max_num_etfs):
    """Generate a random binary selection vector."""
    num_selected = np.random.randint(min_num_etfs, max_num_etfs + 1)
    portfolio = np.zeros(num_etfs, dtype=int)
    selected_indices = np.random.choice(num_etfs, num_selected, replace=False)
    portfolio[selected_indices] = 1
    return portfolio


def calculate_fitness(portfolio, expected_returns, cov_matrix,
                      min_num_etfs, max_num_etfs):
    """Sharpe ratio for an equal-weight portfolio (used during search)."""
    return equal_weight_fitness(portfolio, expected_returns, cov_matrix,
                                min_count=min_num_etfs, max_count=max_num_etfs)


def monte_carlo_search(data, trials, min_num_etfs, max_num_etfs):
    """Run *trials* random portfolio evaluations, return the best."""
    log_returns = calculate_log_returns(data)
    expected_returns = calculate_expected_returns(log_returns).values
    centered = (log_returns - log_returns.mean(axis=0)).values
    T_obs = centered.shape[0]
    num_etfs = data.shape[1]

    best_fitness = float('-inf')
    best_portfolio = None

    for _ in range(trials):
        portfolio = random_portfolio(num_etfs, min_num_etfs, max_num_etfs)
        selected = portfolio == 1
        n = np.sum(selected)
        if n < min_num_etfs or n > max_num_etfs:
            continue
        s = centered[:, selected].sum(axis=1)
        port_var = np.sum(s ** 2) / ((T_obs - 1) * n ** 2) * _TRADING_DAYS
        port_ret = expected_returns[selected].mean()
        fitness = port_ret / np.sqrt(port_var) if port_var > 0 else 0.0
        if fitness > best_fitness:
            best_fitness = fitness
            best_portfolio = portfolio

    return best_portfolio, best_fitness


def parallel_monte_carlo(data, num_trials, num_processes,
                         min_num_etfs, max_num_etfs):
    """Distribute Monte Carlo search across CPU cores."""
    logger.info(
        "Starting Monte Carlo: %d trials across %d processes",
        num_trials, num_processes,
    )
    start = time.time()
    trials_per_process = num_trials // num_processes
    with Pool(num_processes) as pool:
        results = pool.starmap(
            monte_carlo_search,
            [(data, trials_per_process, min_num_etfs, max_num_etfs)
             for _ in range(num_processes)],
        )

    best_solution, best_fitness = max(results, key=lambda x: x[1])
    logger.info("Monte Carlo completed in %.1fs, best fitness=%.6f",
                time.time() - start, best_fitness)
    return best_solution, best_fitness


class MonteCarloOptimiser(BaseOptimiser):
    """Random search portfolio selection with SLSQP weight refinement."""

    def __init__(self, n_trials=10_000_000,
                 min_securities=GA_MIN_SECURITIES,
                 max_securities=GA_MAX_SECURITIES,
                 num_processes=None, seed=None):
        self.n_trials = n_trials
        self.min_securities = min_securities
        self.max_securities = max_securities
        self.num_processes = num_processes or os.cpu_count()
        self.seed = seed

    def optimise(self, prices: pd.DataFrame) -> OptimisationResult:
        if self.seed is not None:
            np.random.seed(self.seed)
        start = time.time()
        if self.seed is not None and self.num_processes == 1:
            # Run in-process for reproducibility (Pool forks lose seed state)
            best_solution, best_fitness = monte_carlo_search(
                prices, self.n_trials, self.min_securities, self.max_securities,
            )
        else:
            best_solution, best_fitness = parallel_monte_carlo(
                prices, self.n_trials, self.num_processes,
                self.min_securities, self.max_securities,
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

        result = optimise_weights(best_solution, prices)
        if result.success:
            weights = result.x
            best_fitness = -result.fun

        return OptimisationResult(
            selected_tickers=selected,
            weights=weights,
            sharpe_ratio=best_fitness,
            metadata={
                'n_trials': self.n_trials,
                'num_processes': self.num_processes,
                'elapsed_seconds': elapsed,
            },
        )


if __name__ == '__main__':
    from src.logging_config import setup_logging
    setup_logging()

    from src.data_loading import load_training_data
    from src.portfolio_utils import save_optimisation_result
    from src import db

    conn = db.get_connection()
    data = load_training_data(exchange='US', csv_fallback=ETF_PRICES_CSV)

    # ── Parameters ────────────────────────────────────────────────────────
    num_trials = MC_NUM_TRIALS
    num_processes = os.cpu_count()
    min_num_etfs = ISLAND_GA_MIN_SECURITIES
    max_num_etfs = ISLAND_GA_MAX_SECURITIES

    # ── Monte Carlo search ────────────────────────────────────────────────
    mc_start = time.time()
    best_solution, best_fitness = parallel_monte_carlo(
        data, num_trials, num_processes, min_num_etfs, max_num_etfs,
    )
    mc_elapsed = time.time() - mc_start

    if best_solution is not None:
        selected_tickers = list(data.columns[best_solution == 1])
        logger.info("Best equal-weight Sharpe from MC: %.6f", best_fitness)
        logger.info("Selected %d securities: %s",
                     len(selected_tickers), selected_tickers)

        # ── SLSQP weight optimisation ────────────────────────────────────
        result = optimise_weights(best_solution, data)
        if result.success:
            final_sharpe = -result.fun
            logger.info("Final optimised Sharpe ratio: %.4f", final_sharpe)
            for ticker, w in zip(selected_tickers, result.x):
                if w > 1e-4:
                    logger.info("  %s: %.1f%%", ticker, w * 100)

            run_id = save_optimisation_result(
                conn, selected_tickers, result.x, data,
                script_name='monte_carlo',
                params={
                    'data_source': 'yahoo_finance',
                    'num_trials': num_trials,
                    'num_processes': num_processes,
                    'min_securities': min_num_etfs,
                    'max_securities': max_num_etfs,
                },
                exchange='US',
                elapsed_seconds=mc_elapsed,
            )
            logger.info("Run saved to database (id=%d)", run_id)
        else:
            logger.warning("SLSQP weight optimisation failed: %s",
                           result.message)
    else:
        logger.error("Monte Carlo did not find a valid solution.")

    conn.close()
