import logging
import os
import time

import numpy as np
import pandas as pd
from multiprocessing import Pool

from src.portfolio_utils import (
    calculate_log_returns,
    calculate_expected_returns,
    calculate_covariance_matrix,
    optimise_weights,
    OptimisationResult,
)
from src.config import (
    DATA_MIN_COVERAGE, DATA_LOOKBACK_DAYS,
    GA_MIN_SECURITIES, GA_MAX_SECURITIES,
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
    selected_indices = portfolio == 1
    num_selected = np.sum(selected_indices)
    if num_selected < min_num_etfs or num_selected > max_num_etfs:
        return -1e4
    if not np.any(selected_indices):
        return 0

    filtered_returns = expected_returns[selected_indices]
    filtered_cov_matrix = cov_matrix[np.ix_(selected_indices, selected_indices)]
    weights = np.ones(num_selected) / num_selected
    portfolio_return = np.dot(weights, filtered_returns)
    portfolio_variance = np.dot(weights, np.dot(filtered_cov_matrix, weights))

    return portfolio_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0


def monte_carlo_search(data, trials, min_num_etfs, max_num_etfs):
    """Run *trials* random portfolio evaluations, return the best."""
    log_returns = calculate_log_returns(data)
    expected_returns = calculate_expected_returns(log_returns).values
    cov_matrix = calculate_covariance_matrix(log_returns).values
    num_etfs = data.shape[1]

    best_fitness = float('-inf')
    best_portfolio = None

    for _ in range(trials):
        portfolio = random_portfolio(num_etfs, min_num_etfs, max_num_etfs)
        fitness = calculate_fitness(portfolio, expected_returns, cov_matrix,
                                    min_num_etfs, max_num_etfs)
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


class MonteCarloOptimiser:
    """Random search portfolio selection with SLSQP weight refinement."""

    def __init__(self, n_trials=10_000_000,
                 min_securities=GA_MIN_SECURITIES,
                 max_securities=GA_MAX_SECURITIES,
                 num_processes=None):
        self.n_trials = n_trials
        self.min_securities = min_securities
        self.max_securities = max_securities
        self.num_processes = num_processes or os.cpu_count()

    def optimise(self, prices: pd.DataFrame) -> OptimisationResult:
        start = time.time()
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Load prices from DB (CSV fallback) ────────────────────────────────
    from src import db

    conn = db.get_connection()
    data = db.load_prices(conn, exchange='US')
    if data.empty:
        logger.info("No data in DB, falling back to CSV")
        from src.portfolio_utils import load_prices_csv
        data = load_prices_csv('Data/ETF_Prices.csv', last_n_days=730)
    else:
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        cutoff = data.index[-1] - pd.Timedelta(days=DATA_LOOKBACK_DAYS)
        data = data[data.index >= cutoff]
        data = data.dropna(axis=1, thresh=int(DATA_MIN_COVERAGE * len(data)))
        data = data.ffill()

    logger.info("Loaded price data: %d rows x %d columns", *data.shape)

    # ── Parameters ────────────────────────────────────────────────────────
    num_trials = 10_000_000
    num_processes = os.cpu_count()
    min_num_etfs = 10
    max_num_etfs = 20

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

            # ── Compute return / volatility for DB ────────────────────────
            log_returns = calculate_log_returns(data[selected_tickers])
            er = calculate_expected_returns(log_returns)
            cov = calculate_covariance_matrix(log_returns)
            portfolio_return = float(np.dot(result.x, er))
            portfolio_vol = float(
                np.sqrt(np.dot(result.x.T, np.dot(cov, result.x)))
            )

            # ── Save to database ──────────────────────────────────────────
            run_id = db.save_optimisation_run(
                conn,
                params={
                    'script': 'monte_carlo_optimisation',
                    'data_source': 'yahoo_finance',
                    'num_trials': num_trials,
                    'num_processes': num_processes,
                    'min_securities': min_num_etfs,
                    'max_securities': max_num_etfs,
                },
                results={
                    'best_sharpe': final_sharpe,
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_vol,
                    'num_selected': len(selected_tickers),
                    'elapsed_seconds': mc_elapsed,
                },
                holdings=list(zip(selected_tickers, result.x)),
                exchange='US',
            )
            logger.info("Run saved to database (id=%d)", run_id)
        else:
            logger.warning("SLSQP weight optimisation failed: %s",
                           result.message)
    else:
        logger.error("Monte Carlo did not find a valid solution.")

    conn.close()
