import logging
import time
import warnings

import numpy as np
import pandas as pd
import pygad
from src.returns import calculate_log_returns
from src.covariance import calculate_covariance_matrix
from src.weights import optimise_weights as _optimise_weights
from src.data_loading import load_data
from src.portfolio_utils import OptimisationResult
from src.optimisers.base import BaseOptimiser
from src.config import (
    GA_MIN_SECURITIES, GA_MAX_SECURITIES,
    GA_MIN_WEIGHT, GA_MAX_WEIGHT,
    GA_TARGET_RETURN, GA_NUM_GENERATIONS, GA_POPULATION_SIZE,
    GA_CROSSOVER_PROBABILITY, GA_THREAD_POOL_SIZE,
    GA_ELITISM_FRACTION, GA_EARLY_STOP_SATURATE,
    GA_SELECTION_TYPE, GA_TOURNAMENT_SIZE, GA_PARENT_FRACTION,
    TRADING_DAYS_PER_YEAR,
    NZ_ETF_PRICES_CSV, VARIANCES_CSV, EXPECTED_RETURNS_CSV,
)

# Suppress known noisy warnings from dependencies, not all warnings globally.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

logger = logging.getLogger(__name__)


# ─── PygadOptimiser ─────────────────────────────────────────────────────────


class PygadOptimiser(BaseOptimiser):
    """PyGAD-based genetic algorithm with copula/CCC covariance support.

    Self-contained optimiser that does not mutate module-level globals.
    Uses closures to pass instance state to PyGAD fitness callbacks.
    """

    def __init__(self, num_children=GA_POPULATION_SIZE,
                 num_generations=GA_NUM_GENERATIONS,
                 min_securities=GA_MIN_SECURITIES,
                 max_securities=GA_MAX_SECURITIES,
                 min_weight=GA_MIN_WEIGHT, max_weight=GA_MAX_WEIGHT,
                 target_return=GA_TARGET_RETURN, target_risk=None,
                 use_copulae=False, use_forecasts=False, conn=None,
                 seed=None):
        self.num_children = num_children
        self.num_generations = num_generations
        self.min_securities = min_securities
        self.max_securities = max_securities
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_return = target_return
        self.target_risk = target_risk
        self.use_copulae = use_copulae
        self.use_forecasts = use_forecasts
        self.conn = conn
        self.seed = seed
        # Instance state — set by _prepare_inputs()
        self._data = None
        self._expected_returns = None
        self._variances = None
        self._forecast_source = None  # 'db', 'csv', or 'historical_fallback'

    def _prepare_inputs(self, prices):
        """Prepare log returns, expected returns, and variances."""
        if prices is None or prices.empty:
            raise ValueError("prices DataFrame is None or empty.")
        self._data = calculate_log_returns(prices)  # dates x tickers (standard)
        if self.use_forecasts:
            self._variances = None
            self._expected_returns = None
            if self.conn is not None:
                from src import db
                er = db.load_expected_returns(self.conn)
                var = db.load_variances(self.conn)
                if not er.empty and not var.empty:
                    self._expected_returns = er
                    self._variances = var
                    self._forecast_source = 'db'
            if self._expected_returns is None:
                try:
                    self._variances = load_data(VARIANCES_CSV)
                    self._expected_returns = load_data(EXPECTED_RETURNS_CSV)['0']
                    self._forecast_source = 'csv'
                except (FileNotFoundError, KeyError) as e:
                    logger.warning(
                        "Forecasts requested but unavailable (%s); "
                        "using historical estimates.", e)
                    self._variances = None
                    self._expected_returns = (
                        self._data.mean() * TRADING_DAYS_PER_YEAR)
                    self._forecast_source = 'historical_fallback'
        else:
            self._variances = None
            self._expected_returns = (
                self._data.mean() * TRADING_DAYS_PER_YEAR)
            self._forecast_source = 'historical'

    def _get_cov_matrix(self, ret_data, use_copulae=False):
        """CCC covariance matrix using instance variances."""
        cov = calculate_covariance_matrix(
            ret_data, annualise=True,
            forecast_variances=self._variances,
            use_copulae=use_copulae,
        )
        return cov.values if hasattr(cov, 'values') else cov

    def _optimize_weights(self, ret_data, initial_weights):
        """SLSQP weight optimisation using instance expected returns."""
        cov_matrix = self._get_cov_matrix(ret_data, self.use_copulae)
        rets = self._expected_returns.loc[ret_data.columns].values
        sol = _optimise_weights(
            expected_returns=rets, cov_matrix=cov_matrix,
            min_weight=self.min_weight, max_weight=self.max_weight,
            target_return=self.target_return, target_risk=self.target_risk,
            initial_weights=initial_weights,
        )
        if not sol.success:
            logger.warning("Optimization did not converge: %s", sol.message)
        return sol

    def _make_fitness_fn(self):
        """Create a closure that captures instance state for PyGAD."""
        inst_data = self._data
        inst_er = self._expected_returns
        min_sec = self.min_securities
        max_sec = self.max_securities
        target_ret = self.target_return
        target_risk = self.target_risk
        max_w = self.max_weight
        min_w = self.min_weight

        def _fitness(ga_instance, solution, solution_idx):
            num_stocks = np.count_nonzero(solution)
            # inst_data is dates x tickers; select columns by boolean mask
            subset = inst_data.iloc[:, np.array(solution).astype(bool)]
            if num_stocks >= 2:
                rng = np.random.default_rng()
                random_w = rng.random(num_stocks)
                random_w /= np.sum(random_w)
                cov_matrix = calculate_covariance_matrix(subset).values
                rets = inst_er.loc[subset.columns].values
                sol = _optimise_weights(
                    expected_returns=rets, cov_matrix=cov_matrix,
                    min_weight=min_w, max_weight=max_w,
                    target_return=target_ret, target_risk=target_risk,
                    initial_weights=random_w,
                )
                base_fitness = -sol.fun
            else:
                base_fitness = -1

            if num_stocks > max_sec:
                return base_fitness - (num_stocks - max_sec) ** 2
            elif num_stocks < min_sec:
                return base_fitness - (min_sec - num_stocks) ** 2
            return base_fitness

        return _fitness

    def _create_individual(self):
        """Create a random binary individual for the GA."""
        n = self._data.shape[1]  # number of tickers
        p = self.max_securities / n
        individual = np.random.binomial(1, p, n)
        while np.count_nonzero(individual) < self.min_securities:
            individual = np.random.binomial(1, p, n)
        return individual

    def optimise(self, prices: pd.DataFrame) -> OptimisationResult:
        if self.seed is not None:
            np.random.seed(self.seed)
        self._prepare_inputs(prices)

        fitness_fn = self._make_fitness_fn()
        initial_pop = np.array([self._create_individual()
                                for _ in range(self.num_children)])

        start = time.time()
        # Disable thread parallelism when seeded for reproducibility
        parallel = None if self.seed is not None else ["thread", GA_THREAD_POOL_SIZE]
        ga_kwargs = dict(
            num_generations=self.num_generations,
            initial_population=initial_pop,
            num_parents_mating=max(2, int(self.num_children * GA_PARENT_FRACTION)),
            gene_type=int,
            init_range_low=0, init_range_high=2,
            parent_selection_type=GA_SELECTION_TYPE,
            K_tournament=GA_TOURNAMENT_SIZE,
            keep_parents=0,
            keep_elitism=max(1, int(self.num_children * GA_ELITISM_FRACTION)),
            random_mutation_min_val=-1, random_mutation_max_val=1,
            mutation_type="random",
            crossover_type="uniform",
            crossover_probability=GA_CROSSOVER_PROBABILITY,
            fitness_func=fitness_fn,
            stop_criteria=f'saturate_{GA_EARLY_STOP_SATURATE}',
        )
        if parallel is not None:
            ga_kwargs['parallel_processing'] = parallel
        if self.seed is not None:
            ga_kwargs['random_seed'] = self.seed
        ga_instance = pygad.GA(**ga_kwargs)
        ga_instance.run()
        elapsed = time.time() - start

        solution, solution_fitness, _ = ga_instance.best_solution(
            ga_instance.last_generation_fitness)
        indices = np.array(solution).astype(bool)
        portfolio = list(self._data.columns[indices])

        if not portfolio:
            return OptimisationResult(
                selected_tickers=[], weights=np.array([]),
                sharpe_ratio=float('-inf'),
                metadata={'error': 'No valid solution found'},
            )

        # SLSQP weight refinement
        log_rets = self._data[portfolio]  # dates x tickers subset
        random_weights = np.random.random(len(portfolio))
        random_weights /= np.sum(random_weights)
        sol = self._optimize_weights(log_rets, random_weights)
        weights = sol.x if sol.success else np.ones(len(portfolio)) / len(portfolio)
        final_sharpe = -sol.fun if sol.success else float(solution_fitness)

        return OptimisationResult(
            selected_tickers=portfolio,
            weights=weights,
            sharpe_ratio=final_sharpe,
            metadata={
                'num_children': self.num_children,
                'num_generations': self.num_generations,
                'use_copulae': self.use_copulae,
                'use_forecasts': self.use_forecasts,
                'forecast_source': self._forecast_source,
                'elapsed_seconds': elapsed,
            },
        )


# ─── CLI entry point ─────────────────────────────────────────────────────────


if __name__ == '__main__':
    from src.logging_config import setup_logging
    setup_logging()

    from src.data_loading import load_training_data
    from src.portfolio_utils import save_optimisation_result

    from src import db as _db
    conn = _db.get_connection()
    prices_df = load_training_data(
        exchange='US', csv_fallback=NZ_ETF_PRICES_CSV, lookback_days=None)

    opt = PygadOptimiser(
        num_children=500,
        use_forecasts=True,
        conn=conn,
    )
    start = time.time()
    result = opt.optimise(prices_df)
    elapsed = time.time() - start

    logger.info(
        "GA completed in %.1fs — Sharpe=%.4f, %d securities",
        elapsed, result.sharpe_ratio, len(result.selected_tickers),
    )
    for ticker, w in zip(result.selected_tickers, result.weights):
        if w > 1e-4:
            logger.info("  %s: %.1f%%", ticker, w * 100)

    run_id = save_optimisation_result(
        conn, result.selected_tickers, result.weights, prices_df,
        script_name='pygad_ga',
        params={
            'data_source': 'yahoo_finance',
            'num_children': 500,
            'min_securities': GA_MIN_SECURITIES,
            'max_securities': GA_MAX_SECURITIES,
            'min_weight': GA_MIN_WEIGHT,
            'max_weight': GA_MAX_WEIGHT,
            'target_return': GA_TARGET_RETURN,
            'use_forecasts': True,
        },
        exchange='US',
        elapsed_seconds=elapsed,
    )
    logger.info("Run saved to database (id=%d)", run_id)
    conn.close()
