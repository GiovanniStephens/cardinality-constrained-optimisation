import logging
import time
import warnings

import numpy as np
import pandas as pd
import pygad
import scipy.optimize as opt
from copulae import GaussianCopula, TCopula
from muarch import MUArch
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.portfolio_utils import (
    load_prices_csv, calculate_log_returns, negative_sharpe_ratio,
    OptimisationResult,
)
from src.optimisers.base import BaseOptimiser
from src.config import (
    GA_MIN_SECURITIES, GA_MAX_SECURITIES,
    GA_MIN_WEIGHT, GA_MAX_WEIGHT,
    GA_TARGET_RETURN, GA_NUM_GENERATIONS, GA_POPULATION_SIZE,
    TRADING_DAYS_PER_YEAR, DATA_MIN_COVERAGE,
    FORECAST_EXPECTED_RETURNS_PATH, FORECAST_VARIANCES_PATH,
)

# Suppress known noisy warnings from dependencies, not all warnings globally.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

logger = logging.getLogger(__name__)


def estimate_corr_using_copulas(data: pd.DataFrame,
                                diagnostics: bool = False) -> np.ndarray:
    """
    Estimates the correlation matrix using the copula method.

    It first models the returns using an AR(1)-GARCH(1, 1)
    with skewt innovations. Then it fits a Student-t copula
    to the standardized residuals and extracts the correlation
    matrix (cop.sigma).

    :data: pandas dataframe of the log returns data.
    :diagnostics: if True, log GARCH residual adequacy tests and
                  copula model comparison (t-copula vs Gaussian).
    :return: numpy array of the correlation matrix.
    """
    try:
        # Estimate GARCH model for each time series.
        # scale=10 multiplies returns before fitting for numerical stability
        # (daily returns are ~0.001), then divides back internally.
        models = MUArch(data.shape[1], mean='AR', lags=1, dist='skewt', scale=10)
        models.fit(data)
        residuals = models.residuals()

        if diagnostics:
            # Ljung-Box test on squared standardized residuals
            # to check GARCH adequacy (H0: no remaining autocorrelation)
            for i, col in enumerate(data.columns):
                sq_resid = residuals[:, i] ** 2
                lb_result = acorr_ljungbox(sq_resid, lags=[10], return_df=True)
                p_value = lb_result['lb_pvalue'].values[0]
                if p_value < 0.05:
                    logger.warning(
                        "GARCH residuals for %s show remaining autocorrelation "
                        "(Ljung-Box p=%.4f < 0.05). Model may be inadequate.",
                        col, p_value)
                else:
                    logger.info(
                        "GARCH residuals for %s pass Ljung-Box test (p=%.4f).",
                        col, p_value)

        # Fit Student-t copula
        cop = TCopula(dim=data.shape[1])
        cop.fit(residuals)

        if diagnostics:
            # Compare t-copula vs Gaussian copula via log-likelihood
            gauss_cop = GaussianCopula(dim=data.shape[1])
            gauss_cop.fit(residuals)
            logger.info(
                "Copula comparison — t-copula log-lik: %.2f, "
                "Gaussian copula log-lik: %.2f",
                cop.log_lik(residuals), gauss_cop.log_lik(residuals))

        return cop.sigma
    except Exception as e:
        logger.warning("Copula estimation failed: %s; falling back to sample correlation.", e)
        logger.debug("Copula traceback:", exc_info=True)
        return data.corr()


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

    def _prepare_inputs(self, prices):
        """Prepare log returns, expected returns, and variances."""
        if prices is None or prices.empty:
            raise ValueError("prices DataFrame is None or empty.")
        self._data = calculate_log_returns(prices).transpose()
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
            if self._expected_returns is None:
                try:
                    self._variances = load_prices_csv(FORECAST_VARIANCES_PATH,
                                                      min_coverage=0.10)
                    er_df = load_prices_csv(FORECAST_EXPECTED_RETURNS_PATH,
                                            min_coverage=0.10)
                    self._expected_returns = er_df['0']
                except (FileNotFoundError, KeyError) as e:
                    logger.warning(
                        "Could not load forecast files (%s); "
                        "falling back to historical estimates.", e)
                    self._variances = None
                    self._expected_returns = (
                        self._data.T.mean() * TRADING_DAYS_PER_YEAR)
        else:
            self._variances = None
            self._expected_returns = (
                self._data.T.mean() * TRADING_DAYS_PER_YEAR)

    def _get_cov_matrix(self, ret_data, use_copulae=False):
        """CCC covariance matrix using instance variances (not globals)."""
        if self._variances is not None:
            missing_cols = set(ret_data.columns) - set(self._variances.index)
            if missing_cols:
                logger.warning(
                    "Columns missing from variances: %s. "
                    "Falling back to historical cov.", missing_cols)
                return ret_data.cov() * TRADING_DAYS_PER_YEAR

        if use_copulae:
            corr = estimate_corr_using_copulas(ret_data)
        else:
            corr = ret_data.corr().values

        D = np.zeros((ret_data.shape[1], ret_data.shape[1]))
        if self._variances is not None:
            var_values = self._variances.loc[ret_data.columns].values.flatten()
            if np.any(var_values < 0):
                logger.warning("Negative forecast variances found; clipping to 0.")
                var_values = np.clip(var_values, 0, None)
            diag = np.sqrt(var_values)
        else:
            diag = ret_data.std().values * np.sqrt(TRADING_DAYS_PER_YEAR)
        np.fill_diagonal(D, diag)
        return np.matmul(np.matmul(D, corr), D)

    def _optimize_weights(self, ret_data, initial_weights,
                          use_copulae=None, max_weight=None,
                          min_weight=None):
        """SLSQP weight optimisation using instance expected returns.

        :param use_copulae: override instance setting (default: self.use_copulae).
        :param max_weight: override instance setting (default: self.max_weight).
        :param min_weight: override instance setting (default: self.min_weight).
        """
        if use_copulae is None:
            use_copulae = self.use_copulae
        if max_weight is None:
            max_weight = self.max_weight
        if min_weight is None:
            min_weight = self.min_weight
        cov_matrix = self._get_cov_matrix(ret_data, use_copulae)
        rets = self._expected_returns.loc[ret_data.columns].values
        cons = [{'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}]
        if self.target_risk is not None and self.target_return is None:
            cons.append({'type': 'eq',
                         'fun': lambda W: self.target_risk -
                         np.sqrt(np.dot(W.T, np.dot(cov_matrix, W)))})
        if self.target_return is not None and self.target_risk is None:
            cons.append({'type': 'eq',
                         'fun': lambda W: self.target_return -
                         np.sum(rets * W)})
        bounds = tuple((min_weight, max_weight)
                        for _ in range(len(initial_weights)))
        sol = opt.minimize(negative_sharpe_ratio, initial_weights,
                           args=(rets, cov_matrix), method='SLSQP',
                           bounds=bounds, constraints=cons)
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
            subset = inst_data.iloc[np.array(solution).astype(bool), :]
            if num_stocks >= 2:
                random_w = np.random.random(num_stocks)
                random_w /= np.sum(random_w)
                cov_matrix = subset.transpose().cov().values * TRADING_DAYS_PER_YEAR
                rets = inst_er.loc[subset.index].values
                cons = [{'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}]
                if target_ret is not None and target_risk is None:
                    cons.append({'type': 'eq',
                                 'fun': lambda W: target_ret - np.sum(rets * W)})
                if target_risk is not None and target_ret is None:
                    cons.append({'type': 'eq',
                                 'fun': lambda W: target_risk -
                                 np.sqrt(np.dot(W.T, np.dot(cov_matrix, W)))})
                bounds = tuple((min_w, max_w) for _ in range(num_stocks))
                sol = opt.minimize(negative_sharpe_ratio, random_w,
                                   args=(rets, cov_matrix), method='SLSQP',
                                   bounds=bounds, constraints=cons)
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
        n = len(self._data)
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
        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            initial_population=initial_pop,
            num_parents_mating=max(2, self.num_children // 10),
            gene_type=int,
            init_range_low=0, init_range_high=2,
            parent_selection_type='rank',
            keep_parents=0,
            random_mutation_min_val=-1, random_mutation_max_val=1,
            mutation_type="random",
            crossover_type="single_point",
            crossover_probability=0.85,
            fitness_func=fitness_fn,
            stop_criteria='saturate_5',
        )
        ga_instance.run()
        elapsed = time.time() - start

        solution, solution_fitness, _ = ga_instance.best_solution(
            ga_instance.last_generation_fitness)
        indices = np.array(solution).astype(bool)
        portfolio = list(self._data.transpose().iloc[:, indices].columns)

        if not portfolio:
            return OptimisationResult(
                selected_tickers=[], weights=np.array([]),
                sharpe_ratio=float('-inf'),
                metadata={'error': 'No valid solution found'},
            )

        # SLSQP weight refinement
        log_rets = self._data.loc[portfolio, :]
        random_weights = np.random.random(len(portfolio))
        random_weights /= np.sum(random_weights)
        sol = self._optimize_weights(log_rets.transpose(), random_weights)
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
                'elapsed_seconds': elapsed,
            },
        )


def main():
    from src import db
    from src.portfolio_utils import load_prices

    conn = db.get_connection()
    prices_df = load_prices(exchange='US', csv_fallback='data/NZ_ETF_Prices.csv',
                            conn=conn)
    logger.info("Loaded price data: %d rows x %d columns", *prices_df.shape)

    optimiser = PygadOptimiser(
        num_children=500,
        use_forecasts=True,
        conn=conn,
    )
    result = optimiser.optimise(prices_df)

    logger.info("Selected tickers: %s", result.selected_tickers)
    logger.info("Weights: %s", result.weights)
    logger.info("Sharpe ratio: %.4f", result.sharpe_ratio)
    logger.info("Metadata: %s", result.metadata)

    # Save to database
    run_id = db.save_optimisation_run(conn,
        params={
            'script': 'pygad_ga',
            'data_source': 'yahoo_finance',
            'num_children': optimiser.num_children,
            'min_securities': optimiser.min_securities,
            'max_securities': optimiser.max_securities,
            'min_weight': optimiser.min_weight,
            'max_weight': optimiser.max_weight,
            'target_return': optimiser.target_return,
            'target_risk': optimiser.target_risk,
            'use_forecasts': optimiser.use_forecasts,
        },
        results={
            'best_sharpe': result.sharpe_ratio,
            'num_selected': len(result.selected_tickers),
            'elapsed_seconds': result.metadata.get('elapsed_seconds', 0),
        },
        holdings=list(zip(result.selected_tickers, result.weights)),
        exchange='US')
    logger.info("Run saved to database (id=%d)", run_id)
    conn.close()


if __name__ == '__main__':
    from src.logging_config import setup_logging
    setup_logging()
    main()
