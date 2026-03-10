import logging
import warnings

import numpy as np
import pandas as pd
import pygad
import scipy.optimize as opt
from copulae import GaussianCopula, TCopula
from muarch import MUArch
from statsmodels.stats.diagnostic import acorr_ljungbox

# Suppress known noisy warnings from dependencies, not all warnings globally.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

MAX_NUM_STOCKS = 15
MIN_NUM_STOCKS = 3
TARGET_RETURN = 0.15
TARGET_RISK = None
MAX_WEIGHT = 0.45
MIN_WEIGHT = 0.05
last_fitness = 0
data = None
variances = None
expected_returns = None


def sharpe_ratio(weights: np.ndarray, returns: np.ndarray, cov: np.ndarray) -> float:
    """
    Calculates the Sharpe ratio of a portfolio.
    The Sharpe ratio is the ratio of the mean return of the portfolio
    to the portfolio standard deviation.

    :weights: numpy array of weights.
    :p_returns: list of the portfolio's expected return.
    :return: float of the negative Sharpe ratio.
    """
    p_returns = np.sum(weights*returns)
    variance = np.dot(weights.T, np.dot(cov, weights))
    if variance <= 0:
        logger.warning("Portfolio variance is non-positive (%.6f), returning 0.", variance)
        return 0.0
    p_volatility = np.sqrt(variance)
    return -p_returns/p_volatility


def load_data(filename: str) -> pd.DataFrame:
    """
    Loads the data from a CSV file in the local directory.

    :filename: string of the filename.
    :return: pandas dataframe of the data.
    """
    prices_df = pd.read_csv(filename, index_col=0)
    if prices_df.empty:
        raise ValueError(f"Loaded CSV '{filename}' is empty.")
    # Remove columns with 10% or more null values
    prices_df = prices_df.dropna(axis=1, thresh=int(len(prices_df)/10))
    # Fill the null values with the previous day's close price
    prices_df = prices_df.ffill()
    return prices_df


def get_cov_matrix(data: pd.DataFrame, use_copulae=False) -> np.ndarray:
    """
    Calculates the covariance matrix of the data using the CCC model.

    Uses Bollerslev's (1990) Constant Conditional Correlation model:
    Cov = D × R × D, where D is a diagonal matrix of volatilities
    and R is the correlation matrix.

    When forecast variances are available, D uses GARCH-forecast volatilities.
    Otherwise, D uses annualised historical sample standard deviations.

    When use_copulae=True, R is estimated via a Student-t copula fitted
    to AR(1)-GARCH(1,1) standardized residuals. Otherwise, R is the
    historical sample correlation matrix.

    (see: Bollerslev, T. (1990). Modelling the Coherence in Short-Run
    Nominal Exchange Rates: A Multivariate Generalized Arch Model.
    The Review of Economics and Statistics,
    72(3), 498–505. https://doi.org/10.2307/2109358)

    :data: pandas dataframe of the returns data.
    :use_copulae: whether to estimate correlations via copula.
    :return: numpy array of the covariance matrix.
    """
    # If variances are available, check for missing columns
    if variances is not None:
        missing_cols = set(data.columns) - set(variances.index)
        if missing_cols:
            logger.warning("Columns missing from variances: %s. Falling back to historical cov.", missing_cols)
            return data.cov() * 252

    # Correlation matrix R
    if use_copulae:
        corr = estimate_corr_using_copulas(data)
    else:
        corr = data.corr().values

    # Diagonal volatility matrix D
    D = np.zeros((data.shape[1], data.shape[1]))
    if variances is not None:
        var_values = variances.loc[data.columns].values.flatten()
        # Guard against negative variances before taking sqrt
        if np.any(var_values < 0):
            logger.warning("Negative forecast variances found; clipping to 0.")
            var_values = np.clip(var_values, 0, None)
        diag = np.sqrt(var_values)
    else:
        diag = data.std().values * np.sqrt(252)
    np.fill_diagonal(D, diag)

    # CCC reconstruction: Cov = D × R × D
    cov_matrix = np.matmul(np.matmul(D, corr), D)
    return cov_matrix


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
        logger.warning("Copula estimation failed (%s); falling back to sample correlation.", e)
        return data.corr()


# risk budgeting optimization
def calculate_portfolio_var(w, V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0, 0]


def calculate_risk_contribution(w, V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    portfolio_var = calculate_portfolio_var(w, V)
    if portfolio_var <= 0:
        return np.zeros_like(w.T)
    sigma = np.sqrt(portfolio_var)
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC, w.T)/sigma
    return RC


def risk_budget_objective(x, pars):
    # calculate portfolio risk
    V = pars[0]     # covariance table
    x_t = pars[1]   # risk target in percent of portfolio risk
    sig_p = np.sqrt(calculate_portfolio_var(x, V))      # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p, x_t))
    asset_RC = calculate_risk_contribution(x, V)
    J = sum(np.square(asset_RC-risk_target.T))[0, 0]    # sum of squared error
    return J


def optimize(data: pd.DataFrame,
             initial_weights: np.array,
             target_risk: float = None,
             target_return: float = None,
             max_weight: float = 0.3333,
             min_weight: float = 0.0000,
             use_copulae: bool = False,
             risk_parity: bool = False) -> float:
    """
    Optimizes the portfolio using the Sharpe ratio.

    :data: pandas dataframe of the log returns data.
    :initial_weights: numpy array of initial weights.
    :target_risk: float of the target risk
                  (annualised portfolio standard deviation).
    :target_return: float of the target return
                    (annualised portfolio mean return).
    :max_weight: float of the maximum weight of any single stock.
    :min_weight: float of the minimum weight of any single stock.
    :use_copulae: boolean of whether to use copulae or not.
    :return: pcipy optimization result.
    """
    if expected_returns is None:
        raise ValueError("expected_returns is not set. Call prepare_opt_inputs() first.")
    if len(initial_weights) != data.shape[1]:
        raise ValueError(
            f"initial_weights length ({len(initial_weights)}) does not match "
            f"number of assets ({data.shape[1]})."
        )
    cov_matrix = get_cov_matrix(data, use_copulae)
    missing = set(data.columns) - set(expected_returns.index)
    if missing:
        raise KeyError(f"Expected returns missing for columns: {missing}")
    rets = expected_returns.loc[data.columns].values
    cons = [{'type': 'eq',
             'fun': lambda x: 1 - np.sum(x)}]
    if target_risk is not None and target_return is None:
        cons.append(
            {'type': 'eq',
             'fun': lambda W: target_risk -
             np.sqrt(np.dot(W.T,
                            np.dot(cov_matrix,
                                   W)))})
    if target_return is not None and target_risk is None:
        cons.append(
            {'type': 'eq',
             'fun': lambda W: target_return -
             np.sum(rets*W)})
    bounds = tuple((min_weight, max_weight) for _ in range(len(initial_weights)))
    if risk_parity:
        risk_proportion = [1/len(initial_weights)]*len(initial_weights)
        sol = opt.minimize(risk_budget_objective,
                           initial_weights,
                           args=([np.matrix(cov_matrix), risk_proportion]),
                           method='SLSQP',
                           bounds=bounds,
                           constraints=cons)
    else:
        sol = opt.minimize(sharpe_ratio,
                           initial_weights,
                           args=(rets, cov_matrix),
                           method='SLSQP',
                           bounds=bounds,
                           constraints=cons)
    if not sol.success:
        logger.warning("Optimization did not converge: %s", sol.message)
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

    This is the max Sharpe Ratio for a given portfolio.
    The the number of ETFs is out of the limits, the fitness
    is set to the negative of the count of the securities.

    :individual: binary array.
    :data: pandas dataframe of the returns data.
    :return: float of the fitness (i.e. Sharpe Ratio)
    """
    num_stocks = np.count_nonzero(individual)
    random_weights = np.random.random(num_stocks)
    random_weights /= np.sum(random_weights)  # Normalize the weights
    subset = data.iloc[np.array(individual).astype(bool), :]

    # Calculate base fitness
    if num_stocks >= 2:
        base_fitness = -optimize(subset.transpose(),
                                 random_weights,
                                 target_return=TARGET_RETURN,
                                 target_risk=TARGET_RISK,
                                 max_weight=MAX_WEIGHT,
                                 min_weight=MIN_WEIGHT,
                                 risk_parity=False)['fun']
    else:
        base_fitness = -1

    # Apply penalties if necessary
    if num_stocks > MAX_NUM_STOCKS:
        excess = num_stocks - MAX_NUM_STOCKS
        penalty = excess**2
        return base_fitness - penalty
    elif num_stocks < MIN_NUM_STOCKS:
        deficit = MIN_NUM_STOCKS - num_stocks
        penalty = deficit**2
        return base_fitness - penalty
    else:
        return base_fitness


def fitness_2(ga_instance, solution: np.array, solution_idx: int) -> float:
    """
    Fitness function for the pygad genetic algorithm.

    :solution: binary array.
    :solution_idx: int of the solution index.
    """
    fit = fitness(solution, data)
    return fit


def generate_random_gene(individual):
    """
    Generates a random gene for the individual.

    :individual: binary array of the individual.
    :return: binary array of the individual.
    """
    for i in range(len(individual)):
        individual[i] = np.random.binomial(1, MAX_NUM_STOCKS/len(individual))
    return individual


def create_individual(data):
    """
    Creates an individual.

    :data: pandas dataframe of the returns data.
    :return: a binary array of the individual.
    """
    individual = np.zeros(len(data))
    individual = generate_random_gene(individual)
    while np.count_nonzero(individual) < MIN_NUM_STOCKS:
        individual = generate_random_gene(individual)
    return individual


def on_generation(ga_instance: pygad.GA) -> None:
    """
    On each generation in the GA, this function is called.

    :ga_instance: the GA instance.
    """
    global last_fitness
    current_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    logger.debug(
        "Generation %d: fitness=%.6f, change=%.6f",
        ga_instance.generations_completed,
        current_fitness,
        current_fitness - last_fitness,
    )
    last_fitness = current_fitness


def prepare_opt_inputs(prices, use_forecasts: bool) -> None:
    """
    Prepares the inputs for the optimisation.

    :prices: pandas dataframe of the prices.
    :use_forecasts: bool of whether to use forecasts.
    """
    global variances, expected_returns, data
    if prices is None or prices.empty:
        raise ValueError("prices DataFrame is None or empty.")
    if use_forecasts:
        data = calculate_returns(prices).transpose()
        try:
            variances = load_data('Data/variances.csv')
            expected_returns = load_data('Data/expected_returns.csv')['0']
        except (FileNotFoundError, KeyError) as e:
            logger.warning(
                "Could not load forecast files (%s); falling back to historical estimates.", e
            )
            variances = None
            expected_returns = data.T.mean() * 252
    else:
        data = calculate_returns(prices).transpose()
        variances = None
        expected_returns = data.T.mean()*252


def cardinality_constrained_optimisation(num_children: int = 1000,
                                         verbose: bool = False):
    """
    Performs the cardinality constrained optimisation.

    :num_children: int of the number of children to create.
    :verbose: bool of whether to print the progress.
    :return: the best Sharpe Ratio and the individual (portfolio).
    """
    if verbose:
        on_gen = on_generation
    else:
        on_gen = None
    ga_instance = pygad.GA(num_generations=6,
                           initial_population=np.array([create_individual(data)
                                                        for _ in range(num_children)]),
                           num_parents_mating=num_children//10,
                           gene_type=int,
                           init_range_low=0,
                           init_range_high=2,
                           parent_selection_type='rank',
                           keep_parents=0,
                           random_mutation_min_val=-1,
                           random_mutation_max_val=1,
                           mutation_type="random",
                           crossover_type="single_point",
                           crossover_probability=0.85,
                           fitness_func=fitness_2,
                           on_generation=on_gen,
                           stop_criteria='saturate_5')
    start = time.time()
    ga_instance.run()
    elapsed = time.time() - start
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    logger.info(
        "GA completed in %.1fs — %d generations, best fitness=%.6f",
        elapsed,
        ga_instance.generations_completed,
        solution_fitness,
    )
    if verbose:
        logger.debug("Best solution params: %s (index %d)", solution, solution_idx)
    return solution


def create_portfolio(num_children: int = 100, verbose: bool = True) -> list:
    """
    Creates a cardinality constrained portfolio.

    :num_children: int of the number of children to create.
    :verbose: bool of whether to print the progress.
    :return: pandas dataframe of the portfolio.
    """
    individual = cardinality_constrained_optimisation(num_children=num_children,
                                                      verbose=verbose)
    indices = np.array(individual).astype(bool)
    portfolio = data.transpose().iloc[:, indices].columns
    return list(portfolio)


def main():
    import time as _time

    # Load the data
    prices_df = load_data('Data/NZ_ETF_Prices.csv')
    logger.info("Loaded price data: %d rows x %d columns", *prices_df.shape)
    # Prepare the inputs for the optimisation
    use_forecasts = True
    prepare_opt_inputs(prices_df, use_forecasts=use_forecasts)

    log_returns = calculate_returns(prices_df)
    # Run the cardinality constrained optimisation
    opt_start = _time.time()
    best_individual = cardinality_constrained_optimisation(num_children=500,
                                                           verbose=True)
    opt_elapsed = _time.time() - opt_start

    indeces = np.array(best_individual).astype(bool)
    # Print the portfolio metrics for the best portfolio we could find.
    best_portfolio_returns = log_returns.iloc[:, indeces]
    random_weights = np.random.random(np.count_nonzero(best_individual))
    random_weights /= np.sum(random_weights)
    sol = optimize(best_portfolio_returns,
                   random_weights,
                   target_return=TARGET_RETURN,
                   target_risk=TARGET_RISK,
                   max_weight=MAX_WEIGHT,
                   min_weight=MIN_WEIGHT)
    if not sol.success:
        logger.warning("Weight optimisation did not converge: %s", sol.message)
    best_weights = sol['x']
    cov = best_portfolio_returns.cov()*252
    risk = float(np.sqrt(np.dot(best_weights.T, np.dot(cov, best_weights))))
    portfolio_ret = float(np.sum(best_weights*(best_portfolio_returns.mean()*252)))
    best_sharpe = float(fitness(best_individual, log_returns.T))
    selected_tickers = list(prices_df.iloc[:, indeces].columns)
    stock_allocations = {ticker: weight for ticker, weight in
                         zip(selected_tickers, sol.x)}
    logger.info("Optimal weights: %s", sol.x)
    logger.info("Portfolio return=%.4f, risk=%.4f, Sharpe=%.4f", portfolio_ret, risk, best_sharpe)
    logger.info("Allocations: %s", stock_allocations)

    # Save to database
    import db
    conn = db.get_connection()
    run_id = db.save_optimisation_run(conn,
        params={
            'script': 'optimisation',
            'data_source': 'yahoo_finance',
            'min_etfs': MIN_NUM_STOCKS,
            'max_etfs': MAX_NUM_STOCKS,
            'min_weight': MIN_WEIGHT,
            'max_weight': MAX_WEIGHT,
            'target_return': TARGET_RETURN,
            'target_risk': TARGET_RISK,
            'use_forecasts': use_forecasts,
        },
        results={
            'best_sharpe': best_sharpe,
            'portfolio_return': portfolio_ret,
            'portfolio_volatility': risk,
            'num_selected': int(np.count_nonzero(best_individual)),
            'elapsed_seconds': opt_elapsed,
        },
        holdings=list(zip(selected_tickers, sol.x)))
    print(f"Run saved to database (id={run_id})")
    conn.close()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    prices_df = load_data('Data/NZ_ETF_Prices.csv')
    prices_df = prices_df.dropna(axis=1, thresh=0.95*len(prices_df))
    logger.info("Loaded price data: %d rows x %d columns", *prices_df.shape)
    prepare_opt_inputs(prices_df, use_forecasts=False)
    log_returns = calculate_returns(prices_df)
    # portfolio = create_portfolio(num_children=100)
    # portfolio = ['QQQ', 'STIP', 'SPTI', 'SMOG', 'VIXM', 'LEAD']
    portfolio = ['USF.NZ', 'NZC.NZ', 'USV.NZ', 'USA.NZ', 'ASF.NZ']
    # portfolio = load_data('Data/3x_leveraged_ETFs.csv').index.to_list()

    logger.info("Portfolio: %s", portfolio)
    data = log_returns.loc[:, portfolio]
    random_weights = np.random.random(len(portfolio))
    random_weights /= np.sum(random_weights)
    res = optimize(data,
                   random_weights,
                   risk_parity=False,
                   max_weight=0.4,
                   target_return=0.15,
                   use_copulae=True)
    logger.info("Optimisation result: %s", res)
