import warnings

import numpy as np
import pandas as pd
import pygad
import scipy.optimize as opt
from copulae import TCopula
from muarch import MUArch

warnings.filterwarnings("ignore")

MAX_NUM_STOCKS = 15
MIN_NUM_STOCKS = 3
TARGET_RETURN = 0.15
TARGET_RISK = None
MAX_WEIGHT = 0.45
MIN_WEIGHT = 0.1
last_fitness = 0
data = None
variances = None
expected_returns = None


def sharpe_ratio(weights: np.array, returns: list, cov: list) -> float:
    """
    Calculates the Sharpe ratio of a portfolio.
    The Sharpe ratio is the ratio of the mean return of the portfolio
    to the portfolio standard deviation.

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
    # Remove columns with 10% or more null values
    prices_df = prices_df.dropna(axis=1, thresh=int(len(prices_df)/10))
    # Fill the null values with the previous day's close price
    prices_df = prices_df.fillna(method='ffill')
    return prices_df


def get_cov_matrix(data: pd.DataFrame, use_copulae=False) -> pd.DataFrame:
    """
    Calculates the covariance matrix of the data.
    If there are forecast variances, the covariance
    matrix gets updated to include them.

    (see: Bollerslev, T. (1990). Modelling the Coherence in Short-Run
    Nominal Exchange Rates: A Multivariate Generalized Arch Model.
    The Review of Economics and Statistics,
    72(3), 498–505. https://doi.org/10.2307/2109358)

    :data: pandas dataframe of the returns data.
    :return: pandas dataframe of the covariance matrix.
    """
    # If we have forecast variances, use the forecast variances
    # to update the covariance matrix.
    if variances is not None:
        D = np.zeros((data.shape[1], data.shape[1]))
        diag = np.sqrt(variances.loc[data.columns].values)
        np.fill_diagonal(D, diag)
        if use_copulae:
            corr = estimate_corr_using_copulas(data)
        else:
            corr = data.corr()
        cov_matrix = np.matmul(np.matmul(D, corr), D)
    else:
        cov_matrix = data.cov()*252  # Historical sample cov
    return cov_matrix


def estimate_corr_using_copulas(data: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates the covariance matrix using the copula method.

    It first models the returns using an AR(1)-GARCH(1, 1)
    with skewt innovations. Then it uses the copula to
    estimate the correlation matrix.

    :data: pandas dataframe of the log returns data.
    :return: pandas dataframe of the covariance matrix.
    """
    # Estimate GARCH model for each time series
    models = MUArch(data.shape[1], mean='AR', lags=1, dist='skewt', scale=10)
    # Estimate GARCH model for each time series
    models.fit(data)
    # Fit residuals into a copula
    residuals = models.residuals()
    cop = TCopula(dim=data.shape[1])
    cop.fit(residuals)
    return cop.sigma


# risk budgeting optimization
def calculate_portfolio_var(w, V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0, 0]


def calculate_risk_contribution(w, V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w, V))
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
    cov_matrix = get_cov_matrix(data, use_copulae)
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
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


def prepare_opt_inputs(prices, use_forecasts: bool) -> None:
    """
    Prepares the inputs for the optimisation.

    :prices: pandas dataframe of the prices.
    :use_forecasts: bool of whether to use forecasts.
    """
    global variances, expected_returns, data
    if use_forecasts:
        data = calculate_returns(prices).transpose()
        variances = load_data('Data/variances.csv')
        expected_returns = load_data('Data/expected_returns.csv')['0']
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
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    if verbose:
        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")
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
    # Load the data
    prices_df = load_data('Data/NZ_ETF_Prices.csv')
    # Prepare the inputs for the optimisation
    prepare_opt_inputs(prices_df, use_forecasts=True)

    log_returns = calculate_returns(prices_df)
    # Run the cardinality constrained optimisation
    best_individual = cardinality_constrained_optimisation(num_children=500,
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
                   max_weight=MAX_WEIGHT,
                   min_weight=MIN_WEIGHT)
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


if __name__ == '__main__':
    prices_df = load_data('Data/NZ_ETF_Prices.csv')
    prices_df = prices_df.dropna(axis=1, thresh=0.95*len(prices_df))
    prepare_opt_inputs(prices_df, use_forecasts=False)
    log_returns = calculate_returns(prices_df)
    # portfolio = create_portfolio(num_children=100)
    # portfolio = ['QQQ', 'STIP', 'SPTI', 'SMOG', 'VIXM', 'LEAD']
    portfolio = ['USF.NZ', 'NZC.NZ', 'USV.NZ', 'USA.NZ', 'ASF.NZ']
    # portfolio = load_data('Data/3x_leveraged_ETFs.csv').index.to_list()

    print(portfolio)
    data = log_returns.loc[:, portfolio]
    random_weights = np.random.random(len(portfolio))
    random_weights /= np.sum(random_weights)
    res = optimize(data,
                   random_weights,
                   risk_parity=False,
                   max_weight=0.4,
                   target_return=0.15,
                   use_copulae=True)
    print(res)
