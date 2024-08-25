import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager
import os

def load_data(filename: str) -> pd.DataFrame:
    prices_df = pd.read_csv(filename, index_col=0)
    prices_df = prices_df.dropna(axis=1, thresh=0.95*len(prices_df))
    prices_df = prices_df.fillna(method='ffill')
    return prices_df

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    return np.log(data / data.shift(1)).fillna(0).replace([np.inf, -np.inf], 0)

def calculate_covariance_matrix(log_returns: pd.DataFrame) -> np.ndarray:
    """Return the covariance matrix as a NumPy array."""
    return log_returns.cov().values * 252

def calculate_expected_returns(log_returns: pd.DataFrame) -> pd.Series:
    return log_returns.mean() * 252

def random_portfolio(num_etfs, max_num_etfs):
    """ Generate a random individual with constraints on the number of ETFs. """
    num_selected = np.random.randint(3, max_num_etfs + 1)  # Ensure between 3 and max_num_etfs ETFs are selected
    portfolio = np.zeros(num_etfs, dtype=int)
    selected_indices = np.random.choice(num_etfs, num_selected, replace=False)
    portfolio[selected_indices] = 1
    return portfolio

def calculate_fitness(portfolio, expected_returns, cov_matrix):
    """Calculate the Sharpe Ratio for the given portfolio, ensuring numpy handling."""
    selected_indices = portfolio == 1
    num_selected_etfs = np.sum(selected_indices)
    if num_selected_etfs < 3 or num_selected_etfs > 10:
        return -1e4  # Penalize solutions that do not meet the holding constraints
    if not np.any(selected_indices):
        return 0  # Avoid division by zero if no ETFs are selected

    # Ensure handling with numpy arrays
    filtered_returns = expected_returns[selected_indices]
    filtered_cov_matrix = cov_matrix[np.ix_(selected_indices, selected_indices)]
    weights = np.ones(num_selected_etfs) / num_selected_etfs
    portfolio_return = np.dot(weights, filtered_returns)
    portfolio_variance = np.dot(weights, np.dot(filtered_cov_matrix, weights))

    return portfolio_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0


def monte_carlo_search(data, trials, max_num_etfs):
    log_returns = calculate_returns(data)
    expected_returns = calculate_expected_returns(log_returns).values  # Convert to numpy array
    cov_matrix = calculate_covariance_matrix(log_returns)
    num_etfs = data.shape[1]

    best_fitness = float('-inf')
    best_portfolio = None

    for _ in range(trials):
        portfolio = random_portfolio(num_etfs, max_num_etfs)
        fitness = calculate_fitness(portfolio, expected_returns, cov_matrix)
        if fitness > best_fitness:
            best_fitness = fitness
            best_portfolio = portfolio

    return best_portfolio, best_fitness

def parallel_monte_carlo(data, num_trials, num_processes, max_num_etfs):
    pool = Pool(num_processes)
    trials_per_process = num_trials // num_processes
    results = pool.starmap(monte_carlo_search, [(data, trials_per_process, max_num_etfs) for _ in range(num_processes)])
    pool.close()
    pool.join()

    # Find the best result across all processes
    best_solution, best_fitness = max(results, key=lambda x: x[1])
    return best_solution, best_fitness

# Load data and run the Monte Carlo simulation
data = load_data('Data/ETF_Prices.csv')
num_trials = 10000000
num_processes = os.cpu_count()
max_num_etfs = 10
best_solution, best_fitness = parallel_monte_carlo(data, num_trials, num_processes, max_num_etfs)

print("Best Solution:", best_solution)
print("Best Sharpe Ratio:", best_fitness)
print("Selected ETFs:", data.columns[best_solution == 1])
