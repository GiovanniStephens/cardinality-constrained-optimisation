import numpy as np
import pandas as pd
import scipy.optimize as opt


def sharpe_ratio(weights: np.array, returns: pd.DataFrame) -> float:
    """
    Calculates the Sharpe ratio of a portfolio.

    :weights: numpy array of weights.
    :returns: numpy array of log returns.
    :return: float of the negative Sharpe ratio.
    """
    p_returns = np.sum((returns.mean()*weights*252))
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    return -p_returns/p_volatility


def load_data(filename: str) -> pd.DataFrame:
    """
    Loads the data from a CSV file.

    :filename: string of the filename.
    :return: pandas dataframe of the data.
    """
    return pd.read_csv(filename)


def optimize(data: pd.DataFrame, initial_weights: np.array,
             max_weight: float=0.5) -> float:
    """
    Optimizes the portfolio using the Sharpe ratio.

    :data: pandas dataframe of the data.
    :initial_weights: numpy array of initial weights.
    :return: numpy array of optimized weights.
    """
    cons = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})
    bounds = tuple((0, max_weight) for x in range(len(initial_weights)))
    sol = opt.minimize(sharpe_ratio, initial_weights, args=(data),
                       method='SLSQP', bounds=bounds, constraints=cons)
    return -sol['fun']


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the log returns of the data.

    :data: pandas dataframe of the data.
    :return: pandas dataframe of the log returns.
    """
    log_returns = np.log(data/data.shift(1))
    log_returns = log_returns.dropna()
    log_returns = log_returns.replace([np.inf, -np.inf], 0)
    return log_returns


if __name__ == '__main__':
    prices_df = load_data('ETF_Prices.csv')
    prices_df = prices_df.drop(prices_df.columns[0], axis=1)
    log_returns = calculate_returns(prices_df)
    num_holdings = 50
    weights = np.array(np.random.random(j))
    weights = weights/np.sum(weights)
    starting_ticker_index = 50
    sol = optimize(log_returns.iloc[:,starting_ticker_index:starting_ticker_index+num_holdings], weights)
    print(sol)