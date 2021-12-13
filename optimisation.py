import numpy as np
import pandas as pd
import scipy.optimize as opt


def sharpe_ratio(weights, returns):
    """
    Calculates the Sharpe ratio of a portfolio.

    :weights: numpy array of weights.
    :returns: numpy array of log returns.
    :return: float of the negative Sharpe ratio.
    """
    p_returns = np.sum((returns.mean()*weights*252))
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    return -p_returns/p_volatility


def load_data(filename):
    """
    Loads the data from a CSV file.

    :filename: string of the filename.
    :return: pandas dataframe of the data.
    """
    return pd.read_csv(filename)


if __name__ == '__main__':
    # Load the data
    data = load_data('ETF_Prices.csv')

    print(data.head())