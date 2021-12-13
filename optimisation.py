import numpy as np

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