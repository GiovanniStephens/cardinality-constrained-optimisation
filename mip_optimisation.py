import pandas as pd
import numpy as np
import pulp


def load_data(filename: str) -> pd.DataFrame:
    prices_df = pd.read_csv(filename, index_col=0)
    prices_df = prices_df.dropna(axis=1, thresh=0.95*len(prices_df))
    prices_df = prices_df.fillna(method='ffill')
    return prices_df


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(data / data.shift(1))
    log_returns = log_returns.fillna(0)
    log_returns = log_returns.replace([np.inf, -np.inf], 0)
    return log_returns


def calculate_variances(log_returns: pd.DataFrame) -> pd.Series:
    return log_returns.var() * 252


def calculate_expected_returns(log_returns: pd.DataFrame) -> pd.Series:
    return log_returns.mean() * 252


def calculate_covariance_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    return log_returns.cov() * 252


def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights, np.dot(cov_matrix.values, weights))


def calculate_portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)


def portfolio_sharpe_ratio(selection_vars, expected_returns, log_returns):
    selected_etfs = [etf for etf in selection_vars if pulp.value(selection_vars[etf]) > 0.5]
    if not selected_etfs:
        return 0
    selected_log_returns = log_returns[selected_etfs]
    cov_matrix = calculate_covariance_matrix(selected_log_returns)
    num_etfs = len(selected_etfs)
    weights = np.array([1/num_etfs] * num_etfs)
    portfolio_variance = calculate_portfolio_variance(weights, cov_matrix)
    portfolio_return = calculate_portfolio_return(weights, expected_returns[selected_etfs])
    sharpe_ratio = portfolio_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0
    return sharpe_ratio


def setup_portfolio_selection_problem(etfs, expected_returns, volatilities, risk_aversion):
    portfolio_problem = pulp.LpProblem("Portfolio_Selection", pulp.LpMaximize)
    selection = pulp.LpVariable.dicts("Select", etfs, 0, 1, pulp.LpBinary)
    portfolio_problem += pulp.lpSum([expected_returns[etf] * selection[etf] - risk_aversion
                                     * volatilities[etf] * selection[etf] for etf in etfs]), "Risk_Adjusted_Return"
    portfolio_problem += pulp.lpSum([selection[etf] for etf in etfs]) <= 10, "Max_ETFs"
    return portfolio_problem, selection


if __name__ == '__main__':
    # Load data
    prices_df = load_data('Data/ETF_Prices.csv')
    prices_df = prices_df.iloc[:-213]
    log_returns = calculate_returns(prices_df)
    expected_returns = calculate_expected_returns(log_returns)
    volatilities = np.sqrt(calculate_variances(log_returns))

    # Define risk aversion coefficient
    risk_aversion = 0.8     # Adjust based on risk preference

    # Setup and solve the MILP problem
    portfolio_problem, selection = setup_portfolio_selection_problem(log_returns.columns, expected_returns,
                                                                     volatilities, risk_aversion)
    portfolio_problem.solve()

    # Output the selected ETFs
    print("Selected ETFs in the Portfolio:")
    for etf in log_returns.columns:
        if pulp.value(selection[etf]) == 1:
            print(etf)
    sharpe_ratio = portfolio_sharpe_ratio(selection, expected_returns, log_returns)
    print(f"Portfolio Sharpe Ratio: {sharpe_ratio}")
