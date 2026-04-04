import logging

import numpy as np
import pandas as pd
import pulp

from src.portfolio_utils import (
    load_prices_csv,
    calculate_log_returns,
    calculate_expected_returns,
    calculate_variances,
    calculate_covariance_matrix,
    optimise_weights,
    OptimisationResult,
)
from src.config import GA_MAX_SECURITIES

logger = logging.getLogger(__name__)


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


class MIPOptimiser:
    """Mixed Integer Linear Programming portfolio selection."""

    def __init__(self, max_securities=GA_MAX_SECURITIES, risk_aversion=0.8):
        self.max_securities = max_securities
        self.risk_aversion = risk_aversion

    def optimise(self, prices: pd.DataFrame) -> OptimisationResult:
        log_returns = calculate_log_returns(prices)
        er = calculate_expected_returns(log_returns)
        volatilities = np.sqrt(calculate_variances(log_returns))

        problem, selection = setup_portfolio_selection_problem(
            log_returns.columns, er, volatilities, self.risk_aversion)
        # Override max ETFs constraint with our setting
        problem.constraints.pop("Max_ETFs", None)
        problem += (
            pulp.lpSum([selection[etf] for etf in log_returns.columns])
            <= self.max_securities, "Max_ETFs"
        )
        problem.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = [etf for etf in log_returns.columns
                     if pulp.value(selection[etf]) > 0.5]
        sharpe = portfolio_sharpe_ratio(selection, er, log_returns)

        # SLSQP weight optimisation on selected subset
        sel_vector = np.array([1 if c in selected else 0
                               for c in prices.columns])
        weights = np.ones(len(selected)) / len(selected)
        if len(selected) >= 2:
            result = optimise_weights(sel_vector, prices)
            if result.success:
                weights = result.x
                sharpe = -result.fun

        return OptimisationResult(
            selected_tickers=selected,
            weights=weights,
            sharpe_ratio=sharpe,
            metadata={
                'risk_aversion': self.risk_aversion,
                'solver_status': pulp.LpStatus[problem.status],
            },
        )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Load data
    prices_df = load_prices_csv('data/ETF_Prices.csv')
    prices_df = prices_df.iloc[:-213]
    logger.info("Loaded price data: %d rows x %d columns", *prices_df.shape)
    log_returns = calculate_log_returns(prices_df)
    expected_returns = calculate_expected_returns(log_returns)
    volatilities = np.sqrt(calculate_variances(log_returns))

    # Define risk aversion coefficient
    risk_aversion = 0.8     # Adjust based on risk preference

    # Setup and solve the MILP problem
    portfolio_problem, selection = setup_portfolio_selection_problem(log_returns.columns, expected_returns,
                                                                     volatilities, risk_aversion)
    portfolio_problem.solve()
    logger.info("MILP solver status: %s", pulp.LpStatus[portfolio_problem.status])

    if portfolio_problem.status != pulp.constants.LpStatusOptimal:
        print(f"Warning: Solver did not find optimal solution. Status: {pulp.LpStatus[portfolio_problem.status]}")

    # Output the selected ETFs
    selected = [etf for etf in log_returns.columns if pulp.value(selection[etf]) == 1]
    logger.info("Selected ETFs in the Portfolio: %s", selected)
    sharpe = portfolio_sharpe_ratio(selection, expected_returns, log_returns)
    logger.info("Portfolio Sharpe Ratio: %.4f", sharpe)
