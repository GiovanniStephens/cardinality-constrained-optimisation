# Cardinality Constrained Optimisation

The goal of this optimisation is to find a subset portfolio of N stocks from a universe of M stocks that maximises the Sharpe Ratio. 

The optimisation is constrained by the maximum number of stocks that can be held (i.e. N). 

The portfolio weightings is constrained to 1 (i.e. no leverage or cash).

Each stock allocation (weighting) is constrained to be between 0 and 1. (i.e. no shorting).

Lastly, if the user wants, he/she should be able to specify a minimum expected return. He/she should also be able to run the optimisation constrained by a maximum level of risk.

# Objective Function

The objective function is defined as: 

obj = E(R)/Std(R)

where E(R) is the stock's expected return and Std(R) is the standard deviation of the returns.