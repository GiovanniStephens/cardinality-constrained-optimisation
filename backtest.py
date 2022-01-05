import optimisation as op
import numpy as np

NUM_PORTFOLIOS = 100
NUM_DAYS_OUT_OF_SAMPLE = 150

# Create a set of cardinality-constrained portfolios
data = op.load_data('ETF_Prices.csv')
log_returns = op.calculate_returns(data)
op.data = log_returns.transpose().iloc[:, :-NUM_DAYS_OUT_OF_SAMPLE-1]
op.TARGET_RETURN = None
portfolios = [list(op.create_portfolio(500)) for _ in range(NUM_PORTFOLIOS)]

# Create a set of random portfolios
random_portfolios = [list(log_returns.iloc[:, op.create_individual(op.data).astype(bool)].columns) for _ in range(NUM_PORTFOLIOS)]


# For each portfolio, get the optimal allocation
def get_random_weights(portfolio):
    random_weights = np.random.random(len(portfolio))
    random_weights /= np.sum(random_weights)
    return random_weights

def optimal_weights(portfolio):
    random_weights = get_random_weights(portfolio)
    return op.optimize(op.data.loc[portfolio, :].transpose(),
                       random_weights,
                       max_weight=max(1/len(portfolio), 0.2))['x']

portfolios_weights = [optimal_weights(portfolio) for portfolio in portfolios]

random_portfolios_weights = [optimal_weights(portfolio) for portfolio in random_portfolios]

random_portfolios_random_weights = [get_random_weights(portfolio) for portfolio in random_portfolios]

# Runs the portfolio on the out-of-sample data with the initial weights
def run_portfolio(portfolio, weights):
    portfolio_returns = []
    start_i = op.data.shape[1] + 1
    for i in range(NUM_DAYS_OUT_OF_SAMPLE):
        portfolio_returns.append(np.sum(weights*log_returns.transpose().loc[portfolio].iloc[:, start_i + i]))
        # update the weights
        weights = weights*np.exp(log_returns.transpose().loc[portfolio].iloc[:, start_i + i])/(1+portfolio_returns[-1])
    return portfolio_returns


def fitness(portfolio_returns):
    return (np.mean(portfolio_returns)*252)/(np.std(portfolio_returns)*np.sqrt(252))

portfolios_fitness = [fitness(run_portfolio(portfolio, weights)) for portfolio, weights in zip(portfolios, portfolios_weights)]
random_portfolios_fitness = [fitness(run_portfolio(portfolio, weights)) for portfolio, weights in zip(random_portfolios, random_portfolios_weights)]
random_portfolios_random_fitness = [fitness(run_portfolio(portfolio, weights)) for portfolio, weights in zip(random_portfolios, random_portfolios_random_weights)]

# After running the portfolio forward, calculate the fitness of the portfolio 
# on the out-of-sample test. 

# The fitness results need to be aggregated by group.
# (Three groups. cardinality-constrained optimal, random, and random optimal)

print(np.array(portfolios_fitness).mean())
print(np.array(portfolios_fitness).std())
print(np.array(random_portfolios_fitness).mean())
print(np.array(random_portfolios_fitness).std())
print(np.array(random_portfolios_random_fitness).mean())
print(np.array(random_portfolios_random_fitness).std())