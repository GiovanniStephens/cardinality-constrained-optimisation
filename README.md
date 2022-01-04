# Cardinality Constrained Optimisation

The goal of this optimisation is to find a subset portfolio of N stocks from a universe of M stocks that maximises the Sharpe Ratio. 

The optimisation is constrained by the maximum number of stocks that can be held (i.e. N). 

The portfolio weightings is constrained to 1 (i.e. no leverage or cash).

Each stock allocation (weighting) is constrained to be between 0 and 1. (i.e. no shorting).

Lastly, if the user wants, he/she should be able to specify a minimum expected return. He/she should also be able to run the optimisation constrained by a maximum level of risk.

## Why Cardinality Constrained?

I want to constrain the number of stocks that can be held in the portfolio so that transaction costs and the effort to rebalance the portfolio is manageable. The other reason for this is that the optimisation relys on a variance-covariance matrix that needs at least T = N time series obseravtions else the variance-covariance matrix cannot be invertible. 

So, for a universe of 500 stocks, I would need at least 500 (2 years worth of daily) observations to be able to acturately estimate the variance-covariance matrix. This poses a challenge when trying to pick an optimal portfolio from a universe of 500+ stocks. 

In terms of ETFs, I have a set of around 1700+ that I would like to pick from. So constraining the portfolio to a maximum number of ETFs is a necessary constraint.

## Implicit Assumptions

When running a mean-variance optimisation, it is assumed implicitly that the variance-covariance structure of the portfolio will remain constant. The same goes for the expected returns. 

Another assumption is that historical average returns and sample variances and covariances are good estimators for the true (and assumed constant) average returns and variances and covariances. This is a faulty assumption because all three variables change over time. 

# Objective Function

The objective function is defined as: 

obj = E(R)/Std(R)

where E(R) is the stock's expected return and Std(R) is the standard deviation of the returns.

## Alternative Objective Function

An alternative is to use a tail risk objective function. For example conditional value at risk (CVaR) or conditional expected shortfall (CES). Although, these functions would likely need to be estimated using numerical estimations.

An issue with this alternative function is that it is slow to estimate. It cannot be repeated quickly for thousands of hypothetical portfolios. The Sharpe Ratio is a fast way to gauge whether the portfolio would be any good. 

# Plan

I want to be able to solve the cardinality constrained problem quickly, so an approximate solution like evolutionalry (done) or particle swarm algorithms could be used. An alternative would be to use mixed integer programming to solve the problem, but I have not been able to work out how to do that yet. 

First, I am going to try create a genetic algorithm (done), and then I will try something else. Maybe I will constrain the problem to take a limited amount of time too... we will see. (done. I used early termination to speed up the process)

Next step will be to create a particle swarm optimisation algorithm to see how it compares to the genetic algorithm for speed and outcome.

Try a different genetic algorithm to see if it is faster. (pyGAD could be a good alternative). (done)

I want to be able to run the optimisation on forward-looking variances and covariances and forecast returns. I am not sure how I would go about doing this, however. (Food for thought!)

## Basic Back-Test Plan

One basic validation for the optimisation method is to run and out of sample backtest. This can be done by creating N optimal portfolios optimised on the data 3 and 2 years ago. You can then compare their performance on the last year against N random optimal portfolios, and N random portfolios. 

My initial guess is that the first group of cardinality-constrained optimal portfolios would perform better than the second group of random optimal portfolios. The random selection of ETFs that have been optimally allocated should perform better than just a random allocation with a random selection of ETFs. 

I would need N >= 30 to be able to compare the performance of the two groups, but I would rather it be done on like 100 or 200 portfolios. The issue is that it takes quite a while to run the cardinality-constrained optimisation at this stage. I may have to just create a quick optimal portfolio with fewer GA children per generations and generations. (Hopefully, this still creates better results than the random optimal portfolios). 

The comparison of the 3 groups would be on Sharpe Ratio, which is the training objective function. The average Sharpe ratio for the cardinality-constrained portfolios should should be greater than the random selection of ETFs optimised, and it needs to be statistically significant.

## Forecasting Returns

One approach could be to fit multiple forecasting models to each of the ETFs to get a forecast of the returns. It would be quite slow, so I would need to pre-estimate all of the expected returns ahead of time and then run the optimisation.

This approach may be appropriate, though, because I am looking at a huge range of ETFs. Some are equities, some are mixed across nations, others are commodities, some are TIPS, some are bonds, others are currencies, or derivatives, or even trading strategies. So fundamental estimates of long-term returns are not appropriate unless you have a whole lot of additional data to estimte GDP growth in each country, ETF constituents, dividend yields for each holding, inflation in each country, FX movements etc. It's pretty impractical to do this.

I am thinking the best approach will be to use univariate forecasting models to get a feel for the returns of each ETF.

### Long-term Expected Returns Research 

There was mention of the following things in an initial Google search:
1. Shiller E/P as a long-term forecast
2. D/P (Dividend payout ratio) as a long-term forecast
3. Broad payout yield
4. Book/(Market*10)
5. CAPM (Capital Asset Pricing Model)
6. Growth in P/E
7. Dividend yield
8. Expected growth in earnings
9. Long-term return from equities = Dividend yield + Inflation + Real earnings growth
10. "The first is to assume a premium over interest rates or bond returns, justified by the
risk-averse behaviour of portfolio investors. The second approach is to project dividend
income assuming a link with inflation and/or parity with gross domestic profit." - [A Review of the Methodology of
Forecasting Long-term Equity Returns](https://fbe.unimelb.edu.au/__data/assets/pdf_file/0003/2591805/184.pdf)


## To do

- [x] I want to clean out the old genetic algorithm code because the new one is better.
- [ ] I want to refactor some of the functions to make it cleaner.