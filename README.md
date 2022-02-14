# Cardinality Constrained Optimisation

The goal of this optimisation is to find a subset portfolio of N stocks from a universe of M stocks that maximises some objective function. The portfolio should test well out-of-sample (i.e. performs well on data the model has not seen yet).

The optimisation is constrained by the maximum number (N) of stocks that can be held (e.g. N=10). 

The sum of the portfolio weightings is constrained to 1 (i.e. no leverage or cash).

Each stock allocation (weighting) is constrained to be between 0 and 1 (or a maximum of choice). (i.e. no shorting).
(This constraint can be changed to anything you want in the optimisation module.)

Lastly, if the user wants, he/she can specify a minimum expected return. He/she can also run the optimisation constrained by a maximum level of risk (portfolio standard deviation).

## Why Cardinality Constrained?

I want to constrain the number of stocks that can be held in the portfolio so that transaction costs and the effort to rebalance the portfolio is manageable. The other reason for this is that the optimisation relys on a variance-covariance matrix that needs at least T = N time series obseravtions, else the variance-covariance matrix cannot be invertible. 

So, for a universe of 500 stocks, I would need at least 500 (2 years worth of daily) observations to be able to acturately estimate the variance-covariance matrix. This poses a challenge when trying to pick an optimal portfolio from a universe of 500+ stocks. 

In terms of ETFs, I have a set of around 1700+ that I would like to pick from. So constraining the portfolio to a maximum number of ETFs is a necessary constraint.

Another reason for this is to minimise transaction costs when rebalancing and entering into N positions. It is quite unfeasible to enter into 50+ positions unless you have a significant amount of capital.

## Implicit Assumptions

When running a mean-variance optimisation, it is assumed implicitly that the variance-covariance structure of the portfolio will remain constant. The same goes for the expected returns. 

Another assumption is that historical average returns and sample variances and covariances are good estimators for the true (and assumed constant) average returns and variances and covariances. This is a faulty assumption because all three variables change over time. 

To mitigate this, I forecast variances, covariances and returns and compare to portfolios optimised using historical data.

# Objective Function

The objective function is defined as: 

obj = E(R)/Std(R)

where E(R) is the portfolio's expected return and Std(R) is the standard deviation of the portfolio returns.

## Alternative Objective Function (not currently used)

An alternative is to use a tail risk objective function. For example conditional value at risk (CVaR) or conditional expected shortfall (CES). Although, these functions would likely need to be estimated using numerical estimations.

One way to estimate this value is to simulate the portfolio using a Copula-GARCH model and then taking the ith percentile.

An issue with this alternative function is that it is slow to estimate. It cannot be repeated quickly for thousands of hypothetical portfolios. The Sharpe Ratio is a fast way to gauge whether the portfolio would be any good. 

# Plan

I want to be able to solve the cardinality constrained problem quickly, so an approximate solution like evolutionalry (done) or particle swarm algorithms could be used. An alternative would be to use mixed integer programming to solve the problem, but I have not been able to work out how to do that yet. 

First, I am going to try create a genetic algorithm (done), and then I will try something else. Maybe I will constrain the problem to take a limited amount of time too... we will see. (done. I used early termination to speed up the process)

Next step will be to create a particle swarm optimisation algorithm to see how it compares to the genetic algorithm for speed and outcome.

Try a different genetic algorithm to see if it is faster. (pyGAD could be a good alternative). (done)

Using historical variances, covariances, and mean returns, the average out-of-sample Sharpe Ratio is significantly greater for optimised cardinality-constrained portfolios than for random selections and allocations. Regardless, it would be interesting to see if I can get even better results using forecast average returns and variances with historical covariances.

I have created the whole model, however, I have not created automated unit tests alogn the way. As a result, I have lost a little of my trust in the model. Next steps are to do a lot of validation and simplificaton of the model.

Additionally, the model is quite slow, so I need to work out how to speed it up further.

## Forecasting Returns

To get an estimate of the expected returns, I am using univariate ARIMA models for each ETF and then projecting the price to get returns. The models are fitted using auto-ARIMA to minimise the AIC (Akaike Information Criterion) value.

The forecast returns are then fed back into the optimisation algorithm.

### Long-term Expected Returns Research 

Though I juse use an ARIMA model to forecast returns, I did look into other expected return models to feed into the optimisation. Here were some of the quick findings from an initial Google search:
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

## Forecasting variances

I have been able to forecast variances using a GARCH model. The variances are, then, loaded into the variance-covariance matrix using Bollerslov's (1990) constant conditional correlation (CCC) model. This can be done by taking a historical (and static) correlation matrix and then transforming it to a covariance matrix and inserting the forecast variances into the matrix.

## Forecasting covariances

Bollerslov (1990) proposes a constant conditional correlation (CCC) model that forecasts the variances using a GARCH model and uses historical correlations to back out the variance-covariance matrix. For the optimisation, I use this method if I am using forecast variances. 

After the portfolios have been selected through the cardinality-constrained optimisation, I optimise the weights using the CCC model using forecast variances and a correlation matrix estimated by fitting skew-t copulas to the residuals from an AR-GARCH model.

The fitted copulas are used to estimate covariance, but it is a bit slow to compute. 

The estimated variance-coverage matrix gives better results than just using historical values on average.

# Backtest

To test whether the cardinality-constrained portfolio does any better than a randomly selected portfolio, I run a backtest. The backtest is conducted as follows:
1. Create N cardinality-constrained portfolios optimised on historical data;
2. Create N randomly selected portfolios optimised on historical data;
3. Create N cardinality-constrained portfolios optimised using forecasted variances and expected returns;
4. For each portfolio, create N random set of weightings and N set of optimal weightings using copulae and using the CCC model;
5. For each portfolio, run them all forward, out-of-sample with the initial weightings;
6. Compare the results of the out-of-sample runs for each of the six groups.

The comparison of the groups is done using a one-tailed t-test. It is assumed that the mean of the out-of-sample Sharpe Ratios is normally distributed, random, and independent. The results in the next section show that the cardinality-constrained portfolios are significantly better than the randomly selected portfolios.

## Backtest results

Sample size = 100

Number of children per generation = 1000

Number of out-of-sample days = 252

Maximum number of holdings = 10

Minimum number of holdings = 3

Maximum weighting per holding = 20%

Minimum weighting per holding = 0%

Cardinality-constrained, optimised portfolio mean:           2.9537768422303015
Cardinality-constrained, optimised portfolio std:           0.4731999673156576
Cardinality-constrained, optimised portfolio using copulae mean:           3.064262233898966
Cardinality-constrained, optimised portfolio using copulae std:           0.4829962345476325
Cardinality-constrained, optimised portfolio w/ forecasts mean:           2.983131149108223
Cardinality-constrained, optimised portfolio w/ forecasts std:           0.40163833672694216
Cardinality-constrained, random weightings portfolio mean:           2.5689569449391905
Cardinality-constrained, random weightings portfolio std:           0.8758641343139768

Random selections, optimised portfolio mean:           0.6269879489226622
Random selections, optimised portfolio std:           0.40837903507269024
Random selections, random weightings portfolio mean:           0.10187350826412626
Random selections, random weightings portfolio std:           0.484911941229262

Cardinality-constrained, optimised portfolio vs. random weightings t-statistic:         -3.8655240526741874
Random selections, optimised portfolio vs. random weightings t-statistic:         -8.283004933819473
Cardinality-constrained, optimised portfolio vs. random selection, optimised t-statistic:         -37.22544687238842
Cardinality-constrained, optimised portfolio vs. random selection, random weightings t-statistic:         -42.09215059012316
Cardinality-constrained, optimised portfolio vs. cardinality-constrained w/ forecast values and optimal weightings t-statistic:         0.4729452837454157

![Out-of-sample Sharpe Ratio Distributions by Portfolio Construction Method](https://github.com/GiovanniStephens/cardinality-constrained-optimisation/blob/main/Images/Out-of-sample%20Sharpe%20Ratio%20Distributions%20by%20Portfolio%20Construction%20Method.png)

# To do

- [x] Check that the comments are appropriate.
- [x] Do some more tidying and linting.