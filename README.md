# Cardinality Constrained Optimisation

The goal of this optimisation is to find a subset portfolio of N stocks from a universe of M stocks that maximises the Sharpe Ratio. 

The optimisation is constrained by the maximum number of stocks that can be held (i.e. N=10). 

The sum of the portfolio weightings is constrained to 1 (i.e. no leverage or cash).

Each stock allocation (weighting) is constrained to be between 0 and 1 (or a maximum of choice). (i.e. no shorting).

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

Using historical variances, covariances, and mean returns, the average out-of-sample Sharpe Ratio is significantly greater for optimised cardinality-constrained portfolios than for random selections and allocations. Regardless, it would be interesting to see if I can get even better results using forecast average returns and variances with historical covariances.

## Forecasting Returns

One approach could be to fit multiple forecasting models to each of the ETFs to get a forecast of the returns. It would be quite slow, so I would need to pre-estimate all of the expected returns ahead of time and then run the optimisation.

This approach may be appropriate, though, because I am looking at a huge range of ETFs. Some are equities, some are mixed across nations, others are commodities, some are TIPS, some are bonds, others are currencies, or derivatives, or even trading strategies. So fundamental estimates of long-term returns are not appropriate unless you have a whole lot of additional data to estimte GDP growth in each country, ETF constituents, dividend yields for each holding, inflation in each country, FX movements etc. It's pretty impractical to do this.

I am thinking the best approach will be to use univariate forecasting models to get a feel for the returns of each ETF.

For now, I have just used an ARIMA model to forecast returns for each of the ETFS.

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

## Forecasting variances

I have been able to forecast variances using a GARCH model; however, I am not too sure about how to incorporate the forecasts into the historical variance-covariance matrix. I have a hunch that just changing the diagonal elements of the covariance matrix does not make any sense.

After I validate that I cannot just update the diagonal elements of the covariance matrix, I will look into forecasting covariances too. I think that there is a lot of research on forecasting the variance-covariance matrix... 

## Forecasting covariances

I think there is a constant conditional correlation (CCC) model that forecasts the variances using a GARCH model and uses historical correlations to back out the variance-covariance matrix. That's pretty much what I am planning to do. I just want to check whether replacing the diagonal of a variance-covariance matrix with the forecast variances is the same as the CCC variance-covariance matrix. 

If this is not looking the same, I could look into forecasting the variance-covariance matrix using a GARCH DCC. DCC models have been pretty heavily criticised as not really correct time-varying correlations. 

Fitted copulas can be used to estimate covariance, but it may be a bit slow to compute. 

CCC is probably the fastest, but copulas might give a good result. Worth testing the two against each other. 

# Backtest

To test whether the cardinality-constrained portfolio does any better than a randomly selected portfolio, I will run a backtest. The backtest is conducted as follows:
1. Create N cardinality-constrained portfolios;
2. Create N randomly selected portfolios;
3. For each portfolio, create N random set of weightings and N set of optimal weightings;
4. For each portfolio, run them all forward, out-of-sample with the initial weightings;
5. Compare the results of the out-of-sample runs for each of the four groups.

The comparison of the groups is done using a one-tailed t-test. It is assumed that the mean of the out-of-sample Sharpe Ratios is normally distributed, random, and independent. The results in the next section show that the cardinality-constrained portfolios are significantly better than the randomly selected portfolios.

## Backtest results

Sample size = 100

Number of children per generation = 1000

Number of out-of-sample days = 150

Maximum number of holdings = 10

Maximum weighting per holding = 20%

Cardinality-constrained, optimised portfolio mean: 1.0469363694638476

Cardinality-constrained, optimised portfolio std: 0.5833230941274853

Cardinality-constrained, random weightings portfolio mean: 0.7076622398936138

Cardinality-constrained, random weightings portfolio std: 0.8433489843501207

Random selections, optimised portfolio mean: 0.38039021527564487

Random selections, optimised portfolio std: 0.6119541460273398

Random selections, random weightings portfolio mean: 0.2942966480176118

Random selections, random weightings portfolio std: 0.7356430516803205

Cardinality-constrained, optimised portfolio vs. random weightings t-statistic: 1.8121996564583602

Random selections, optimised portfolio vs. random weightings t-statistic: 0.4927930187163222

Cardinality-constrained, optimised portfolio vs. random selection, optimised t-statistic: 4.318298361944475

Cardinality-constrained, optimised portfolio vs. random selection, random weightings t-statistic: 4.390886784250938

![Out-of-sample Sharpe Ratio Distributions by Portfolio Construction Method](https://github.com/GiovanniStephens/cardinality-constrained-optimisation/blob/main/Images/Out-of-sample%20Sharpe%20Ratio%20Distributions%20by%20Portfolio%20Construction%20Method.png)

# To do

- [x] Forecast returns using ARIMA models for each ETF.
- [x] Forecast variance for each ETF using AR-GARCH models.
- [x] Update the optimisation algorithm to use the forecasted returns and variances.
- [x] Since I will be using historical and forecast variances and returns, I would like to be able to compare the historical estimates vs. the forecasted estimates. As a result, the optimisation will need a toggle to go back and forth between the two.
- [x] Validate whether I can just update the diagonal elements of a variance-covariance matrix with forecast variances.
- [x] If I cannot just update the diagonal elements of a variance-covariance matrix with forecast variances, I will need to forecast covariances.
- [ ] Run the backtest with another group of portfolios optimised using the forecasted returns and variances.

