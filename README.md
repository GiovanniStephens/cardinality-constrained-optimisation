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
- [x] Run the backtest with another group of portfolios optimised using the forecasted returns and variances.
- [x] Create some unit tests and validations to check that the optimisation algorithm and backtest is working as expected.
- [ ] I want to maybe use time indeces rather than just numbers to make sure that I am using the right days in the backtest.
- [x] Maybe do some exploratory data analysis (EDA) on the input pricess to understand price data better.
- [x] There is an issue with the ending results when you fill the blanks with zero-valued returns. The results were as expected when I just dropped the nulls. I need to think about how I am going to go about solving this issue. Maybe I can impute the price data using an ARIMA model or something similar before calculating returns. I need to understand the nature of the nulls in the prices though. Maybe there are heaps that are blank at the beginning of the data for a few ETFs. I need to look into this.
- [x] It seems like the constrained portfolios are not performing well when compared to random portfolios when there is a long historical data for training. I do not know what that is, tbh. I want to see what happens when we train derive the variance-covariance matrix from just 1 year of data as opposed to the full historical data. 
- [x] I will create an optimisation where I get the weightings using the copula-estimated correlations with forecast variances to create an estimate variance-covariance matrix for the optimisation. I want to see if this does better than the CCC model and the historical covariances.
- [ ] Another thing that would be cool to do is to create a moving optimisation that 'rebalances' each n days. So you would run the cardinality optimisation once, and then a mean-variance optimisation every n days. This would allow you to see how the constrained portfolios perform over time.