import optimisation as op
import backtest
import pmdarima as pmd
from arch import arch_model
import numpy as np
import pandas as pd
import tqdm

data = op.load_data('ETF_Prices.csv')
training_data = data.iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE, :]
log_returns = op.calculate_returns(training_data)
test_log_returns = op.calculate_returns(data.iloc[-backtest.NUM_DAYS_OUT_OF_SAMPLE:, :])

# Forecast returns
n_periods = 252
print('Forecasting returns...')
expected_returns = {}
for ticker in tqdm.tqdm(data.columns):
    autoarima_model = pmd.auto_arima(training_data[ticker].dropna(),
                                    start_p=1,
                                    start_q=1,
                                    max_p=5,
                                    max_q=5,
                                    seasonal=False,
                                    trace=False,
                                    error_action='ignore',
                                    suppress_warnings=True, 
                                    n_jobs=-1, 
                                    stepwise=False)
    forecast = autoarima_model.predict(n_periods=n_periods, return_conf_int=False)
    expected_returns[ticker] = np.log(max(0.0001,forecast[-1])/forecast[0])
expected_returns = pd.DataFrame.from_dict(expected_returns, orient='index')
expected_returns.to_csv('expected_returns.csv')

# Forecast volatility
print('\nForecasting volatility...')
volatilities = {}
for ticker in tqdm.tqdm(data.columns):
    am = arch_model(100*log_returns[ticker], vol="Garch", p=1, o=1, q=1, dist="skewt", rescale=False)
    res = am.fit(disp='off')
    forecast = res.forecast(horizon=n_periods, reindex=False)
    volatilities[ticker] = forecast.residual_variance.iloc[-1].mean()/np.power(100,2)*252
volatilities = pd.DataFrame.from_dict(volatilities, orient='index')
volatilities.to_csv('variances.csv')
