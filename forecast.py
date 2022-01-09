import optimisation as op
import backtest
import pmdarima as pmd
from arch import arch_model
import numpy as np

# Forecast returns
data = op.load_data('ETF_Prices.csv')
training_data = data.iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE, :]
log_returns = op.calculate_returns(training_data)
test_log_returns = op.calculate_returns(data.iloc[-backtest.NUM_DAYS_OUT_OF_SAMPLE:, :])

autoarima_model = pmd.auto_arima(training_data['QQQ'],
                                 start_p=1,
                                 start_q=1,
                                 max_p=3,
                                 max_q=3,
                                 seasonal=False,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True, 
                                 n_jobs=-1, 
                                 stepwise=False)

n_periods = 252
forecast = autoarima_model.predict(n_periods=n_periods, return_conf_int=False)
print(autoarima_model.summary())
print(f'Forecast annualized returns: {np.log(forecast[-1]/forecast[0])}')

# Forecast volatility
am = arch_model(log_returns['QQQ'], mean='AR', vol="Garch", p=1, o=1, q=1, dist="skewt")
res = am.fit(disp='off')
forecast = res.forecast(horizon=n_periods, reindex=False)
print(f'Forecast annualized volatility: {sum(forecast.variance.iloc[0])}')

# Create a covariance matrix with historical covariances, and update the diagonal with forecast vol.
cov_matrix = np.cov(log_returns.T)

