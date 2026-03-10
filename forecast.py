import logging
import time

import numpy as np
import pandas as pd
import pmdarima as pmd
import tqdm
from arch import arch_model

import backtest
import optimisation as op

logger = logging.getLogger(__name__)


def main():
    start_time = time.time()

    data = op.load_data('Data/ETF_Prices.csv')
    data = data.dropna(axis=1, thresh=0.95*len(data))
    training_data = data.iloc[:-backtest.NUM_DAYS_OUT_OF_SAMPLE, :]

    if len(training_data) < 30:
        raise ValueError(
            f"Insufficient training data: {len(training_data)} rows (need at least 30)."
        )

    log_returns = op.calculate_returns(training_data)

    # Forecast returns
    n_periods = 252
    print('Forecasting returns...')
    expected_returns = {}
    failed_return_tickers = []
    for ticker in tqdm.tqdm(data.columns):
        try:
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
            forecast = autoarima_model.predict(n_periods=n_periods,
                                               return_conf_int=False)
            if forecast.iloc[0] <= 0:
                logger.warning("Ticker %s: ARIMA forecast starts at %.4f; using historical mean.", ticker, forecast.iloc[0])
                expected_returns[ticker] = log_returns[ticker].mean() * 252
            else:
                expected_returns[ticker] = np.log(max(0.0001,
                                                       forecast.iloc[-1])/forecast.iloc[0])
        except Exception as e:
            logger.warning("ARIMA forecast failed for %s (%s); using historical mean.", ticker, e)
            expected_returns[ticker] = log_returns[ticker].mean() * 252
            failed_return_tickers.append(ticker)

    if failed_return_tickers:
        logger.info("ARIMA forecasts failed for %d tickers.", len(failed_return_tickers))

    expected_returns = pd.DataFrame.from_dict(expected_returns,
                                              orient='index')
    expected_returns.to_csv('Data/expected_returns.csv')

    # Forecast volatility
    print('\nForecasting volatility...')
    volatilities = {}
    failed_vol_tickers = []
    for ticker in tqdm.tqdm(data.columns):
        try:
            am = arch_model(100*log_returns[ticker],
                            vol="Garch",
                            p=1,
                            o=1,
                            q=1,
                            dist="skewt",
                            rescale=False)
            res = am.fit(disp='off')
            forecast = res.forecast(horizon=n_periods,
                                    reindex=False)
            vol = forecast.residual_variance.iloc[-1].mean() \
                / np.power(100, 2)*252
            if np.isnan(vol) or vol <= 0:
                logger.warning("Ticker %s: GARCH forecast produced invalid variance (%.6f); using sample variance.", ticker, vol)
                volatilities[ticker] = log_returns[ticker].var() * 252
            else:
                volatilities[ticker] = vol
        except Exception as e:
            logger.warning("GARCH forecast failed for %s (%s); using sample variance.", ticker, e)
            volatilities[ticker] = log_returns[ticker].var() * 252
            failed_vol_tickers.append(ticker)

    if failed_vol_tickers:
        logger.info("GARCH forecasts failed for %d tickers.", len(failed_vol_tickers))

    volatilities = pd.DataFrame.from_dict(volatilities,
                                          orient='index')
    volatilities.to_csv('Data/variances.csv')

    elapsed = time.time() - start_time

    # Save to database
    import db
    conn = db.get_connection()
    run_id = db.save_forecast_results(conn, expected_returns[0], volatilities[0],
                                      n_periods=n_periods,
                                      elapsed_seconds=elapsed)
    print(f"\nForecasts saved to database (forecast_run id={run_id})")
    conn.close()


if __name__ == '__main__':
    main()
