import logging
import time

import numpy as np
import pandas as pd
import pmdarima as pmd
import tqdm
from arch import arch_model

from src.portfolio_utils import load_prices_csv, calculate_log_returns
from src.config import (
    BACKTEST_NUM_DAYS_OOS, TRADING_DAYS_PER_YEAR, DATA_MIN_COVERAGE,
    FORECAST_EXPECTED_RETURNS_PATH, FORECAST_VARIANCES_PATH,
    ARIMA_START_P, ARIMA_START_Q, ARIMA_MAX_P, ARIMA_MAX_Q,
    GARCH_SCALE, MIN_TRAINING_DAYS,
)

logger = logging.getLogger(__name__)


def main():
    from src.logging_config import setup_logging
    setup_logging()
    start_time = time.time()

    from src.portfolio_utils import load_prices
    data = load_prices(exchange='US', csv_fallback='data/ETF_Prices.csv')
    logger.info("Loaded price data: %d rows x %d tickers", *data.shape)
    training_data = data.iloc[:-BACKTEST_NUM_DAYS_OOS, :]

    if len(training_data) < MIN_TRAINING_DAYS:
        raise ValueError(
            f"Insufficient training data: {len(training_data)} rows "
            f"(need at least {MIN_TRAINING_DAYS})."
        )

    log_returns = calculate_log_returns(training_data)

    # Forecast returns
    n_periods = TRADING_DAYS_PER_YEAR
    logger.info('Forecasting returns for %d tickers...', len(data.columns))
    forecast_start = time.time()
    expected_returns = {}
    failed_return_tickers = []
    for ticker in tqdm.tqdm(data.columns):
        try:
            autoarima_model = pmd.auto_arima(training_data[ticker].dropna(),
                                             start_p=ARIMA_START_P,
                                             start_q=ARIMA_START_Q,
                                             max_p=ARIMA_MAX_P,
                                             max_q=ARIMA_MAX_Q,
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
                expected_returns[ticker] = log_returns[ticker].mean() * TRADING_DAYS_PER_YEAR
            else:
                expected_returns[ticker] = np.log(max(0.0001,
                                                       forecast.iloc[-1])/forecast.iloc[0])
        except Exception as e:
            logger.warning("ARIMA forecast failed for %s: %s", ticker, e)
            logger.debug("ARIMA traceback for %s:", ticker, exc_info=True)
            expected_returns[ticker] = log_returns[ticker].mean() * TRADING_DAYS_PER_YEAR
            failed_return_tickers.append(ticker)

    if failed_return_tickers:
        logger.info("ARIMA forecasts failed for %d tickers.", len(failed_return_tickers))

    expected_returns = pd.DataFrame.from_dict(expected_returns,
                                              orient='index')
    expected_returns.to_csv(FORECAST_EXPECTED_RETURNS_PATH)
    logger.info("Return forecasting completed in %.1fs", time.time() - forecast_start)

    # Forecast volatility
    logger.info('Forecasting volatility for %d tickers...', len(data.columns))
    vol_start = time.time()
    volatilities = {}
    failed_vol_tickers = []
    for ticker in tqdm.tqdm(data.columns):
        try:
            # Scale returns by GARCH_SCALE for numerical conditioning:
            # GARCH estimation is more stable with values around 1-10
            # rather than the ~0.001 magnitude of daily log returns.
            am = arch_model(GARCH_SCALE * log_returns[ticker],
                            vol="Garch",
                            p=1,
                            o=1,
                            q=1,
                            dist="skewt",
                            rescale=False)
            res = am.fit(disp='off')
            forecast = res.forecast(horizon=n_periods,
                                    reindex=False)
            # Reverse the scaling: variance scales with GARCH_SCALE^2.
            vol = forecast.residual_variance.iloc[-1].mean() \
                / (GARCH_SCALE ** 2) * TRADING_DAYS_PER_YEAR
            if np.isnan(vol) or vol <= 0:
                logger.warning("Ticker %s: GARCH forecast produced invalid variance (%.6f); using sample variance.", ticker, vol)
                volatilities[ticker] = log_returns[ticker].var() * TRADING_DAYS_PER_YEAR
            else:
                volatilities[ticker] = vol
        except Exception as e:
            logger.warning("GARCH forecast failed for %s: %s", ticker, e)
            logger.debug("GARCH traceback for %s:", ticker, exc_info=True)
            volatilities[ticker] = log_returns[ticker].var() * TRADING_DAYS_PER_YEAR
            failed_vol_tickers.append(ticker)

    if failed_vol_tickers:
        logger.info("GARCH forecasts failed for %d tickers.", len(failed_vol_tickers))

    volatilities = pd.DataFrame.from_dict(volatilities,
                                          orient='index')
    volatilities.to_csv(FORECAST_VARIANCES_PATH)
    logger.info("Volatility forecasting completed in %.1fs", time.time() - vol_start)

    elapsed = time.time() - start_time

    # Save to database
    from src import db
    conn = db.get_connection()
    run_id = db.save_forecast_results(conn, expected_returns[0], volatilities[0],
                                      n_periods=n_periods,
                                      elapsed_seconds=elapsed)
    logger.info("Forecasts saved to database (forecast_run id=%d)", run_id)
    conn.close()


if __name__ == '__main__':
    main()
