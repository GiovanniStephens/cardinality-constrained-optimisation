import logging
import time

import numpy as np
import pandas as pd
import pmdarima as pmd
import tqdm
from arch import arch_model

from src.portfolio_utils import calculate_log_returns
from src.config import (
    BACKTEST_NUM_DAYS_OOS, TRADING_DAYS_PER_YEAR,
    FORECAST_MIN_OBSERVATIONS,
    FORECAST_ARIMA_START_P, FORECAST_ARIMA_START_Q,
    FORECAST_ARIMA_MAX_P, FORECAST_ARIMA_MAX_Q,
    FORECAST_GARCH_P, FORECAST_GARCH_O, FORECAST_GARCH_Q,
    FORECAST_GARCH_DIST, FORECAST_GARCH_VOL,
    ETF_PRICES_CSV, EXPECTED_RETURNS_CSV, VARIANCES_CSV,
)

logger = logging.getLogger(__name__)


def fit_arima_forecast(training_prices_series, n_periods):
    """Fit an ARIMA model to a single ticker's price series and return expected return.

    :param training_prices_series: pandas Series of prices (not returns) for one ticker.
    :param n_periods: number of periods to forecast.
    :return: annualised expected log return (float).
    :raises ValueError: if the series has fewer than 30 non-null observations.
    """
    clean = training_prices_series.dropna()
    if len(clean) < FORECAST_MIN_OBSERVATIONS:
        raise ValueError(f"Insufficient data: {len(clean)} observations (need >= {FORECAST_MIN_OBSERVATIONS})")

    log_returns = np.log(clean / clean.shift(1)).dropna().replace([np.inf, -np.inf], 0)

    autoarima_model = pmd.auto_arima(
        clean,
        start_p=FORECAST_ARIMA_START_P, start_q=FORECAST_ARIMA_START_Q,
        max_p=FORECAST_ARIMA_MAX_P, max_q=FORECAST_ARIMA_MAX_Q,
        seasonal=False, trace=False,
        error_action='ignore', suppress_warnings=True,
        n_jobs=-1, stepwise=False,
    )
    forecast = autoarima_model.predict(n_periods=n_periods, return_conf_int=False)
    if forecast.iloc[0] <= 0:
        logger.warning(
            "ARIMA forecast starts at %.4f; using historical mean.",
            forecast.iloc[0],
        )
        return float(log_returns.mean() * TRADING_DAYS_PER_YEAR)
    return float(np.log(max(0.0001, forecast.iloc[-1]) / forecast.iloc[0]))


def fit_garch_forecast(log_returns_series, n_periods):
    """Fit a GARCH(1,1) model to a single ticker's log returns and return forecast variance.

    :param log_returns_series: pandas Series of log returns for one ticker.
    :param n_periods: number of periods to forecast.
    :return: annualised forecast variance (float).
    :raises ValueError: if the series has fewer than 30 non-null observations.
    """
    clean = log_returns_series.dropna()
    if len(clean) < FORECAST_MIN_OBSERVATIONS:
        raise ValueError(f"Insufficient data: {len(clean)} observations (need >= {FORECAST_MIN_OBSERVATIONS})")

    am = arch_model(
        100 * clean,
        vol=FORECAST_GARCH_VOL, p=FORECAST_GARCH_P, o=FORECAST_GARCH_O,
        q=FORECAST_GARCH_Q, dist=FORECAST_GARCH_DIST, rescale=False,
    )
    res = am.fit(disp='off')
    forecast = res.forecast(horizon=n_periods, reindex=False)
    vol = forecast.residual_variance.iloc[-1].mean() / np.power(100, 2) * TRADING_DAYS_PER_YEAR

    if np.isnan(vol) or vol <= 0:
        logger.warning(
            "GARCH forecast produced invalid variance (%.6f); using sample variance.",
            vol,
        )
        return float(clean.var() * TRADING_DAYS_PER_YEAR)
    return float(vol)


def main():
    from src.logging_config import setup_logging
    setup_logging()
    start_time = time.time()

    from src.portfolio_utils import load_training_data
    data = load_training_data(
        exchange='US', csv_fallback=ETF_PRICES_CSV, lookback_days=None)
    training_data = data.iloc[:-BACKTEST_NUM_DAYS_OOS, :]

    if len(training_data) < FORECAST_MIN_OBSERVATIONS:
        raise ValueError(
            f"Insufficient training data: {len(training_data)} rows (need at least {FORECAST_MIN_OBSERVATIONS})."
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
            expected_returns[ticker] = fit_arima_forecast(
                training_data[ticker], n_periods)
        except Exception as e:
            logger.warning("ARIMA forecast failed for %s: %s", ticker, e)
            logger.debug("ARIMA traceback for %s:", ticker, exc_info=True)
            expected_returns[ticker] = log_returns[ticker].mean() * TRADING_DAYS_PER_YEAR
            failed_return_tickers.append(ticker)

    if failed_return_tickers:
        logger.info("ARIMA forecasts failed for %d tickers.", len(failed_return_tickers))

    expected_returns = pd.DataFrame.from_dict(expected_returns,
                                              orient='index')
    expected_returns.to_csv(EXPECTED_RETURNS_CSV)
    logger.info("Return forecasting completed in %.1fs", time.time() - forecast_start)

    # Forecast volatility
    logger.info('Forecasting volatility for %d tickers...', len(data.columns))
    vol_start = time.time()
    volatilities = {}
    failed_vol_tickers = []
    for ticker in tqdm.tqdm(data.columns):
        try:
            volatilities[ticker] = fit_garch_forecast(
                log_returns[ticker], n_periods)
        except Exception as e:
            logger.warning("GARCH forecast failed for %s: %s", ticker, e)
            logger.debug("GARCH traceback for %s:", ticker, exc_info=True)
            volatilities[ticker] = log_returns[ticker].var() * TRADING_DAYS_PER_YEAR
            failed_vol_tickers.append(ticker)

    if failed_vol_tickers:
        logger.info("GARCH forecasts failed for %d tickers.", len(failed_vol_tickers))

    volatilities = pd.DataFrame.from_dict(volatilities,
                                          orient='index')
    volatilities.to_csv(VARIANCES_CSV)
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
