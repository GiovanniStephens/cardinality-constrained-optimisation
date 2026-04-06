"""Shared test utilities for unit and integration tests."""

import numpy as np
import pandas as pd

from src import db


def make_synthetic_prices(n_days=500, n_tickers=30, seed=42,
                          start='2018-01-01', daily_drift=0.0002,
                          daily_vol=0.01):
    """Deterministic synthetic GBM prices for tests.

    :param n_days: number of business days.
    :param n_tickers: number of tickers.
    :param seed: numpy random seed for reproducibility.
    :param start: start date string.
    :param daily_drift: mean daily log return.
    :param daily_vol: daily log-return standard deviation.
    :return: DataFrame with dates as index, ticker names as columns.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start, periods=n_days, freq='B')
    tickers = [f'S{i}' for i in range(n_tickers)]
    log_rets = np.random.randn(n_days, n_tickers) * daily_vol + daily_drift
    prices = 100 * np.exp(log_rets.cumsum(axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def make_arima_series(n_days=500, ar_coef=0.3, noise_std=1.0, seed=42):
    """Synthetic AR(1) series for forecast tests.

    :param n_days: number of observations.
    :param ar_coef: AR(1) coefficient (|ar_coef| < 1 for stationarity).
    :param noise_std: standard deviation of the noise term.
    :param seed: numpy random seed.
    :return: pandas Series.
    """
    np.random.seed(seed)
    noise = np.random.randn(n_days) * noise_std
    series = np.zeros(n_days)
    series[0] = noise[0]
    for t in range(1, n_days):
        series[t] = ar_coef * series[t - 1] + noise[t]
    dates = pd.bdate_range('2020-01-01', periods=n_days, freq='B')
    return pd.Series(series, index=dates)


def get_memory_db():
    """Return a :memory: SQLite connection with the full project schema."""
    return db.get_connection(':memory:')
