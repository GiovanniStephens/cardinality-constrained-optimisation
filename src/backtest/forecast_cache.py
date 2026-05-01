"""Per-window ARIMA / GARCH forecast cache for the rolling backtest.

The legacy `python -m src.forecast` pipeline fits forecasts once on the
full dataset and writes them to CSV / DB. That contaminates a forward-walk
backtest because the fits see the OOS period.

This module fits forecasts lazily per training window — only for the union
of tickers a window's GA selection actually touches — and caches them
keyed by ``(ticker, train_end)`` so re-evaluating the same window inside
a single Python process is free.

Failure handling matches the historical-fallback contract used in
:mod:`src.forecast`: ARIMA failures fall back to the annualised mean log
return on the training window; GARCH failures fall back to the annualised
sample variance. Cached fallback values are indistinguishable from
successful fits at lookup time so callers don't need to special-case.
"""

import logging
import multiprocessing as mp
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR
from src.forecast import fit_arima_forecast, fit_garch_forecast

logger = logging.getLogger(__name__)

# Module-level caches. Process-local: each Python process holds its own.
_arima_cache: dict[tuple[str, pd.Timestamp], float] = {}
_garch_cache: dict[tuple[str, pd.Timestamp], float] = {}


def _key(ticker: str, train_end) -> tuple[str, pd.Timestamp]:
    """Normalise the cache key so str/Timestamp inputs collide deterministically."""
    return str(ticker), pd.Timestamp(train_end)


def clear_caches() -> None:
    """Drop both caches. Primarily for tests."""
    _arima_cache.clear()
    _garch_cache.clear()


def get_arima_er(ticker: str, train_end) -> float:
    """Return the cached ARIMA expected return for the (ticker, train_end) pair.

    Raises ``KeyError`` if the cache hasn't been warmed for this pair.
    """
    return _arima_cache[_key(ticker, train_end)]


def get_garch_var(ticker: str, train_end) -> float:
    """Return the cached GARCH variance forecast.

    Raises ``KeyError`` if the cache hasn't been warmed for this pair.
    """
    return _garch_cache[_key(ticker, train_end)]


def arima_er_series_for_window(tickers: Iterable[str],
                               train_end) -> pd.Series:
    """Build an annualised ER Series indexed by ticker for the window."""
    tickers = list(tickers)
    return pd.Series(
        [get_arima_er(t, train_end) for t in tickers],
        index=tickers,
        dtype=float,
    )


def garch_var_series_for_window(tickers: Iterable[str],
                                train_end) -> pd.Series:
    """Build an annualised variance Series indexed by ticker for the window."""
    tickers = list(tickers)
    return pd.Series(
        [get_garch_var(t, train_end) for t in tickers],
        index=tickers,
        dtype=float,
    )


def _arima_worker(args):
    """Top-level worker for ARIMA fits.

    Catches any exception and returns the historical-mean fallback so a
    single bad ticker doesn't poison the whole window.
    """
    ticker, prices_series, log_returns_series, n_periods = args
    try:
        return ticker, fit_arima_forecast(prices_series, n_periods)
    except Exception as exc:  # noqa: BLE001 — log + degrade, don't crash
        logger.warning(
            "ARIMA forecast failed for %s: %s; using historical mean.",
            ticker, exc,
        )
        clean = log_returns_series.dropna()
        fallback = float(clean.mean() * TRADING_DAYS_PER_YEAR) if len(clean) \
            else 0.0
        return ticker, fallback


def _garch_worker(args):
    """Top-level worker for GARCH(1,1) variance forecasts."""
    ticker, log_returns_series, n_periods = args
    try:
        return ticker, fit_garch_forecast(log_returns_series, n_periods)
    except Exception as exc:  # noqa: BLE001 — log + degrade, don't crash
        logger.warning(
            "GARCH forecast failed for %s: %s; using sample variance.",
            ticker, exc,
        )
        clean = log_returns_series.dropna()
        fallback = float(clean.var() * TRADING_DAYS_PER_YEAR) if len(clean) > 1 \
            else 0.0
        return ticker, fallback


def warm_cache_for_window(tickers: Iterable[str],
                          train_prices: pd.DataFrame,
                          train_log_returns: pd.DataFrame,
                          train_end,
                          n_periods: int,
                          n_workers: int = 1) -> None:
    """Fit ARIMA + GARCH forecasts for the missing ``(ticker, train_end)`` keys.

    Only fits tickers that aren't already cached for this ``train_end``,
    making repeated calls within the same process effectively free.

    The worker payload contains only the per-ticker price/log-return
    Series — never the full training DataFrame — so worker pickling cost
    is bounded by the union size, not the universe size.

    :param tickers: tickers to forecast (union of GA-selected portfolios).
    :param train_prices: DataFrame of prices on the training window only.
        Index must be sorted ascending; OOS rows must not be present.
    :param train_log_returns: DataFrame of log returns on the training
        window only.
    :param train_end: training-window end Timestamp. Used as the second
        component of the cache key, so repeated windows don't collide.
    :param n_periods: forecast horizon (typically the OOS test length).
    :param n_workers: number of worker processes. Set to 1 for tests or
        in-process diagnostics.
    """
    tickers = sorted(set(tickers))
    train_end_ts = pd.Timestamp(train_end)
    arima_targets = [t for t in tickers
                     if (str(t), train_end_ts) not in _arima_cache]
    garch_targets = [t for t in tickers
                     if (str(t), train_end_ts) not in _garch_cache]

    arima_payload = [
        (t, train_prices[t], train_log_returns[t], n_periods)
        for t in arima_targets if t in train_prices.columns
    ]
    garch_payload = [
        (t, train_log_returns[t], n_periods)
        for t in garch_targets if t in train_log_returns.columns
    ]

    if not arima_payload and not garch_payload:
        return

    if n_workers <= 1 or len(arima_payload) + len(garch_payload) <= 1:
        # Sequential path — used in tests and as a safe fallback.
        for payload in arima_payload:
            ticker, er = _arima_worker(payload)
            _arima_cache[_key(ticker, train_end_ts)] = er
        for payload in garch_payload:
            ticker, var = _garch_worker(payload)
            _garch_cache[_key(ticker, train_end_ts)] = var
        return

    # Parallel path: one pool per kind so a stuck ARIMA fit doesn't block
    # the GARCH queue. ARIMA is the dominant cost, so it gets the larger
    # share of workers when both are running.
    if arima_payload:
        with mp.Pool(processes=min(n_workers, len(arima_payload))) as pool:
            for ticker, er in pool.imap_unordered(_arima_worker, arima_payload):
                _arima_cache[_key(ticker, train_end_ts)] = er
    if garch_payload:
        with mp.Pool(processes=min(n_workers, len(garch_payload))) as pool:
            for ticker, var in pool.imap_unordered(_garch_worker, garch_payload):
                _garch_cache[_key(ticker, train_end_ts)] = var
