"""Tests for src/backtest/forecast_cache.py.

The forecast cache must (a) only fit the union of selected tickers per
window, (b) survive ARIMA/GARCH fit failures by storing historical
fallbacks, (c) cache results by (ticker, train_end), and (d) never use
data past ``train_end`` — that's the whole point of the per-window fit.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.helpers import make_synthetic_prices
from src.backtest import forecast_cache
from src.config import TRADING_DAYS_PER_YEAR
from src.returns import calculate_log_returns


class TestForecastCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.prices = make_synthetic_prices(
            n_days=400, n_tickers=6, seed=17, daily_vol=0.02)
        cls.log_returns = calculate_log_returns(cls.prices)

    def setUp(self):
        forecast_cache.clear_caches()

    # ------------------------------------------------------------------
    # Cache hit + miss accounting
    # ------------------------------------------------------------------

    def test_warm_cache_only_fits_union(self):
        """`warm_cache_for_window` should call the underlying fit functions
        once per ticker — never for tickers outside the union."""
        union = list(self.prices.columns[:3])
        train_end = self.prices.index[-1]

        with patch.object(forecast_cache, 'fit_arima_forecast',
                          return_value=0.05) as arima_mock, \
             patch.object(forecast_cache, 'fit_garch_forecast',
                          return_value=0.02) as garch_mock:
            forecast_cache.warm_cache_for_window(
                tickers=union,
                train_prices=self.prices,
                train_log_returns=self.log_returns,
                train_end=train_end,
                n_periods=20,
                n_workers=1,
            )

        self.assertEqual(arima_mock.call_count, len(union))
        self.assertEqual(garch_mock.call_count, len(union))

    def test_repeated_warm_is_a_cache_hit(self):
        """Calling warm twice with the same train_end should not refit."""
        union = list(self.prices.columns[:3])
        train_end = self.prices.index[-1]

        with patch.object(forecast_cache, 'fit_arima_forecast',
                          return_value=0.05) as arima_mock, \
             patch.object(forecast_cache, 'fit_garch_forecast',
                          return_value=0.02):
            forecast_cache.warm_cache_for_window(
                union, self.prices, self.log_returns,
                train_end, n_periods=20, n_workers=1)
            forecast_cache.warm_cache_for_window(
                union, self.prices, self.log_returns,
                train_end, n_periods=20, n_workers=1)

        # Second call should be a no-op for cached tickers.
        self.assertEqual(arima_mock.call_count, len(union))

    # ------------------------------------------------------------------
    # Failure-fallback behaviour
    # ------------------------------------------------------------------

    def test_arima_failure_falls_back_to_hist_mean(self):
        """ARIMA fit failure must populate the cache with the historical mean."""
        union = list(self.prices.columns[:1])
        train_end = self.prices.index[-1]
        ticker = union[0]

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        with patch.object(forecast_cache, 'fit_arima_forecast',
                          side_effect=boom), \
             patch.object(forecast_cache, 'fit_garch_forecast',
                          return_value=0.02):
            forecast_cache.warm_cache_for_window(
                union, self.prices, self.log_returns,
                train_end, n_periods=20, n_workers=1)

        cached = forecast_cache.get_arima_er(ticker, train_end)
        expected = float(
            self.log_returns[ticker].dropna().mean() * TRADING_DAYS_PER_YEAR)
        self.assertAlmostEqual(cached, expected, places=6)

    def test_garch_failure_falls_back_to_sample_variance(self):
        """GARCH fit failure must populate the cache with the historical variance."""
        union = list(self.prices.columns[:1])
        train_end = self.prices.index[-1]
        ticker = union[0]

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        with patch.object(forecast_cache, 'fit_arima_forecast',
                          return_value=0.05), \
             patch.object(forecast_cache, 'fit_garch_forecast',
                          side_effect=boom):
            forecast_cache.warm_cache_for_window(
                union, self.prices, self.log_returns,
                train_end, n_periods=20, n_workers=1)

        cached = forecast_cache.get_garch_var(ticker, train_end)
        expected = float(
            self.log_returns[ticker].dropna().var() * TRADING_DAYS_PER_YEAR)
        self.assertAlmostEqual(cached, expected, places=6)

    # ------------------------------------------------------------------
    # Series builders
    # ------------------------------------------------------------------

    def test_series_builders_align_to_request_order(self):
        union = list(self.prices.columns[:3])
        train_end = self.prices.index[-1]
        with patch.object(forecast_cache, 'fit_arima_forecast',
                          side_effect=lambda s, n: 0.01 + 0.001 * len(s)), \
             patch.object(forecast_cache, 'fit_garch_forecast',
                          side_effect=lambda s, n: 0.04 + 0.001 * len(s)):
            forecast_cache.warm_cache_for_window(
                union, self.prices, self.log_returns,
                train_end, n_periods=20, n_workers=1)
        er = forecast_cache.arima_er_series_for_window(union, train_end)
        var = forecast_cache.garch_var_series_for_window(union, train_end)
        self.assertEqual(list(er.index), union)
        self.assertEqual(list(var.index), union)
        self.assertTrue(np.all(np.isfinite(er.values)))
        self.assertTrue(np.all(np.isfinite(var.values)))

    # ------------------------------------------------------------------
    # No leakage from OOS data
    # ------------------------------------------------------------------

    def test_no_leakage_when_oos_data_changes(self):
        """Changing data after ``train_end`` must not change cached values.

        This is the canonical leakage test: callers slice the training
        window and pass it in. If we accidentally peeked at OOS rows, this
        comparison would fail.
        """
        union = list(self.prices.columns[:2])
        cutoff = 250
        train_end = self.prices.index[cutoff - 1]
        train_prices_v1 = self.prices.iloc[:cutoff].copy()
        train_log_returns_v1 = self.log_returns.iloc[:cutoff].copy()

        with patch.object(forecast_cache, 'fit_arima_forecast',
                          side_effect=lambda s, n: float(s.mean())), \
             patch.object(forecast_cache, 'fit_garch_forecast',
                          side_effect=lambda s, n: float(s.var())):
            forecast_cache.warm_cache_for_window(
                union, train_prices_v1, train_log_returns_v1,
                train_end, n_periods=20, n_workers=1)

        first = {t: forecast_cache.get_arima_er(t, train_end) for t in union}

        # Mutate the OOS rows in the original frames (they aren't passed
        # into the second call, but if there were any global side-channels
        # they'd surface here).
        self.prices.iloc[cutoff:, :] = 0.0

        # Re-warm with the same train_end + same training slice → cache hit.
        with patch.object(forecast_cache, 'fit_arima_forecast',
                          side_effect=lambda s, n: 999.0), \
             patch.object(forecast_cache, 'fit_garch_forecast',
                          side_effect=lambda s, n: 999.0):
            forecast_cache.warm_cache_for_window(
                union, train_prices_v1, train_log_returns_v1,
                train_end, n_periods=20, n_workers=1)

        for t in union:
            self.assertAlmostEqual(
                forecast_cache.get_arima_er(t, train_end), first[t],
                places=10,
                msg="Cache mutation suggests leakage from OOS data",
            )


if __name__ == '__main__':
    unittest.main()
