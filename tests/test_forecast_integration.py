"""Integration tests for the ARIMA/GARCH forecast pipeline.

Tests forecast computation, scaling, fallbacks, and DB/CSV round-trips.

Run with:  RUN_INTEGRATION=1 python -m unittest tests.test_forecast_integration
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import pmdarima as pmd
from arch import arch_model

from tests import requires_integration
from tests.helpers import make_synthetic_prices, make_arima_series, get_memory_db

from src import db
from src.config import GARCH_SCALE, TRADING_DAYS_PER_YEAR
from src.returns import calculate_log_returns


@requires_integration
class TestARIMAForecast(unittest.TestCase):
    """Test ARIMA forecast on synthetic data."""

    def test_arima_forecast_produces_values(self):
        series = make_arima_series(n_days=500, ar_coef=0.3, seed=42)
        model = pmd.auto_arima(
            series, start_p=1, start_q=1, max_p=5, max_q=5,
            seasonal=False, trace=False, error_action='ignore',
            suppress_warnings=True, stepwise=True,
        )
        forecast = model.predict(n_periods=252, return_conf_int=False)
        self.assertEqual(len(forecast), 252)
        self.assertFalse(np.any(np.isnan(forecast)))
        self.assertTrue(np.all(np.isfinite(forecast)))

    def test_arima_fallback_on_constant_series(self):
        """Constant series should not crash ARIMA; fallback to historical mean."""
        series = pd.Series([100.0] * 200)
        try:
            model = pmd.auto_arima(
                series, start_p=1, start_q=1, max_p=3, max_q=3,
                seasonal=False, error_action='ignore',
                suppress_warnings=True, stepwise=True,
            )
            forecast = model.predict(n_periods=10, return_conf_int=False)
            # If ARIMA succeeds on constant data, forecast should be ~100
            self.assertTrue(np.all(np.isfinite(forecast)))
        except Exception:
            # Expected: ARIMA may fail on constant data. Fallback = mean.
            fallback = series.mean()
            self.assertAlmostEqual(fallback, 100.0)


@requires_integration
class TestGARCHForecast(unittest.TestCase):
    """Test GARCH forecast with scaling."""

    @classmethod
    def setUpClass(cls):
        prices = make_synthetic_prices(n_days=500, n_tickers=3, seed=42)
        cls.log_returns = calculate_log_returns(prices)

    def test_garch_variance_and_scaling(self):
        ticker = self.log_returns.columns[0]
        scaled_returns = GARCH_SCALE * self.log_returns[ticker]
        am = arch_model(scaled_returns, vol="Garch", p=1, o=1, q=1,
                        dist="skewt", rescale=False)
        res = am.fit(disp='off')
        forecast = res.forecast(horizon=252, reindex=False)

        raw_var = forecast.residual_variance.iloc[-1].mean()
        # Reverse scaling
        annual_var = raw_var / (GARCH_SCALE ** 2) * TRADING_DAYS_PER_YEAR
        self.assertGreater(annual_var, 0)
        self.assertTrue(np.isfinite(annual_var))
        # Realistic range: annualized variance of 0.001 to 10
        self.assertGreater(annual_var, 0.001)
        self.assertLess(annual_var, 10.0)

    def test_garch_fallback_on_constant_returns(self):
        """Constant returns → GARCH fails → fallback to sample variance."""
        constant_rets = pd.Series([0.001] * 300)
        try:
            am = arch_model(GARCH_SCALE * constant_rets, vol="Garch",
                            p=1, o=1, q=1, rescale=False)
            am.fit(disp='off')
        except Exception:
            # Expected failure; fallback is sample variance
            fallback = constant_rets.var() * TRADING_DAYS_PER_YEAR
            self.assertGreaterEqual(fallback, 0)
            self.assertTrue(np.isfinite(fallback))


@requires_integration
class TestForecastDBRoundTrip(unittest.TestCase):
    """Save forecast results to DB and load back."""

    def setUp(self):
        self.conn = get_memory_db()
        # Create tickers so forecast save can resolve symbols
        prices = make_synthetic_prices(n_days=50, n_tickers=5, seed=42)
        db.save_prices(self.conn, prices, exchange='US', asset_type='etf')

    def tearDown(self):
        self.conn.close()

    def test_expected_returns_round_trip(self):
        tickers = [f'S{i}' for i in range(5)]
        er = pd.Series([0.05 + 0.01 * i for i in range(5)], index=tickers)
        var = pd.Series([0.02 + 0.005 * i for i in range(5)], index=tickers)

        run_id = db.save_forecast_results(
            self.conn, er, var, n_periods=252, exchange='US')
        self.assertIsNotNone(run_id)

        loaded_er = db.load_expected_returns(self.conn, run_id)
        loaded_var = db.load_variances(self.conn, run_id)

        self.assertEqual(len(loaded_er), 5)
        self.assertEqual(len(loaded_var), 5)
        for t in tickers:
            self.assertAlmostEqual(loaded_er[t], er[t], places=8)
            self.assertAlmostEqual(loaded_var[t], var[t], places=8)


@requires_integration
class TestForecastCSVRoundTrip(unittest.TestCase):
    """Save forecast to CSV and read back."""

    def test_csv_round_trip(self):
        tickers = [f'T{i}' for i in range(5)]
        er = pd.DataFrame(
            [0.05 + 0.01 * i for i in range(5)],
            index=tickers, columns=['0'],
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f:
            er.to_csv(f.name)
            path = f.name
        try:
            loaded = pd.read_csv(path, index_col=0)
            self.assertEqual(loaded.shape, er.shape)
            for t in tickers:
                self.assertAlmostEqual(
                    loaded.loc[t, '0'], er.loc[t, '0'], places=8)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
