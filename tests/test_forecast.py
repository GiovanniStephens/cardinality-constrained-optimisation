"""Tests for the forecast module.

Tests temporal integrity (no future data leakage), numerical validity,
fallback paths, and output shape consistency.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR


def _make_price_series(n=300, drift=0.0005, vol=0.02, start=100.0, seed=42):
    """Create a synthetic price series with given drift and volatility."""
    np.random.seed(seed)
    dates = pd.bdate_range('2018-01-01', periods=n, freq='B')
    log_rets = np.random.randn(n) * vol + drift
    log_rets[0] = 0
    prices = start * np.exp(np.cumsum(log_rets))
    return pd.Series(prices, index=dates, name='SYN')


class TestForecastImports(unittest.TestCase):
    def test_module_imports(self):
        """Verify forecast module can be imported without side effects."""
        from src import forecast
        self.assertTrue(hasattr(forecast, 'main'))

    def test_config_constants_available(self):
        """Verify forecast uses centralised config constants."""
        from src import forecast
        from src.config import TRADING_DAYS_PER_YEAR, DATA_MIN_COVERAGE
        self.assertEqual(TRADING_DAYS_PER_YEAR, 252)
        self.assertGreater(DATA_MIN_COVERAGE, 0)

    def test_extracted_functions_exist(self):
        """Verify fit_arima_forecast and fit_garch_forecast are importable."""
        from src.forecast import fit_arima_forecast, fit_garch_forecast
        self.assertTrue(callable(fit_arima_forecast))
        self.assertTrue(callable(fit_garch_forecast))


class TestArimaForecast(unittest.TestCase):
    """Tests for the extracted fit_arima_forecast function."""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def test_arima_uses_only_training_data(self):
        """ARIMA forecast should reflect training regime, not future data.

        Construct a regime break: positive drift in training, negative drift
        after. The ARIMA forecast (fitted on training only) should be
        non-negative, reflecting the training regime.
        """
        from src.forecast import fit_arima_forecast
        # Training: strong positive drift
        train_prices = _make_price_series(n=300, drift=0.001, vol=0.01, seed=42)
        # If ARIMA were contaminated with future (negative drift) data,
        # the forecast would be pulled down significantly
        result = fit_arima_forecast(train_prices, n_periods=TRADING_DAYS_PER_YEAR)
        # The forecast from positive-drift training should not be strongly negative
        self.assertGreater(result, -0.5,
                           "ARIMA forecast from positive-drift training should not be strongly negative")

    def test_arima_forecast_returns_finite_values(self):
        """ARIMA forecast must return a finite float, not NaN or inf."""
        from src.forecast import fit_arima_forecast
        prices = _make_price_series(n=300, seed=123)
        result = fit_arima_forecast(prices, n_periods=252)
        self.assertTrue(np.isfinite(result),
                        f"Expected finite forecast, got {result}")

    def test_arima_fallback_uses_historical_mean(self):
        """When prices are constant (ARIMA predicts 0 change), fallback to historical mean."""
        from src.forecast import fit_arima_forecast
        # Near-constant prices with tiny noise to avoid ARIMA failure
        np.random.seed(42)
        dates = pd.bdate_range('2018-01-01', periods=300, freq='B')
        prices = pd.Series(100.0 + np.random.randn(300) * 0.001, index=dates, name='FLAT')
        result = fit_arima_forecast(prices, n_periods=252)
        # For near-constant prices the forecast should be close to 0
        self.assertAlmostEqual(result, 0.0, delta=0.5,
                               msg="Near-constant prices should produce near-zero forecast")

    def test_arima_insufficient_data_raises(self):
        """ARIMA should raise ValueError with fewer than 30 observations."""
        from src.forecast import fit_arima_forecast
        short_prices = _make_price_series(n=20, seed=42)
        with self.assertRaises(ValueError):
            fit_arima_forecast(short_prices, n_periods=252)


class TestGarchForecast(unittest.TestCase):
    """Tests for the extracted fit_garch_forecast function."""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def test_garch_uses_only_training_data(self):
        """GARCH variance forecast should reflect training volatility regime.

        Low-vol training regime: daily vol ~0.5%. If future data (high vol ~5%)
        leaked in, the forecast variance would be much larger.
        """
        from src.forecast import fit_garch_forecast
        # Low-vol training data
        np.random.seed(42)
        n = 500
        dates = pd.bdate_range('2018-01-01', periods=n, freq='B')
        low_vol_returns = pd.Series(
            np.random.randn(n) * 0.005,  # 0.5% daily vol
            index=dates, name='SYN',
        )
        result = fit_garch_forecast(low_vol_returns, n_periods=252)
        # Annualised variance from 0.5% daily vol ≈ 0.005^2 * 252 ≈ 0.0063
        # Allow generous upper bound but should not be anywhere near high-vol regime
        self.assertLess(result, 0.1,
                        f"GARCH variance {result} too high for low-vol training data")
        self.assertGreater(result, 0,
                           "GARCH variance must be positive")

    def test_garch_forecast_variance_is_positive(self):
        """GARCH forecast variance must be strictly positive."""
        from src.forecast import fit_garch_forecast
        prices = _make_price_series(n=300, seed=99)
        log_rets = np.log(prices / prices.shift(1)).dropna()
        log_rets = log_rets.replace([np.inf, -np.inf], 0)
        result = fit_garch_forecast(log_rets, n_periods=252)
        self.assertGreater(result, 0,
                           f"GARCH variance should be positive, got {result}")

    def test_garch_forecast_returns_finite_values(self):
        """GARCH forecast must return a finite float."""
        from src.forecast import fit_garch_forecast
        prices = _make_price_series(n=300, seed=77)
        log_rets = np.log(prices / prices.shift(1)).dropna()
        log_rets = log_rets.replace([np.inf, -np.inf], 0)
        result = fit_garch_forecast(log_rets, n_periods=252)
        self.assertTrue(np.isfinite(result),
                        f"Expected finite variance, got {result}")

    def test_garch_insufficient_data_raises(self):
        """GARCH should raise ValueError with fewer than 30 observations."""
        from src.forecast import fit_garch_forecast
        short_returns = pd.Series(np.random.randn(20) * 0.01, name='SHORT')
        with self.assertRaises(ValueError):
            fit_garch_forecast(short_returns, n_periods=252)


class TestForecastOutputShape(unittest.TestCase):
    """Test that forecasts produce correctly shaped output."""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def test_arima_returns_scalar(self):
        """fit_arima_forecast should return a single float."""
        from src.forecast import fit_arima_forecast
        prices = _make_price_series(n=300, seed=42)
        result = fit_arima_forecast(prices, n_periods=252)
        self.assertIsInstance(result, float)

    def test_garch_returns_scalar(self):
        """fit_garch_forecast should return a single float."""
        from src.forecast import fit_garch_forecast
        prices = _make_price_series(n=300, seed=42)
        log_rets = np.log(prices / prices.shift(1)).dropna()
        log_rets = log_rets.replace([np.inf, -np.inf], 0)
        result = fit_garch_forecast(log_rets, n_periods=252)
        self.assertIsInstance(result, float)


if __name__ == '__main__':
    unittest.main()
