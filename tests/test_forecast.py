"""Minimal tests for forecast module (smoke tests only).

The forecast pipeline runs ARIMA/GARCH on real data and is too slow for
unit tests. These tests verify the module can be imported and its
dependencies resolve correctly.
"""

import unittest


class TestForecastImports(unittest.TestCase):
    def test_module_imports(self):
        """Verify forecast module can be imported without side effects."""
        from src import forecast
        self.assertTrue(hasattr(forecast, 'main'))

    def test_config_constants_available(self):
        """Verify forecast uses centralised config constants."""
        from src import forecast
        from src.config import TRADING_DAYS_PER_YEAR, DATA_MIN_COVERAGE
        # These are used inside main() — just verify they're importable
        self.assertEqual(TRADING_DAYS_PER_YEAR, 252)
        self.assertGreater(DATA_MIN_COVERAGE, 0)


if __name__ == '__main__':
    unittest.main()
