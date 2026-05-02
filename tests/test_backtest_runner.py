"""Tests for src/backtest/runner.py — evaluate_window orchestrator."""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from tests.helpers import make_synthetic_prices
from src.backtest.types import WindowSpec, WindowResult, MethodResults, PortfolioResult


def _make_window(prices):
    """Build a WindowSpec from synthetic prices."""
    idx = prices.index
    mid = len(idx) // 2
    return WindowSpec(
        train_start=idx[0],
        train_end=idx[mid - 1],
        test_start=idx[mid],
        test_end=idx[-1],
        label='test-window',
    )


class _MockPool:
    """Drop-in replacement for multiprocessing.Pool that runs sequentially."""

    def __init__(self, *args, **kwargs):
        init_fn = kwargs.get('initializer')
        init_args = kwargs.get('initargs', ())
        if init_fn:
            init_fn(*init_args)

    def map(self, fn, iterable, chunksize=None):
        return [fn(x) for x in iterable]

    def imap_unordered(self, fn, iterable, chunksize=None):
        return iter([fn(x) for x in iterable])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestEvaluateWindow(unittest.TestCase):
    """Tests for evaluate_window with mocked parallel + optimisation layers."""

    @classmethod
    def setUpClass(cls):
        cls.prices = make_synthetic_prices(n_days=400, n_tickers=10, seed=99)
        cls.window = _make_window(cls.prices)
        cls.tickers_sample = list(cls.prices.columns[:5])

    def _run_evaluate(self, tickers_sample=None,
                      run_forecast_strategies=False, **kwargs):
        """Run evaluate_window with all heavy computation mocked out.

        Forecast strategies are disabled by default so existing assertions
        don't pay the ARIMA/GARCH fit cost on synthetic data. Tests that
        explicitly want them enabled pass ``run_forecast_strategies=True``
        and patch warm_cache_for_window themselves.
        """
        from src.backtest.runner import evaluate_window

        if tickers_sample is None:
            tickers_sample = self.tickers_sample
        # Snapshot for the closure to avoid late-binding issues.
        sample = list(tickers_sample)

        def fake_create_portfolio(n_children):
            return list(sample)

        def fake_mc_search(prices, trials, **kw):
            sol = np.zeros(prices.shape[1])
            sol[:5] = 1
            return sol, 1.0

        def fake_weights(task):
            # _compute_weights_for_portfolio accepts both the legacy
            # 2-tuple (portfolio, mode) and the new 3-tuple form
            # (portfolio, mode, kwargs). Handle both for forward
            # compatibility with the dispatcher refactor.
            portfolio = task[0]
            return np.ones(len(portfolio)) / len(portfolio)

        defaults = dict(
            window=self.window,
            full_prices=self.prices,
            conn=MagicMock(),
            num_portfolios=2,
            num_children=10,
            mc_trials=100,
        )
        defaults.update(kwargs)

        def fake_random_portfolios(columns, n, **kw):
            return [list(columns[:5]) for _ in range(n)]

        with patch('src.backtest.runner.mp.Pool', _MockPool), \
             patch('src.backtest.simulation.create_portfolio',
                   side_effect=fake_create_portfolio), \
             patch('src.backtest.runner.create_portfolio',
                   side_effect=fake_create_portfolio), \
             patch('src.backtest.runner.create_random_portfolios',
                   side_effect=fake_random_portfolios), \
             patch('src.optimisers.monte_carlo.monte_carlo_search',
                   side_effect=fake_mc_search), \
             patch('src.backtest.simulation._compute_weights_for_portfolio',
                   side_effect=fake_weights), \
             patch('src.backtest.runner._compute_weights_for_portfolio',
                   side_effect=fake_weights), \
             patch('src.backtest.runner.BACKTEST_RUN_FORECAST_STRATEGIES',
                   run_forecast_strategies), \
             patch('src.backtest.runner.tqdm', side_effect=lambda x, **kw: x):
            return evaluate_window(**defaults)

    def test_returns_window_result(self):
        result = self._run_evaluate()
        self.assertIsInstance(result, WindowResult)

    def test_all_categories_present(self):
        result = self._run_evaluate()
        # Phase 1 categories that should be present regardless of the
        # forecast-strategy config flag.
        expected = {
            'cc_optimised', 'cc_copulae', 'cc_random_weights',
            'cc_ccc_baseline', 'cc_equal_weight', 'cc_min_variance',
            'mc_optimised', 'mc_random_weights',
            'random_optimised', 'random_random',
        }
        # Benchmarks (bench_spy, bench_6040) are conditional on SPY/AGG
        # being in the universe; the synthetic fixture has neither, so
        # they're skipped (verified separately in test_benchmarks_skipped).
        self.assertTrue(expected.issubset(set(result.method_results.keys())))

    def test_forecast_categories_when_enabled(self):
        """When BACKTEST_RUN_FORECAST_STRATEGIES=True, the three fast
        forecast strategies populate. The two copula+forecast variants
        are gated behind a separate flag (default off) and not expected
        here. Patch warm_cache_for_window to avoid expensive ARIMA/GARCH
        fits in unit tests."""
        from src.backtest import forecast_cache

        def fake_warm(tickers, train_prices, train_log_returns,
                      train_end, n_periods, n_workers):
            for t in tickers:
                key = (str(t), pd.Timestamp(train_end))
                forecast_cache._arima_cache[key] = 0.05
                forecast_cache._garch_cache[key] = 0.04

        with patch.object(forecast_cache, 'warm_cache_for_window',
                          side_effect=fake_warm):
            forecast_cache.clear_caches()
            result = self._run_evaluate(run_forecast_strategies=True)
        fast_forecast_cats = {
            'cc_arima_er', 'cc_garch_var', 'cc_arima_garch',
        }
        self.assertTrue(
            fast_forecast_cats.issubset(set(result.method_results.keys())))
        # Copula+forecast variants are gated and must NOT appear by default.
        copula_forecast_cats = {'cc_garch_copula', 'cc_arima_garch_copula'}
        self.assertEqual(
            copula_forecast_cats & set(result.method_results.keys()),
            set(),
            "Copula+forecast variants should be gated off by default")

    def test_forecast_categories_skipped_when_disabled(self):
        """With the flag off, none of the 5 forecast strategies appear."""
        result = self._run_evaluate(run_forecast_strategies=False)
        forecast_cats = {
            'cc_arima_er', 'cc_garch_var', 'cc_garch_copula',
            'cc_arima_garch', 'cc_arima_garch_copula',
        }
        self.assertEqual(
            forecast_cats & set(result.method_results.keys()),
            set(),
        )

    def test_new_weight_strategies_present(self):
        """Tier 1/2 weighting strategies (inverse_vol, risk_parity,
        max_diversification) should populate alongside the existing
        cc_* variants."""
        result = self._run_evaluate()
        for cat in ('cc_inverse_vol', 'cc_risk_parity',
                    'cc_max_diversification'):
            self.assertIn(cat, result.method_results,
                f"{cat} not populated; check evaluate_window wiring")

    def test_benchmarks_skipped_when_tickers_missing(self):
        """Synthetic fixture has no SPY/AGG → benchmark categories absent."""
        result = self._run_evaluate()
        self.assertNotIn('bench_spy', result.method_results)
        self.assertNotIn('bench_6040', result.method_results)

    def test_benchmarks_present_when_tickers_available(self):
        """When SPY and AGG are in the price fixture, both benchmarks run."""
        prices_with_bench = self.prices.copy()
        prices_with_bench = prices_with_bench.rename(columns={
            prices_with_bench.columns[0]: 'SPY',
            prices_with_bench.columns[1]: 'AGG',
        })
        # Replace fixture window so the renamed prices are picked up.
        window = _make_window(prices_with_bench)
        # Rebuild tickers_sample against the renamed columns so the fake
        # GA selection actually returns tickers that exist in the prices.
        sample = list(prices_with_bench.columns[:5])
        result = self._run_evaluate(
            tickers_sample=sample,
            full_prices=prices_with_bench, window=window)
        self.assertIn('bench_spy', result.method_results)
        self.assertIn('bench_6040', result.method_results)
        spy_mr = result.method_results['bench_spy']
        self.assertEqual(len(spy_mr.portfolios), 1)
        self.assertEqual(spy_mr.portfolios[0].portfolio, ['SPY'])
        self.assertTrue(np.allclose(spy_mr.portfolios[0].weights, [1.0]))
        s60 = result.method_results['bench_6040']
        self.assertEqual(s60.portfolios[0].portfolio, ['SPY', 'AGG'])
        self.assertTrue(np.allclose(s60.portfolios[0].weights, [0.6, 0.4]))

    def test_elapsed_time_recorded(self):
        result = self._run_evaluate()
        self.assertGreater(result.elapsed_seconds, 0)

    def test_portfolios_have_metrics(self):
        result = self._run_evaluate()
        for cat, mr in result.method_results.items():
            for pr in mr.portfolios:
                self.assertIn('sharpe_ratio', pr.metrics,
                              f"Portfolio in {cat} missing sharpe_ratio")


if __name__ == '__main__':
    unittest.main()
