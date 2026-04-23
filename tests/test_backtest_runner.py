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

    def map(self, fn, iterable):
        return [fn(x) for x in iterable]

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

    def _run_evaluate(self, **kwargs):
        """Run evaluate_window with all heavy computation mocked out."""
        from src.backtest.runner import evaluate_window

        def fake_create_portfolio(n_children):
            return list(self.tickers_sample)

        def fake_mc_search(prices, trials, **kw):
            sol = np.zeros(prices.shape[1])
            sol[:5] = 1
            return sol, 1.0

        def fake_weights(task):
            portfolio, mode = task
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
             patch('src.backtest.runner.tqdm', side_effect=lambda x, **kw: x):
            return evaluate_window(**defaults)

    def test_returns_window_result(self):
        result = self._run_evaluate()
        self.assertIsInstance(result, WindowResult)

    def test_all_categories_present(self):
        result = self._run_evaluate()
        expected = {
            'cc_optimised', 'cc_copulae', 'cc_random_weights',
            'mc_optimised', 'mc_random_weights',
            'random_optimised', 'random_random',
        }
        self.assertTrue(expected.issubset(set(result.method_results.keys())))

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
