"""Integration tests for the backtest pipeline.

Tests the end-to-end flow: window generation -> data slicing -> portfolio
evaluation -> DB save/load -> cross-window aggregation -> statistical tests.

Uses random portfolios (not GA) to keep runtime under 30 seconds.
GA correctness is covered by test_optimisers.py.

Run with:  RUN_INTEGRATION=1 python -m unittest tests.test_backtest_integration
"""

import unittest

import numpy as np
import pandas as pd

from tests import requires_integration
from tests.helpers import make_synthetic_prices, get_memory_db

from src import db
from src.backtest import (
    generate_windows,
    slice_window_data,
    create_random_portfolios,
    get_random_weights,
    evaluate_portfolios,
    aggregate_cross_window,
    paired_t_test,
    friedman_test,
    get_statistics,
    WindowResult,
)
from src.portfolio_utils import calculate_log_returns, optimise_weights


def _run_window(prices, window, conn):
    """Evaluate a window using random + weight-optimised portfolios (no GA)."""
    train_prices, oos_log_returns = slice_window_data(window, prices)
    train_log_returns = calculate_log_returns(train_prices)
    result = WindowResult(window=window)

    random_portfolios = create_random_portfolios(
        train_prices.columns, num_portfolios=3,
        min_securities=3, max_securities=6)

    categories = [
        ('random_optimised', random_portfolios,
         lambda p: _safe_optimise(p, train_prices)),
        ('random_random', random_portfolios, get_random_weights),
    ]
    for cat_name, portfolios, weight_fn in categories:
        weights_list = [weight_fn(p) for p in portfolios]
        result.method_results[cat_name] = evaluate_portfolios(
            portfolios, weights_list, oos_log_returns,
            train_log_returns, cat_name)

    return result


def _safe_optimise(portfolio, prices):
    """Optimise weights, fall back to equal weight on failure."""
    selection = np.array([1 if c in portfolio else 0
                          for c in prices.columns])
    result = optimise_weights(selection, prices)
    if result.success:
        return result.x
    return get_random_weights(portfolio)


# 8 years of synthetic data for multiple windows.
_PRICES = None
_WINDOWS = None
_RESULTS = None
_CONN = None


def setUpModule():
    global _PRICES, _WINDOWS, _RESULTS, _CONN
    _PRICES = make_synthetic_prices(n_days=2016, n_tickers=15, seed=42)
    _CONN = get_memory_db()
    _WINDOWS = generate_windows(
        _PRICES.index, train_days=1260, test_days=252, step_days=252)[:3]
    _RESULTS = [_run_window(_PRICES, w, _CONN) for w in _WINDOWS]


def tearDownModule():
    global _CONN
    if _CONN:
        _CONN.close()


@requires_integration
class TestWindowEvaluation(unittest.TestCase):
    """Verify window evaluation produces correct outputs."""

    def test_method_categories_present(self):
        for wr in _RESULTS:
            self.assertIn('random_optimised', wr.method_results)
            self.assertIn('random_random', wr.method_results)

    def test_portfolio_count(self):
        for wr in _RESULTS:
            for cat, mr in wr.method_results.items():
                self.assertEqual(len(mr.portfolios), 3,
                                 f"Window {wr.window.label} {cat}")

    def test_metrics_finite(self):
        for wr in _RESULTS:
            for cat, mr in wr.method_results.items():
                for pr in mr.portfolios:
                    for name, val in pr.metrics.items():
                        self.assertTrue(
                            np.isfinite(val),
                            f"{wr.window.label} {cat} {name}={val}",
                        )

    def test_is_sharpe_populated(self):
        for wr in _RESULTS:
            for cat, mr in wr.method_results.items():
                for pr in mr.portfolios:
                    self.assertIsNotNone(pr.is_sharpe)


@requires_integration
class TestDBSaveLoad(unittest.TestCase):
    """Save backtest results to DB and load back."""

    @classmethod
    def setUpClass(cls):
        cls.session_ids = []
        for wr in _RESULTS:
            sid = db.save_backtest_session(_CONN, {
                'data_source': 'synthetic',
                'num_portfolios': 3,
                'num_days_oos': 252,
                'use_forecast': False,
                'optimiser_params': {'method': 'random'},
                'elapsed_seconds': 0.1,
                'window_label': wr.window.label,
            })
            for cat, mr in wr.method_results.items():
                for i, pr in enumerate(mr.portfolios):
                    db.save_backtest_result(_CONN, sid, cat, i,
                                           metrics=pr.metrics)
            cls.session_ids.append(sid)

    def test_sessions_saved(self):
        recent = db.get_recent_backtests(_CONN, n=10)
        self.assertGreaterEqual(len(recent), 3)

    def test_results_count(self):
        for sid in self.session_ids:
            rows = db.get_backtest_results(_CONN, sid)
            # 2 categories x 3 portfolios = 6
            self.assertEqual(len(rows), 6)

    def test_sharpe_round_trip(self):
        sid = self.session_ids[0]
        rows = db.get_backtest_results(_CONN, sid)
        saved = {(r['category'], r['portfolio_index']): r['sharpe_ratio']
                 for r in rows}
        wr = _RESULTS[0]
        for cat, mr in wr.method_results.items():
            for i, pr in enumerate(mr.portfolios):
                self.assertAlmostEqual(
                    saved[(cat, i)], pr.metrics['sharpe_ratio'], places=6)


@requires_integration
class TestCrossWindowStats(unittest.TestCase):
    """Cross-window aggregation and statistical tests."""

    def test_aggregation_shape(self):
        df = aggregate_cross_window(_RESULTS)
        self.assertEqual(df.shape[0], 2)  # 2 categories
        self.assertIn('mean', df.columns)
        self.assertIn('std', df.columns)

    def test_aggregation_mean_correct(self):
        df = aggregate_cross_window(_RESULTS)
        for cat in df.index:
            window_vals = [
                df.loc[cat, wr.window.label] for wr in _RESULTS
            ]
            self.assertAlmostEqual(
                df.loc[cat, 'mean'], np.mean(window_vals), places=10)

    def test_paired_t_test(self):
        a = {wr.window.label: wr.method_results['random_optimised'].mean_sharpe
             for wr in _RESULTS}
        b = {wr.window.label: wr.method_results['random_random'].mean_sharpe
             for wr in _RESULTS}
        t_stat, p_val = paired_t_test(a, b)
        self.assertTrue(np.isfinite(t_stat))
        self.assertGreaterEqual(p_val, 0.0)
        self.assertLessEqual(p_val, 1.0)


@requires_integration
class TestNoLookAhead(unittest.TestCase):
    """Verify window slicing does not leak future data."""

    def test_train_test_no_date_overlap(self):
        for w in _WINDOWS:
            train_prices, oos_log_returns = slice_window_data(w, _PRICES)
            self.assertLess(train_prices.index[-1], oos_log_returns.index[0])

    def test_first_oos_return_nonzero(self):
        """Boundary price prepending should produce nonzero first return."""
        for w in _WINDOWS:
            _, oos_log_returns = slice_window_data(w, _PRICES)
            first_row = oos_log_returns.iloc[0]
            self.assertFalse((first_row == 0.0).all())


if __name__ == '__main__':
    unittest.main()
