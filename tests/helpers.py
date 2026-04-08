"""Shared test utilities for unit and integration tests."""

import os
import shutil
import tempfile
import unittest
from itertools import combinations

import numpy as np
import pandas as pd

from src import db
from src.portfolio_utils import OptimisationResult  # defined there


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


def make_small_divergent_prices(n_days=300, seed=42):
    """5 tickers with clearly separated return/risk profiles.

    S0: high return, low vol (best risk-adjusted)
    S1: medium return, low vol (second best)
    S2: low return, high vol (poor risk-adjusted)
    S3: negative return (worst)
    S4: medium return, medium vol
    """
    np.random.seed(seed)
    dates = pd.bdate_range('2018-01-01', periods=n_days, freq='B')
    profiles = [
        (0.0008, 0.005),   # S0: high ret, low vol
        (0.0004, 0.006),   # S1: medium ret, low vol
        (0.0001, 0.020),   # S2: low ret, high vol
        (-0.0003, 0.015),  # S3: negative ret
        (0.0003, 0.010),   # S4: medium ret, medium vol
    ]
    tickers = [f'S{i}' for i in range(5)]
    all_prices = []
    for drift, vol in profiles:
        log_rets = np.random.randn(n_days) * vol + drift
        prices = 100 * np.exp(log_rets.cumsum())
        all_prices.append(prices)
    return pd.DataFrame(
        np.column_stack(all_prices), index=dates, columns=tickers,
    )


def brute_force_optimal(prices, k):
    """Evaluate all C(n,k) combinations, return the best OptimisationResult.

    Uses SLSQP via optimise_weights for each combo. Feasible for small n.
    """
    from src.returns import calculate_log_returns, calculate_expected_returns
    from src.covariance import calculate_covariance_matrix
    from src.weights import optimise_weights
    tickers = list(prices.columns)
    log_returns = calculate_log_returns(prices)
    best_sharpe = -np.inf
    best_result = None

    for combo in combinations(range(len(tickers)), k):
        sel = np.zeros(len(tickers), dtype=int)
        sel[list(combo)] = 1
        selected = [tickers[i] for i in combo]
        sub_lr = log_returns[selected]
        er = calculate_expected_returns(sub_lr).values
        cov = calculate_covariance_matrix(sub_lr).values
        opt = optimise_weights(expected_returns=er, cov_matrix=cov)
        w = opt.x
        port_ret = np.dot(er, w)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        sharpe = port_ret / port_vol if port_vol > 0 else -1e4
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_result = OptimisationResult(
                selected_tickers=selected,
                weights=w,
                sharpe_ratio=sharpe,
                metadata={'method': 'brute_force'},
            )
    return best_result


def assert_result_integrity(tc, result, prices, min_weight=0.0, max_weight=1.0):
    """Bundle of assertions for OptimisationResult correctness.

    :param tc: unittest.TestCase instance (for assertion methods).
    :param result: OptimisationResult to validate.
    :param prices: DataFrame used for optimisation.
    :param min_weight: expected lower weight bound.
    :param max_weight: expected upper weight bound.
    """
    tc.assertIsInstance(result, OptimisationResult)
    # Tickers exist in data
    for t in result.selected_tickers:
        tc.assertIn(t, prices.columns)
    # Weights match tickers
    tc.assertEqual(len(result.weights), len(result.selected_tickers))
    # No NaN/inf in weights
    tc.assertTrue(np.all(np.isfinite(result.weights)),
                  "weights contain NaN or inf")
    # Weights sum to ~1
    tc.assertAlmostEqual(float(np.sum(result.weights)), 1.0, places=3)
    # Each weight within bounds (with tolerance)
    tol = 1e-6
    for i, w in enumerate(result.weights):
        tc.assertGreaterEqual(w, min_weight - tol,
                              f"weight[{i}]={w} below min_weight={min_weight}")
        tc.assertLessEqual(w, max_weight + tol,
                           f"weight[{i}]={w} above max_weight={max_weight}")
    # Non-negative weights
    tc.assertTrue(np.all(result.weights >= -tol), "negative weights found")
    # Sharpe is finite
    tc.assertTrue(np.isfinite(result.sharpe_ratio), "sharpe_ratio not finite")


def get_memory_db():
    """Return a :memory: SQLite connection with the full project schema."""
    return db.get_connection(':memory:')


# ─── Test Base Classes ────────────────────────────────────────────────────────


class BaseDBTest(unittest.TestCase):
    """Mixin providing an in-memory SQLite DB with the full schema.

    Subclasses get ``self.conn`` in setUp and automatic cleanup in tearDown.
    """

    def setUp(self):
        self.conn = db.get_connection(':memory:')

    def tearDown(self):
        self.conn.close()


class BaseTmpDirTest(unittest.TestCase):
    """Mixin providing a temporary directory in setUp with cleanup in tearDown.

    Subclasses get ``self.tmpdir`` as a fresh temp directory.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class OptimiserTestMixin:
    """Common assertions for all optimiser test classes.

    Subclasses must define ``_run(**kwargs)`` returning an ``OptimisationResult``
    and ``prices`` as a class attribute.
    """

    def test_returns_optimisation_result(self):
        result = self._run()
        self.assertIsInstance(result, OptimisationResult)

    def test_selected_tickers_subset(self):
        result = self._run()
        for t in result.selected_tickers:
            self.assertIn(t, self.prices.columns)

    def test_weights_length_matches_tickers(self):
        result = self._run()
        self.assertEqual(len(result.weights), len(result.selected_tickers))

    def test_sharpe_is_finite(self):
        result = self._run()
        self.assertTrue(np.isfinite(result.sharpe_ratio))
