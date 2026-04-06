"""Test numerical equivalence between C++ and Python fitness calculations.

Generates a small seeded returns matrix, writes it as binary data, runs the
C++ binary, and compares the reported fitness against the Python
equal_weight_fitness() function.
"""

import json
import os
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.portfolio_utils import (
    calculate_log_returns,
    calculate_expected_returns,
    calculate_covariance_matrix,
    equal_weight_fitness,
    write_binary_data,
)

CPP_BINARY = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'optimisation')


def _make_test_data(num_days=300, num_assets=20, seed=42):
    """Generate reproducible synthetic price data and log returns."""
    rng = np.random.RandomState(seed)
    # Simulate GBM prices so log returns are well-behaved
    daily_returns = rng.normal(0.0003, 0.015, (num_days, num_assets))
    prices = 100.0 * np.exp(np.cumsum(daily_returns, axis=0))
    tickers = [f'T{i:04d}' for i in range(num_assets)]
    dates = pd.date_range('2020-01-01', periods=num_days, freq='B')
    prices_df = pd.DataFrame(prices, index=dates, columns=tickers)
    log_returns = calculate_log_returns(prices_df)
    return prices_df, log_returns, tickers


@unittest.skipUnless(os.path.isfile(CPP_BINARY),
                     f'C++ binary not found at {CPP_BINARY}')
class TestCppEquivalence(unittest.TestCase):
    """Compare C++ GA output fitness against Python equal_weight_fitness."""

    @classmethod
    def setUpClass(cls):
        cls.prices_df, cls.log_returns, cls.tickers = _make_test_data()
        cls.expected_returns = calculate_expected_returns(cls.log_returns).values
        cls.cov_matrix = calculate_covariance_matrix(cls.log_returns).values

    def _run_cpp(self, mode='ga', extra_args=None, seed=42, time_budget=5,
                 generations=2, pop_size=20, num_islands=1):
        """Write binary data, run C++ binary, return parsed JSON."""
        tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
        tmp.close()
        try:
            write_binary_data(self.log_returns, tmp.name)
            cmd = [
                CPP_BINARY, '--binary',
                '--data', tmp.name,
                '--mode', mode,
                '--seed', str(seed),
                '--time-budget', str(time_budget),
                '--pop-size', str(pop_size),
                '--generations', str(generations),
                '--min-etfs', '3',
                '--max-etfs', '15',
                '--num-islands', str(num_islands),
                '--risk-free-rate', '0.0',
            ]
            if extra_args:
                cmd.extend(extra_args)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0,
                             f'C++ binary failed:\n{result.stderr}')
            return json.loads(result.stdout)
        finally:
            os.unlink(tmp.name)

    def _python_fitness_for_tickers(self, selected_tickers):
        """Compute equal-weight fitness in Python for a set of tickers."""
        all_tickers = list(self.log_returns.columns)
        selection = np.zeros(len(all_tickers), dtype=int)
        for t in selected_tickers:
            idx = all_tickers.index(t)
            selection[idx] = 1
        return equal_weight_fitness(
            selection, self.expected_returns, self.cov_matrix,
            min_count=3, max_count=15,
        )

    def test_ga_fitness_matches_python(self):
        """C++ GA best_fitness should match Python equal_weight_fitness."""
        result = self._run_cpp(mode='ga', seed=42, generations=5, pop_size=50)
        cpp_fitness = result['best_fitness']
        selected = result['selected_tickers']

        if cpp_fitness <= -1e3:
            self.skipTest('C++ found no valid solution')

        py_fitness = self._python_fitness_for_tickers(selected)

        # Allow small floating-point divergence
        self.assertAlmostEqual(
            cpp_fitness, py_fitness, places=6,
            msg=f'C++ fitness {cpp_fitness} != Python fitness {py_fitness}',
        )

    def test_mc_fitness_matches_python(self):
        """C++ MC best_fitness should match Python equal_weight_fitness."""
        result = self._run_cpp(mode='mc', seed=42, time_budget=3)
        cpp_fitness = result['best_fitness']
        selected = result['selected_tickers']

        if cpp_fitness <= -1e3:
            self.skipTest('C++ found no valid solution')

        py_fitness = self._python_fitness_for_tickers(selected)

        self.assertAlmostEqual(
            cpp_fitness, py_fitness, places=6,
            msg=f'C++ fitness {cpp_fitness} != Python fitness {py_fitness}',
        )

    def test_top_solutions_all_match(self):
        """All top-K solutions from C++ should match Python fitness."""
        result = self._run_cpp(mode='ga', seed=123, generations=5, pop_size=50)
        for sol in result.get('top_solutions', []):
            cpp_fit = sol['fitness']
            tickers = sol['tickers']
            if cpp_fit <= -1e3 or len(tickers) < 3:
                continue
            py_fit = self._python_fitness_for_tickers(tickers)
            self.assertAlmostEqual(
                cpp_fit, py_fit, places=6,
                msg=f'Top-K mismatch for {tickers}: C++={cpp_fit} Python={py_fit}',
            )

    def test_num_instruments_correct(self):
        """C++ should report correct number of instruments."""
        result = self._run_cpp(mode='ga', seed=42, generations=1, pop_size=10)
        self.assertEqual(result['num_instruments'], len(self.tickers))


if __name__ == '__main__':
    unittest.main()
