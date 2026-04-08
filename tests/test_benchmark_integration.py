"""Integration tests for the benchmark framework.

Tests adapter execution, runner orchestration, resume, and serialization.

Run with:  RUN_INTEGRATION=1 python -m unittest tests.test_benchmark_integration
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from tests import requires_integration
from tests.helpers import make_synthetic_prices

from benchmark.results import BenchmarkResult, BenchmarkSuite, ConvergenceRecord


@requires_integration
class TestAdapterReturnsResult(unittest.TestCase):
    """Each Python adapter should return a valid BenchmarkResult."""

    @classmethod
    def setUpClass(cls):
        cls.prices = make_synthetic_prices(n_days=200, n_tickers=10, seed=42)

    def _run_adapter(self, adapter_cls, **kwargs):
        adapter = adapter_cls(**kwargs)
        return adapter.run(self.prices, time_budget=15, seed=42, run_id=0)

    def test_monte_carlo_adapter(self):
        from benchmark.adapters import MonteCarloAdapter
        result = self._run_adapter(
            MonteCarloAdapter, log_interval=500,
            min_securities=2, max_securities=5)
        self._assert_valid(result)

    def test_mip_adapter(self):
        from benchmark.adapters import MIPAdapter
        result = self._run_adapter(MIPAdapter, max_securities=5)
        self._assert_valid(result)

    def test_pygad_adapter(self):
        from benchmark.adapters import PygadGAAdapter
        result = self._run_adapter(
            PygadGAAdapter, num_generations=3, population_size=50,
            min_securities=2, max_securities=5)
        self._assert_valid(result)

    def _assert_valid(self, result):
        self.assertIsInstance(result, BenchmarkResult)
        self.assertTrue(np.isfinite(result.best_fitness),
                        f"best_fitness={result.best_fitness} is not finite")
        self.assertGreater(len(result.convergence), 0)
        self.assertIsNotNone(result.selected_etfs)
        self.assertGreater(len(result.selected_etfs), 0)


@requires_integration
class TestBenchmarkRunnerEndToEnd(unittest.TestCase):
    """Run BenchmarkRunner with a single adapter and verify outputs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_runner_produces_suite(self):
        from benchmark.adapters import MIPAdapter
        from benchmark.runner import BenchmarkRunner

        prices = make_synthetic_prices(n_days=200, n_tickers=10, seed=42)
        adapter = MIPAdapter(max_securities=5)
        runner = BenchmarkRunner(
            adapters=[adapter], data=prices,
            time_budget=10, num_runs=2, base_seed=42,
            output_dir=self.tmpdir,
        )
        suite = runner.run()
        self.assertIsInstance(suite, BenchmarkSuite)
        self.assertIn(adapter.name, suite.results)
        self.assertEqual(len(suite.results[adapter.name]), 2)

    def test_saved_files_exist(self):
        from benchmark.adapters import MIPAdapter
        from benchmark.runner import BenchmarkRunner

        prices = make_synthetic_prices(n_days=200, n_tickers=10, seed=42)
        runner = BenchmarkRunner(
            adapters=[MIPAdapter(max_securities=5)], data=prices,
            time_budget=10, num_runs=1, base_seed=42,
            output_dir=self.tmpdir,
        )
        runner.run()
        # Should produce pickle and JSON files
        files = os.listdir(self.tmpdir)
        pkl_files = [f for f in files if f.endswith('.pkl')]
        json_files = [f for f in files if f.endswith('.json')]
        self.assertGreater(len(pkl_files), 0, "No .pkl files saved")
        self.assertGreater(len(json_files), 0, "No .json files saved")


@requires_integration
class TestBenchmarkResume(unittest.TestCase):
    """Run 1 of 2 seeds, resume, verify only missing seed runs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resume_skips_completed(self):
        from benchmark.adapters import MIPAdapter
        from benchmark.runner import BenchmarkRunner, load_suite

        prices = make_synthetic_prices(n_days=200, n_tickers=10, seed=42)
        adapter = MIPAdapter(max_securities=5)

        # Run 1 of 2
        runner1 = BenchmarkRunner(
            adapters=[adapter], data=prices,
            time_budget=10, num_runs=1, base_seed=42,
            output_dir=self.tmpdir,
        )
        suite1 = runner1.run()
        self.assertEqual(len(suite1.results[adapter.name]), 1)

        # Resume with 2 total runs
        runner2 = BenchmarkRunner(
            adapters=[adapter], data=prices,
            time_budget=10, num_runs=2, base_seed=42,
            output_dir=self.tmpdir,
        )
        suite2 = runner2.run(resume_suite=suite1)
        self.assertEqual(len(suite2.results[adapter.name]), 2)


@requires_integration
class TestSuiteSerialization(unittest.TestCase):
    """BenchmarkSuite.to_dict() round-trip."""

    def test_to_dict_contains_all_fields(self):
        suite = BenchmarkSuite()
        result = BenchmarkResult(
            algorithm="test",
            seed=42,
            time_budget=10.0,
            convergence=[
                ConvergenceRecord(
                    wall_clock_seconds=1.0, function_evaluations=100,
                    best_fitness=1.5, mean_fitness=0.5, generation=1,
                ),
            ],
            best_fitness=1.5,
            selected_etfs=["A", "B"],
        )
        suite.add_result(result)
        d = suite.to_dict()
        self.assertIn('test', d)
        self.assertEqual(len(d['test']), 1)
        self.assertEqual(d['test'][0]['seed'], 42)
        self.assertAlmostEqual(d['test'][0]['best_fitness'], 1.5)
        self.assertEqual(len(d['test'][0]['convergence']), 1)


if __name__ == '__main__':
    unittest.main()
