"""Tests for benchmark/analysis.py — pure statistical functions."""

import unittest

import numpy as np

from benchmark.results import BenchmarkResult, BenchmarkSuite, ConvergenceRecord


def _make_suite(algo_fitnesses: dict[str, list[float]],
                seeds: list[int] | None = None) -> BenchmarkSuite:
    """Build a BenchmarkSuite from {algo: [fitness_per_seed]}."""
    suite = BenchmarkSuite()
    max_runs = max(len(v) for v in algo_fitnesses.values())
    if seeds is None:
        seeds = list(range(max_runs))
    for algo, fitnesses in algo_fitnesses.items():
        for i, f in enumerate(fitnesses):
            suite.add_result(BenchmarkResult(
                algorithm=algo,
                seed=seeds[i],
                time_budget=10.0,
                convergence=[],
                best_fitness=f,
            ))
    return suite


# ---------------------------------------------------------------------------
# summary_table
# ---------------------------------------------------------------------------

class TestSummaryTable(unittest.TestCase):
    def setUp(self):
        from benchmark.analysis import summary_table
        self.summary_table = summary_table
        self.suite = _make_suite({
            'ga': [1.0, 1.5, 2.0, 2.5, 3.0],
            'mc': [0.5, 0.8, 1.0, 1.2, 1.5],
        })

    def test_returns_dataframe(self):
        import pandas as pd
        df = self.summary_table(self.suite)
        self.assertIsInstance(df, pd.DataFrame)

    def test_correct_statistics(self):
        df = self.summary_table(self.suite)
        ga_row = df.loc['ga']
        self.assertEqual(ga_row['Runs'], 5)
        self.assertAlmostEqual(ga_row['Median'], 2.0)
        self.assertAlmostEqual(ga_row['Best'], 3.0)
        self.assertAlmostEqual(ga_row['Worst'], 1.0)

    def test_filters_penalty_values(self):
        """Fitness values < -1e3 (penalised) should be excluded."""
        suite = _make_suite({'algo': [1.0, 2.0, -1e5]})
        df = self.summary_table(suite)
        self.assertEqual(df.loc['algo']['Runs'], 2)


# ---------------------------------------------------------------------------
# friedman_test
# ---------------------------------------------------------------------------

class TestFriedmanTest(unittest.TestCase):
    def setUp(self):
        from benchmark.analysis import friedman_test
        self.friedman_test = friedman_test

    def test_needs_3_algorithms(self):
        suite = _make_suite({
            'a': [1.0, 2.0, 3.0],
            'b': [1.5, 2.5, 3.5],
        })
        result = self.friedman_test(suite)
        self.assertIn('error', result)

    def test_needs_3_seeds(self):
        suite = _make_suite({
            'a': [1.0, 2.0],
            'b': [1.5, 2.5],
            'c': [0.5, 1.0],
        })
        result = self.friedman_test(suite)
        self.assertIn('error', result)

    def test_correct_keys(self):
        suite = _make_suite({
            'a': [1.0, 2.0, 3.0],
            'b': [1.5, 2.5, 3.5],
            'c': [0.5, 1.0, 1.5],
        })
        result = self.friedman_test(suite)
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('rankings', result)

    def test_rankings_sum(self):
        """Mean ranks across k algorithms should sum to k*(k+1)/2."""
        suite = _make_suite({
            'a': [3.0, 3.0, 3.0],
            'b': [2.0, 2.0, 2.0],
            'c': [1.0, 1.0, 1.0],
        })
        result = self.friedman_test(suite)
        rankings = result['rankings']
        k = len(rankings)
        total = sum(rankings.values())
        self.assertAlmostEqual(total, k * (k + 1) / 2, places=5)


# ---------------------------------------------------------------------------
# wilcoxon_pairwise
# ---------------------------------------------------------------------------

class TestWilcoxonPairwise(unittest.TestCase):
    def setUp(self):
        from benchmark.analysis import wilcoxon_pairwise
        self.wilcoxon_pairwise = wilcoxon_pairwise

    def test_needs_5_common_seeds(self):
        suite = _make_suite({
            'a': [1.0, 2.0, 3.0],
            'b': [1.5, 2.5, 3.5],
        })
        result = self.wilcoxon_pairwise(suite, 'a', 'b')
        self.assertIn('error', result)

    def test_correct_keys(self):
        suite = _make_suite({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [0.5, 1.0, 1.5, 2.0, 2.5],
        })
        result = self.wilcoxon_pairwise(suite, 'a', 'b')
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('num_pairs', result)
        self.assertEqual(result['num_pairs'], 5)


# ---------------------------------------------------------------------------
# aocc
# ---------------------------------------------------------------------------

class TestAocc(unittest.TestCase):
    def setUp(self):
        from benchmark.analysis import aocc
        self.aocc = aocc

    def test_perfect_convergence(self):
        """Immediate convergence to reference → AOCC ≈ 0."""
        records = [ConvergenceRecord(
            wall_clock_seconds=0.0,
            function_evaluations=1,
            best_fitness=5.0,
            mean_fitness=5.0,
            generation=0,
        )]
        result = self.aocc(records, time_budget=10.0, reference_fitness=5.0)
        self.assertAlmostEqual(result, 0.0)

    def test_no_convergence(self):
        """Fitness stuck at 0 while reference is 5 → large AOCC."""
        records = [ConvergenceRecord(
            wall_clock_seconds=0.0,
            function_evaluations=1,
            best_fitness=0.0,
            mean_fitness=0.0,
            generation=0,
        )]
        result = self.aocc(records, time_budget=10.0, reference_fitness=5.0)
        self.assertGreater(result, 0)

    def test_empty_records(self):
        result = self.aocc([], time_budget=10.0, reference_fitness=5.0)
        self.assertEqual(result, float('inf'))

    def test_zero_reference(self):
        records = [ConvergenceRecord(
            wall_clock_seconds=0.0,
            function_evaluations=1,
            best_fitness=1.0,
            mean_fitness=1.0,
            generation=0,
        )]
        result = self.aocc(records, time_budget=10.0, reference_fitness=0.0)
        self.assertEqual(result, float('inf'))

    def test_gradual_convergence(self):
        """Stepwise convergence should have AOCC between 0 and no-convergence."""
        records = [
            ConvergenceRecord(0.0, 1, 0.0, 0.0, 0),
            ConvergenceRecord(5.0, 100, 2.5, 2.0, 50),
            ConvergenceRecord(10.0, 200, 5.0, 4.0, 100),
        ]
        none_result = self.aocc(
            [ConvergenceRecord(0.0, 1, 0.0, 0.0, 0)],
            time_budget=10.0, reference_fitness=5.0,
        )
        gradual_result = self.aocc(records, time_budget=10.0, reference_fitness=5.0)
        self.assertGreater(gradual_result, 0)
        self.assertLess(gradual_result, none_result)


if __name__ == '__main__':
    unittest.main()
