"""Orchestrates benchmark runs across all adapters."""

import json
import os
import pickle
import time
from typing import List, Optional

import pandas as pd

from benchmark.adapters import OptimiserAdapter
from benchmark.results import BenchmarkSuite


class BenchmarkRunner:
    """Runs each adapter for each seed, prints progress, and saves results."""

    def __init__(self, adapters: List[OptimiserAdapter], data: pd.DataFrame,
                 time_budget: float = 120.0, num_runs: int = 30,
                 base_seed: int = 42, output_dir: str = 'benchmark_results'):
        self.adapters = adapters
        self.data = data
        self.time_budget = time_budget
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.output_dir = output_dir
        self.seeds = list(range(base_seed, base_seed + num_runs))

    def run(self, resume_suite: Optional[BenchmarkSuite] = None) -> BenchmarkSuite:
        os.makedirs(self.output_dir, exist_ok=True)
        suite = resume_suite or BenchmarkSuite()
        total_runs = len(self.adapters) * self.num_runs
        current = 0

        for adapter in self.adapters:
            # Check which seeds are already done for this adapter
            done_seeds = set()
            if adapter.name in suite.results:
                done_seeds = {r.seed for r in suite.results[adapter.name]}

            print(f"\n{'='*60}")
            print(f"Algorithm: {adapter.name}")
            print(f"{'='*60}")

            for i, seed in enumerate(self.seeds):
                current += 1
                if seed in done_seeds:
                    print(f"  Run {i+1}/{self.num_runs} (seed={seed}) "
                          f"[{current}/{total_runs}]... SKIPPED (already done)")
                    continue

                print(f"  Run {i+1}/{self.num_runs} (seed={seed}) "
                      f"[{current}/{total_runs}]...", end=' ', flush=True)

                t0 = time.time()
                try:
                    result = adapter.run(self.data, self.time_budget, seed, i)
                    elapsed = time.time() - t0
                    suite.add_result(result)
                    print(f"OK | best={result.best_fitness:.4f} | "
                          f"time={elapsed:.1f}s | "
                          f"convergence_pts={len(result.convergence)}")
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"FAILED ({elapsed:.1f}s): {e}")

            # Save incrementally after each algorithm
            self._save(suite)
            print(f"  Saved results for {adapter.name}")

        return suite

    def _save(self, suite: BenchmarkSuite):
        # Pickle for full fidelity
        pkl_path = os.path.join(self.output_dir, 'benchmark_suite.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(suite, f)

        # JSON summary
        json_path = os.path.join(self.output_dir, 'benchmark_final.json')
        with open(json_path, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2, default=str)


def load_suite(output_dir: str = 'benchmark_results') -> Optional[BenchmarkSuite]:
    """Load a previously saved BenchmarkSuite from pickle."""
    pkl_path = os.path.join(output_dir, 'benchmark_suite.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return None
