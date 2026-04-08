"""Orchestrates benchmark runs across all adapters."""

import concurrent.futures
import json
import logging
import os
import pickle
from typing import List, Optional

import pandas as pd

from benchmark.adapters import OptimiserAdapter
from benchmark.results import BenchmarkSuite
from src.metrics import warn_if_sharpe_suspicious

logger = logging.getLogger(__name__)


def _run_single_seed(adapter, data, time_budget, seed, run_id):
    """Run a single adapter+seed combination. Top-level for pickling."""
    return adapter.run(data, time_budget, seed, run_id)


# Max parallel seed runs per adapter type, based on internal parallelism.
# Adapters that already use all CPU cores should run 1-2 seeds at a time.
_MAX_PARALLEL_SEEDS = {
    'Island GA (Python)': 2,   # uses ~4 cores internally via mp.Pool
    'Island GA (C++)': 1,      # uses all cores via std::thread
    'Monte Carlo (C++)': 1,    # uses all cores via std::thread
    'Pygad GA': 8,             # single-threaded
    'Monte Carlo': 8,          # single-threaded
    'MILP': 8,                 # single-threaded (CBC solver)
}
_DEFAULT_MAX_PARALLEL = 4


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
            done_seeds = set()
            if adapter.name in suite.results:
                done_seeds = {r.seed for r in suite.results[adapter.name]}

            logger.info("=" * 60)
            logger.info("Algorithm: %s", adapter.name)
            logger.info("=" * 60)

            # Determine parallelism for this adapter
            max_workers = _MAX_PARALLEL_SEEDS.get(
                adapter.name, _DEFAULT_MAX_PARALLEL)

            # Collect seeds to run
            pending_seeds = []
            for i, seed in enumerate(self.seeds):
                current += 1
                if seed in done_seeds:
                    logger.info("  Run %d/%d (seed=%d) [%d/%d]... SKIPPED (already done)",
                                i + 1, self.num_runs, seed, current, total_runs)
                    continue
                pending_seeds.append((i, seed, current))

            if not pending_seeds:
                self._save(suite)
                continue

            # Run seeds in parallel
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_info = {}
                for i, seed, cur in pending_seeds:
                    logger.info("  Submitting run %d/%d (seed=%d) [%d/%d]...",
                                i + 1, self.num_runs, seed, cur, total_runs)
                    future = executor.submit(
                        _run_single_seed, adapter, self.data,
                        self.time_budget, seed, i,
                    )
                    future_to_info[future] = (i, seed, cur)

                for future in concurrent.futures.as_completed(future_to_info):
                    i, seed, cur = future_to_info[future]
                    try:
                        result = future.result()
                        suite.add_result(result)
                        logger.info("    OK | seed=%d | best=%.4f | convergence_pts=%d",
                                    seed, result.best_fitness, len(result.convergence))
                        warn_if_sharpe_suspicious(
                            result.best_fitness,
                            f"{adapter.name} seed={seed} IS",
                            logger,
                        )
                    except Exception as e:
                        logger.error("    FAILED (seed=%d): %s", seed, e, exc_info=True)

            # Save incrementally after each algorithm
            self._save(suite)
            logger.info("  Saved results for %s", adapter.name)

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
