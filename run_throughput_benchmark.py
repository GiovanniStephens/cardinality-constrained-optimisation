#!/usr/bin/env python3
"""Throughput benchmark: portfolio evaluations per second across all methods.

Measures raw evaluation throughput using synthetic data matching real
dimensions (M=1800 instruments, T=1260 days, n=15 selected securities).

Usage:
    python run_throughput_benchmark.py           # Full benchmark
    python run_throughput_benchmark.py --quick   # Shorter time budgets
    python run_throughput_benchmark.py --skip-cpp # Python only
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.helpers import make_synthetic_prices
from src.portfolio_utils import (
    calculate_log_returns,
    calculate_expected_returns,
    calculate_covariance_matrix,
    equal_weight_fitness,
    sharpe_ratio,
    optimise_weights,
    write_binary_data,
)
from src.optimisers.island_ga import batch_fitness, calculate_fitness


# ─── Data structures ────────────────────────────────────────────────────────

@dataclass
class ThroughputResult:
    method: str
    evals_per_sec: float
    total_evals: int
    elapsed_sec: float


# ─── Defaults ────────────────────────────────────────────────────────────────

M = 1800   # universe size
T = 1261   # price rows (1260 returns)
N = 15     # cardinality
SEED = 42
MIN_TIME = 2.0        # minimum timed-run duration (seconds)
WARMUP = 50           # warmup iterations
NUM_RUNS = 3          # repeat runs, report median
BATCH_SIZE = 2000     # population size for batch_fitness
PYGAD_CAP = 50        # max evals for slow SLSQP path
CPP_TIME_BUDGET = 10  # seconds for C++ MC


# ─── Data generation ─────────────────────────────────────────────────────────

def generate_test_data(m=M, t=T, n=N, seed=SEED):
    """Generate synthetic prices and derived arrays.

    Returns the full-universe arrays (no full M×M cov — that's rank-deficient)
    plus a pre-computed sub-covariance for a single n-asset portfolio for
    benchmarks that need one.
    """
    prices = make_synthetic_prices(n_days=t, n_tickers=m, seed=seed)
    log_returns = calculate_log_returns(prices)
    expected_returns = calculate_expected_returns(log_returns).values
    centered = (log_returns - log_returns.mean(axis=0)).values
    T_obs = centered.shape[0]

    # Sub-covariance for the first n assets (well-conditioned: T >> n)
    sub_log_returns = log_returns.iloc[:, :n]
    sub_cov = calculate_covariance_matrix(sub_log_returns).values
    sub_er = expected_returns[:n]

    return prices, log_returns, expected_returns, centered, T_obs, sub_cov, sub_er


def generate_test_portfolios(m, n, count, seed=SEED):
    """Generate random binary selection vectors with exactly n bits set."""
    rng = np.random.default_rng(seed)
    portfolios = np.zeros((count, m), dtype=np.float64)
    for i in range(count):
        indices = rng.choice(m, n, replace=False)
        portfolios[i, indices] = 1.0
    return portfolios


# ─── Timing helper ───────────────────────────────────────────────────────────

def timed_run(fn, min_time=MIN_TIME, min_iters=10):
    """Run fn() in a loop for at least min_time seconds, return (total_evals, elapsed)."""
    total_evals = 0
    start = time.perf_counter()
    while True:
        evals = fn()
        total_evals += evals
        elapsed = time.perf_counter() - start
        if elapsed >= min_time and total_evals >= min_iters:
            break
    return total_evals, elapsed


def benchmark_method(name, fn, warmup_fn=None, warmup_iters=WARMUP,
                     min_time=MIN_TIME, num_runs=NUM_RUNS):
    """Warmup + timed runs, return ThroughputResult with median evals/sec."""
    # Warmup
    if warmup_fn is not None:
        for _ in range(warmup_iters):
            warmup_fn()
    else:
        for _ in range(warmup_iters):
            fn()

    # Timed runs
    rates = []
    total_evals_best = 0
    elapsed_best = 0
    for _ in range(num_runs):
        total_evals, elapsed = timed_run(fn, min_time=min_time)
        rate = total_evals / elapsed if elapsed > 0 else 0
        rates.append(rate)
        if rate == sorted(rates)[len(rates) // 2]:  # track median's raw values
            total_evals_best = total_evals
            elapsed_best = elapsed

    median_rate = sorted(rates)[len(rates) // 2]

    # Use the run closest to median for total/elapsed reporting
    if elapsed_best == 0:
        total_evals_best = sum(r * min_time for r in rates) // num_runs
        elapsed_best = min_time

    return ThroughputResult(
        method=name,
        evals_per_sec=median_rate,
        total_evals=int(total_evals_best),
        elapsed_sec=round(elapsed_best, 3),
    )


# ─── Individual benchmarks ──────────────────────────────────────────────────

def bench_batch_fitness(expected_returns, centered, T_obs, portfolios,
                        min_time=MIN_TIME):
    """batch_fitness() with a population of BATCH_SIZE."""
    pop = portfolios[:BATCH_SIZE]

    def fn():
        batch_fitness(pop, expected_returns, centered, T_obs,
                      min_etfs=N, max_etfs=N, min_return=None)
        return BATCH_SIZE

    return benchmark_method(
        f"batch_fitness (pop={BATCH_SIZE})", fn,
        warmup_fn=lambda: fn(), warmup_iters=10,
        min_time=min_time,
    )


def bench_island_single_fitness(expected_returns, centered, T_obs, portfolios,
                                min_time=MIN_TIME):
    """calculate_fitness() — single portfolio wrapper around batch_fitness."""
    individual = portfolios[0]

    def fn():
        calculate_fitness(individual, expected_returns, centered, T_obs,
                          min_etfs=N, max_etfs=N, min_return=None)
        return 1

    return benchmark_method(
        "island_ga single fitness", fn,
        min_time=min_time,
    )


def bench_equal_weight_fitness(sub_er, sub_cov, min_time=MIN_TIME):
    """equal_weight_fitness() — sub-covariance extraction path.

    Uses a pre-computed n×n sub-covariance (as would happen in practice when
    evaluating a single candidate portfolio from the full universe).
    """
    # Build a trivial "all selected" mask of length n
    individual = np.ones(len(sub_er), dtype=int)

    def fn():
        equal_weight_fitness(individual, sub_er, sub_cov,
                             min_count=N, max_count=N)
        return 1

    return benchmark_method(
        "equal_weight_fitness", fn,
        min_time=min_time,
    )


def bench_sharpe_ratio(sub_er, sub_cov, min_time=MIN_TIME):
    """sharpe_ratio() with pre-computed sub-covariance and equal weights."""
    weights = np.ones(N) / N

    def fn():
        sharpe_ratio(weights, sub_er, sub_cov)
        return 1

    return benchmark_method(
        "sharpe_ratio (dot-product only)", fn,
        min_time=min_time,
    )


def bench_pygad_fitness(prices, portfolios, min_time=MIN_TIME):
    """Pygad fitness path — includes SLSQP weight optimisation per eval."""
    import logging as _logging
    import warnings
    from src.optimisers import pygad_ga

    # Prepare module-level state for the legacy fitness() function
    pygad_ga.prepare_opt_inputs(prices, use_forecasts=False)
    pygad_ga.MIN_NUM_STOCKS = N
    pygad_ga.MAX_NUM_STOCKS = N

    individual = portfolios[0].astype(int)

    # Suppress noisy convergence warnings from SLSQP and CCC overflow
    pygad_logger = _logging.getLogger('src.optimisers.pygad_ga')
    pu_logger = _logging.getLogger('src.portfolio_utils')
    prev_pygad = pygad_logger.level
    prev_pu = pu_logger.level
    pygad_logger.setLevel(_logging.CRITICAL)
    pu_logger.setLevel(_logging.CRITICAL)

    def fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pygad_ga.fitness(individual, pygad_ga._ctx.data)
        return 1

    try:
        # Very slow — reduce warmup and runs
        return benchmark_method(
            "Pygad fitness (SLSQP)", fn,
            warmup_fn=lambda: fn(), warmup_iters=3,
            min_time=min_time, num_runs=2,
        )
    finally:
        pygad_logger.setLevel(prev_pygad)
        pu_logger.setLevel(prev_pu)


def bench_cpp_mc(log_returns, num_threads, time_budget=CPP_TIME_BUDGET):
    """C++ Monte Carlo benchmark via subprocess."""
    cpp_binary = os.path.join(os.path.dirname(__file__), 'cpp', 'optimisation')
    if not os.path.isfile(cpp_binary):
        return None

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_binary_data(log_returns, tmp_path)

        cmd = [
            cpp_binary,
            '--binary',
            '--mode', 'mc',
            '--data', tmp_path,
            '--time-budget', str(time_budget),
            '--num-islands', str(num_threads),
            '--seed', '42',
            '--min-etfs', str(N),
            '--max-etfs', str(N),
        ]

        label = f"C++ MC ({num_threads} thread{'s' if num_threads > 1 else ''})"
        print(f"  Running {label} (time budget: {time_budget}s)...", flush=True)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=time_budget + 30)

        if result.returncode != 0:
            print(f"  WARNING: C++ binary exited with code {result.returncode}", flush=True)
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}", flush=True)
            return None

        # Parse JSON output
        output = result.stdout.strip()
        # The JSON may have trailing content after the closing brace
        brace_depth = 0
        json_end = 0
        for i, ch in enumerate(output):
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    json_end = i + 1
                    break

        parsed = json.loads(output[:json_end])
        total_trials = parsed.get('total_trials', 0)
        elapsed = parsed.get('elapsed_seconds', time_budget)

        if total_trials == 0:
            print(f"  WARNING: C++ reported 0 trials", flush=True)
            return None

        evals_per_sec = total_trials / elapsed if elapsed > 0 else 0

        return ThroughputResult(
            method=label,
            evals_per_sec=evals_per_sec,
            total_evals=total_trials,
            elapsed_sec=round(elapsed, 3),
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  WARNING: C++ benchmark failed: {e}", flush=True)
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def bench_cpp_ga(log_returns, num_threads, time_budget=CPP_TIME_BUDGET,
                 label=None, extra_args=None):
    """C++ island GA benchmark via subprocess.

    Uses --time-budget so the GA runs for the same duration as MC, and
    reads actual total_trials from the binary output.
    """
    cpp_binary = os.path.join(os.path.dirname(__file__), 'cpp', 'optimisation')
    if not os.path.isfile(cpp_binary):
        return None

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_binary_data(log_returns, tmp_path)

        # Use a large generation count so the time budget is the binding constraint
        cmd = [
            cpp_binary,
            '--binary',
            '--mode', 'ga',
            '--data', tmp_path,
            '--pop-size', '1000',
            '--generations', '100000',
            '--time-budget', str(time_budget),
            '--num-islands', str(num_threads),
            '--seed', '42',
            '--min-etfs', str(N),
            '--max-etfs', str(N),
        ]
        if extra_args:
            cmd.extend(extra_args)

        if label is None:
            label = f"C++ GA ({num_threads} island{'s' if num_threads > 1 else ''})"
        print(f"  Running {label} (time budget: {time_budget}s)...", flush=True)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=time_budget + 30)

        if result.returncode != 0:
            print(f"  WARNING: C++ binary exited with code {result.returncode}", flush=True)
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}", flush=True)
            return None

        # Parse JSON output
        output = result.stdout.strip()
        brace_depth = 0
        json_end = 0
        for i, ch in enumerate(output):
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    json_end = i + 1
                    break

        parsed = json.loads(output[:json_end])
        total_trials = parsed.get('total_trials', 0)
        elapsed = parsed.get('elapsed_seconds', 0)

        if total_trials == 0 or elapsed <= 0:
            print(f"  WARNING: C++ reported 0 trials or elapsed time", flush=True)
            return None

        evals_per_sec = total_trials / elapsed

        return ThroughputResult(
            method=label,
            evals_per_sec=evals_per_sec,
            total_evals=total_trials,
            elapsed_sec=round(elapsed, 3),
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  WARNING: C++ GA benchmark failed: {e}", flush=True)
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─── Output formatting ──────────────────────────────────────────────────────

def format_results_table(results: List[ThroughputResult], m=M, t=T-1, n=N):
    """Format results as a console table with relative performance."""
    # Find baseline (batch_fitness)
    baseline_rate = None
    for r in results:
        if 'batch_fitness' in r.method:
            baseline_rate = r.evals_per_sec
            break
    if baseline_rate is None and results:
        baseline_rate = results[0].evals_per_sec

    lines = []
    lines.append(f"\nThroughput Benchmark (M={m}, T={t}, n={n})")
    lines.append("=" * 65)
    lines.append(f"{'Method':<35} {'Evals/sec':>12}   {'Relative':>10}")
    lines.append("-" * 65)

    # Sort by evals/sec descending
    sorted_results = sorted(results, key=lambda r: r.evals_per_sec, reverse=True)

    for r in sorted_results:
        rate_str = f"{r.evals_per_sec:>12,.0f}"
        if baseline_rate and baseline_rate > 0:
            relative = r.evals_per_sec / baseline_rate
            rel_str = f"{relative:>8.2f}x"
            if 'batch_fitness' in r.method:
                rel_str += " (base)"
        else:
            rel_str = ""
        lines.append(f"{r.method:<35} {rate_str}   {rel_str}")

    lines.append("-" * 65)
    return "\n".join(lines)


def save_json(results: List[ThroughputResult], path):
    """Save results to JSON."""
    data = {
        'config': {
            'universe_size': M,
            'time_horizon': T - 1,
            'cardinality': N,
            'seed': SEED,
        },
        'results': [asdict(r) for r in results],
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio evaluation throughput benchmark")
    parser.add_argument('--quick', action='store_true',
                        help='Reduce time budgets for faster execution')
    parser.add_argument('--skip-cpp', action='store_true',
                        help='Skip C++ benchmarks')
    args = parser.parse_args()

    # Adjust time budgets
    if args.quick:
        min_time = 0.5
        cpp_time = 3
        pygad_min_time = 0.5
    else:
        min_time = MIN_TIME
        cpp_time = CPP_TIME_BUDGET
        pygad_min_time = 2.0

    print(f"Generating synthetic data (M={M}, T={T}, seed={SEED})...", flush=True)
    (prices, log_returns, expected_returns, centered,
     T_obs, sub_cov, sub_er) = generate_test_data()
    print(f"  Prices: {prices.shape}, Log returns: {log_returns.shape}")
    print(f"  Generating {max(BATCH_SIZE, 100)} test portfolios (n={N})...", flush=True)
    portfolios = generate_test_portfolios(M, N, max(BATCH_SIZE, 100))

    results: List[ThroughputResult] = []

    # ── C++ benchmarks (run first since they're independent subprocesses) ──
    if not args.skip_cpp:
        print("\n--- C++ Benchmarks ---", flush=True)

        # Single thread
        r = bench_cpp_mc(log_returns, num_threads=1, time_budget=cpp_time)
        if r:
            results.append(r)
            print(f"  {r.method}: {r.evals_per_sec:,.0f} evals/sec")
        else:
            print("  C++ MC (1 thread): skipped (binary not found or failed)")

        # All threads
        cpu_count = os.cpu_count() or 1
        if cpu_count > 1:
            r = bench_cpp_mc(log_returns, num_threads=cpu_count, time_budget=cpp_time)
            if r:
                results.append(r)
                print(f"  {r.method}: {r.evals_per_sec:,.0f} evals/sec")
            else:
                print(f"  C++ MC ({cpu_count} threads): skipped")

        # C++ GA — includes selection/crossover/mutation overhead per generation
        r = bench_cpp_ga(log_returns, num_threads=1, time_budget=cpp_time)
        if r:
            results.append(r)
            print(f"  {r.method}: {r.evals_per_sec:,.0f} evals/sec")
        else:
            print("  C++ GA (1 island): skipped")

        if cpu_count > 1:
            r = bench_cpp_ga(log_returns, num_threads=cpu_count, time_budget=cpp_time)
            if r:
                results.append(r)
                print(f"  {r.method}: {r.evals_per_sec:,.0f} evals/sec")
            else:
                print(f"  C++ GA ({cpu_count} islands): skipped")

    # ── Python benchmarks ──────────────────────────────────────────────────
    print("\n--- Python Benchmarks ---", flush=True)

    print("  batch_fitness...", flush=True)
    r = bench_batch_fitness(expected_returns, centered, T_obs, portfolios, min_time=min_time)
    results.append(r)
    print(f"  {r.method}: {r.evals_per_sec:,.0f} evals/sec")

    print("  island_ga single fitness...", flush=True)
    r = bench_island_single_fitness(expected_returns, centered, T_obs, portfolios, min_time=min_time)
    results.append(r)
    print(f"  {r.method}: {r.evals_per_sec:,.0f} evals/sec")

    print("  equal_weight_fitness...", flush=True)
    r = bench_equal_weight_fitness(sub_er, sub_cov, min_time=min_time)
    results.append(r)
    print(f"  {r.method}: {r.evals_per_sec:,.0f} evals/sec")

    print("  sharpe_ratio...", flush=True)
    r = bench_sharpe_ratio(sub_er, sub_cov, min_time=min_time)
    results.append(r)
    print(f"  {r.method}: {r.evals_per_sec:,.0f} evals/sec")

    print("  Pygad fitness (SLSQP)...", flush=True)
    r = bench_pygad_fitness(prices, portfolios, min_time=pygad_min_time)
    results.append(r)
    print(f"  {r.method}: {r.evals_per_sec:,.0f} evals/sec")

    # ── Output ─────────────────────────────────────────────────────────────
    print(format_results_table(results))

    output_path = os.path.join(os.path.dirname(__file__),
                               'benchmark_results', 'throughput_results.json')
    save_json(results, output_path)


if __name__ == '__main__':
    main()
