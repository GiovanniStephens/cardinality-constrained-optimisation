#!/usr/bin/env python3
"""CLI entry point for the portfolio optimisation benchmarking framework.

Usage:
    python run_benchmark.py                          # Full: 120s budget, 30 runs
    python run_benchmark.py --quick                  # Quick: 30s budget, 5 runs
    python run_benchmark.py --time-budget 60 --runs 10
    python run_benchmark.py --algorithms "Island GA (Python)" "Monte Carlo"
    python run_benchmark.py --analyze-only           # Re-plot from saved results
"""

import argparse
import sys

from benchmark.adapters import ALL_ADAPTERS, DEFAULT_ADAPTERS
from benchmark.analysis import generate_full_report
from benchmark.runner import BenchmarkRunner, load_suite


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark portfolio optimisation algorithms'
    )
    parser.add_argument('--time-budget', type=float, default=120.0,
                        help='Time budget per run in seconds (default: 120)')
    parser.add_argument('--runs', type=int, default=30,
                        help='Number of runs per algorithm (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 30s budget, 5 runs')
    parser.add_argument('--algorithms', nargs='+', default=None,
                        help='Algorithms to benchmark (default: all except C++)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Skip runs, only re-analyze saved results')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory (default: benchmark_results)')
    parser.add_argument('--data-file', type=str,
                        default='Data/time_series_20251016_113257.csv',
                        help='Path to price data CSV')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.time_budget = 30.0
        args.runs = 5

    output_dir = args.output_dir

    # Analyze-only mode
    if args.analyze_only:
        suite = load_suite(output_dir)
        if suite is None:
            print(f"No saved results found in {output_dir}/")
            sys.exit(1)
        generate_full_report(suite, args.time_budget, output_dir)
        return

    # Load data
    from simple_ga_optimisation import load_data
    print(f"Loading data from {args.data_file}...")
    data = load_data(args.data_file)
    print(f"  {data.shape[1]} instruments, {data.shape[0]} days")

    # Select adapters
    algo_names = args.algorithms or DEFAULT_ADAPTERS
    adapters = []
    for name in algo_names:
        if name not in ALL_ADAPTERS:
            print(f"Unknown algorithm: {name}")
            print(f"Available: {list(ALL_ADAPTERS.keys())}")
            sys.exit(1)
        adapters.append(ALL_ADAPTERS[name]())

    print(f"\nBenchmark configuration:")
    print(f"  Time budget: {args.time_budget}s per run")
    print(f"  Runs: {args.runs}")
    print(f"  Base seed: {args.seed}")
    print(f"  Algorithms: {[a.name for a in adapters]}")
    print(f"  Output: {output_dir}/")

    # Run benchmarks
    runner = BenchmarkRunner(
        adapters=adapters,
        data=data,
        time_budget=args.time_budget,
        num_runs=args.runs,
        base_seed=args.seed,
        output_dir=output_dir,
    )

    # Resume from saved results if available
    existing = load_suite(output_dir)
    if existing:
        print(f"\nResuming from {len(existing.algorithms)} saved algorithm(s)")
    suite = runner.run(resume_suite=existing)

    # Analyze
    generate_full_report(suite, args.time_budget, output_dir)


if __name__ == '__main__':
    main()
