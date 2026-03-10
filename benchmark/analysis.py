"""Statistical analysis and plotting for benchmark results."""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from benchmark.results import BenchmarkResult, BenchmarkSuite


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_table(suite: BenchmarkSuite) -> pd.DataFrame:
    """Median, IQR, best, worst Sharpe per algorithm."""
    rows = []
    for algo in suite.algorithms:
        fitnesses = [r.best_fitness for r in suite.results[algo]
                     if r.best_fitness > -1e3]
        if not fitnesses:
            rows.append({'Algorithm': algo, 'Runs': 0})
            continue
        rows.append({
            'Algorithm': algo,
            'Runs': len(fitnesses),
            'Median': np.median(fitnesses),
            'IQR_25': np.percentile(fitnesses, 25),
            'IQR_75': np.percentile(fitnesses, 75),
            'Best': np.max(fitnesses),
            'Worst': np.min(fitnesses),
            'Mean': np.mean(fitnesses),
            'Std': np.std(fitnesses),
        })
    return pd.DataFrame(rows).set_index('Algorithm')


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def friedman_test(suite: BenchmarkSuite) -> dict:
    """Friedman rank-sum test across 3+ algorithms (paired by seed).

    Returns dict with 'statistic', 'p_value', and 'rankings'.
    """
    algos = suite.algorithms
    if len(algos) < 3:
        return {'error': 'Need at least 3 algorithms for Friedman test'}

    # Build seed -> {algo: fitness} mapping
    seed_results = {}
    for algo in algos:
        for r in suite.results[algo]:
            if r.seed not in seed_results:
                seed_results[r.seed] = {}
            seed_results[r.seed][algo] = r.best_fitness

    # Only keep seeds where all algorithms have results
    common_seeds = [s for s in seed_results
                    if all(a in seed_results[s] for a in algos)]
    if len(common_seeds) < 3:
        return {'error': f'Only {len(common_seeds)} common seeds (need >= 3)'}

    groups = []
    for algo in algos:
        groups.append([seed_results[s][algo] for s in common_seeds])

    stat, p_value = stats.friedmanchisquare(*groups)

    # Mean ranks
    ranks = np.zeros((len(common_seeds), len(algos)))
    for i, seed in enumerate(common_seeds):
        values = [seed_results[seed][a] for a in algos]
        ranks[i] = stats.rankdata([-v for v in values])  # negate for descending
    mean_ranks = ranks.mean(axis=0)

    return {
        'statistic': stat,
        'p_value': p_value,
        'rankings': dict(zip(algos, mean_ranks)),
        'num_seeds': len(common_seeds),
    }


def nemenyi_posthoc(suite: BenchmarkSuite) -> Optional[pd.DataFrame]:
    """Nemenyi post-hoc test. Returns pairwise p-value matrix or None."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("scikit-posthocs not installed. Skipping Nemenyi test.")
        return None

    algos = suite.algorithms
    if len(algos) < 3:
        return None

    seed_results = {}
    for algo in algos:
        for r in suite.results[algo]:
            if r.seed not in seed_results:
                seed_results[r.seed] = {}
            seed_results[r.seed][algo] = r.best_fitness

    common_seeds = [s for s in seed_results
                    if all(a in seed_results[s] for a in algos)]
    if len(common_seeds) < 3:
        return None

    # Build long-form DataFrame for scikit-posthocs
    rows = []
    for seed in common_seeds:
        for algo in algos:
            rows.append({'seed': seed, 'algorithm': algo,
                         'fitness': float(seed_results[seed][algo])})
    df = pd.DataFrame(rows)
    df['fitness'] = pd.to_numeric(df['fitness'])
    df['seed'] = pd.to_numeric(df['seed'])

    result = sp.posthoc_nemenyi_friedman(
        df, y_col='fitness', group_col='algorithm', block_col='seed',
        melted=True
    )
    return result


def wilcoxon_pairwise(suite: BenchmarkSuite, algo_a: str, algo_b: str) -> dict:
    """Wilcoxon signed-rank test between two algorithms (paired by seed)."""
    seeds_a = {r.seed: r.best_fitness for r in suite.results.get(algo_a, [])}
    seeds_b = {r.seed: r.best_fitness for r in suite.results.get(algo_b, [])}
    common = sorted(set(seeds_a) & set(seeds_b))
    if len(common) < 5:
        return {'error': f'Only {len(common)} common seeds'}

    vals_a = [seeds_a[s] for s in common]
    vals_b = [seeds_b[s] for s in common]
    stat, p_value = stats.wilcoxon(vals_a, vals_b)
    return {'statistic': stat, 'p_value': p_value, 'num_pairs': len(common)}


def aocc(convergence_records, time_budget: float,
         reference_fitness: float) -> float:
    """Area Over the Convergence Curve.

    Measures the gap between the reference fitness and the convergence curve,
    normalised by time_budget * reference_fitness. Lower = faster convergence.
    """
    if not convergence_records or reference_fitness <= 0:
        return float('inf')

    times = [0.0] + [c.wall_clock_seconds for c in convergence_records]
    fitnesses = [0.0] + [c.best_fitness for c in convergence_records]
    # Extend to time_budget
    if times[-1] < time_budget:
        times.append(time_budget)
        fitnesses.append(fitnesses[-1])

    area_over = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        gap = max(0, reference_fitness - fitnesses[i - 1])
        area_over += gap * dt

    return area_over / (time_budget * reference_fitness)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _interpolate_convergence(runs: List[BenchmarkResult],
                             time_grid: np.ndarray,
                             penalty_threshold: float = -1e3) -> np.ndarray:
    """Interpolate each run's best-so-far onto a common time grid.

    Penalty values (below penalty_threshold) are treated as NaN so they
    don't distort convergence plots. The curve starts from the first
    feasible solution.

    Returns shape (num_runs, len(time_grid)).
    """
    matrix = np.full((len(runs), len(time_grid)), np.nan)
    for i, run in enumerate(runs):
        if not run.convergence:
            continue
        # Filter to only feasible convergence points
        feasible = [(c.wall_clock_seconds, c.best_fitness)
                    for c in run.convergence
                    if c.best_fitness > penalty_threshold]
        if not feasible:
            continue
        times = [f[0] for f in feasible]
        fits = [f[1] for f in feasible]
        # Step-function interpolation (forward fill)
        for j, t in enumerate(time_grid):
            idx = np.searchsorted(times, t, side='right') - 1
            if idx >= 0:
                matrix[i, j] = fits[idx]
    return matrix


def plot_convergence_curves(suite: BenchmarkSuite, time_budget: float,
                            output_dir: str = 'benchmark_results'):
    """Median best-so-far Sharpe vs wall-clock time with IQR bands."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    time_grid = np.linspace(0, time_budget, 500)

    for algo in suite.algorithms:
        runs = suite.results[algo]
        matrix = _interpolate_convergence(runs, time_grid)
        if np.all(np.isnan(matrix)):
            continue
        median = np.nanmedian(matrix, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0)
        q75 = np.nanpercentile(matrix, 75, axis=0)
        ax.plot(time_grid, median, label=algo, linewidth=2)
        ax.fill_between(time_grid, q25, q75, alpha=0.2)

    ax.set_xlabel('Wall-clock time (seconds)')
    ax.set_ylabel('Best Sharpe ratio')
    ax.set_title('Convergence Curves (median with 25th-75th percentile)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'convergence_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_boxplots(suite: BenchmarkSuite, output_dir: str = 'benchmark_results'):
    """Box plots of final Sharpe ratios per algorithm."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    data_to_plot = []
    labels = []
    for algo in suite.algorithms:
        fitnesses = [r.best_fitness for r in suite.results[algo]
                     if r.best_fitness > -1e3]
        if fitnesses:
            data_to_plot.append(fitnesses)
            labels.append(algo)

    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('Sharpe ratio')
        ax.set_title('Final Sharpe Ratio Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15, ha='right')

    fig.tight_layout()
    path = os.path.join(output_dir, 'boxplots.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_performance_profile(suite: BenchmarkSuite, time_budget: float,
                             target_fraction: float = 0.95,
                             output_dir: str = 'benchmark_results'):
    """Dolan-More performance profile.

    For each seed, find the best fitness across all algorithms.
    Then for each algorithm, find the time to reach target_fraction * best.
    Plot the CDF of performance ratios (time_algo / time_best).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    algos = suite.algorithms
    # Collect per-seed best fitness
    seed_best = {}
    for algo in algos:
        for r in suite.results[algo]:
            if r.seed not in seed_best:
                seed_best[r.seed] = float('-inf')
            if r.best_fitness > seed_best[r.seed]:
                seed_best[r.seed] = r.best_fitness

    def time_to_target(run: BenchmarkResult, target: float) -> float:
        """Wall-clock time to first reach target fitness."""
        for c in run.convergence:
            if c.best_fitness >= target:
                return c.wall_clock_seconds
        return time_budget * 2  # never reached

    # Build performance ratios per seed
    all_ratios = {algo: [] for algo in algos}
    for seed, best in seed_best.items():
        target = target_fraction * best
        if target <= 0:
            continue

        times = {}
        for algo in algos:
            runs = [r for r in suite.results[algo] if r.seed == seed]
            if runs:
                times[algo] = time_to_target(runs[0], target)
            else:
                times[algo] = time_budget * 2

        best_time = max(min(times.values()), 1e-6)
        for algo in algos:
            all_ratios[algo].append(times[algo] / best_time)

    fig, ax = plt.subplots(figsize=(10, 6))
    tau_max = 10.0

    for algo in algos:
        ratios = sorted(all_ratios[algo])
        if not ratios:
            continue
        n = len(ratios)
        taus = np.linspace(1, tau_max, 500)
        cdf = [sum(1 for r in ratios if r <= t) / n for t in taus]
        ax.plot(taus, cdf, label=algo, linewidth=2)

    ax.set_xlabel(f'Performance ratio (time / best time)')
    ax.set_ylabel('Fraction of runs')
    ax.set_title(f'Performance Profile (target = {target_fraction:.0%} of best)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, tau_max)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(output_dir, 'performance_profile.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def generate_full_report(suite: BenchmarkSuite, time_budget: float,
                         output_dir: str = 'benchmark_results'):
    """Run all analyses and save outputs."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("BENCHMARK ANALYSIS")
    print("=" * 60)

    # Summary table
    table = summary_table(suite)
    print("\nSummary Statistics:")
    print(table.to_string())
    csv_path = os.path.join(output_dir, 'summary_table.csv')
    table.to_csv(csv_path)
    print(f"\n  Saved {csv_path}")

    # AOCC
    best_overall = max(
        r.best_fitness
        for runs in suite.results.values()
        for r in runs
        if r.best_fitness > -1e3
    )
    print(f"\nArea Over Convergence Curve (ref={best_overall:.4f}):")
    for algo in suite.algorithms:
        aocc_values = []
        for r in suite.results[algo]:
            if r.convergence:
                aocc_values.append(aocc(r.convergence, time_budget, best_overall))
        if aocc_values:
            print(f"  {algo}: median={np.median(aocc_values):.4f}, "
                  f"mean={np.mean(aocc_values):.4f}")

    # Friedman test
    if len(suite.algorithms) >= 3:
        print("\nFriedman Test:")
        fr = friedman_test(suite)
        if 'error' in fr:
            print(f"  {fr['error']}")
        else:
            print(f"  Statistic: {fr['statistic']:.4f}")
            print(f"  p-value: {fr['p_value']:.6f}")
            print(f"  Seeds: {fr['num_seeds']}")
            print(f"  Mean rankings (lower=better):")
            for algo, rank in sorted(fr['rankings'].items(), key=lambda x: x[1]):
                print(f"    {algo}: {rank:.2f}")

            # Nemenyi post-hoc if significant
            if fr['p_value'] < 0.05:
                print("\nNemenyi Post-hoc Test (p-values):")
                nem = nemenyi_posthoc(suite)
                if nem is not None:
                    print(nem.to_string())

    # Pairwise Wilcoxon for all pairs
    algos = suite.algorithms
    if len(algos) >= 2:
        print("\nWilcoxon Pairwise Tests:")
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                w = wilcoxon_pairwise(suite, algos[i], algos[j])
                if 'error' in w:
                    print(f"  {algos[i]} vs {algos[j]}: {w['error']}")
                else:
                    sig = "*" if w['p_value'] < 0.05 else ""
                    print(f"  {algos[i]} vs {algos[j]}: "
                          f"p={w['p_value']:.6f}{sig} (n={w['num_pairs']})")

    # Plots
    print("\nGenerating plots...")
    plot_convergence_curves(suite, time_budget, output_dir)
    plot_boxplots(suite, output_dir)
    plot_performance_profile(suite, time_budget, output_dir=output_dir)

    print("\nAnalysis complete.")
