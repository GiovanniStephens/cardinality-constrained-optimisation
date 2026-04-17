"""Adapter classes wrapping each optimisation algorithm for benchmarking."""

import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd

from benchmark.results import BenchmarkResult, ConvergenceRecord


class OptimiserAdapter(ABC):
    """Common interface for all optimiser adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, data: pd.DataFrame, time_budget: float,
            seed: int, run_id: int) -> BenchmarkResult:
        ...


class SimpleGAAdapter(OptimiserAdapter):
    """Wraps optimisers/island_ga.py (parallel island-based GA)."""

    name = "Island GA (Python)"

    def __init__(self, num_generations=200, total_population_size=2000,
                 num_elites=50, migration_interval=10, migration_rate=0.1,
                 min_etfs=3, max_etfs=15, min_return=None):
        self.num_generations = num_generations
        self.total_population_size = total_population_size
        self.num_elites = num_elites
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.min_etfs = min_etfs
        self.max_etfs = max_etfs
        self.min_return = min_return

    def run(self, data: pd.DataFrame, time_budget: float,
            seed: int, run_id: int) -> BenchmarkResult:
        from src.optimisers.island_ga import genetic_algorithm
        from src.portfolio_utils import optimise_weights
        np.random.seed(seed)
        num_islands = min(os.cpu_count(), 4)
        manager = Manager()
        return_dict = manager.dict()
        convergence_log = manager.list()
        start_time = time.time()
        mutation_rate = 1 / data.shape[1]
        island_pop_size = self.total_population_size // num_islands

        def init_random_state():
            np.random.seed(None)

        with Pool(num_islands, initializer=init_random_state) as pool:
            args = [
                (i, num_islands, data, self.num_generations, island_pop_size,
                 mutation_rate, self.num_elites, self.migration_interval,
                 self.migration_rate, return_dict,
                 convergence_log, start_time, time_budget,
                 self.min_etfs, self.max_etfs, self.min_return)
                for i in range(num_islands)
            ]
            results = pool.starmap(genetic_algorithm, args)

        elapsed = time.time() - start_time
        timed_out = elapsed >= time_budget

        # Find best solution across islands
        best_fitness = float('-inf')
        best_solution = None
        for result in results:
            if result is not None:
                solution, fitness = result
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution

        # Build convergence curve: best-so-far across all islands
        raw_logs = list(convergence_log)
        raw_logs.sort(key=lambda x: x[0])  # sort by wall_clock
        convergence = []
        best_so_far = float('-inf')
        for wall_clock, gen, best_fit, mean_fit, island_id in raw_logs:
            if best_fit > best_so_far:
                best_so_far = best_fit
            convergence.append(ConvergenceRecord(
                wall_clock_seconds=wall_clock,
                function_evaluations=(gen + 1) * island_pop_size * num_islands,
                best_fitness=best_so_far,
                mean_fitness=mean_fit,
                generation=gen,
            ))

        # Try SLSQP weight optimisation if we have a solution and time remains
        selected_etfs = None
        optimised_weights = None
        if best_solution is not None:
            selected_etfs = list(data.columns[best_solution == 1])
            remaining = time_budget - (time.time() - start_time)
            if remaining > 1.0:
                try:
                    opt_result = optimise_weights(best_solution, data,
                                                    min_return=self.min_return)
                    if opt_result.success:
                        best_fitness = -opt_result.fun
                        optimised_weights = opt_result.x
                except Exception:
                    pass

        # Record final convergence point including SLSQP refinement time
        total_elapsed = time.time() - start_time
        last_evals = convergence[-1].function_evaluations if convergence else 0
        last_gen = convergence[-1].generation if convergence else 0
        convergence.append(ConvergenceRecord(
            wall_clock_seconds=total_elapsed,
            function_evaluations=last_evals,
            best_fitness=best_fitness,
            mean_fitness=best_fitness,
            generation=last_gen,
        ))

        return BenchmarkResult(
            algorithm=self.name,
            seed=seed,
            time_budget=time_budget,
            convergence=convergence,
            best_fitness=best_fitness,
            selected_etfs=selected_etfs,
            optimised_weights=optimised_weights,
            timed_out=timed_out,
        )


class PygadGAAdapter(OptimiserAdapter):
    """Wraps optimisation.py (pygad-based GA with copula support)."""

    name = "Pygad GA"

    def __init__(self, num_generations=200, population_size=50,
                 min_etfs=3, max_etfs=15):
        self.num_generations = num_generations
        self.population_size = population_size
        self.min_etfs = min_etfs
        self.max_etfs = max_etfs

    def run(self, data: pd.DataFrame, time_budget: float,
            seed: int, run_id: int) -> BenchmarkResult:
        import pygad
        from src.optimisers import pygad_ga as opt_mod

        np.random.seed(seed)
        start_time = time.time()

        # Prepare global state in optimisation module
        opt_mod.prepare_opt_inputs(data, use_forecasts=False)
        saved_max = opt_mod.MAX_NUM_STOCKS
        saved_min = opt_mod.MIN_NUM_STOCKS
        opt_mod.MAX_NUM_STOCKS = self.max_etfs
        opt_mod.MIN_NUM_STOCKS = self.min_etfs

        convergence = []
        best_so_far = float('-inf')

        # Wrap fitness to bail out early when time is up
        def timed_fitness(ga_instance, solution, solution_idx):
            if (time.time() - start_time) > time_budget:
                return -1e6
            return opt_mod.fitness_2(ga_instance, solution, solution_idx)

        def on_gen_callback(ga_instance):
            nonlocal best_so_far
            elapsed = time.time() - start_time
            gen = ga_instance.generations_completed
            pop_fitness = ga_instance.last_generation_fitness
            current_best = float(max(pop_fitness))
            mean_fit = float(np.mean(pop_fitness))
            if current_best > best_so_far:
                best_so_far = current_best
            convergence.append(ConvergenceRecord(
                wall_clock_seconds=elapsed,
                function_evaluations=gen * self.population_size,
                best_fitness=best_so_far,
                mean_fitness=mean_fit,
                generation=gen,
            ))
            if elapsed > time_budget:
                return "stop"

        try:
            initial_pop = np.array([
                opt_mod.create_individual(opt_mod.data)
                for _ in range(self.population_size)
            ])

            ga_instance = pygad.GA(
                num_generations=self.num_generations,
                initial_population=initial_pop,
                num_parents_mating=max(2, self.population_size // 10),
                gene_type=int,
                init_range_low=0,
                init_range_high=2,
                parent_selection_type='rank',
                keep_parents=0,
                random_mutation_min_val=-1,
                random_mutation_max_val=1,
                mutation_type="random",
                crossover_type="single_point",
                crossover_probability=0.85,
                fitness_func=timed_fitness,
                on_generation=on_gen_callback,
                stop_criteria='saturate_5',
            )
            ga_instance.run()

            solution, solution_fitness, _ = ga_instance.best_solution(
                ga_instance.last_generation_fitness
            )
            best_fitness = float(solution_fitness)

            # Extract selected ETFs
            indices = np.array(solution).astype(bool)
            all_tickers = list(opt_mod.data.columns)
            selected_etfs = [all_tickers[i] for i in range(len(indices)) if indices[i]]

            # Try SLSQP weight optimisation
            optimised_weights = None
            remaining = time_budget - (time.time() - start_time)
            if remaining > 1.0 and len(selected_etfs) >= 2:
                try:
                    subset = opt_mod.data.iloc[indices, :]
                    random_weights = np.random.random(np.count_nonzero(solution))
                    random_weights /= np.sum(random_weights)
                    sol = opt_mod.optimize(
                        subset.transpose(),
                        random_weights,
                        target_return=opt_mod.TARGET_RETURN,
                        target_risk=opt_mod.TARGET_RISK,
                        max_weight=opt_mod.MAX_WEIGHT,
                        min_weight=opt_mod.MIN_WEIGHT,
                    )
                    if sol.success:
                        best_fitness = -sol.fun
                        optimised_weights = sol.x
                except Exception:
                    pass

        except Exception as e:
            best_fitness = float('-inf')
            selected_etfs = None
            optimised_weights = None
        finally:
            opt_mod.MAX_NUM_STOCKS = saved_max
            opt_mod.MIN_NUM_STOCKS = saved_min

        # Record final convergence point including SLSQP refinement time
        elapsed = time.time() - start_time
        last_evals = convergence[-1].function_evaluations if convergence else 0
        last_gen = convergence[-1].generation if convergence else 0
        convergence.append(ConvergenceRecord(
            wall_clock_seconds=elapsed,
            function_evaluations=last_evals,
            best_fitness=best_fitness,
            mean_fitness=best_fitness,
            generation=last_gen,
        ))

        return BenchmarkResult(
            algorithm=self.name,
            seed=seed,
            time_budget=time_budget,
            convergence=convergence,
            best_fitness=best_fitness,
            selected_etfs=selected_etfs,
            optimised_weights=optimised_weights,
            timed_out=elapsed >= time_budget,
        )


class MonteCarloAdapter(OptimiserAdapter):
    """Wraps optimisers/monte_carlo.py (random search)."""

    name = "Monte Carlo"

    def __init__(self, min_etfs=3, max_etfs=15, log_interval=5000):
        self.min_etfs = min_etfs
        self.max_etfs = max_etfs
        self.log_interval = log_interval

    def run(self, data: pd.DataFrame, time_budget: float,
            seed: int, run_id: int) -> BenchmarkResult:
        from src.portfolio_utils import (
            calculate_log_returns as calculate_returns,
            calculate_expected_returns,
            calculate_covariance_matrix,
        )

        np.random.seed(seed)
        start_time = time.time()

        log_returns = calculate_returns(data)
        expected_returns = calculate_expected_returns(log_returns).values
        cov_matrix = calculate_covariance_matrix(log_returns).values
        num_etfs = data.shape[1]

        convergence = []
        best_fitness = float('-inf')
        best_portfolio = None
        trial = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                break

            # Generate random portfolio
            num_selected = np.random.randint(self.min_etfs, self.max_etfs + 1)
            portfolio = np.zeros(num_etfs, dtype=int)
            selected_indices = np.random.choice(num_etfs, num_selected, replace=False)
            portfolio[selected_indices] = 1

            # Calculate fitness inline with our constraints
            sel = portfolio == 1
            n_sel = np.sum(sel)
            if n_sel < self.min_etfs or n_sel > self.max_etfs:
                trial += 1
                continue
            filtered_returns = expected_returns[sel]
            filtered_cov = cov_matrix[np.ix_(sel, sel)]
            weights = np.ones(n_sel) / n_sel
            p_return = np.dot(weights, filtered_returns)
            p_variance = np.dot(weights, np.dot(filtered_cov, weights))
            fitness = p_return / np.sqrt(p_variance) if p_variance > 0 else 0

            if fitness > best_fitness:
                best_fitness = fitness
                best_portfolio = portfolio.copy()

            trial += 1
            if trial % self.log_interval == 0:
                convergence.append(ConvergenceRecord(
                    wall_clock_seconds=time.time() - start_time,
                    function_evaluations=trial,
                    best_fitness=best_fitness,
                    mean_fitness=best_fitness,  # MC has no population mean
                    generation=trial,
                ))

        # Final log point
        convergence.append(ConvergenceRecord(
            wall_clock_seconds=time.time() - start_time,
            function_evaluations=trial,
            best_fitness=best_fitness,
            mean_fitness=best_fitness,
            generation=trial,
        ))

        selected_etfs = None
        if best_portfolio is not None:
            selected_etfs = list(data.columns[best_portfolio == 1])

        return BenchmarkResult(
            algorithm=self.name,
            seed=seed,
            time_budget=time_budget,
            convergence=convergence,
            best_fitness=best_fitness,
            selected_etfs=selected_etfs,
            timed_out=True,  # MC always runs until timeout
        )


class MIPAdapter(OptimiserAdapter):
    """Wraps optimisers/mip.py (Mixed Integer Linear Programming)."""

    name = "MILP"

    def __init__(self, max_etfs=15, risk_aversion=0.8):
        self.max_etfs = max_etfs
        self.risk_aversion = risk_aversion

    def run(self, data: pd.DataFrame, time_budget: float,
            seed: int, run_id: int) -> BenchmarkResult:
        import pulp
        from src.portfolio_utils import (
            calculate_log_returns as calculate_returns,
            calculate_expected_returns,
            calculate_variances,
        )
        from src.optimisers.mip import portfolio_sharpe_ratio

        start_time = time.time()

        log_returns = calculate_returns(data)
        expected_returns = calculate_expected_returns(log_returns)
        volatilities = np.sqrt(calculate_variances(log_returns))
        etfs = log_returns.columns

        # Build MILP problem directly (to override max_etfs constraint)
        problem = pulp.LpProblem("Portfolio_Selection", pulp.LpMaximize)
        selection = pulp.LpVariable.dicts("Select", etfs, 0, 1, pulp.LpBinary)
        problem += pulp.lpSum([
            expected_returns[etf] * selection[etf]
            - self.risk_aversion * volatilities[etf] * selection[etf]
            for etf in etfs
        ]), "Risk_Adjusted_Return"
        problem += pulp.lpSum([selection[etf] for etf in etfs]) <= self.max_etfs, "Max_ETFs"

        # Solve with time limit
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_budget)
        problem.solve(solver)
        elapsed = time.time() - start_time

        # Calculate Sharpe ratio of selected portfolio
        best_fitness = portfolio_sharpe_ratio(selection, expected_returns, log_returns)
        selected_etfs = [etf for etf in etfs if pulp.value(selection[etf]) > 0.5]

        convergence = [ConvergenceRecord(
            wall_clock_seconds=elapsed,
            function_evaluations=1,
            best_fitness=best_fitness,
            mean_fitness=best_fitness,
            generation=1,
        )]

        return BenchmarkResult(
            algorithm=self.name,
            seed=seed,
            time_budget=time_budget,
            convergence=convergence,
            best_fitness=best_fitness,
            selected_etfs=selected_etfs,
            timed_out=elapsed >= time_budget,
            metadata={'note': 'Deterministic — identical across seeds'},
        )


def _run_cpp_binary(binary_path, cmd, start_time, time_budget, stderr_pattern,
                    evals_fn):
    """Shared logic for running the C++ binary and collecting results.

    :param binary_path: path to the compiled binary.
    :param cmd: full command list to execute.
    :param start_time: wall-clock start time.
    :param time_budget: time budget in seconds.
    :param stderr_pattern: compiled regex for parsing stderr convergence lines.
    :param evals_fn: callable(match) -> int, extracts function_evaluations from
        a regex match object.
    :return: (convergence, best_so_far, result_json or None).
    """
    import json as json_mod
    import threading

    convergence = []
    best_so_far = float('-inf')

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    stdout_data = []

    def read_stdout():
        for line in proc.stdout:
            stdout_data.append(line)

    stdout_thread = threading.Thread(target=read_stdout)
    stdout_thread.start()

    for line in proc.stderr:
        elapsed = time.time() - start_time
        match = stderr_pattern.match(line.strip())
        if match:
            fitness = float(match.group(3))
            if fitness > best_so_far:
                best_so_far = fitness
            convergence.append(ConvergenceRecord(
                wall_clock_seconds=elapsed,
                function_evaluations=evals_fn(match),
                best_fitness=best_so_far,
                mean_fitness=fitness,
                generation=int(match.group(2)),
            ))

    proc.wait(timeout=time_budget + 30)
    stdout_thread.join(timeout=5)

    stdout_text = ''.join(stdout_data)
    result_json = None
    try:
        result_json = json_mod.loads(stdout_text)
    except (json_mod.JSONDecodeError, ValueError):
        pass

    return convergence, best_so_far, result_json


def _slsqp_refine_topk(result_json, data):
    """Run SLSQP weight optimisation on top-K solutions from C++ output.

    Returns (best_fitness, selected_etfs, optimised_weights).
    """
    from src.portfolio_utils import optimise_weights, calculate_log_returns

    top_solutions = result_json.get('top_solutions', [])
    if not top_solutions:
        return None, None, None

    best_fitness = float('-inf')
    best_etfs = None
    best_weights = None

    all_tickers = list(data.columns)
    for sol in top_solutions:
        tickers = sol.get('tickers', [])
        if len(tickers) < 2:
            continue
        # Build selection vector
        selection = np.zeros(len(all_tickers), dtype=int)
        for t in tickers:
            if t in all_tickers:
                selection[all_tickers.index(t)] = 1
        if np.sum(selection) < 2:
            continue
        try:
            opt = optimise_weights(selection, data)
            if opt.success and -opt.fun > best_fitness:
                best_fitness = -opt.fun
                best_etfs = tickers
                best_weights = opt.x
        except Exception:
            continue

    if best_etfs is None:
        return None, None, None
    return best_fitness, best_etfs, best_weights


class CppGAAdapter(OptimiserAdapter):
    """Wraps the compiled C++ island GA binary with SLSQP weight refinement."""

    name = "Island GA (C++)"

    def __init__(self, binary_path='./cpp/optimisation',
                 num_generations=200, total_population_size=2000,
                 num_elites=50, migration_interval=10, migration_rate=0.1,
                 min_etfs=3, max_etfs=15, min_return=None,
                 num_islands=4, use_svd=False, svd_components=200):
        self.binary_path = binary_path
        self.num_generations = num_generations
        self.pop_size = total_population_size // num_islands
        self.num_elites = num_elites
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.min_etfs = min_etfs
        self.max_etfs = max_etfs
        self.min_return = min_return
        self.num_islands = num_islands
        self.use_svd = use_svd
        self.svd_components = svd_components

    def run(self, data: pd.DataFrame, time_budget: float,
            seed: int, run_id: int) -> BenchmarkResult:
        import tempfile
        from src.portfolio_utils import calculate_log_returns, write_binary_data

        start_time = time.time()

        if not os.path.isfile(self.binary_path):
            return BenchmarkResult(
                algorithm=self.name, seed=seed, time_budget=time_budget,
                convergence=[], best_fitness=float('-inf'),
                metadata={'error': f'Binary not found: {self.binary_path}'},
            )

        tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
        tmp.close()
        try:
            # Write binary data (log returns)
            log_returns = calculate_log_returns(data)
            write_binary_data(log_returns, tmp.name)

            cmd = [
                self.binary_path,
                '--binary',
                '--data', tmp.name,
                '--mode', 'ga',
                '--seed', str(seed),
                '--time-budget', str(time_budget),
                '--pop-size', str(self.pop_size),
                '--generations', str(self.num_generations),
                '--min-etfs', str(self.min_etfs),
                '--max-etfs', str(self.max_etfs),
                '--num-islands', str(self.num_islands),
                '--num-elites', str(self.num_elites),
                '--migration-interval', str(self.migration_interval),
                '--migration-rate', str(self.migration_rate),
                '--risk-free-rate', '0.0',
            ]
            if self.min_return is not None:
                cmd += ['--min-return', str(self.min_return)]
            if self.use_svd:
                cmd += ['--svd', '--svd-components', str(self.svd_components)]

            pattern = re.compile(
                r'Island\s+(\d+):\s+Generation\s+(\d+):\s+'
                r'Best fitness\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
            )
            pop = self.pop_size
            convergence, best_so_far, result_json = _run_cpp_binary(
                self.binary_path, cmd, start_time, time_budget, pattern,
                evals_fn=lambda m: (int(m.group(2)) + 1) * pop,
            )

            # Extract equal-weight fitness from C++ output
            selected_etfs = None
            best_fitness = best_so_far
            optimised_weights = None
            if result_json:
                cpp_fitness = result_json.get('best_fitness', best_so_far)
                selected_etfs = result_json.get('selected_tickers')
                if cpp_fitness > -1e8:
                    best_fitness = cpp_fitness

                # SLSQP weight refinement on top-K solutions
                slsqp_fitness, slsqp_etfs, slsqp_weights = \
                    _slsqp_refine_topk(result_json, data)
                if slsqp_fitness is not None and slsqp_fitness > best_fitness:
                    best_fitness = slsqp_fitness
                    selected_etfs = slsqp_etfs
                    optimised_weights = slsqp_weights

            # Record final convergence point including SLSQP refinement time
            # so the convergence curve honestly reflects total wall time used
            total_elapsed = time.time() - start_time
            last_evals = convergence[-1].function_evaluations if convergence else 0
            last_gen = convergence[-1].generation if convergence else 0
            convergence.append(ConvergenceRecord(
                wall_clock_seconds=total_elapsed,
                function_evaluations=last_evals,
                best_fitness=best_fitness,
                mean_fitness=best_fitness,
                generation=last_gen,
            ))

        except Exception as e:
            return BenchmarkResult(
                algorithm=self.name, seed=seed, time_budget=time_budget,
                convergence=[], best_fitness=float('-inf'),
                metadata={'error': str(e)},
            )
        finally:
            os.unlink(tmp.name)

        elapsed = time.time() - start_time
        return BenchmarkResult(
            algorithm=self.name,
            seed=seed,
            time_budget=time_budget,
            convergence=convergence,
            best_fitness=best_fitness,
            selected_etfs=selected_etfs,
            optimised_weights=optimised_weights,
            timed_out=elapsed >= time_budget,
        )


class CppMonteCarloAdapter(OptimiserAdapter):
    """Wraps the C++ binary in Monte Carlo mode with SLSQP weight refinement."""

    name = "Monte Carlo (C++)"

    def __init__(self, binary_path='./cpp/optimisation',
                 min_etfs=3, max_etfs=15, min_return=None,
                 num_threads=None, mc_log_interval=5000):
        self.binary_path = binary_path
        self.min_etfs = min_etfs
        self.max_etfs = max_etfs
        self.min_return = min_return
        self.num_threads = num_threads or min(os.cpu_count(), 8)
        self.mc_log_interval = mc_log_interval

    def run(self, data: pd.DataFrame, time_budget: float,
            seed: int, run_id: int) -> BenchmarkResult:
        import tempfile
        from src.portfolio_utils import calculate_log_returns, write_binary_data

        start_time = time.time()

        if not os.path.isfile(self.binary_path):
            return BenchmarkResult(
                algorithm=self.name, seed=seed, time_budget=time_budget,
                convergence=[], best_fitness=float('-inf'),
                metadata={'error': f'Binary not found: {self.binary_path}'},
            )

        tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
        tmp.close()
        try:
            log_returns = calculate_log_returns(data)
            write_binary_data(log_returns, tmp.name)

            cmd = [
                self.binary_path,
                '--binary',
                '--mode', 'mc',
                '--data', tmp.name,
                '--seed', str(seed),
                '--time-budget', str(time_budget),
                '--min-etfs', str(self.min_etfs),
                '--max-etfs', str(self.max_etfs),
                '--num-islands', str(self.num_threads),
                '--risk-free-rate', '0.0',
                '--mc-log-interval', str(self.mc_log_interval),
            ]
            if self.min_return is not None:
                cmd += ['--min-return', str(self.min_return)]

            pattern = re.compile(
                r'MC worker\s+(\d+):\s+Trial\s+(\d+):\s+'
                r'Best fitness\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
            )
            convergence, best_so_far, result_json = _run_cpp_binary(
                self.binary_path, cmd, start_time, time_budget, pattern,
                evals_fn=lambda m: int(m.group(2)),
            )

            selected_etfs = None
            best_fitness = best_so_far
            optimised_weights = None
            if result_json:
                cpp_fitness = result_json.get('best_fitness', best_so_far)
                selected_etfs = result_json.get('selected_tickers')
                if cpp_fitness > -1e8:
                    best_fitness = cpp_fitness

                # SLSQP weight refinement on top-K solutions
                slsqp_fitness, slsqp_etfs, slsqp_weights = \
                    _slsqp_refine_topk(result_json, data)
                if slsqp_fitness is not None and slsqp_fitness > best_fitness:
                    best_fitness = slsqp_fitness
                    selected_etfs = slsqp_etfs
                    optimised_weights = slsqp_weights

            # Record final convergence point including SLSQP refinement time
            # so the convergence curve honestly reflects total wall time used
            total_elapsed = time.time() - start_time
            last_evals = convergence[-1].function_evaluations if convergence else 0
            last_gen = convergence[-1].generation if convergence else 0
            convergence.append(ConvergenceRecord(
                wall_clock_seconds=total_elapsed,
                function_evaluations=last_evals,
                best_fitness=best_fitness,
                mean_fitness=best_fitness,
                generation=last_gen,
            ))

        except Exception as e:
            return BenchmarkResult(
                algorithm=self.name, seed=seed, time_budget=time_budget,
                convergence=[], best_fitness=float('-inf'),
                metadata={'error': str(e)},
            )
        finally:
            os.unlink(tmp.name)

        elapsed = time.time() - start_time
        return BenchmarkResult(
            algorithm=self.name,
            seed=seed,
            time_budget=time_budget,
            convergence=convergence,
            best_fitness=best_fitness,
            selected_etfs=selected_etfs,
            optimised_weights=optimised_weights,
            timed_out=elapsed >= time_budget,
        )


# Registry of all available adapters
ALL_ADAPTERS = {
    'Island GA (Python)': SimpleGAAdapter,
    'Pygad GA': PygadGAAdapter,
    'Monte Carlo': MonteCarloAdapter,
    'MILP': MIPAdapter,
    'Island GA (C++)': CppGAAdapter,
    'Monte Carlo (C++)': CppMonteCarloAdapter,
}

DEFAULT_ADAPTERS = ['Island GA (Python)', 'Pygad GA', 'Monte Carlo', 'MILP']
