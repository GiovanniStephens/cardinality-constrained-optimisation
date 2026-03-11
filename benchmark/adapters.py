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
    """Wraps simple_ga_optimisation.py (parallel island-based GA)."""

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
        from src.simple_ga_optimisation import (
            genetic_algorithm, optimise_weights,
        )
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
        from src import optimisation as opt_mod

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


class MonteCarloAdapter(OptimiserAdapter):
    """Wraps monte_carlo_optimisation.py (random search)."""

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
    """Wraps mip_optimisation.py (Mixed Integer Linear Programming)."""

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
        from src.mip_optimisation import portfolio_sharpe_ratio

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


class CppGAAdapter(OptimiserAdapter):
    """Wraps the compiled C++ island GA binary (./optimisation)."""

    name = "Island GA (C++)"

    def __init__(self, binary_path='./cpp/optimisation'):
        self.binary_path = binary_path

    def run(self, data: pd.DataFrame, time_budget: float,
            seed: int, run_id: int) -> BenchmarkResult:
        start_time = time.time()
        convergence = []
        best_so_far = float('-inf')
        pattern = re.compile(
            r'Island\s+(\d+):\s+Generation\s+(\d+):\s+Best fitness\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
        )

        if not os.path.isfile(self.binary_path):
            return BenchmarkResult(
                algorithm=self.name, seed=seed, time_budget=time_budget,
                convergence=[], best_fitness=float('-inf'),
                metadata={'error': f'Binary not found: {self.binary_path}'},
            )

        try:
            proc = subprocess.Popen(
                [self.binary_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )

            for line in proc.stdout:
                elapsed = time.time() - start_time
                match = pattern.match(line.strip())
                if match:
                    island_id = int(match.group(1))
                    gen = int(match.group(2))
                    fitness = float(match.group(3))
                    if fitness > best_so_far:
                        best_so_far = fitness
                    convergence.append(ConvergenceRecord(
                        wall_clock_seconds=elapsed,
                        function_evaluations=(gen + 1) * 1000,  # approx
                        best_fitness=best_so_far,
                        mean_fitness=fitness,
                        generation=gen,
                    ))

                if elapsed > time_budget:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

            proc.wait(timeout=10)

        except Exception as e:
            return BenchmarkResult(
                algorithm=self.name, seed=seed, time_budget=time_budget,
                convergence=convergence, best_fitness=best_so_far,
                metadata={'error': str(e)},
            )

        elapsed = time.time() - start_time
        return BenchmarkResult(
            algorithm=self.name,
            seed=seed,
            time_budget=time_budget,
            convergence=convergence,
            best_fitness=best_so_far,
            timed_out=elapsed >= time_budget,
            metadata={
                'note': 'C++ binary uses hardcoded paths (Data/ETF_Prices.csv)',
            },
        )


# Registry of all available adapters
ALL_ADAPTERS = {
    'Island GA (Python)': SimpleGAAdapter,
    'Pygad GA': PygadGAAdapter,
    'Monte Carlo': MonteCarloAdapter,
    'MILP': MIPAdapter,
    'Island GA (C++)': CppGAAdapter,
}

DEFAULT_ADAPTERS = ['Island GA (Python)', 'Pygad GA', 'Monte Carlo', 'MILP']
