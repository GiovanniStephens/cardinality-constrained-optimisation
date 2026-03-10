"""Data structures for benchmark results."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ConvergenceRecord:
    """A single snapshot of optimiser progress."""
    wall_clock_seconds: float
    function_evaluations: int
    best_fitness: float
    mean_fitness: float
    generation: int


@dataclass
class BenchmarkResult:
    """Result of a single algorithm run."""
    algorithm: str
    seed: int
    time_budget: float
    convergence: List[ConvergenceRecord]
    best_fitness: float
    selected_etfs: Optional[List[str]] = None
    optimised_weights: Optional[np.ndarray] = None
    timed_out: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of all benchmark results, keyed by algorithm name."""
    results: Dict[str, List[BenchmarkResult]] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult):
        if result.algorithm not in self.results:
            self.results[result.algorithm] = []
        self.results[result.algorithm].append(result)

    @property
    def algorithms(self) -> List[str]:
        return list(self.results.keys())

    def best_fitness_per_algorithm(self) -> Dict[str, float]:
        return {
            algo: max(r.best_fitness for r in runs)
            for algo, runs in self.results.items()
        }

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        out = {}
        for algo, runs in self.results.items():
            out[algo] = []
            for r in runs:
                out[algo].append({
                    'seed': r.seed,
                    'time_budget': r.time_budget,
                    'best_fitness': r.best_fitness,
                    'timed_out': r.timed_out,
                    'num_convergence_points': len(r.convergence),
                    'selected_etfs': r.selected_etfs,
                    'metadata': r.metadata,
                    'convergence': [
                        {
                            'wall_clock_seconds': c.wall_clock_seconds,
                            'function_evaluations': c.function_evaluations,
                            'best_fitness': c.best_fitness,
                            'mean_fitness': c.mean_fitness,
                            'generation': c.generation,
                        }
                        for c in r.convergence
                    ],
                })
        return out
