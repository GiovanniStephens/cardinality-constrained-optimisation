"""Benchmarking framework for portfolio optimisation algorithms."""

from benchmark.results import BenchmarkResult, BenchmarkSuite, ConvergenceRecord
from benchmark.adapters import (
    ALL_ADAPTERS,
    DEFAULT_ADAPTERS,
    CppGAAdapter,
    MIPAdapter,
    MonteCarloAdapter,
    OptimiserAdapter,
    PygadGAAdapter,
    SimpleGAAdapter,
)
from benchmark.runner import BenchmarkRunner, load_suite
from benchmark.analysis import generate_full_report, summary_table

__all__ = [
    'BenchmarkResult', 'BenchmarkSuite', 'ConvergenceRecord',
    'OptimiserAdapter', 'SimpleGAAdapter', 'PygadGAAdapter',
    'MonteCarloAdapter', 'MIPAdapter', 'CppGAAdapter',
    'ALL_ADAPTERS', 'DEFAULT_ADAPTERS',
    'BenchmarkRunner', 'load_suite',
    'generate_full_report', 'summary_table',
]
