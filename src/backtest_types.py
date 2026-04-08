"""Data structures used by the backtest module."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class WindowSpec:
    """Defines a single train/test window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    label: str


@dataclass
class PortfolioResult:
    """Result for a single portfolio within a single window and method."""
    portfolio: List[str]
    weights: np.ndarray
    metrics: Dict[str, float]
    is_sharpe: Optional[float] = None  # in-sample Sharpe (biased upward)


@dataclass
class MethodResults:
    """All portfolio results for one method in one window."""
    category: str
    portfolios: List[PortfolioResult] = field(default_factory=list)

    @property
    def sharpe_ratios(self) -> np.ndarray:
        return np.array([p.metrics['sharpe_ratio'] for p in self.portfolios])

    @property
    def mean_sharpe(self) -> float:
        return float(self.sharpe_ratios.mean())


@dataclass
class WindowResult:
    """All method results for one window."""
    window: WindowSpec
    method_results: Dict[str, MethodResults] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
