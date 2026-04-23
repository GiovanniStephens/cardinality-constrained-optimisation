"""Abstract base class for all portfolio optimisers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from src.config import NUMERICAL_TOLERANCE


@dataclass
class OptimisationResult:
    """Standard output from any optimiser.

    WARNING: sharpe_ratio is an in-sample value computed on the training data.
    It is biased upward by selection bias — typical IS-to-OOS degradation is
    30-50%. See CLAUDE.md "Sharpe Ratio Overfitting" for details.
    """
    selected_tickers: List[str]
    weights: np.ndarray
    sharpe_ratio: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate result integrity (skip for empty-result sentinels)."""
        if len(self.weights) == 0:
            return
        if len(self.weights) != len(self.selected_tickers):
            raise ValueError(
                f"weights length ({len(self.weights)}) != "
                f"selected_tickers length ({len(self.selected_tickers)})"
            )
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contain NaN or inf values")
        if np.any(self.weights < -NUMERICAL_TOLERANCE):
            raise ValueError("weights contain negative values")
        if not np.isfinite(self.sharpe_ratio):
            raise ValueError(f"sharpe_ratio is not finite: {self.sharpe_ratio}")


class BaseOptimiser(ABC):
    """Interface that all optimisers must implement.

    Each optimiser selects a subset of instruments from a price DataFrame
    and determines portfolio weights that maximise the Sharpe ratio.
    """

    @abstractmethod
    def optimise(self, prices: pd.DataFrame) -> OptimisationResult:
        """Run optimisation on the given price data.

        Args:
            prices: DataFrame with dates as index, tickers as columns.

        Returns:
            OptimisationResult with selected tickers, weights, Sharpe ratio,
            and method-specific metadata.

        WARNING: The sharpe_ratio in OptimisationResult is an IN-SAMPLE value
        computed on the training data passed to optimise(). It is biased upward
        due to selection bias and should NOT be interpreted as expected OOS
        performance. Typical degradation is 30-50%. Use backtest.py for OOS
        validation. See CLAUDE.md "Sharpe Ratio Overfitting" section.
        """
        ...
