"""Abstract base class for all portfolio optimisers."""

from abc import ABC, abstractmethod

import pandas as pd

from src.portfolio_utils import OptimisationResult


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
