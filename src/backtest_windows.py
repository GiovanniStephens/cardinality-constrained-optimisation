"""Window generation and aggregation for forward-walk backtesting."""

import logging
from typing import List

import numpy as np
import pandas as pd

from src.backtest_types import WindowSpec
from src.config import (
    BACKTEST_TRAIN_YEARS,
    BACKTEST_TEST_DAYS,
    BACKTEST_STEP_DAYS,
    TRADING_DAYS_PER_YEAR,
)
from src.returns import calculate_log_returns

logger = logging.getLogger(__name__)


def generate_windows(
    date_index: pd.DatetimeIndex,
    train_days: int = BACKTEST_TRAIN_YEARS * TRADING_DAYS_PER_YEAR,
    test_days: int = BACKTEST_TEST_DAYS,
    step_days: int = BACKTEST_STEP_DAYS,
) -> List[WindowSpec]:
    """
    Generate non-overlapping rolling forward-walk windows from a date index.

    :param date_index: sorted DatetimeIndex of trading days.
    :param train_days: number of trading days for training.
    :param test_days: number of trading days for OOS testing.
    :param step_days: step size in trading days between windows.
    :return: list of WindowSpec objects.
    """
    dates = date_index.sort_values()
    n = len(dates)
    min_required = train_days + test_days
    if n < min_required:
        raise ValueError(
            f"Need at least {min_required} trading days, got {n}"
        )

    windows = []
    start = 0
    while start + min_required <= n:
        train_start = dates[start]
        train_end = dates[start + train_days - 1]
        test_start = dates[start + train_days]
        test_end_idx = min(start + train_days + test_days - 1, n - 1)
        test_end = dates[test_end_idx]

        label = f"{train_start.year}-{train_end.year}/{test_start.year}"
        windows.append(WindowSpec(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            label=label,
        ))
        start += step_days

    return windows


def slice_window_data(window, full_prices):
    """Slice full prices into train/test sets and compute OOS log returns.

    :param window: WindowSpec defining train/test boundaries.
    :param full_prices: complete price DataFrame.
    :return: (train_prices, oos_log_returns) tuple.
    """
    train_prices = full_prices.loc[window.train_start:window.train_end]
    test_prices = full_prices.loc[window.test_start:window.test_end]
    assert train_prices.index.max() < test_prices.index.min(), (
        f"Window {window.label}: train data ends at {train_prices.index.max()} "
        f"but test data starts at {test_prices.index.min()}. "
        f"This would leak test-period data into training."
    )
    boundary_price = train_prices.iloc[[-1]]
    test_with_boundary = pd.concat([boundary_price, test_prices])
    oos_log_returns = calculate_log_returns(test_with_boundary).iloc[1:]
    logger.info(
        "  Window %s: train=%d rows, test=%d rows, %d tickers",
        window.label, len(train_prices), len(test_prices),
        train_prices.shape[1],
    )
    return train_prices, oos_log_returns


def aggregate_cross_window(all_results):
    """
    Build a summary table of mean Sharpe per method per window.

    :param all_results: list of WindowResult objects.
    :return: DataFrame with methods as rows, windows + mean + std as columns.
    """
    data = {}
    all_categories = set()
    for wr in all_results:
        for cat in wr.method_results:
            all_categories.add(cat)

    for cat in sorted(all_categories):
        row = {}
        values = []
        for wr in all_results:
            if cat in wr.method_results:
                val = wr.method_results[cat].mean_sharpe
                row[wr.window.label] = val
                values.append(val)
            else:
                row[wr.window.label] = np.nan
        row['mean'] = np.nanmean(values) if values else np.nan
        row['std'] = np.nanstd(values) if values else np.nan
        data[cat] = row

    return pd.DataFrame(data).T
