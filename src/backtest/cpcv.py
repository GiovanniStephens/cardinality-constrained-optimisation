"""Combinatorially Purged Cross-Validation (López de Prado, 2018).

Generates many (train, test) date partitions from a single time series:
the data is split into ``N`` non-overlapping groups, every choice of
``k_test`` groups becomes a test set, the remaining ``N − k_test`` form
train. **Purging** drops train observations within an autocorrelation
window of any test boundary; **embargo** drops train observations a
short period after each test group, preventing forward-looking leakage.

Used together with the **Probability of Backtest Overfitting (PBO)**:
the fraction of CPCV splits where the in-sample-best strategy ranks
*below median* on its OOS half. PBO ≥ 0.5 → systematic overfitting.

Reference: López de Prado, M. (2018), *Advances in Financial Machine
Learning*, Wiley, Chapter 12.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CpcvSplit:
    """One purged train/test partition.

    :param train_dates: dates assigned to training (after purge+embargo).
    :param test_dates: dates assigned to OOS evaluation.
    :param test_group_ids: indices of the test groups (in [0, N)).
    :param label: human-readable identifier.
    """
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    test_group_ids: tuple
    label: str


def _split_into_groups(date_index: pd.DatetimeIndex,
                       n_groups: int) -> List[pd.DatetimeIndex]:
    """Partition ``date_index`` into ``n_groups`` contiguous DatetimeIndexes.

    The first ``n_extra`` groups receive one extra date when the total
    isn't evenly divisible. All groups remain in chronological order.
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    n = len(date_index)
    if n_groups > n:
        raise ValueError(
            f"n_groups ({n_groups}) cannot exceed number of dates ({n})")
    base, extra = divmod(n, n_groups)
    groups = []
    start = 0
    for i in range(n_groups):
        size = base + (1 if i < extra else 0)
        groups.append(date_index[start:start + size])
        start += size
    return groups


def generate_cpcv_splits(
    date_index: pd.DatetimeIndex,
    n_groups: int,
    k_test_groups: int = 2,
    purge_days: int = 5,
    embargo_days: int = 5,
) -> List[CpcvSplit]:
    """Yield all ``C(n_groups, k_test_groups)`` purged train/test splits.

    For each combination of ``k_test_groups`` chosen as the test set:
      1. Mark those dates as test.
      2. From the remaining (train) dates, **purge** any date within
         ``purge_days`` of a test boundary.
      3. **Embargo** ``embargo_days`` after each test group.
      4. Return the surviving train dates and the full test dates.

    :param date_index: sorted DatetimeIndex of trading days.
    :param n_groups: number of contiguous groups to split into.
    :param k_test_groups: how many groups to draw per test set.
    :param purge_days: purge window (trading days) on each side of every
        test group.
    :param embargo_days: embargo window AFTER each test group only.
    :return: list of :class:`CpcvSplit` of length
        ``C(n_groups, k_test_groups)``.
    """
    if k_test_groups < 1 or k_test_groups >= n_groups:
        raise ValueError(
            f"k_test_groups must be in [1, n_groups), got {k_test_groups}")
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge_days and embargo_days must be >= 0")
    dates = pd.DatetimeIndex(date_index).sort_values()
    groups = _split_into_groups(dates, n_groups)

    splits: List[CpcvSplit] = []
    all_combos = list(itertools.combinations(range(n_groups), k_test_groups))
    for combo in all_combos:
        test_groups = [groups[i] for i in combo]
        test_index = pd.DatetimeIndex(
            np.unique(np.concatenate([g.values for g in test_groups])))

        # Compute purge intervals: ±purge_days around each test group,
        # plus +embargo_days only on the right edge.
        train_mask = ~dates.isin(test_index)
        # Find positional indices of each test date in the full date index.
        # Then expand each test group's [first, last] by purge/embargo.
        test_positions = np.flatnonzero(dates.isin(test_index))
        if test_positions.size > 0:
            # Find contiguous test runs (test groups may be adjacent).
            breaks = np.where(np.diff(test_positions) > 1)[0]
            run_starts = np.concatenate([[0], breaks + 1])
            run_ends = np.concatenate([breaks, [test_positions.size - 1]])
            for s, e in zip(run_starts, run_ends):
                first = test_positions[s] - purge_days
                last = test_positions[e] + purge_days + embargo_days
                first = max(first, 0)
                last = min(last, len(dates) - 1)
                # Purge train within [first, last]
                train_mask[first:last + 1] = False

        train_index = dates[train_mask]
        label = "test=" + ",".join(str(c) for c in combo)
        splits.append(CpcvSplit(
            train_dates=train_index,
            test_dates=test_index,
            test_group_ids=combo,
            label=label,
        ))
    return splits


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting
# ---------------------------------------------------------------------------


def compute_pbo(is_sharpes_per_split: List[List[float]],
                oos_sharpes_per_split: List[List[float]]) -> float:
    """López de Prado's Probability of Backtest Overfitting.

    For each split, identify the strategy with the highest in-sample
    Sharpe; compute the rank of that strategy's OOS Sharpe within the
    same split (1 = best, len = worst). PBO is the fraction of splits
    where the IS-best strategy ranked **below the median** OOS — i.e.
    the IS leader was a worse-than-typical pick OOS.

    A strategy family that is robust will have PBO close to 0; a heavily
    overfit family will have PBO ≥ 0.5.

    :param is_sharpes_per_split: shape ``[n_splits, n_strategies]``.
    :param oos_sharpes_per_split: shape ``[n_splits, n_strategies]``.
    :return: PBO in [0, 1].
    """
    is_arr = np.asarray(is_sharpes_per_split, dtype=np.float64)
    oos_arr = np.asarray(oos_sharpes_per_split, dtype=np.float64)
    if is_arr.shape != oos_arr.shape:
        raise ValueError(
            f"shape mismatch: IS {is_arr.shape}, OOS {oos_arr.shape}")
    if is_arr.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got {is_arr.ndim}-D")
    n_splits, n_strategies = is_arr.shape
    if n_strategies < 2:
        raise ValueError("PBO requires >= 2 strategies per split")

    # In each split, the IS-best strategy index, then its OOS rank.
    is_best_idx = np.argmax(is_arr, axis=1)
    # For each split, rank OOS Sharpes (highest = rank 1).
    oos_ranks = (-oos_arr).argsort(axis=1).argsort(axis=1) + 1
    is_best_oos_rank = oos_ranks[np.arange(n_splits), is_best_idx]
    median_rank = (n_strategies + 1) / 2.0
    # PBO = fraction of splits where IS-best ranked WORSE than median OOS
    pbo = float(np.mean(is_best_oos_rank > median_rank))
    return pbo


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


@dataclass
class CpcvMethodSummary:
    """Aggregate CPCV results for one method across all splits."""
    method: str
    is_sharpes: List[float] = field(default_factory=list)   # one per split
    oos_sharpes: List[float] = field(default_factory=list)  # one per split
    pbo: Optional[float] = None
    mean_oos: Optional[float] = None
    std_oos: Optional[float] = None
    ci95_oos_low: Optional[float] = None
    ci95_oos_high: Optional[float] = None


def summarise_method_across_splits(method: str,
                                    is_sharpes: Sequence[float],
                                    oos_sharpes: Sequence[float]) -> CpcvMethodSummary:
    """Distribution of OOS Sharpe across CPCV splits for one method.

    Computes mean, std, and a 95% CI on the OOS Sharpe across splits.
    PBO is left ``None`` here — it requires the full strategy-vs-strategy
    comparison and should be computed separately via :func:`compute_pbo`.
    """
    is_arr = np.asarray(list(is_sharpes), dtype=np.float64)
    oos_arr = np.asarray(list(oos_sharpes), dtype=np.float64)
    if oos_arr.size == 0:
        return CpcvMethodSummary(method=method)
    mean = float(oos_arr.mean())
    std = float(oos_arr.std(ddof=1)) if oos_arr.size > 1 else 0.0
    n = oos_arr.size
    sem = std / np.sqrt(n) if n > 1 else 0.0
    return CpcvMethodSummary(
        method=method,
        is_sharpes=list(is_arr),
        oos_sharpes=list(oos_arr),
        mean_oos=mean,
        std_oos=std,
        ci95_oos_low=mean - 1.96 * sem,
        ci95_oos_high=mean + 1.96 * sem,
    )
