"""Statistical testing and cross-window aggregation for backtests."""

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel

from .types import WindowResult


def difference_of_means_hypothesis_test(sample_1, sample_2):
    """
    Calculates the t statistic for the difference of means.

    Second sample mean minus the first. (i.e. if positive,
    the second is greater than the first.)

    :sample_1: The first sample. List of floats.
    :sample_2: The second sample. List of floats.
    :return: The t statistic.
    """
    if not sample_1 or not sample_2:
        raise ValueError("Both samples must be non-empty")
    denominator = np.sqrt(
        np.var(sample_1) / len(sample_1) + np.var(sample_2) / len(sample_2)
    )
    if denominator == 0:
        raise ValueError(
            "t-statistic is undefined: both samples have zero variance"
        )
    return (np.mean(sample_2) - np.mean(sample_1)) / denominator


def paired_t_test(sharpes_a, sharpes_b):
    """
    Paired t-test across windows (same window, different methods).

    Controls for market-regime effects by pairing observations from
    the same OOS period.

    :param sharpes_a: dict {window_label: mean_sharpe} for method A.
    :param sharpes_b: dict {window_label: mean_sharpe} for method B.
    :return: (t_statistic, p_value) tuple.
    """
    common = sorted(set(sharpes_a) & set(sharpes_b))
    if len(common) < 2:
        raise ValueError(
            f"Need at least 2 common windows for paired test, got {len(common)}"
        )
    a = [sharpes_a[w] for w in common]
    b = [sharpes_b[w] for w in common]
    # ttest_rel computes first - second; swap so positive t = b > a
    return ttest_rel(b, a)


def friedman_test(all_results: List[WindowResult], categories):
    """
    Non-parametric Friedman test for comparing K methods across W windows.

    :param all_results: list of WindowResult objects.
    :param categories: list of category names to compare.
    :return: (chi2_statistic, p_value) tuple.
    """
    # Build matrix: one column per method, one row per window
    # Value = mean Sharpe of that method in that window
    columns = {}
    for cat in categories:
        values = []
        for wr in all_results:
            if cat in wr.method_results:
                values.append(wr.method_results[cat].mean_sharpe)
        columns[cat] = values

    # All methods must have the same number of windows
    n_windows = len(all_results)
    valid_cats = [c for c in categories if len(columns.get(c, [])) == n_windows]
    if len(valid_cats) < 3:
        raise ValueError(
            f"Friedman test requires >= 3 methods present in all windows, "
            f"got {len(valid_cats)}"
        )
    arrays = [columns[c] for c in valid_cats]
    return friedmanchisquare(*arrays)


def aggregate_cross_window(all_results: List[WindowResult]):
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
