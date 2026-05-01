"""Unit tests for src/backtest/cpcv.py."""

import math
import unittest

import numpy as np
import pandas as pd

from src.backtest.cpcv import (
    CpcvSplit,
    _split_into_groups,
    compute_pbo,
    generate_cpcv_splits,
    summarise_method_across_splits,
)


def _make_dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n, freq="B")


class TestSplitIntoGroups(unittest.TestCase):

    def test_evenly_divisible(self):
        dates = _make_dates(100)
        groups = _split_into_groups(dates, n_groups=10)
        self.assertEqual(len(groups), 10)
        for g in groups:
            self.assertEqual(len(g), 10)
        # Concatenated, should reconstruct the original.
        rebuilt = pd.DatetimeIndex(np.concatenate([g.values for g in groups]))
        self.assertTrue(rebuilt.equals(dates))

    def test_uneven_distributes_remainders_to_first_groups(self):
        # 13 dates / 5 groups = 2 base + 3 extra → groups [3,3,3,2,2].
        dates = _make_dates(13)
        groups = _split_into_groups(dates, n_groups=5)
        self.assertEqual([len(g) for g in groups], [3, 3, 3, 2, 2])

    def test_groups_are_contiguous_and_in_order(self):
        dates = _make_dates(50)
        groups = _split_into_groups(dates, n_groups=5)
        for i in range(1, len(groups)):
            self.assertGreater(groups[i][0], groups[i - 1][-1])

    def test_too_few_groups_raises(self):
        with self.assertRaises(ValueError):
            _split_into_groups(_make_dates(10), n_groups=1)

    def test_too_many_groups_raises(self):
        with self.assertRaises(ValueError):
            _split_into_groups(_make_dates(5), n_groups=10)


class TestGenerateCpcvSplits(unittest.TestCase):

    def test_correct_number_of_splits(self):
        dates = _make_dates(100)
        splits = generate_cpcv_splits(dates, n_groups=6, k_test_groups=2,
                                      purge_days=0, embargo_days=0)
        self.assertEqual(len(splits), math.comb(6, 2))  # = 15

    def test_no_temporal_overlap(self):
        # Strict no-leakage: train and test must be disjoint.
        dates = _make_dates(60)
        splits = generate_cpcv_splits(dates, n_groups=6, k_test_groups=2,
                                      purge_days=2, embargo_days=2)
        for split in splits:
            train = set(split.train_dates)
            test = set(split.test_dates)
            self.assertEqual(train & test, set(),
                f"Overlap found in split {split.label}")

    def test_purge_removes_adjacent_train_dates(self):
        # If we purge=3, then the 3 train dates immediately before any test
        # group must NOT appear in train.
        dates = _make_dates(60)
        splits = generate_cpcv_splits(dates, n_groups=6, k_test_groups=1,
                                      purge_days=3, embargo_days=0)
        for split in splits:
            test_positions = np.flatnonzero(dates.isin(split.test_dates))
            test_first = test_positions.min()
            test_last = test_positions.max()
            train_positions = np.flatnonzero(dates.isin(split.train_dates))
            # The 3 positions before test_first should NOT be in train.
            for offset in (1, 2, 3):
                purged = test_first - offset
                if purged >= 0:
                    self.assertNotIn(purged, train_positions,
                        f"purge_days=3 failed: pos {purged} in train of {split.label}")
            # And 3 positions after test_last (purge, no embargo).
            for offset in (1, 2, 3):
                purged = test_last + offset
                if purged < len(dates):
                    self.assertNotIn(purged, train_positions,
                        f"purge_days=3 failed: pos {purged} in train of {split.label}")

    def test_embargo_only_extends_right_side(self):
        # purge=0, embargo=5: only forward-looking positions removed.
        dates = _make_dates(60)
        splits = generate_cpcv_splits(dates, n_groups=6, k_test_groups=1,
                                      purge_days=0, embargo_days=5)
        for split in splits:
            test_positions = np.flatnonzero(dates.isin(split.test_dates))
            train_positions = np.flatnonzero(dates.isin(split.train_dates))
            test_first = test_positions.min()
            test_last = test_positions.max()
            # Position immediately before test (test_first - 1) IS in train.
            if test_first > 0:
                self.assertIn(test_first - 1, train_positions,
                    f"with purge=0, position before test should be train: "
                    f"split {split.label}")
            # Position 5 after test_last NOT in train (embargo).
            if test_last + 5 < len(dates):
                self.assertNotIn(test_last + 5, train_positions,
                    f"embargo=5 failed for {split.label}")

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            generate_cpcv_splits(_make_dates(60), n_groups=6, k_test_groups=0)
        with self.assertRaises(ValueError):
            generate_cpcv_splits(_make_dates(60), n_groups=6, k_test_groups=6)

    def test_invalid_purge(self):
        with self.assertRaises(ValueError):
            generate_cpcv_splits(_make_dates(60), n_groups=6,
                                  k_test_groups=2, purge_days=-1)


class TestComputePbo(unittest.TestCase):

    def test_robust_strategy_low_pbo(self):
        # Strategy 0 always best in both IS and OOS → PBO = 0.
        n_splits, n_strategies = 20, 5
        rng = np.random.default_rng(0)
        is_sharpes = rng.normal(size=(n_splits, n_strategies))
        oos_sharpes = rng.normal(size=(n_splits, n_strategies))
        # Force strategy 0 to dominate
        is_sharpes[:, 0] = 5.0
        oos_sharpes[:, 0] = 5.0
        pbo = compute_pbo(is_sharpes, oos_sharpes)
        self.assertEqual(pbo, 0.0)

    def test_overfit_strategy_high_pbo(self):
        # Strategy 0 best IS but consistently worst OOS → PBO = 1.
        n_splits, n_strategies = 20, 5
        rng = np.random.default_rng(1)
        is_sharpes = rng.normal(size=(n_splits, n_strategies))
        oos_sharpes = rng.normal(size=(n_splits, n_strategies))
        is_sharpes[:, 0] = 10.0   # IS-best every split
        oos_sharpes[:, 0] = -10.0  # OOS-worst every split
        pbo = compute_pbo(is_sharpes, oos_sharpes)
        self.assertEqual(pbo, 1.0)

    def test_random_strategy_pbo_around_half(self):
        # Independent IS and OOS Sharpes → IS-best is no better than
        # random pick OOS. With even n_strategies the theoretical null
        # PBO is exactly 0.5; with odd n it's floor(n/2)/n. Use n=6 so
        # the test target is unambiguous.
        n_splits, n_strategies = 1000, 6
        rng = np.random.default_rng(2)
        is_sharpes = rng.normal(size=(n_splits, n_strategies))
        oos_sharpes = rng.normal(size=(n_splits, n_strategies))
        pbo = compute_pbo(is_sharpes, oos_sharpes)
        # Standard error of mean(Bernoulli(0.5), n=1000) ≈ 0.016, so
        # ±0.05 (~3σ) is a safe band.
        self.assertGreater(pbo, 0.45)
        self.assertLess(pbo, 0.55)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_pbo([[1, 2]], [[1, 2, 3]])


class TestSummariseMethodAcrossSplits(unittest.TestCase):

    def test_basic_aggregation(self):
        is_sharpes = [1.5, 1.7, 1.4]
        oos_sharpes = [1.0, 0.9, 1.1]
        s = summarise_method_across_splits('cc_optimised',
                                            is_sharpes, oos_sharpes)
        self.assertEqual(s.method, 'cc_optimised')
        self.assertAlmostEqual(s.mean_oos, 1.0, places=10)
        self.assertGreater(s.std_oos, 0.0)
        self.assertLess(s.ci95_oos_low, s.mean_oos)
        self.assertGreater(s.ci95_oos_high, s.mean_oos)


if __name__ == '__main__':
    unittest.main()
