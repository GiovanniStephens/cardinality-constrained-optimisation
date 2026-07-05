"""Tests for src/liquidity.py — the rebalance liquidity/tradeability filter."""

import numpy as np
import pandas as pd

from src import db
from src.liquidity import filter_by_liquidity
from tests.helpers import BaseDBTest


class TestFilterByLiquidity(BaseDBTest):
    """Seed a tiny US universe with mixed listing/liquidity, then filter it.

    Columns:
      LIQUID     US-listed, $100M ADV  -> always kept
      THIN       US-listed, $10k ADV   -> dropped by the ADV floor
      NOVOL      US-listed, no volume  -> unknown ADV, dropped (tradeability-safe)
      329660.KS  foreign (dot), no vol -> dropped as a foreign listing
      PROT.KS    foreign (dot), no vol -> dropped unless protected
    """

    def setUp(self):
        super().setUp()
        dates = pd.bdate_range('2024-01-01', periods=130, freq='B')
        closes = {'LIQUID': 100.0, 'THIN': 10.0, 'NOVOL': 50.0,
                  '329660.KS': 20.0, 'PROT.KS': 30.0}
        vols = {'LIQUID': 1_000_000, 'THIN': 1_000, 'NOVOL': np.nan,
                '329660.KS': np.nan, 'PROT.KS': np.nan}
        self.symbols = list(closes)
        self.prices = pd.DataFrame({s: closes[s] for s in self.symbols}, index=dates)
        volumes_df = pd.DataFrame({s: vols[s] for s in self.symbols}, index=dates)
        db.save_prices(self.conn, self.prices, exchange='US', asset_type='etf',
                       volumes_df=volumes_df)

    def test_keeps_only_liquid_us_listing(self):
        # LIQUID ($100M ADV) survives; THIN (below floor), NOVOL (unknown ADV),
        # and both foreign listings are removed.
        out = filter_by_liquidity(self.prices, self.conn, 1_000_000)
        self.assertEqual(list(out.columns), ['LIQUID'])

    def test_unknown_adv_is_dropped(self):
        out = filter_by_liquidity(self.prices, self.conn, 1_000_000)
        self.assertNotIn('NOVOL', out.columns)

    def test_below_floor_is_dropped(self):
        out = filter_by_liquidity(self.prices, self.conn, 1_000_000)
        self.assertNotIn('THIN', out.columns)

    def test_protect_keeps_foreign_and_illiquid(self):
        out = filter_by_liquidity(self.prices, self.conn, 1_000_000,
                                  protect=['PROT.KS'])
        self.assertIn('PROT.KS', out.columns)   # protected despite foreign + no vol
        self.assertIn('LIQUID', out.columns)
        self.assertNotIn('329660.KS', out.columns)  # unprotected foreign still dropped

    def test_zero_floor_is_suffix_only(self):
        # min_adv=0 disables the ADV stage: keep all US (no-dot) names, drop foreign.
        out = filter_by_liquidity(self.prices, self.conn, 0)
        self.assertEqual(set(out.columns), {'LIQUID', 'THIN', 'NOVOL'})

    def test_exclude_foreign_false_and_no_floor_keeps_all(self):
        out = filter_by_liquidity(self.prices, self.conn, 0, exclude_foreign=False)
        self.assertEqual(set(out.columns), set(self.symbols))

    def test_preserves_column_order(self):
        out = filter_by_liquidity(self.prices, self.conn, 0)
        self.assertEqual(list(out.columns),
                         [c for c in self.prices.columns if '.' not in c])


class TestMinHistoryAdvisoryFlags(BaseDBTest):
    """allow_min_history_flags keeps min_history-flagged tickers (advisory for
    the production 2y admission) while hard flags still exclude."""

    def setUp(self):
        super().setUp()
        dates = pd.bdate_range('2024-01-01', periods=60, freq='B')
        prices = pd.DataFrame({'CLEAN': 100.0, 'YOUNG': 50.0, 'BAD': 10.0},
                              index=dates)
        db.save_prices(self.conn, prices, exchange='US', asset_type='etf')
        self.conn.execute("UPDATE tickers SET excluded='min_history:500_days' "
                          "WHERE symbol='YOUNG'")
        self.conn.execute("UPDATE tickers SET excluded='frozen_price:run=30_days' "
                          "WHERE symbol='BAD'")
        self.conn.commit()

    def test_default_excludes_all_flagged(self):
        out = db.load_prices(self.conn, exchange='US', min_coverage=0)
        self.assertEqual(set(out.columns), {'CLEAN'})

    def test_advisory_keeps_min_history_only(self):
        out = db.load_prices(self.conn, exchange='US', min_coverage=0,
                             allow_min_history_flags=True)
        self.assertEqual(set(out.columns), {'CLEAN', 'YOUNG'})  # BAD still out
