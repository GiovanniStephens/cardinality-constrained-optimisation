"""Tests for data quality validation checks."""

import unittest

import numpy as np
import pandas as pd

from src import db
from src import data_quality as dq


class _BaseQualityTest(unittest.TestCase):
    """Shared setup: in-memory DB with exchange seeded."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')
        self.exchange_id = db._get_exchange_id(self.conn, 'US')

    def tearDown(self):
        self.conn.close()

    def _save(self, symbol, prices, asset_type='stock'):
        dates = pd.date_range('2020-01-01', periods=len(prices), freq='B')
        df = pd.DataFrame({symbol: prices}, index=dates)
        db.save_prices(self.conn, df, exchange='US', asset_type=asset_type)


class TestMinHistory(_BaseQualityTest):

    def test_short_history_flagged(self):
        self._save('SHORT', [100.0 + i for i in range(50)])
        result = dq._check_min_history(self.conn, self.exchange_id, min_days=1260)
        flagged_ids = [r[0] for r in result]
        ticker_id = self.conn.execute(
            "SELECT id FROM tickers WHERE symbol='SHORT'").fetchone()[0]
        self.assertIn(ticker_id, flagged_ids)

    def test_sufficient_history_not_flagged(self):
        self._save('LONG', [100.0 + i * 0.01 for i in range(1300)])
        result = dq._check_min_history(self.conn, self.exchange_id, min_days=1260)
        flagged_ids = [r[0] for r in result]
        ticker_id = self.conn.execute(
            "SELECT id FROM tickers WHERE symbol='LONG'").fetchone()[0]
        self.assertNotIn(ticker_id, flagged_ids)


class TestStalePrices(_BaseQualityTest):

    def test_stale_ticker_flagged(self):
        # FRESH: prices up to recently
        self._save('FRESH', [100.0 + i for i in range(1300)])
        # STALE: only 100 days of prices (ends ~2020-05)
        self._save('STALE', [50.0 + i for i in range(100)])
        result = dq._check_stale_prices(self.conn, self.exchange_id,
                                         max_staleness_days=30)
        flagged_symbols = set()
        for tid, reason in result:
            row = self.conn.execute(
                "SELECT symbol FROM tickers WHERE id=?", (tid,)).fetchone()
            flagged_symbols.add(row['symbol'])
        self.assertIn('STALE', flagged_symbols)
        self.assertNotIn('FRESH', flagged_symbols)


class TestZeroVariance(_BaseQualityTest):

    def test_constant_price_flagged(self):
        self._save('FLAT', [100.0] * 300)
        pbt = dq._load_prices_by_ticker(self.conn, self.exchange_id)
        result = dq._check_zero_variance(pbt, min_annual_vol=0.001)
        self.assertTrue(len(result) > 0)
        self.assertIn('zero_variance', result[0][1])

    def test_volatile_ticker_not_flagged(self):
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.02, 300)))
        self._save('VOLATILE', prices.tolist())
        pbt = dq._load_prices_by_ticker(self.conn, self.exchange_id)
        result = dq._check_zero_variance(pbt, min_annual_vol=0.001)
        flagged_ids = [r[0] for r in result]
        ticker_id = self.conn.execute(
            "SELECT id FROM tickers WHERE symbol='VOLATILE'").fetchone()[0]
        self.assertNotIn(ticker_id, flagged_ids)


class TestFrozenPrices(_BaseQualityTest):

    def test_frozen_run_flagged(self):
        # 25 days of identical prices in the middle
        prices = [100.0 + i for i in range(100)]
        prices[40:65] = [150.0] * 25
        self._save('FROZEN', prices)
        pbt = dq._load_prices_by_ticker(self.conn, self.exchange_id)
        result = dq._check_frozen_prices(pbt, max_consecutive_same=20)
        self.assertTrue(len(result) > 0)
        self.assertIn('frozen_price', result[0][1])

    def test_normal_ticker_not_flagged(self):
        prices = [100.0 + i * 0.5 for i in range(100)]
        self._save('NORMAL', prices)
        pbt = dq._load_prices_by_ticker(self.conn, self.exchange_id)
        result = dq._check_frozen_prices(pbt, max_consecutive_same=20)
        flagged_ids = [r[0] for r in result]
        ticker_id = self.conn.execute(
            "SELECT id FROM tickers WHERE symbol='NORMAL'").fetchone()[0]
        self.assertNotIn(ticker_id, flagged_ids)


class TestExtremeReturns(_BaseQualityTest):

    def test_spiky_ticker_flagged(self):
        np.random.seed(42)
        # Normal prices with 8% of days having extreme jumps (>10 std of normal days)
        # Normal daily vol ~0.01, so 10x = 0.10. We inject jumps of 0.50.
        n = 500
        normal_rets = np.random.normal(0, 0.01, n - 1)
        # Inject extreme returns into 8% of days
        spike_indices = np.random.choice(n - 1, size=int(0.08 * n), replace=False)
        normal_rets[spike_indices] = np.random.choice([-0.5, 0.5], size=len(spike_indices))
        prices = [100.0]
        for r in normal_rets:
            prices.append(prices[-1] * np.exp(r))
        self._save('SPIKY', prices)
        pbt = dq._load_prices_by_ticker(self.conn, self.exchange_id)
        result = dq._check_extreme_returns(pbt, max_extreme_pct=0.05)
        flagged_ids = [r[0] for r in result]
        ticker_id = self.conn.execute(
            "SELECT id FROM tickers WHERE symbol='SPIKY'").fetchone()[0]
        self.assertIn(ticker_id, flagged_ids)

    def test_normal_ticker_not_flagged(self):
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
        self._save('CALM', prices.tolist())
        pbt = dq._load_prices_by_ticker(self.conn, self.exchange_id)
        result = dq._check_extreme_returns(pbt, max_extreme_pct=0.05)
        flagged_ids = [r[0] for r in result]
        ticker_id = self.conn.execute(
            "SELECT id FROM tickers WHERE symbol='CALM'").fetchone()[0]
        self.assertNotIn(ticker_id, flagged_ids)


class TestLongestConstantRun(unittest.TestCase):

    def test_no_run(self):
        self.assertEqual(dq._longest_constant_run([1, 2, 3, 4]), 1)

    def test_full_run(self):
        self.assertEqual(dq._longest_constant_run([5, 5, 5, 5]), 4)

    def test_run_in_middle(self):
        self.assertEqual(dq._longest_constant_run([1, 2, 2, 2, 3]), 3)

    def test_empty(self):
        self.assertEqual(dq._longest_constant_run([]), 0)


class TestValidateUniverse(_BaseQualityTest):

    def test_dry_run_does_not_modify_db(self):
        self._save('FLAT', [100.0] * 300)
        summary = dq.validate_universe(self.conn, exchange='US', dry_run=True)
        self.assertGreater(summary['total_excluded'], 0)
        # Should NOT be written to DB
        excluded = db.get_excluded_tickers(self.conn, exchange='US')
        self.assertEqual(len(excluded), 0)

    def test_validate_flags_bad_ticker(self):
        self._save('FLAT', [100.0] * 300)
        summary = dq.validate_universe(self.conn, exchange='US', dry_run=False)
        self.assertGreater(summary['total_excluded'], 0)
        excluded = db.get_excluded_tickers(self.conn, exchange='US')
        self.assertGreater(len(excluded), 0)

    def test_validate_clears_previous_exclusions(self):
        self._save('FLAT', [100.0] * 300)
        # First run: flag it
        dq.validate_universe(self.conn, exchange='US')
        excluded_before = db.get_excluded_tickers(self.conn, exchange='US')
        self.assertGreater(len(excluded_before), 0)
        # Change the data to be good, re-validate
        self.conn.execute("DELETE FROM prices")
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.02, 1300)))
        dates = pd.date_range('2020-01-01', periods=1300, freq='B')
        df = pd.DataFrame({'FLAT': prices}, index=dates)
        db.save_prices(self.conn, df, exchange='US', asset_type='stock')
        dq.validate_universe(self.conn, exchange='US')
        excluded_after = db.get_excluded_tickers(self.conn, exchange='US')
        self.assertEqual(len(excluded_after), 0)


class TestLoadPricesExcludeFlagged(_BaseQualityTest):

    def test_excluded_ticker_not_loaded(self):
        self._save('GOOD', [100.0 + i for i in range(50)])
        self._save('BAD', [200.0 + i for i in range(50)])
        # Flag BAD
        tid = self.conn.execute(
            "SELECT id FROM tickers WHERE symbol='BAD'").fetchone()[0]
        db.set_ticker_excluded(self.conn, tid, 'test_reason')
        self.conn.commit()
        result = db.load_prices(self.conn, exchange='US', exclude_flagged=True)
        self.assertIn('GOOD', result.columns)
        self.assertNotIn('BAD', result.columns)

    def test_excluded_ticker_loaded_when_flag_off(self):
        self._save('GOOD', [100.0 + i for i in range(50)])
        self._save('BAD', [200.0 + i for i in range(50)])
        tid = self.conn.execute(
            "SELECT id FROM tickers WHERE symbol='BAD'").fetchone()[0]
        db.set_ticker_excluded(self.conn, tid, 'test_reason')
        self.conn.commit()
        result = db.load_prices(self.conn, exchange='US', exclude_flagged=False,
                                min_coverage=0)
        self.assertIn('GOOD', result.columns)
        self.assertIn('BAD', result.columns)


if __name__ == '__main__':
    unittest.main()
