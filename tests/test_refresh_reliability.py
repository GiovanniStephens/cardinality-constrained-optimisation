"""Tests for the June 2026 data-refresh reliability hardening:

  * bad-ticker cache: protected watchlist, TTL self-heal, config-driven threshold
  * incremental validation: validate_universe(skip_min_history=True)

See docs/DATA_REFRESH.md.
"""

import datetime as dt
import os
import tempfile
import unittest

import pandas as pd

from src import db, config
from src.db import bad_tickers as bt


class TestBadTickerCacheHardening(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mktemp(suffix='.db')
        self.conn = db.get_connection(self.tmp)
        # Hermetic protected watchlist (don't depend on data/core_etfs.csv).
        self.prot_csv = tempfile.mktemp(suffix='.csv')
        with open(self.prot_csv, 'w') as f:
            f.write('Tickers\nPROT1\nPROT2\n')
        self._orig_csv = config.PIPELINE_PROTECTED_TICKERS_CSV
        config.PIPELINE_PROTECTED_TICKERS_CSV = self.prot_csv
        bt._protected_cache = None

    def tearDown(self):
        config.PIPELINE_PROTECTED_TICKERS_CSV = self._orig_csv
        bt._protected_cache = None
        self.conn.close()
        for p in (self.tmp, self.prot_csv):
            try:
                os.remove(p)
            except OSError:
                pass

    def test_protected_ticker_never_cached(self):
        bt.save_known_bad_tickers(self.conn, ['PROT1', 'BAD1'], exchange='US')
        rows = {r[0] for r in self.conn.execute(
            "SELECT symbol FROM known_bad_tickers")}
        self.assertNotIn('PROT1', rows)   # protected: never written
        self.assertIn('BAD1', rows)

    def test_protected_ticker_never_loaded(self):
        # Even if a protected ticker is somehow already in the table, load drops it.
        ex = db._get_exchange_id(self.conn, 'US')
        self.conn.execute(
            "INSERT INTO known_bad_tickers "
            "(symbol, exchange_id, failure_count, first_failed, last_failed) "
            "VALUES ('PROT1', ?, 9, 'x', 'x')", (ex,))
        self.conn.commit()
        bad = bt.load_known_bad_tickers(self.conn, exchange='US', min_failures=1)
        self.assertNotIn('PROT1', bad)

    def test_ttl_expired_entry_purged_and_retried(self):
        ex = db._get_exchange_id(self.conn, 'US')
        past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).isoformat()
        self.conn.execute(
            "INSERT INTO known_bad_tickers "
            "(symbol, exchange_id, failure_count, first_failed, last_failed, expires_at) "
            "VALUES ('OLD', ?, 5, 'x', 'x', ?)", (ex, past))
        self.conn.commit()
        bad = bt.load_known_bad_tickers(self.conn, exchange='US', min_failures=1)
        self.assertNotIn('OLD', bad)                      # not returned (expired)
        remaining = self.conn.execute(
            "SELECT COUNT(*) FROM known_bad_tickers WHERE symbol='OLD'").fetchone()[0]
        self.assertEqual(remaining, 0)                    # purged for retry

    def test_legacy_null_expiry_heals_off_last_failed(self):
        ex = db._get_exchange_id(self.conn, 'US')
        old = (dt.datetime.now(dt.timezone.utc)
               - dt.timedelta(days=config.PIPELINE_BAD_TICKER_TTL_DAYS + 5)).isoformat()
        self.conn.execute(
            "INSERT INTO known_bad_tickers "
            "(symbol, exchange_id, failure_count, first_failed, last_failed, expires_at) "
            "VALUES ('LEGACY', ?, 9, 'x', ?, NULL)", (ex, old))
        self.conn.commit()
        bad = bt.load_known_bad_tickers(self.conn, exchange='US', min_failures=1)
        self.assertNotIn('LEGACY', bad)                   # legacy entry aged out

    def test_default_threshold_from_config(self):
        bt.save_known_bad_tickers(self.conn, ['B1'], exchange='US')   # one failure
        # default min_failures = PIPELINE_BAD_CACHE_MIN_FAILURES (3) -> not skipped
        self.assertNotIn('B1', bt.load_known_bad_tickers(self.conn, exchange='US'))
        # explicit min_failures=1 -> skipped
        self.assertIn('B1', bt.load_known_bad_tickers(self.conn, exchange='US',
                                                      min_failures=1))


class TestIncrementalValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mktemp(suffix='.db')
        self.conn = db.get_connection(self.tmp)

    def tearDown(self):
        self.conn.close()
        try:
            os.remove(self.tmp)
        except OSError:
            pass

    def test_skip_min_history_keeps_short_history_ticker(self):
        from src.data_quality import validate_universe
        # A ticker with only ~30 recent rows (simulates incremental staging).
        dates = pd.bdate_range('2026-05-01', '2026-06-12')
        df = pd.DataFrame({'AAA': range(len(dates))}, index=dates)
        db.save_prices(self.conn, df, exchange='US', asset_type='etf')

        full = validate_universe(self.conn, exchange='US')
        self.assertEqual(full['total_active'], 0)          # min_history excludes it

        inc = validate_universe(self.conn, exchange='US', skip_min_history=True)
        self.assertEqual(inc['total_active'], 1)           # survives incremental
        self.assertNotIn('min_history', inc)               # check not run


if __name__ == '__main__':
    unittest.main()
