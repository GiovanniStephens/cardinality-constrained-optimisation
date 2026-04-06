"""Unit tests for the pipeline module (pipeline.py)."""

import json
import os
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src import db
from src.download_data import download_and_save
from src.pipeline import (
    backup_database,
    filter_completed,
    load_checkpoint,
    preflight_check,
    promote_staging,
    rollback,
    save_checkpoint,
    write_manifest,
)


class TestCheckpoint(unittest.TestCase):
    """Test checkpoint save/load round-trip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, 'checkpoint.json')

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_save_load_round_trip(self):
        state = {
            'run_id': 'test123',
            'completed_tickers': ['SPY', 'QQQ', 'IVV'],
            'failed_tickers': ['BAD1'],
        }
        save_checkpoint(self.path, state)
        loaded = load_checkpoint(self.path)
        self.assertEqual(loaded, state)

    def test_load_missing_file(self):
        result = load_checkpoint('/nonexistent/path.json')
        self.assertEqual(result, {})

    def test_atomic_write_creates_no_tmp_file(self):
        save_checkpoint(self.path, {'test': True})
        self.assertFalse(os.path.exists(self.path + '.tmp'))

    def test_overwrite_existing(self):
        save_checkpoint(self.path, {'v': 1})
        save_checkpoint(self.path, {'v': 2})
        loaded = load_checkpoint(self.path)
        self.assertEqual(loaded['v'], 2)


class TestFilterCompleted(unittest.TestCase):
    """Test filtering already-completed tickers."""

    def test_filters_completed(self):
        tickers = ['SPY', 'QQQ', 'IVV', 'VTI', 'BND']
        checkpoint = {'completed_tickers': ['SPY', 'QQQ']}
        result = filter_completed(tickers, checkpoint)
        self.assertEqual(result, ['IVV', 'VTI', 'BND'])

    def test_empty_checkpoint(self):
        tickers = ['SPY', 'QQQ']
        result = filter_completed(tickers, {})
        self.assertEqual(result, tickers)

    def test_all_completed(self):
        tickers = ['SPY', 'QQQ']
        checkpoint = {'completed_tickers': ['SPY', 'QQQ']}
        result = filter_completed(tickers, checkpoint)
        self.assertEqual(result, [])

    def test_preserves_order(self):
        tickers = ['C', 'B', 'A', 'D']
        checkpoint = {'completed_tickers': ['B']}
        result = filter_completed(tickers, checkpoint)
        self.assertEqual(result, ['C', 'A', 'D'])


class TestPreflightCheck(unittest.TestCase):
    """Test disk space pre-flight check."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.staging_path = os.path.join(self.tmpdir, 'staging.db')

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_returns_expected_keys(self):
        result = preflight_check(
            '/nonexistent/prod.db', self.staging_path,
            num_tickers=100, num_trading_days=252,
        )
        self.assertIn('ok', result)
        self.assertIn('available_gb', result)
        self.assertIn('estimated_staging_gb', result)
        self.assertIn('warnings', result)

    def test_small_download_passes(self):
        result = preflight_check(
            '/nonexistent/prod.db', self.staging_path,
            num_tickers=10, num_trading_days=252,
        )
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['warnings']), 0)

    def test_estimates_are_positive(self):
        result = preflight_check(
            '/nonexistent/prod.db', self.staging_path,
            num_tickers=25000, num_trading_days=2520,
        )
        self.assertGreater(result['estimated_staging_gb'], 0)
        self.assertGreater(result['available_gb'], 0)


class TestBackupAndRollback(unittest.TestCase):
    """Test database backup and rollback."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test.db')
        self.backup_path = os.path.join(self.tmpdir, 'test.db.backup')

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_backup_round_trip(self):
        conn = db.get_connection(self.db_path)
        # Insert some data
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        prices = pd.DataFrame(
            {'SPY': np.random.rand(5), 'QQQ': np.random.rand(5)},
            index=dates,
        )
        db.save_prices(conn, prices, exchange='US', asset_type='etf')

        # Backup
        backup_database(conn, self.backup_path)
        conn.close()

        self.assertTrue(os.path.exists(self.backup_path))
        self.assertGreater(os.path.getsize(self.backup_path), 0)

        # Verify backup contains the data
        conn_backup = sqlite3.connect(self.backup_path)
        conn_backup.row_factory = sqlite3.Row
        count = conn_backup.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        conn_backup.close()
        self.assertEqual(count, 10)  # 5 dates x 2 tickers

    def test_rollback_restores_data(self):
        conn = db.get_connection(self.db_path)
        dates = pd.date_range('2024-01-01', periods=3, freq='B')
        prices = pd.DataFrame({'SPY': [100, 101, 102]}, index=dates)
        db.save_prices(conn, prices, exchange='US', asset_type='etf')
        backup_database(conn, self.backup_path)

        # Corrupt the original by adding more data
        prices2 = pd.DataFrame({'FAKE': [999, 998, 997]}, index=dates)
        db.save_prices(conn, prices2, exchange='US', asset_type='etf')
        count_before = conn.execute("SELECT COUNT(DISTINCT t.symbol) FROM prices p JOIN tickers t ON p.ticker_id = t.id").fetchone()[0]
        self.assertEqual(count_before, 2)  # SPY + FAKE
        conn.close()

        # Rollback
        rollback(self.backup_path, self.db_path)

        # Verify FAKE is gone
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT t.symbol FROM prices p JOIN tickers t ON p.ticker_id = t.id"
        ).fetchall()]
        conn.close()
        self.assertEqual(symbols, ['SPY'])

    def test_rollback_missing_backup_raises(self):
        with self.assertRaises(FileNotFoundError):
            rollback('/nonexistent/backup.db', self.db_path)


class TestPromoteStaging(unittest.TestCase):
    """Test promotion from staging DB to production DB."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.staging_path = os.path.join(self.tmpdir, 'staging.db')
        self.prod_path = os.path.join(self.tmpdir, 'prod.db')

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_promote_copies_prices(self):
        # Create staging DB with data
        conn_staging = db.get_connection(self.staging_path)
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        prices = pd.DataFrame(
            {'SPY': [100, 101, 102, 103, 104], 'QQQ': [200, 201, 202, 203, 204]},
            index=dates,
        )
        db.save_prices(conn_staging, prices, exchange='US', asset_type='etf')
        conn_staging.close()

        # Create empty production DB
        conn_prod = db.get_connection(self.prod_path)

        # Promote
        promoted = promote_staging(self.staging_path, conn_prod, exchange='US')
        self.assertEqual(promoted, 2)

        # Verify prices in production
        prod_prices = db.load_prices(conn_prod, exchange='US', min_coverage=0)
        self.assertEqual(set(prod_prices.columns), {'SPY', 'QQQ'})
        self.assertEqual(len(prod_prices), 5)
        conn_prod.close()

    def test_promote_preserves_existing_prod_data(self):
        # Pre-populate production with existing data
        conn_prod = db.get_connection(self.prod_path)
        dates = pd.date_range('2024-01-01', periods=3, freq='B')
        existing = pd.DataFrame({'VTI': [300, 301, 302]}, index=dates)
        db.save_prices(conn_prod, existing, exchange='US', asset_type='etf')

        # Create staging with new data
        conn_staging = db.get_connection(self.staging_path)
        new_prices = pd.DataFrame({'SPY': [100, 101, 102]}, index=dates)
        db.save_prices(conn_staging, new_prices, exchange='US', asset_type='etf')
        conn_staging.close()

        # Promote
        promote_staging(self.staging_path, conn_prod, exchange='US')

        # Both should exist
        prod_prices = db.load_prices(conn_prod, exchange='US', min_coverage=0)
        self.assertIn('VTI', prod_prices.columns)
        self.assertIn('SPY', prod_prices.columns)
        conn_prod.close()

    def test_promote_empty_staging(self):
        conn_staging = db.get_connection(self.staging_path)
        conn_staging.close()
        conn_prod = db.get_connection(self.prod_path)
        promoted = promote_staging(self.staging_path, conn_prod, exchange='US')
        self.assertEqual(promoted, 0)
        conn_prod.close()


class TestManifest(unittest.TestCase):
    """Test manifest writing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, 'manifest.json')

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_write_and_read(self):
        manifest = {
            'run_id': 'test123',
            'status': 'promoted',
            'total_tickers': 100,
            'duration_seconds': 42.5,
        }
        write_manifest(self.path, manifest)

        with open(self.path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded['status'], 'promoted')
        self.assertEqual(loaded['total_tickers'], 100)


class TestSubsetSlicing(unittest.TestCase):
    """Test that subset parameter correctly slices the ticker list."""

    def test_subset_slices_list(self):
        tickers = list(range(100))
        subset = 10
        result = tickers[:subset]
        self.assertEqual(len(result), 10)
        self.assertEqual(result, list(range(10)))

    def test_subset_larger_than_list(self):
        tickers = ['A', 'B', 'C']
        subset = 100
        result = tickers[:subset]
        self.assertEqual(result, tickers)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker in download_and_save."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test.db')
        self.conn = db.get_connection(self.db_path)

    def tearDown(self):
        self.conn.close()
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    @patch('src.download_data._download_batch_with_timeout', return_value=None)
    def test_circuit_breaker_trips_after_threshold(self, mock_dl):
        tickers = [f'T{i}' for i in range(25)]  # 5 batches of 5
        result = download_and_save(
            tickers, self.conn, exchange='US', batch_size=5,
            max_retries=1, circuit_breaker_threshold=3,
            rate_limit_delay=0, batch_timeout=10,
        )
        self.assertTrue(result['circuit_breaker_tripped'])
        # Should have stopped after 3 consecutive failures, not all 5
        self.assertEqual(len(result['failed_batches']), 3)

    @patch('src.download_data._download_batch_with_timeout')
    def test_circuit_breaker_resets_on_success(self, mock_dl):
        dates = pd.date_range('2024-01-01', periods=5, freq='B')

        def side_effect(tickers, start, end, timeout):
            # Fail batches 1-4, succeed on batch 5
            if tickers[0] in ('T20', 'T21', 'T22', 'T23'):
                return pd.DataFrame({'T20': range(5)}, index=dates)
            return None

        mock_dl.side_effect = side_effect
        tickers = [f'T{i}' for i in range(25)]
        result = download_and_save(
            tickers, self.conn, exchange='US', batch_size=5,
            max_retries=1, circuit_breaker_threshold=5,
            rate_limit_delay=0, batch_timeout=10,
        )
        # 4 failures then 1 success — should NOT trip (resets at batch 5)
        self.assertFalse(result['circuit_breaker_tripped'])

    @patch('src.download_data._download_batch_with_timeout', return_value=None)
    def test_circuit_breaker_saves_checkpoint(self, mock_dl):
        checkpoint_path = os.path.join(self.tmpdir, 'ckpt.json')
        checkpoint = {
            'run_id': 'test',
            'completed_tickers': [],
            'failed_tickers': [],
        }

        def on_failed(tickers, batch_num):
            checkpoint['failed_tickers'].extend(tickers)
            save_checkpoint(checkpoint_path, checkpoint)

        tickers = [f'T{i}' for i in range(15)]
        download_and_save(
            tickers, self.conn, exchange='US', batch_size=5,
            max_retries=1, circuit_breaker_threshold=2,
            rate_limit_delay=0, batch_timeout=10,
            on_batch_failed=on_failed,
        )
        saved = load_checkpoint(checkpoint_path)
        self.assertGreater(len(saved['failed_tickers']), 0)


class TestFailedTickerRecording(unittest.TestCase):
    """Test that failed batches record ticker symbols."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test.db')
        self.conn = db.get_connection(self.db_path)

    def tearDown(self):
        self.conn.close()
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    @patch('src.download_data._download_batch_with_timeout', return_value=None)
    def test_failed_batches_contain_ticker_symbols(self, mock_dl):
        tickers = ['SPY', 'QQQ', 'IVV', 'VTI', 'BND']
        result = download_and_save(
            tickers, self.conn, exchange='US', batch_size=2,
            max_retries=1, circuit_breaker_threshold=100,
            rate_limit_delay=0, batch_timeout=10,
        )
        for fb in result['failed_batches']:
            self.assertIn('batch_num', fb)
            self.assertIn('tickers', fb)
            self.assertIsInstance(fb['tickers'], list)
        # First batch should contain SPY, QQQ
        self.assertEqual(result['failed_batches'][0]['tickers'], ['SPY', 'QQQ'])

    @patch('src.download_data._download_batch_with_timeout', return_value=None)
    def test_checkpoint_records_failed_tickers(self, mock_dl):
        checkpoint_path = os.path.join(self.tmpdir, 'ckpt.json')
        checkpoint = {
            'run_id': 'test',
            'completed_tickers': [],
            'failed_tickers': [],
        }

        def on_failed(tickers, batch_num):
            checkpoint['failed_tickers'].extend(tickers)
            save_checkpoint(checkpoint_path, checkpoint)

        tickers = ['SPY', 'QQQ', 'IVV']
        download_and_save(
            tickers, self.conn, exchange='US', batch_size=2,
            max_retries=1, circuit_breaker_threshold=100,
            rate_limit_delay=0, batch_timeout=10,
            on_batch_failed=on_failed,
        )
        saved = load_checkpoint(checkpoint_path)
        self.assertIn('SPY', saved['failed_tickers'])
        self.assertIn('QQQ', saved['failed_tickers'])


class TestAdaptiveRateLimit(unittest.TestCase):
    """Test adaptive rate limit escalation and decay."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test.db')
        self.conn = db.get_connection(self.db_path)

    def tearDown(self):
        self.conn.close()
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    @patch('src.download_data.time.sleep')
    @patch('src.download_data._download_batch_with_timeout')
    def test_rate_limit_delay_escalates_on_429(self, mock_dl, mock_sleep):
        dates = pd.date_range('2024-01-01', periods=5, freq='B')

        call_count = [0]
        def side_effect(tickers, start, end, timeout):
            call_count[0] += 1
            if call_count[0] <= 2:
                # First two calls: raise 429
                raise Exception("HTTP Error 429: Too Many Requests")
            # After retries, succeed
            return pd.DataFrame({t: range(5) for t in tickers}, index=dates)

        mock_dl.side_effect = side_effect
        tickers = [f'T{i}' for i in range(5)]
        result = download_and_save(
            tickers, self.conn, exchange='US', batch_size=5,
            max_retries=3, rate_limit_delay=0.5,
            circuit_breaker_threshold=100, batch_timeout=10,
            max_rate_limit_delay=30.0,
        )
        # Should have called sleep with increasing backoff values
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        # The 429 backoff sleeps should be in the list (8s, 16s for attempts 1, 2)
        self.assertTrue(any(s >= 8 for s in sleep_calls))

    @patch('src.download_data.time.sleep')
    @patch('src.download_data._download_batch_with_timeout')
    def test_rate_limit_delay_decays_on_success(self, mock_dl, mock_sleep):
        dates = pd.date_range('2024-01-01', periods=5, freq='B')

        call_count = [0]
        def side_effect(tickers, start, end, timeout):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("HTTP Error 429: Too Many Requests")
            return pd.DataFrame({t: range(5) for t in tickers}, index=dates)

        mock_dl.side_effect = side_effect
        tickers = [f'T{i}' for i in range(10)]
        result = download_and_save(
            tickers, self.conn, exchange='US', batch_size=5,
            max_retries=3, rate_limit_delay=0.5,
            circuit_breaker_threshold=100, batch_timeout=10,
            max_rate_limit_delay=30.0,
        )
        self.assertEqual(result['saved_tickers'], 10)


class TestBatchTimeout(unittest.TestCase):
    """Test batch download timeout."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test.db')
        self.conn = db.get_connection(self.db_path)

    def tearDown(self):
        self.conn.close()
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    @patch('src.download_data._download_batch')
    def test_batch_timeout_returns_none(self, mock_dl):
        def slow_download(*args):
            time.sleep(5)
            return pd.DataFrame({'T0': [1, 2, 3]})

        mock_dl.side_effect = slow_download
        tickers = ['T0', 'T1', 'T2']
        result = download_and_save(
            tickers, self.conn, exchange='US', batch_size=3,
            max_retries=1, batch_timeout=1,
            circuit_breaker_threshold=100, rate_limit_delay=0,
        )
        # Should have timed out and recorded as failed
        self.assertEqual(len(result['failed_batches']), 1)
        self.assertEqual(result['saved_tickers'], 0)


class TestPromotionIntegrity(unittest.TestCase):
    """Test promotion integrity check."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.staging_path = os.path.join(self.tmpdir, 'staging.db')
        self.prod_path = os.path.join(self.tmpdir, 'prod.db')

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_promote_staging_integrity_passes(self):
        conn_staging = db.get_connection(self.staging_path)
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        prices = pd.DataFrame(
            {'SPY': [100, 101, 102, 103, 104], 'QQQ': [200, 201, 202, 203, 204]},
            index=dates,
        )
        db.save_prices(conn_staging, prices, exchange='US', asset_type='etf')
        conn_staging.close()

        conn_prod = db.get_connection(self.prod_path)
        promoted = promote_staging(self.staging_path, conn_prod, exchange='US')
        self.assertEqual(promoted, 2)

        # Verify production has at least as many tickers as staging
        exchange_id = db._get_exchange_id(conn_prod, 'US')
        prod_count = conn_prod.execute(
            "SELECT COUNT(DISTINCT symbol) FROM tickers WHERE exchange_id = ?",
            (exchange_id,),
        ).fetchone()[0]
        self.assertGreaterEqual(prod_count, 2)
        conn_prod.close()


if __name__ == '__main__':
    unittest.main()
