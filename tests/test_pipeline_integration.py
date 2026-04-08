"""Integration tests for the data pipeline orchestration.

Tests the full flow: download → validate → promote, checkpoint resume,
validation exclusions, and backup/rollback. All downloads are mocked.

Run with:  RUN_INTEGRATION=1 python -m unittest tests.test_pipeline_integration
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests import requires_integration
from tests.helpers import get_memory_db

from src import db, pipeline, data_quality


@requires_integration
class TestPipelineStagePromote(unittest.TestCase):
    """Full pipeline: mock download → staging → validate → promote."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.staging_path = os.path.join(self.tmpdir, 'staging.db')
        self.prod_path = os.path.join(self.tmpdir, 'prod.db')
        self.checkpoint_path = os.path.join(self.tmpdir, 'checkpoint.json')
        # Create production DB with schema
        prod_conn = db.get_connection(self.prod_path)
        prod_conn.close()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mock_prices(self, tickers, n_days=300):
        """Create a realistic price DataFrame for mocking downloads."""
        np.random.seed(42)
        dates = pd.bdate_range('2020-01-01', periods=n_days, freq='B')
        data = {}
        for t in tickers:
            log_rets = np.random.randn(n_days) * 0.02 + 0.0002
            data[t] = 100 * np.exp(log_rets.cumsum())
        return pd.DataFrame(data, index=dates)

    def test_stage_and_promote(self):
        tickers = [f'T{i}' for i in range(5)]
        mock_prices = self._make_mock_prices(tickers)

        # Create staging DB and populate it directly (mock the download)
        staging_conn = db.get_connection(self.staging_path)
        db.save_prices(staging_conn, mock_prices, exchange='US', asset_type='etf')
        staging_conn.close()

        # Promote staging → production
        prod_conn = db.get_connection(self.prod_path)
        n_promoted = pipeline.promote_staging(
            self.staging_path, prod_conn, exchange='US')
        self.assertEqual(n_promoted, 5)

        # Verify production has data
        loaded = db.load_prices(prod_conn, exchange='US', min_coverage=0)
        self.assertEqual(len(loaded.columns), 5)
        for t in tickers:
            self.assertIn(t, loaded.columns)
        prod_conn.close()


@requires_integration
class TestCheckpointResume(unittest.TestCase):
    """Test checkpoint save/load and filter_completed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.tmpdir, 'checkpoint.json')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_checkpoint(self):
        state = {
            'completed_tickers': ['AAPL', 'MSFT', 'GOOG'],
            'batch_index': 1,
        }
        pipeline.save_checkpoint(self.checkpoint_path, state)
        loaded = pipeline.load_checkpoint(self.checkpoint_path)
        self.assertEqual(loaded['completed_tickers'], ['AAPL', 'MSFT', 'GOOG'])
        self.assertEqual(loaded['batch_index'], 1)

    def test_filter_completed(self):
        all_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
        checkpoint = {'completed_tickers': ['AAPL', 'MSFT']}
        remaining = pipeline.filter_completed(all_tickers, checkpoint)
        self.assertEqual(remaining, ['GOOG', 'AMZN', 'META'])

    def test_load_missing_checkpoint(self):
        loaded = pipeline.load_checkpoint('/nonexistent/path.json')
        self.assertEqual(loaded, {})


@requires_integration
class TestValidationExclusion(unittest.TestCase):
    """Validate universe excludes bad tickers before promotion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.staging_path = os.path.join(self.tmpdir, 'staging.db')
        self.prod_path = os.path.join(self.tmpdir, 'prod.db')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_constant_price_excluded(self):
        staging_conn = db.get_connection(self.staging_path)
        # Good ticker: 1300 days (above MIN_HISTORY_DAYS=1260) with normal prices
        np.random.seed(42)
        n_days = 1300
        good_prices = 100 * np.exp(
            np.cumsum(np.random.randn(n_days) * 0.02 + 0.0002))
        dates = pd.bdate_range('2018-01-01', periods=n_days, freq='B')
        df = pd.DataFrame({'GOOD': good_prices, 'BAD': [100.0] * n_days},
                          index=dates)
        db.save_prices(staging_conn, df, exchange='US', asset_type='etf')

        # Validate — BAD should be flagged
        summary = data_quality.validate_universe(
            staging_conn, exchange='US', dry_run=False)
        self.assertGreater(summary['total_excluded'], 0)

        # Load prices excluding flagged — BAD should be gone
        loaded = db.load_prices(staging_conn, exchange='US',
                                exclude_flagged=True, min_coverage=0)
        self.assertIn('GOOD', loaded.columns)
        self.assertNotIn('BAD', loaded.columns)
        staging_conn.close()


@requires_integration
class TestBackupAndRollback(unittest.TestCase):
    """Backup production DB and verify rollback restores it."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.prod_path = os.path.join(self.tmpdir, 'prod.db')
        self.backup_path = os.path.join(self.tmpdir, 'prod_backup.db')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_backup_and_rollback(self):
        # Create production DB with initial data
        conn = db.get_connection(self.prod_path)
        dates = pd.bdate_range('2020-01-01', periods=50, freq='B')
        df = pd.DataFrame({'ORIG': [100.0 + i for i in range(50)]},
                          index=dates)
        db.save_prices(conn, df, exchange='US', asset_type='etf')

        # Backup
        pipeline.backup_database(conn, self.backup_path)
        conn.close()
        self.assertTrue(os.path.exists(self.backup_path))

        # Corrupt production by adding new data
        conn2 = db.get_connection(self.prod_path)
        df2 = pd.DataFrame({'NEW': [200.0 + i for i in range(50)]},
                           index=dates)
        db.save_prices(conn2, df2, exchange='US', asset_type='etf')
        loaded_before = db.load_prices(conn2, exchange='US', min_coverage=0)
        self.assertIn('NEW', loaded_before.columns)
        conn2.close()

        # Rollback
        pipeline.rollback(self.backup_path, self.prod_path)

        # Verify restored state
        conn3 = db.get_connection(self.prod_path)
        loaded_after = db.load_prices(conn3, exchange='US', min_coverage=0)
        self.assertIn('ORIG', loaded_after.columns)
        self.assertNotIn('NEW', loaded_after.columns)
        conn3.close()


if __name__ == '__main__':
    unittest.main()
