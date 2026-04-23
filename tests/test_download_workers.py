"""Tests for src/download/workers.py — concurrent download infrastructure."""

import queue
import sqlite3
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from tests.helpers import BaseDBTest
from src import db


# ---------------------------------------------------------------------------
# P0.1  _partition_tickers
# ---------------------------------------------------------------------------

class TestPartitionTickers(unittest.TestCase):
    """Pure-function tests for _partition_tickers."""

    def _partition(self, tickers, n):
        from src.download.workers import _partition_tickers
        return _partition_tickers(tickers, n)

    def test_even_split(self):
        parts = self._partition(list(range(10)), 2)
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[0]), 5)
        self.assertEqual(len(parts[1]), 5)

    def test_no_tickers_lost(self):
        tickers = list(range(17))
        parts = self._partition(tickers, 3)
        flat = [t for p in parts for t in p]
        self.assertEqual(sorted(flat), sorted(tickers))

    def test_uneven_split(self):
        parts = self._partition(list(range(7)), 3)
        lengths = sorted([len(p) for p in parts])
        self.assertEqual(lengths, [2, 2, 3])

    def test_more_workers_than_tickers(self):
        parts = self._partition(['A', 'B'], 5)
        self.assertEqual(len(parts), 2)
        for p in parts:
            self.assertEqual(len(p), 1)

    def test_zero_tickers(self):
        parts = self._partition([], 3)
        self.assertEqual(parts, [])

    def test_single_worker(self):
        tickers = ['A', 'B', 'C']
        parts = self._partition(tickers, 1)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], tickers)

    def test_contiguous_partitions(self):
        """Each partition should be a contiguous slice of the input."""
        tickers = list(range(20))
        parts = self._partition(tickers, 4)
        flat = [t for p in parts for t in p]
        self.assertEqual(flat, tickers)


# ---------------------------------------------------------------------------
# P0.1  _db_writer
# ---------------------------------------------------------------------------

class TestDbWriter(BaseDBTest):
    """Test the queue-based DB writer consumer."""

    def _run_writer(self, result_queue, n_workers, conn=None,
                    on_batch_complete=None, on_batch_failed=None):
        from src.download.workers import _db_writer
        return _db_writer(
            result_queue=result_queue,
            conn=conn or self.conn,
            exchange='US',
            asset_type='etf',
            on_batch_complete=on_batch_complete,
            on_batch_failed=on_batch_failed,
            pbar=None,
            n_workers=n_workers,
        )

    def test_processes_data_items(self):
        q = queue.Queue()
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        df = pd.DataFrame({'SPY': [100.0, 101.0, 102.0]}, index=dates)
        metadata = {
            'names': {}, 'countries': {}, 'sectors': {},
            'industries': {}, 'category_groups': {}, 'categories': {},
        }
        # Protocol: ('data', batch_df, metadata_dict, saved_tickers_list, batch_num)
        q.put(('data', df, metadata, ['SPY'], 0))
        q.put(None)  # sentinel

        total_saved, failed = self._run_writer(q, n_workers=1)
        self.assertEqual(total_saved, 1)
        self.assertEqual(failed, [])

        loaded = db.load_prices(self.conn, exchange='US')
        self.assertIn('SPY', loaded.columns)

    def test_processes_failed_items(self):
        q = queue.Queue()
        # Protocol: ('failed', failed_tickers_list, batch_num)
        q.put(('failed', ['BAD1', 'BAD2'], 1))
        q.put(None)

        total_saved, failed = self._run_writer(q, n_workers=1)
        self.assertEqual(total_saved, 0)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['tickers'], ['BAD1', 'BAD2'])

    def test_multiple_sentinels(self):
        """Writer waits for n_workers sentinels before exiting."""
        q = queue.Queue()
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        df = pd.DataFrame({'AAA': [10.0, 11.0, 12.0]}, index=dates)
        metadata = {
            'names': {}, 'countries': {}, 'sectors': {},
            'industries': {}, 'category_groups': {}, 'categories': {},
        }
        q.put(('data', df, metadata, ['AAA'], 0))
        q.put(None)
        q.put(None)

        total_saved, failed = self._run_writer(q, n_workers=2)
        self.assertEqual(total_saved, 1)

    def test_callbacks_invoked(self):
        q = queue.Queue()
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        df = pd.DataFrame({'ZZZ': [50.0, 51.0, 52.0]}, index=dates)
        metadata = {
            'names': {}, 'countries': {}, 'sectors': {},
            'industries': {}, 'category_groups': {}, 'categories': {},
        }
        q.put(('data', df, metadata, ['ZZZ'], 0))
        q.put(('failed', ['F'], 1))
        q.put(None)

        on_complete = MagicMock()
        on_failed = MagicMock()
        self._run_writer(q, n_workers=1,
                         on_batch_complete=on_complete,
                         on_batch_failed=on_failed)
        on_complete.assert_called_once()
        on_failed.assert_called_once()


# ---------------------------------------------------------------------------
# P0.1  concurrent_download_and_save (orchestration with mocks)
# ---------------------------------------------------------------------------

class TestConcurrentDownloadAndSave(BaseDBTest):
    """Integration test for the top-level concurrent_download_and_save."""

    @patch('src.download.workers._sess._make_session')
    @patch('src.download.workers._dd._download_batch')
    @patch('src.download.workers.time.sleep', return_value=None)
    def test_deduplicates_tickers(self, _sleep, mock_dl, _mock_sess):
        """Duplicate tickers in input should be deduplicated."""
        from src.download.workers import concurrent_download_and_save

        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]}, index=dates)
        mock_dl.return_value = df

        result = concurrent_download_and_save(
            tickers=['A', 'A', 'A'],
            conn=self.conn,
            exchange='US',
            n_workers=1,
            batch_size=10,
        )
        self.assertEqual(result['total_tickers'], 1)

    @patch('src.download.workers._sess._proxy_url', None)
    @patch('src.download.workers._sess._make_session')
    @patch('src.download.workers._dd._download_batch')
    @patch('src.download.workers.time.sleep', return_value=None)
    def test_thread_path_saves_to_db(self, _sleep, mock_dl, _mock_sess):
        """Thread path should save downloaded data to the database."""
        from src.download.workers import concurrent_download_and_save

        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'T1': [10.0, 11.0, 12.0, 13.0, 14.0],
            'T2': [20.0, 21.0, 22.0, 23.0, 24.0],
        }, index=dates)
        mock_dl.return_value = df

        result = concurrent_download_and_save(
            tickers=['T1', 'T2'],
            conn=self.conn,
            exchange='US',
            n_workers=1,
            batch_size=10,
        )
        self.assertIn('saved_tickers', result)
        self.assertIn('failed_batches', result)

    @patch('src.download.workers._sess._make_session')
    @patch('src.download.workers._dd._download_batch')
    @patch('src.download.workers.time.sleep', return_value=None)
    def test_returns_expected_structure(self, _sleep, mock_dl, _mock_sess):
        """Return dict should have all expected keys."""
        from src.download.workers import concurrent_download_and_save

        mock_dl.return_value = None  # simulate empty download

        result = concurrent_download_and_save(
            tickers=['X'],
            conn=self.conn,
            exchange='US',
            n_workers=1,
            batch_size=10,
        )
        for key in ('total_tickers', 'saved_tickers', 'failed_batches',
                     'circuit_breaker_tripped', 'circuit_breaker_trip_count'):
            self.assertIn(key, result)


if __name__ == '__main__':
    unittest.main()
