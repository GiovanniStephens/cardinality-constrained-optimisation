"""Tests for src.download_validate — ticker validation and batch retry-with-splitting."""

import json
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from tests.helpers import BaseTmpDirTest


class TestValidateTickersAllValid(BaseTmpDirTest):
    """All tickers return data and end up in the valid set."""

    @patch('src.download_validate._sess._get_state', return_value=False)
    @patch('src.download_validate._sess._rotate_tor_circuit')
    @patch('src.download_data._download_batch_with_timeout')
    def test_validate_tickers_all_valid(self, mock_download, mock_rotate,
                                        mock_get_state):
        from src.download_validate import validate_tickers

        tickers = ['AAPL', 'MSFT', 'GOOG']
        mock_df = pd.DataFrame(
            {'AAPL': [100.0], 'MSFT': [200.0], 'GOOG': [300.0]},
            index=pd.DatetimeIndex(['2024-07-01']),
        )
        mock_download.return_value = mock_df

        valid, invalid, unvalidated = validate_tickers(
            tickers,
            batch_size=10,
            delay=0,
            max_retries=0,
            timeout=10,
            validation_windows=[('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir,
            max_cache_hours=0,
        )

        self.assertEqual(valid, {'AAPL', 'MSFT', 'GOOG'})
        self.assertEqual(invalid, set())
        self.assertEqual(unvalidated, set())


class TestValidateTickersSomeInvalid(BaseTmpDirTest):
    """Some tickers return data, others do not."""

    @patch('src.download_validate._sess._get_state', return_value=False)
    @patch('src.download_validate._sess._rotate_tor_circuit')
    @patch('src.download_data._download_batch_with_timeout')
    def test_validate_tickers_some_invalid(self, mock_download, mock_rotate,
                                           mock_get_state):
        from src.download_validate import validate_tickers

        tickers = ['AAPL', 'BADTICKER', 'GOOG']
        # Only AAPL and GOOG have data
        mock_df = pd.DataFrame(
            {'AAPL': [100.0], 'GOOG': [300.0]},
            index=pd.DatetimeIndex(['2024-07-01']),
        )
        mock_download.return_value = mock_df

        valid, invalid, unvalidated = validate_tickers(
            tickers,
            batch_size=10,
            delay=0,
            max_retries=0,
            timeout=10,
            validation_windows=[('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir,
            max_cache_hours=0,
        )

        self.assertEqual(valid, {'AAPL', 'GOOG'})
        self.assertIn('BADTICKER', invalid)
        self.assertEqual(unvalidated, set())


class TestValidateTickersCacheHit(BaseTmpDirTest):
    """Pre-existing fresh cache file is loaded, download skipped."""

    @patch('src.download_validate._sess._get_state', return_value=False)
    @patch('src.download_validate._sess._rotate_tor_circuit')
    @patch('src.download_data._download_batch_with_timeout')
    def test_validate_tickers_cache_hit(self, mock_download, mock_rotate,
                                        mock_get_state):
        from src.download_validate import validate_tickers

        # Write a fresh cache file
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'valid': ['AAPL', 'MSFT'],
            'invalid': ['BADTICKER'],
            'unvalidated': [],
        }
        cache_path = os.path.join(self.tmpdir, 'validated_tickers.json')
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        tickers = ['AAPL', 'MSFT', 'BADTICKER']
        valid, invalid, unvalidated = validate_tickers(
            tickers,
            batch_size=10,
            delay=0,
            max_retries=0,
            timeout=10,
            validation_windows=[('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir,
            max_cache_hours=168,
        )

        # Should use cache, not call download
        mock_download.assert_not_called()
        self.assertEqual(valid, {'AAPL', 'MSFT'})
        self.assertIn('BADTICKER', invalid)


class TestValidateTickersCacheExpired(BaseTmpDirTest):
    """Stale cache triggers re-validation."""

    @patch('src.download_validate._sess._get_state', return_value=False)
    @patch('src.download_validate._sess._rotate_tor_circuit')
    @patch('src.download_data._download_batch_with_timeout')
    def test_validate_tickers_cache_expired(self, mock_download, mock_rotate,
                                            mock_get_state):
        from src.download_validate import validate_tickers

        # Write a stale cache (200 hours old > 168h max)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        cache_data = {
            'timestamp': old_ts,
            'valid': ['AAPL'],
            'invalid': ['BADTICKER'],
            'unvalidated': [],
        }
        cache_path = os.path.join(self.tmpdir, 'validated_tickers.json')
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        tickers = ['AAPL', 'MSFT']
        mock_df = pd.DataFrame(
            {'AAPL': [100.0], 'MSFT': [200.0]},
            index=pd.DatetimeIndex(['2024-07-01']),
        )
        mock_download.return_value = mock_df

        valid, invalid, unvalidated = validate_tickers(
            tickers,
            batch_size=10,
            delay=0,
            max_retries=0,
            timeout=10,
            validation_windows=[('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir,
            max_cache_hours=168,
        )

        # Stale + complete cache => re-validate from scratch
        mock_download.assert_called()
        self.assertEqual(valid, {'AAPL', 'MSFT'})


class TestRetryWithSplitting(unittest.TestCase):
    """Test _retry_with_splitting recursive splitting logic."""

    @patch('src.download_validate.time.sleep')
    @patch('src.download_data._download_batch_with_timeout')
    def test_retry_with_splitting(self, mock_download, mock_sleep):
        """Fails on large batches, succeeds on smaller ones."""
        from src.download_validate import _retry_with_splitting

        # Fail on batches > 2 tickers, succeed on 1-2 ticker batches
        def download_side_effect(tickers, start, end, timeout):
            if len(tickers) > 2:
                return None
            return pd.DataFrame(
                {t: [100.0 + i] for i, t in enumerate(tickers)},
                index=pd.DatetimeIndex(['2024-01-01']),
            )

        mock_download.side_effect = download_side_effect

        tickers = ['A', 'B', 'C', 'D']
        delay_state = [1.0, 10.0, 1.0]

        df, failed = _retry_with_splitting(
            tickers, '2024-01-01', '2024-01-08', 30,
            min_batch_size=1, delay_state=delay_state,
        )

        self.assertIsNotNone(df)
        self.assertEqual(len(failed), 0)
        # All tickers should be in the combined result
        self.assertEqual(set(df.columns), {'A', 'B', 'C', 'D'})

    @patch('src.download_validate.time.sleep')
    @patch('src.download_data._download_batch_with_timeout')
    def test_retry_with_splitting_min_batch(self, mock_download, mock_sleep):
        """Splitting stops at min_sub_batch_size, returns failures."""
        from src.download_validate import _retry_with_splitting

        # Always fail
        mock_download.return_value = None

        tickers = ['A', 'B', 'C', 'D']
        delay_state = [1.0, 10.0, 1.0]

        df, failed = _retry_with_splitting(
            tickers, '2024-01-01', '2024-01-08', 30,
            min_batch_size=3, delay_state=delay_state,
        )

        self.assertIsNone(df)
        # All tickers should be in the failed list
        self.assertEqual(set(failed), {'A', 'B', 'C', 'D'})


if __name__ == '__main__':
    unittest.main()
