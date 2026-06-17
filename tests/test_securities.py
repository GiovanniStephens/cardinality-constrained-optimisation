import multiprocessing
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src import db
from src import download as dd
from src.download.core import _download_batch
from src.download.session import _rotate_tor_circuit
from src.download.workers import _partition_tickers, _reset_yf_singleton, _subprocess_worker
import src.download.session as dl_sess
from tests import requires_network


@requires_network
class TestSecurityUniverse(unittest.TestCase):
    """Tests for retrieving securities from FinanceDatabase."""

    def test_get_equities_returns_dataframe(self):
        result = dd.get_equities(countries='United States')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_get_equities_has_name_column(self):
        result = dd.get_equities(countries='United States')
        self.assertIn('name', result.columns)

    def test_get_equities_index_contains_tickers(self):
        result = dd.get_equities(countries='United States')
        self.assertTrue(len(result.index) > 0)
        self.assertIsInstance(result.index[0], str)

    def test_get_etfs_returns_dataframe(self):
        result = dd.get_etfs()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_get_funds_returns_dataframe(self):
        result = dd.get_funds()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_build_universe_equities_only(self):
        result = dd.build_security_universe(
            asset_types=['equities'],
            countries='United States'
        )
        self.assertIn('Tickers', result.columns)
        self.assertIn('Name', result.columns)
        self.assertIn('AssetType', result.columns)
        self.assertTrue((result['AssetType'] == 'equity').all())

    def test_build_universe_etfs_only(self):
        result = dd.build_security_universe(asset_types=['etfs'])
        self.assertTrue((result['AssetType'] == 'etf').all())

    def test_build_universe_mixed(self):
        result = dd.build_security_universe(
            asset_types=['equities', 'etfs'],
            countries='United States'
        )
        asset_types = result['AssetType'].unique()
        self.assertIn('equity', asset_types)
        self.assertIn('etf', asset_types)

    def test_build_universe_no_duplicates(self):
        result = dd.build_security_universe(
            asset_types=['equities'],
            countries='United States'
        )
        self.assertEqual(len(result), len(result.drop_duplicates(subset='Tickers')))


class TestLoadTickers(unittest.TestCase):
    """Tests for loading tickers from CSV files."""

    def test_load_tickers_from_csv(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            f.write('Tickers\nAAPL\nMSFT\n')
            f.flush()
            result = dd.load_tickers(f.name)
        os.unlink(f.name)
        self.assertEqual(list(result['Tickers']), ['AAPL', 'MSFT'])

    def test_load_tickers_wrong_column_raises(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            f.write('Symbol\nAAPL\n')
            f.flush()
        with self.assertRaises(ValueError):
            dd.load_tickers(f.name)
        os.unlink(f.name)

    def test_load_tickers_custom_column(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            f.write('Symbol\nAAPL\nMSFT\n')
            f.flush()
            result = dd.load_tickers(f.name, ticker_column='Symbol')
        os.unlink(f.name)
        self.assertEqual(list(result['Symbol']), ['AAPL', 'MSFT'])


@requires_network
class TestDownloadData(unittest.TestCase):
    """Tests for downloading price data from Yahoo Finance."""

    def test_download_equity_prices(self):
        tickers_df = pd.DataFrame({'Tickers': ['AAPL', 'MSFT']})
        prices = dd.download_data(tickers_df, start='2024-01-01',
                                  end='2024-02-01')
        self.assertIsInstance(prices, pd.DataFrame)
        self.assertGreater(len(prices), 0)
        self.assertIn('AAPL', prices.columns)
        self.assertIn('MSFT', prices.columns)

    def test_download_etf_prices(self):
        tickers_df = pd.DataFrame({'Tickers': ['SPY']})
        prices = dd.download_data(tickers_df, start='2024-01-01',
                                  end='2024-02-01')
        self.assertIsInstance(prices, pd.DataFrame)
        self.assertGreater(len(prices), 0)
        self.assertIn('SPY', prices.columns)

    def test_download_skips_invalid_ticker(self):
        tickers_df = pd.DataFrame({'Tickers': ['AAPL', 'ZZZZZNOTREAL99']})
        prices = dd.download_data(tickers_df, start='2024-01-01',
                                  end='2024-02-01')
        self.assertIn('AAPL', prices.columns)

    def test_download_returns_numeric_data(self):
        tickers_df = pd.DataFrame({'Tickers': ['AAPL']})
        prices = dd.download_data(tickers_df, start='2024-01-01',
                                  end='2024-02-01')
        self.assertTrue(prices['AAPL'].dtype in ['float64', 'float32'])


@requires_network
class TestEndToEnd(unittest.TestCase):
    """End-to-end: FinanceDatabase tickers -> Yahoo Finance prices."""

    def test_equity_tickers_to_prices(self):
        securities = dd.build_security_universe(
            asset_types=['equities'],
            countries='United States'
        )
        sample = securities.head(2)
        prices = dd.download_data(sample, ticker_column='Tickers',
                                  start='2024-01-01', end='2024-02-01')
        self.assertIsInstance(prices, pd.DataFrame)
        self.assertGreater(len(prices), 0)
        self.assertGreater(len(prices.columns), 0)

    def test_save_and_reload_securities_csv(self):
        securities = dd.build_security_universe(
            asset_types=['equities'],
            countries='United States'
        )
        sample = securities.head(5)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            sample.to_csv(f.name, index=False)
            reloaded = dd.load_tickers(f.name, ticker_column='Tickers')
        os.unlink(f.name)
        self.assertEqual(list(reloaded['Tickers']), list(sample['Tickers']))


class TestDownloadBatchValidation(unittest.TestCase):
    """Test DataFrame structure validation in _download_batch."""

    @patch('src.download.core.yf.download')
    def test_handles_empty_dataframe(self, mock_yf):
        mock_yf.return_value = pd.DataFrame()
        result = _download_batch(['SPY'], '2024-01-01', '2024-02-01')
        self.assertIsNone(result)

    @patch('src.download.core.yf.download')
    def test_handles_none_return(self, mock_yf):
        mock_yf.return_value = None
        result = _download_batch(['SPY'], '2024-01-01', '2024-02-01')
        self.assertIsNone(result)

    @patch('src.download.core.yf.download')
    def test_handles_non_datetime_index(self, mock_yf):
        # Return a DataFrame with integer index that can be coerced to datetime
        dates = pd.date_range('2024-01-01', periods=3, freq='B')
        df = pd.DataFrame({'Close': [100, 101, 102]}, index=dates.strftime('%Y-%m-%d'))
        mock_yf.return_value = df
        result = _download_batch(['SPY'], '2024-01-01', '2024-01-05')
        # Should succeed after coercing index
        self.assertIsNotNone(result)

    @patch('src.download.core.yf.download')
    def test_handles_unconvertible_index(self, mock_yf):
        df = pd.DataFrame({'Close': [100, 101]}, index=['not-a-date', 'also-not'])
        mock_yf.return_value = df
        result = _download_batch(['SPY'], '2024-01-01', '2024-01-05')
        self.assertIsNone(result)


class TestDeduplication(unittest.TestCase):
    """Test ticker deduplication in download_and_save."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test.db')
        self.conn = db.get_connection(self.db_path)

    def tearDown(self):
        self.conn.close()
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    @patch('src.download.core._download_batch_with_timeout')
    def test_download_and_save_deduplicates_tickers(self, mock_dl):
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        mock_dl.return_value = pd.DataFrame(
            {'SPY': range(5), 'QQQ': range(5)}, index=dates)

        # Pass duplicates
        tickers = ['SPY', 'QQQ', 'SPY', 'QQQ', 'SPY']
        result = dd.download_and_save(
            tickers, self.conn, exchange='US', batch_size=5,
            max_retries=1, circuit_breaker_threshold=100,
            rate_limit_delay=0, batch_timeout=10,
        )
        # Should only process 2 unique tickers
        self.assertEqual(result['total_tickers'], 2)


class TestLogAggregation(unittest.TestCase):
    """Test that skipped tickers are logged as a summary, not individually."""

    @patch('src.download.core.yf.download')
    def test_batch_skipped_tickers_logged_as_summary(self, mock_yf):
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        # Return a DataFrame that only has data for one ticker out of many
        df = pd.DataFrame(
            {('SPY', 'Close'): range(5)},
            index=dates,
        )
        df.columns = pd.MultiIndex.from_tuples([('SPY', 'Close')])
        mock_yf.return_value = df

        with self.assertLogs('src.download.core', level='INFO') as cm:
            # Request many tickers but only SPY has data
            tickers = ['SPY'] + [f'BAD{i}' for i in range(20)]
            _download_batch(tickers, '2024-01-01', '2024-02-01')

        # Should have exactly one log line about skipped tickers (not 20)
        skipped_logs = [l for l in cm.output if 'had no data' in l]
        self.assertEqual(len(skipped_logs), 1)
        self.assertIn('20/21', skipped_logs[0])


class TestFilterUnwantedTickers(unittest.TestCase):
    """Tests for the regex pre-filter that removes warrants, units, etc."""

    def _make_df(self, tickers, names=None):
        data = {'Tickers': tickers}
        if names is not None:
            data['Name'] = names
        return pd.DataFrame(data)

    def test_removes_warrants(self):
        df = self._make_df(['AAPL', 'ACIC-WT', 'MSFT', 'FOO.WS', 'BAR-WTA'])
        filtered, removed = dd.filter_unwanted_tickers(df)
        self.assertEqual(sorted(filtered['Tickers'].tolist()), ['AAPL', 'MSFT'])
        self.assertEqual(sorted(removed['Tickers'].tolist()),
                         ['ACIC-WT', 'BAR-WTA', 'FOO.WS'])

    def test_removes_units(self):
        df = self._make_df(['SPY', 'AAQC-UN', 'QQQ'])
        filtered, removed = dd.filter_unwanted_tickers(df)
        self.assertEqual(sorted(filtered['Tickers'].tolist()), ['QQQ', 'SPY'])
        self.assertEqual(removed['Tickers'].tolist(), ['AAQC-UN'])

    def test_removes_preferred(self):
        df = self._make_df(['AAPL', 'ABR-PA', 'ABR-PB', 'MSFT'])
        filtered, removed = dd.filter_unwanted_tickers(df)
        self.assertEqual(sorted(filtered['Tickers'].tolist()), ['AAPL', 'MSFT'])
        self.assertEqual(sorted(removed['Tickers'].tolist()), ['ABR-PA', 'ABR-PB'])

    def test_removes_rights(self):
        df = self._make_df(['AAPL', 'FOO-RT', 'BAR.RI'])
        filtered, removed = dd.filter_unwanted_tickers(df)
        self.assertEqual(filtered['Tickers'].tolist(), ['AAPL'])
        self.assertEqual(sorted(removed['Tickers'].tolist()), ['BAR.RI', 'FOO-RT'])

    def test_removes_spac_names(self):
        df = self._make_df(
            ['ARES', 'AAPL', 'BLNK', 'MSFT'],
            ['Ares Acquisition Corp', 'Apple Inc', 'Blank Check Co', 'Microsoft Corp'],
        )
        filtered, removed = dd.filter_unwanted_tickers(df)
        self.assertEqual(sorted(filtered['Tickers'].tolist()), ['AAPL', 'MSFT'])
        self.assertEqual(sorted(removed['Tickers'].tolist()), ['ARES', 'BLNK'])

    def test_preserves_normal_tickers(self):
        df = self._make_df(
            ['AAPL', 'MSFT', 'GOOG', 'SPY', 'TLT'],
            ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc', 'SPDR S&P 500', 'iShares 20+ Year'],
        )
        filtered, removed = dd.filter_unwanted_tickers(df)
        self.assertEqual(len(filtered), 5)
        self.assertEqual(len(removed), 0)

    def test_skip_suffix_filter(self):
        df = self._make_df(['AAPL', 'ACIC-WT', 'MSFT'])
        filtered, removed = dd.filter_unwanted_tickers(df, skip_suffix_filter=True)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(len(removed), 0)

    def test_skip_name_filter(self):
        df = self._make_df(
            ['ARES', 'AAPL'],
            ['Ares Acquisition Corp', 'Apple Inc'],
        )
        filtered, removed = dd.filter_unwanted_tickers(df, skip_name_filter=True)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(len(removed), 0)

    def test_no_name_column(self):
        """Gracefully skips name filter when Name column is absent."""
        df = self._make_df(['AAPL', 'ACIC-WT', 'MSFT'])
        # No Name column — should still filter by suffix
        filtered, removed = dd.filter_unwanted_tickers(df)
        self.assertEqual(sorted(filtered['Tickers'].tolist()), ['AAPL', 'MSFT'])
        self.assertEqual(removed['Tickers'].tolist(), ['ACIC-WT'])

    def test_empty_dataframe(self):
        df = self._make_df([])
        filtered, removed = dd.filter_unwanted_tickers(df)
        self.assertEqual(len(filtered), 0)
        self.assertEqual(len(removed), 0)

    def test_dot_separated_suffixes(self):
        """Tickers using dots instead of dashes are also caught."""
        df = self._make_df(['ABR.PA', 'FOO.WT', 'MSFT'])
        filtered, removed = dd.filter_unwanted_tickers(df)
        self.assertEqual(filtered['Tickers'].tolist(), ['MSFT'])
        self.assertEqual(sorted(removed['Tickers'].tolist()), ['ABR.PA', 'FOO.WT'])


class TestValidateTickers(unittest.TestCase):
    """Tests for the yfinance ticker validation pass."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_result(self, tickers, dates=None):
        """Helper to build a fake download result DataFrame."""
        if dates is None:
            dates = pd.date_range('2024-07-01', periods=5, freq='B')
        return pd.DataFrame(
            {t: range(len(dates)) for t in tickers}, index=dates
        )

    @patch('src.download.core._download_batch_with_timeout')
    def test_valid_tickers_from_successful_batch(self, mock_dl):
        """Tickers present in result columns are returned as valid."""
        mock_dl.return_value = self._make_result(['AAPL', 'MSFT'])
        valid, invalid, unvalidated = dd.validate_tickers(
            ['AAPL', 'MSFT', 'GOOG'],
            batch_size=10, delay=0, max_retries=0, timeout=10,
            validation_windows=[('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir, max_cache_hours=None,
        )
        self.assertIn('AAPL', valid)
        self.assertIn('MSFT', valid)

    @patch('src.download.core._download_batch_with_timeout')
    def test_invalid_tickers_from_successful_batch(self, mock_dl):
        """Tickers NOT in result columns are returned as invalid."""
        mock_dl.return_value = self._make_result(['AAPL'])
        valid, invalid, unvalidated = dd.validate_tickers(
            ['AAPL', 'BADTICKER'],
            batch_size=10, delay=0, max_retries=0, timeout=10,
            validation_windows=[('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir, max_cache_hours=None,
        )
        self.assertIn('AAPL', valid)
        self.assertIn('BADTICKER', invalid)
        self.assertNotIn('BADTICKER', valid)

    @patch('src.download.core._download_batch_with_timeout')
    def test_failed_batch_tickers_are_unvalidated(self, mock_dl):
        """When batch returns None, tickers go to unvalidated (not invalid)."""
        mock_dl.return_value = None  # all retries fail
        valid, invalid, unvalidated = dd.validate_tickers(
            ['AAPL', 'MSFT'],
            batch_size=10, delay=0, max_retries=1, timeout=10,
            validation_windows=[('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir, max_cache_hours=None,
        )
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(invalid), 0)
        self.assertIn('AAPL', unvalidated)
        self.assertIn('MSFT', unvalidated)

    @patch('src.download.core._download_batch_with_timeout')
    def test_two_windows_union(self, mock_dl):
        """Ticker valid in window 2 but not window 1 → still valid."""
        # Per-window data is produced by side_effect below (AAPL in window 1,
        # GOOG in window 2).
        def side_effect(tickers, start, end, timeout):
            if start == '2019-07-01':
                returned = [t for t in tickers if t == 'AAPL']
                if returned:
                    return self._make_result(returned,
                                              pd.date_range('2019-07-01', periods=5, freq='B'))
                return self._make_result([], pd.date_range('2019-07-01', periods=5, freq='B'))
            else:
                returned = [t for t in tickers if t == 'GOOG']
                if returned:
                    return self._make_result(returned,
                                              pd.date_range('2024-07-01', periods=5, freq='B'))
                return self._make_result([], pd.date_range('2024-07-01', periods=5, freq='B'))

        mock_dl.side_effect = side_effect
        valid, invalid, unvalidated = dd.validate_tickers(
            ['AAPL', 'GOOG'],
            batch_size=10, delay=0, max_retries=0, timeout=10,
            validation_windows=[('2019-07-01', '2019-07-08'),
                                ('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir, max_cache_hours=None,
        )
        self.assertIn('AAPL', valid)
        self.assertIn('GOOG', valid)
        self.assertEqual(len(invalid), 0)

    @patch('src.download.core._download_batch_with_timeout')
    def test_uses_cache_when_fresh(self, mock_dl):
        """Loads from JSON cache instead of calling yfinance."""
        import json
        from datetime import datetime as dt_cls
        cache_path = os.path.join(self.tmpdir, 'validated_tickers.json')
        cache_data = {
            'timestamp': dt_cls.now().isoformat(),
            'valid': ['AAPL', 'MSFT'],
            'invalid': ['BADTICKER'],
            'unvalidated': [],
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        valid, invalid, unvalidated = dd.validate_tickers(
            ['AAPL', 'MSFT', 'BADTICKER'],
            batch_size=10, delay=0, max_retries=0, timeout=10,
            cache_dir=self.tmpdir, max_cache_hours=168,
        )
        self.assertIn('AAPL', valid)
        self.assertIn('MSFT', valid)
        self.assertIn('BADTICKER', invalid)
        # Should not have called yfinance at all
        mock_dl.assert_not_called()

    @patch('src.download.core._download_batch_with_timeout')
    def test_refreshes_stale_cache(self, mock_dl):
        """Re-validates when cache is older than max_cache_hours."""
        import json
        from datetime import datetime as dt_cls, timedelta
        cache_path = os.path.join(self.tmpdir, 'validated_tickers.json')
        old_ts = (dt_cls.now() - timedelta(hours=200)).isoformat()
        cache_data = {
            'timestamp': old_ts,
            'valid': ['AAPL'],
            'invalid': ['BADTICKER'],
            'unvalidated': [],
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        mock_dl.return_value = self._make_result(['AAPL', 'MSFT'])
        valid, invalid, unvalidated = dd.validate_tickers(
            ['AAPL', 'MSFT'],
            batch_size=10, delay=0, max_retries=0, timeout=10,
            validation_windows=[('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir, max_cache_hours=168,
        )
        # Should have called yfinance because cache is stale
        mock_dl.assert_called()
        self.assertIn('AAPL', valid)
        self.assertIn('MSFT', valid)

    def test_empty_ticker_list(self):
        """Returns empty sets for empty ticker list."""
        valid, invalid, unvalidated = dd.validate_tickers(
            [],
            batch_size=10, delay=0, max_retries=0, timeout=10,
            validation_windows=[('2024-07-01', '2024-07-08')],
            cache_dir=self.tmpdir, max_cache_hours=None,
        )
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(invalid), 0)
        self.assertEqual(len(unvalidated), 0)


class TestProxyIntegration(unittest.TestCase):
    """Tests for proxy and Tor integration in download session."""

    def setUp(self):
        self._orig_proxy = dl_sess._proxy_url
        self._orig_tor = dl_sess._tor_enabled
        dl_sess._proxy_url = None
        dl_sess._tor_enabled = False

    def tearDown(self):
        dl_sess._proxy_url = self._orig_proxy
        dl_sess._tor_enabled = self._orig_tor

    def test_make_session_with_proxy_adds_proxy(self):
        """When _proxy_url is set, session should have proxy dict set."""
        dl_sess._proxy_url = 'socks5://user:pass@proxy.example.com:1080'
        session = dl_sess._make_session()
        self.assertEqual(session.proxies['http'], dl_sess._proxy_url)
        self.assertEqual(session.proxies['https'], dl_sess._proxy_url)

    def test_make_session_with_tor_adds_proxy(self):
        """When _proxy_url is Tor SOCKS5, session should have proxy dict set."""
        dl_sess._proxy_url = 'socks5://127.0.0.1:9050'
        dl_sess._tor_enabled = True
        session = dl_sess._make_session()
        self.assertIn('socks5://', session.proxies['http'])

    def test_make_session_without_proxy_no_proxy(self):
        """When _proxy_url is None, session should have no proxies."""
        dl_sess._proxy_url = None
        session = dl_sess._make_session()
        self.assertFalse(getattr(session, 'proxies', None))

    def test_rotate_tor_circuit_calls_newnym(self):
        """_rotate_tor_circuit should authenticate and send NEWNYM."""
        import sys
        import unittest.mock as mock

        mock_signal = mock.MagicMock()
        mock_signal.NEWNYM = 'NEWNYM'

        mock_controller_instance = mock.MagicMock()
        mock_controller_cls = mock.MagicMock()
        mock_controller_cls.from_port.return_value.__enter__ = mock.MagicMock(
            return_value=mock_controller_instance)
        mock_controller_cls.from_port.return_value.__exit__ = mock.MagicMock(
            return_value=False)

        mock_stem = mock.MagicMock()
        mock_stem.Signal = mock_signal
        mock_stem_control = mock.MagicMock()
        mock_stem_control.Controller = mock_controller_cls

        with patch.dict(sys.modules, {
            'stem': mock_stem,
            'stem.control': mock_stem_control,
        }):
            _rotate_tor_circuit()

        mock_controller_instance.authenticate.assert_called_once()
        mock_controller_instance.signal.assert_called_once_with('NEWNYM')


class TestPartitionTickers(unittest.TestCase):
    """Tests for _partition_tickers helper."""

    def test_even_split(self):
        tickers = [f'T{i}' for i in range(20)]
        parts = _partition_tickers(tickers, 4)
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(len(p) == 5 for p in parts))
        # All tickers accounted for
        self.assertEqual(sorted(sum(parts, [])), sorted(tickers))

    def test_uneven_split(self):
        tickers = [f'T{i}' for i in range(22)]
        parts = _partition_tickers(tickers, 4)
        self.assertEqual(len(parts), 4)
        sizes = sorted([len(p) for p in parts])
        self.assertEqual(sizes, [5, 5, 6, 6])
        self.assertEqual(sorted(sum(parts, [])), sorted(tickers))

    def test_fewer_tickers_than_workers(self):
        tickers = ['A', 'B']
        parts = _partition_tickers(tickers, 4)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(len(p) == 1 for p in parts))

    def test_single_worker(self):
        tickers = [f'T{i}' for i in range(10)]
        parts = _partition_tickers(tickers, 1)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], tickers)

    def test_empty_tickers(self):
        parts = _partition_tickers([], 4)
        self.assertEqual(len(parts), 0)


class TestConcurrentDownload(unittest.TestCase):
    """Tests for concurrent_download_and_save."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test.db')
        self.conn = db.get_connection(self.db_path)

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch('src.download.core._download_batch')
    def test_concurrent_download_basic(self, mock_dl):
        """Run with 2 workers, verify all tickers saved."""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')

        def fake_download(tickers, start, end, session=None):
            return pd.DataFrame(
                {t: range(5) for t in tickers}, index=dates)

        mock_dl.side_effect = fake_download
        tickers = [f'T{i}' for i in range(10)]
        saved_tickers = []

        def on_complete(saved_list, batch_num):
            saved_tickers.extend(saved_list)

        result = dd.concurrent_download_and_save(
            tickers, self.conn, exchange='US', n_workers=2,
            batch_size=3, max_retries=1, rate_limit_delay=0,
            batch_timeout=10, circuit_breaker_threshold=100,
            on_batch_complete=on_complete,
        )
        self.assertEqual(result['total_tickers'], 10)
        self.assertEqual(result['saved_tickers'], 10)
        self.assertEqual(sorted(saved_tickers), sorted(tickers))

    @patch('src.download.core._download_batch')
    def test_concurrent_download_worker_failure_isolation(self, mock_dl):
        """One worker's batches fail, the other continues."""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        # Worker 0 gets T0-T4, Worker 1 gets T5-T9
        # Fail tickers starting with T0-T4 (worker 0's partition)
        fail_tickers = {f'T{i}' for i in range(5)}

        def fake_download(tickers, start, end, session=None):
            if any(t in fail_tickers for t in tickers):
                return None
            return pd.DataFrame(
                {t: range(5) for t in tickers}, index=dates)

        mock_dl.side_effect = fake_download

        tickers = [f'T{i}' for i in range(10)]
        result = dd.concurrent_download_and_save(
            tickers, self.conn, exchange='US', n_workers=2,
            batch_size=3, max_retries=1, rate_limit_delay=0,
            batch_timeout=10, circuit_breaker_threshold=100,
        )
        # Worker 1's tickers should all be saved
        self.assertEqual(result['saved_tickers'], 5)
        self.assertGreater(len(result['failed_batches']), 0)


class TestSubprocessWorker(unittest.TestCase):
    """Tests for subprocess-based worker and singleton reset."""

    def test_reset_yf_singleton_destroys_instance(self):
        """_reset_yf_singleton should destroy the singleton instance."""
        import yfinance.data as yd

        # Ensure singleton exists
        _ = yd.YfData()
        self.assertIn(yd.YfData, yd.SingletonMeta._instances)

        _reset_yf_singleton()

        self.assertNotIn(yd.YfData, yd.SingletonMeta._instances)

    def test_subprocess_worker_sends_sentinel(self):
        """_subprocess_worker always sends a None sentinel, even with empty tickers."""
        result_queue = multiprocessing.Queue()

        _subprocess_worker(
            worker_id=0, tickers=[], proxy_url=None,
            proxy_counter_start=0, result_queue=result_queue,
            start='2024-01-01', end='2024-02-01', batch_size=5,
            null_threshold=0.9, names=None, countries=None,
            sectors=None, industries=None, category_groups=None,
            categories=None, max_retries=1, rate_limit_delay=0,
            batch_timeout=10, circuit_breaker_threshold=100,
            max_rate_limit_delay=60, circuit_breaker_max_trips=3,
            circuit_breaker_cooldown=10, tor_enabled=False,
            session_rotate_interval=50,
        )

        sentinel = result_queue.get(timeout=5)
        self.assertIsNone(sentinel)

    def test_proxy_triggers_subprocess_path(self):
        """When _proxy_url is set and n_workers > 1, subprocess path is used."""
        from src.download import workers as dw
        old_proxy = dl_sess._proxy_url
        try:
            dl_sess._proxy_url = 'http://user-1:pass@proxy.example.com:8080'

            # Patch _concurrent_subprocess_download to verify it's called
            with patch.object(dw, '_concurrent_subprocess_download',
                              return_value={'total_tickers': 0,
                                            'saved_tickers': 0,
                                            'failed_batches': [],
                                            'circuit_breaker_tripped': False,
                                            'circuit_breaker_trip_count': 0}
                              ) as mock_sub:
                tmpdir = tempfile.mkdtemp()
                db_path = os.path.join(tmpdir, 'test.db')
                conn = db.get_connection(db_path)
                try:
                    dw.concurrent_download_and_save(
                        ['A', 'B'], conn, exchange='US', n_workers=2,
                        batch_size=5, rate_limit_delay=0, batch_timeout=10,
                        circuit_breaker_threshold=100,
                    )
                    mock_sub.assert_called_once()
                finally:
                    conn.close()
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
        finally:
            dl_sess._proxy_url = old_proxy

    def test_no_proxy_uses_thread_path(self):
        """When _proxy_url is None, thread path is used even with multiple workers."""
        from src.download import workers as dw
        old_proxy = dl_sess._proxy_url
        try:
            dl_sess._proxy_url = None

            with patch.object(dw, '_concurrent_thread_download',
                              return_value={'total_tickers': 0,
                                            'saved_tickers': 0,
                                            'failed_batches': [],
                                            'circuit_breaker_tripped': False,
                                            'circuit_breaker_trip_count': 0}
                              ) as mock_thread:
                tmpdir = tempfile.mkdtemp()
                db_path = os.path.join(tmpdir, 'test.db')
                conn = db.get_connection(db_path)
                try:
                    dw.concurrent_download_and_save(
                        ['A', 'B'], conn, exchange='US', n_workers=2,
                        batch_size=5, rate_limit_delay=0, batch_timeout=10,
                        circuit_breaker_threshold=100,
                    )
                    mock_thread.assert_called_once()
                finally:
                    conn.close()
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
        finally:
            dl_sess._proxy_url = old_proxy


if __name__ == '__main__':
    unittest.main()
