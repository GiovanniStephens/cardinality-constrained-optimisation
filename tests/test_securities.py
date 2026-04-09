import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src import db
from src import download_data as dd
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

    @patch('src.download_data.yf.download')
    def test_handles_empty_dataframe(self, mock_yf):
        mock_yf.return_value = pd.DataFrame()
        result = dd._download_batch(['SPY'], '2024-01-01', '2024-02-01')
        self.assertIsNone(result)

    @patch('src.download_data.yf.download')
    def test_handles_none_return(self, mock_yf):
        mock_yf.return_value = None
        result = dd._download_batch(['SPY'], '2024-01-01', '2024-02-01')
        self.assertIsNone(result)

    @patch('src.download_data.yf.download')
    def test_handles_non_datetime_index(self, mock_yf):
        # Return a DataFrame with integer index that can be coerced to datetime
        dates = pd.date_range('2024-01-01', periods=3, freq='B')
        df = pd.DataFrame({'Close': [100, 101, 102]}, index=dates.strftime('%Y-%m-%d'))
        mock_yf.return_value = df
        result = dd._download_batch(['SPY'], '2024-01-01', '2024-01-05')
        # Should succeed after coercing index
        self.assertIsNotNone(result)

    @patch('src.download_data.yf.download')
    def test_handles_unconvertible_index(self, mock_yf):
        df = pd.DataFrame({'Close': [100, 101]}, index=['not-a-date', 'also-not'])
        mock_yf.return_value = df
        result = dd._download_batch(['SPY'], '2024-01-01', '2024-01-05')
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

    @patch('src.download_data._download_batch_with_timeout')
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

    @patch('src.download_data.yf.download')
    def test_batch_skipped_tickers_logged_as_summary(self, mock_yf):
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        # Return a DataFrame that only has data for one ticker out of many
        df = pd.DataFrame(
            {('SPY', 'Close'): range(5)},
            index=dates,
        )
        df.columns = pd.MultiIndex.from_tuples([('SPY', 'Close')])
        mock_yf.return_value = df

        with self.assertLogs('src.download_data', level='INFO') as cm:
            # Request many tickers but only SPY has data
            tickers = ['SPY'] + [f'BAD{i}' for i in range(20)]
            dd._download_batch(tickers, '2024-01-01', '2024-02-01')

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


if __name__ == '__main__':
    unittest.main()
