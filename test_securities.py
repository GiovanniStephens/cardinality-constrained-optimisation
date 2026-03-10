import unittest
import os
import tempfile

import pandas as pd

import list_of_securities as ls
import download_data as dd


class TestListOfSecurities(unittest.TestCase):
    """Tests for retrieving securities from FinanceDatabase."""

    def test_get_equities_returns_dataframe(self):
        result = ls.get_equities(countries='United States')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_get_equities_has_name_column(self):
        result = ls.get_equities(countries='United States')
        self.assertIn('name', result.columns)

    def test_get_equities_index_contains_tickers(self):
        result = ls.get_equities(countries='United States')
        self.assertTrue(len(result.index) > 0)
        # Tickers should be strings like 'AAPL', 'MSFT', etc.
        self.assertIsInstance(result.index[0], str)

    def test_get_etfs_returns_dataframe(self):
        result = ls.get_etfs()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_get_funds_returns_dataframe(self):
        result = ls.get_funds()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_build_universe_equities_only(self):
        result = ls.build_security_universe(
            asset_types=['equities'],
            countries='United States'
        )
        self.assertIn('Tickers', result.columns)
        self.assertIn('Name', result.columns)
        self.assertIn('AssetType', result.columns)
        self.assertTrue((result['AssetType'] == 'equity').all())

    def test_build_universe_etfs_only(self):
        result = ls.build_security_universe(asset_types=['etfs'])
        self.assertTrue((result['AssetType'] == 'etf').all())

    def test_build_universe_mixed(self):
        result = ls.build_security_universe(
            asset_types=['equities', 'etfs'],
            countries='United States'
        )
        asset_types = result['AssetType'].unique()
        self.assertIn('equity', asset_types)
        self.assertIn('etf', asset_types)

    def test_build_universe_no_duplicates(self):
        result = ls.build_security_universe(
            asset_types=['equities'],
            countries='United States'
        )
        self.assertEqual(len(result), len(result.drop_duplicates(subset='Tickers')))


class TestDownloadData(unittest.TestCase):
    """Tests for downloading price data from Yahoo Finance."""

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

    def test_load_etfs_backward_compat(self):
        """Ensure the old load_etfs alias still works."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            f.write('Tickers\nSPY\n')
            f.flush()
            result = dd.load_etfs(f.name)
        os.unlink(f.name)
        self.assertEqual(list(result['Tickers']), ['SPY'])

    def test_download_equity_prices(self):
        """Download real price data for a couple of well-known stocks."""
        tickers_df = pd.DataFrame({'Tickers': ['AAPL', 'MSFT']})
        prices = dd.download_data(tickers_df, start='2024-01-01',
                                  end='2024-02-01')
        self.assertIsInstance(prices, pd.DataFrame)
        self.assertGreater(len(prices), 0)
        self.assertIn('AAPL', prices.columns)
        self.assertIn('MSFT', prices.columns)

    def test_download_etf_prices(self):
        """Download real price data for an ETF to confirm parity."""
        tickers_df = pd.DataFrame({'Tickers': ['SPY']})
        prices = dd.download_data(tickers_df, start='2024-01-01',
                                  end='2024-02-01')
        self.assertIsInstance(prices, pd.DataFrame)
        self.assertGreater(len(prices), 0)
        self.assertIn('SPY', prices.columns)

    def test_download_skips_invalid_ticker(self):
        """Invalid tickers should be skipped, not crash."""
        tickers_df = pd.DataFrame({'Tickers': ['AAPL', 'ZZZZZNOTREAL99']})
        prices = dd.download_data(tickers_df, start='2024-01-01',
                                  end='2024-02-01')
        self.assertIn('AAPL', prices.columns)

    def test_download_returns_numeric_data(self):
        """Price data should be numeric, not strings."""
        tickers_df = pd.DataFrame({'Tickers': ['AAPL']})
        prices = dd.download_data(tickers_df, start='2024-01-01',
                                  end='2024-02-01')
        self.assertTrue(prices['AAPL'].dtype in ['float64', 'float32'])


class TestEndToEnd(unittest.TestCase):
    """End-to-end: FinanceDatabase tickers -> Yahoo Finance prices."""

    def test_equity_tickers_to_prices(self):
        """Build a small equity universe and download prices for two tickers."""
        securities = ls.build_security_universe(
            asset_types=['equities'],
            countries='United States'
        )
        # Pick two tickers from the universe
        sample = securities.head(2)
        prices = dd.download_data(sample, ticker_column='Tickers',
                                  start='2024-01-01', end='2024-02-01')
        self.assertIsInstance(prices, pd.DataFrame)
        self.assertGreater(len(prices), 0)
        self.assertGreater(len(prices.columns), 0)

    def test_save_and_reload_securities_csv(self):
        """Securities CSV round-trips correctly for downstream consumption."""
        securities = ls.build_security_universe(
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


if __name__ == '__main__':
    unittest.main()
