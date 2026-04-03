"""Unit tests for the database module (db.py)."""

import json
import unittest

import numpy as np
import pandas as pd

from src import db


class TestDBConnection(unittest.TestCase):
    """Test database connection and schema creation."""

    def test_get_connection_creates_tables(self):
        conn = db.get_connection(':memory:')
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t['name'] for t in tables]
        expected = [
            'backtest_holdings', 'backtest_results', 'backtest_sessions',
            'data_sources', 'exchanges', 'expected_returns', 'forecast_runs',
            'optimisation_runs', 'portfolio_holdings', 'prices',
            'tickers', 'variances',
        ]
        for name in expected:
            self.assertIn(name, table_names)
        conn.close()

    def test_schema_creation_is_idempotent(self):
        conn = db.get_connection(':memory:')
        # Run schema again — should not raise
        conn.executescript(db.SCHEMA_SQL)
        conn.close()

    def test_exchanges_seeded(self):
        conn = db.get_connection(':memory:')
        rows = conn.execute("SELECT code FROM exchanges ORDER BY code").fetchall()
        codes = [r['code'] for r in rows]
        self.assertEqual(codes, ['ASX', 'NZX', 'US'])
        conn.close()

    def test_exchanges_not_duplicated_on_reconnect(self):
        conn = db.get_connection(':memory:')
        count_before = conn.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0]
        count_after = conn.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0]
        self.assertEqual(count_before, count_after)
        conn.close()

    def test_foreign_keys_enforced(self):
        conn = db.get_connection(':memory:')
        with self.assertRaises(Exception):
            conn.execute(
                "INSERT INTO tickers (symbol, exchange_id, asset_type, created_at, updated_at) "
                "VALUES ('TEST', 9999, 'etf', '2025-01-01', '2025-01-01')"
            )
        conn.close()


class TestPrices(unittest.TestCase):
    """Test save_prices and load_prices round-trip."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        self.prices_df = pd.DataFrame({
            'SPY': [100.0, 101.0, 102.0, 103.0, 104.0],
            'QQQ': [200.0, 201.0, 202.0, 203.0, 204.0],
            'VTI': [300.0, np.nan, 302.0, 303.0, 304.0],
        }, index=dates)

    def tearDown(self):
        self.conn.close()

    def test_save_and_load_roundtrip(self):
        ds_id = db.save_prices(self.conn, self.prices_df, exchange='US', asset_type='etf')
        self.assertIsInstance(ds_id, int)
        self.assertGreater(ds_id, 0)

        loaded = db.load_prices(self.conn, exchange='US')
        self.assertEqual(set(loaded.columns), {'SPY', 'QQQ', 'VTI'})
        self.assertEqual(len(loaded), 5)

        self.assertAlmostEqual(loaded.loc['2024-01-01', 'SPY'], 100.0)
        self.assertAlmostEqual(loaded.loc['2024-01-05', 'QQQ'], 204.0)

    def test_load_with_date_filter(self):
        db.save_prices(self.conn, self.prices_df, exchange='US')
        loaded = db.load_prices(self.conn, exchange='US',
                                start='2024-01-03', end='2024-01-04')
        self.assertEqual(len(loaded), 2)

    def test_load_with_ticker_filter(self):
        db.save_prices(self.conn, self.prices_df, exchange='US')
        loaded = db.load_prices(self.conn, exchange='US', tickers=['SPY'])
        self.assertEqual(list(loaded.columns), ['SPY'])

    def test_load_min_coverage_filter(self):
        db.save_prices(self.conn, self.prices_df, exchange='US')
        loaded = db.load_prices(self.conn, exchange='US', min_coverage=1.0)
        self.assertNotIn('VTI', loaded.columns)
        self.assertIn('SPY', loaded.columns)

    def test_save_records_data_source(self):
        db.save_prices(self.conn, self.prices_df, exchange='US')
        ds = db.get_latest_data_source(self.conn)
        self.assertIsNotNone(ds)
        self.assertEqual(ds['num_tickers'], 3)

    def test_tickers_created(self):
        db.save_prices(self.conn, self.prices_df, exchange='US', asset_type='etf')
        rows = self.conn.execute("SELECT symbol FROM tickers ORDER BY symbol").fetchall()
        symbols = [r['symbol'] for r in rows]
        self.assertEqual(symbols, ['QQQ', 'SPY', 'VTI'])

    def test_upsert_prices(self):
        db.save_prices(self.conn, self.prices_df, exchange='US')
        updated = self.prices_df.copy()
        updated.loc[updated.index[0], 'SPY'] = 999.0
        db.save_prices(self.conn, updated, exchange='US')
        loaded = db.load_prices(self.conn, exchange='US')
        self.assertAlmostEqual(loaded.loc['2024-01-01', 'SPY'], 999.0)


class TestPricesDateConstraint(unittest.TestCase):
    """Test that prices.date rejects non-date strings."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')

    def tearDown(self):
        self.conn.close()

    def test_invalid_date_rejected(self):
        self.conn.execute(
            "INSERT INTO tickers (symbol, exchange_id, asset_type, created_at, updated_at) "
            "VALUES ('TEST', 1, 'etf', '2025-01-01', '2025-01-01')"
        )
        ticker_id = self.conn.execute("SELECT id FROM tickers WHERE symbol='TEST'").fetchone()[0]
        with self.assertRaises(Exception):
            self.conn.execute(
                "INSERT INTO prices (ticker_id, date, close) VALUES (?, ?, ?)",
                (ticker_id, 'not-a-date', 100.0),
            )

    def test_integer_date_rejected(self):
        self.conn.execute(
            "INSERT INTO tickers (symbol, exchange_id, asset_type, created_at, updated_at) "
            "VALUES ('TEST', 1, 'etf', '2025-01-01', '2025-01-01')"
        )
        ticker_id = self.conn.execute("SELECT id FROM tickers WHERE symbol='TEST'").fetchone()[0]
        with self.assertRaises(Exception):
            self.conn.execute(
                "INSERT INTO prices (ticker_id, date, close) VALUES (?, ?, ?)",
                (ticker_id, '42', 100.0),
            )


class TestAssetTypeConstraint(unittest.TestCase):
    """Test tickers.asset_type NOT NULL and CHECK constraint."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')

    def tearDown(self):
        self.conn.close()

    def test_invalid_asset_type_rejected(self):
        with self.assertRaises(Exception):
            self.conn.execute(
                "INSERT INTO tickers (symbol, exchange_id, asset_type, created_at, updated_at) "
                "VALUES ('TEST', 1, 'invalid_type', '2025-01-01', '2025-01-01')"
            )

    def test_valid_asset_types_accepted(self):
        for i, at in enumerate(('etf', 'stock', 'fund', 'managed_fund')):
            self.conn.execute(
                "INSERT INTO tickers (symbol, exchange_id, asset_type, created_at, updated_at) "
                "VALUES (?, 1, ?, '2025-01-01', '2025-01-01')",
                (f'TEST{i}', at),
            )
        count = self.conn.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
        self.assertEqual(count, 4)


class TestForecasts(unittest.TestCase):
    """Test forecast save/load round-trip."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')
        self.er = pd.Series({'SPY': 0.08, 'QQQ': 0.12, 'VTI': 0.07})
        self.var = pd.Series({'SPY': 0.02, 'QQQ': 0.04, 'VTI': 0.015})

    def tearDown(self):
        self.conn.close()

    def test_save_and_load_expected_returns(self):
        run_id = db.save_forecast_results(self.conn, self.er, self.var, n_periods=252)
        loaded = db.load_expected_returns(self.conn, run_id)
        self.assertEqual(len(loaded), 3)
        self.assertAlmostEqual(loaded['SPY'], 0.08)
        self.assertAlmostEqual(loaded['QQQ'], 0.12)

    def test_save_and_load_variances(self):
        run_id = db.save_forecast_results(self.conn, self.er, self.var, n_periods=252)
        loaded = db.load_variances(self.conn, run_id)
        self.assertEqual(len(loaded), 3)
        self.assertAlmostEqual(loaded['SPY'], 0.02)

    def test_load_latest_forecast(self):
        db.save_forecast_results(self.conn, self.er, self.var, n_periods=252)
        er2 = pd.Series({'SPY': 0.10, 'QQQ': 0.15})
        var2 = pd.Series({'SPY': 0.03, 'QQQ': 0.05})
        db.save_forecast_results(self.conn, er2, var2, n_periods=126)

        loaded = db.load_expected_returns(self.conn)
        self.assertEqual(len(loaded), 2)
        self.assertAlmostEqual(loaded['SPY'], 0.10)

    def test_get_latest_forecast(self):
        db.save_forecast_results(self.conn, self.er, self.var, n_periods=252)
        latest = db.get_latest_forecast(self.conn)
        self.assertIsNotNone(latest)
        self.assertEqual(latest['n_periods'], 252)
        self.assertEqual(latest['num_tickers'], 3)

    def test_forecast_stores_exchange_id(self):
        db.save_forecast_results(self.conn, self.er, self.var,
                                 n_periods=252, exchange='US')
        latest = db.get_latest_forecast(self.conn)
        us_id = db._get_exchange_id(self.conn, 'US')
        self.assertEqual(latest['exchange_id'], us_id)

    def test_empty_forecast(self):
        loaded = db.load_expected_returns(self.conn)
        self.assertEqual(len(loaded), 0)


class TestOptimisationRuns(unittest.TestCase):
    """Test optimisation run save/retrieve."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')
        # Pre-create tickers so holdings can resolve
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        prices = pd.DataFrame({
            'SPY': [100.0, 101.0, 102.0],
            'QQQ': [200.0, 201.0, 202.0],
            'VTI': [300.0, 301.0, 302.0],
        }, index=dates)
        db.save_prices(self.conn, prices, exchange='US', asset_type='etf')

    def tearDown(self):
        self.conn.close()

    def test_save_and_get_run(self):
        run_id = db.save_optimisation_run(
            self.conn,
            params={
                'script': 'simple_ga_optimisation',
                'data_source': 'investnow',
                'num_generations': 70,
                'total_population_size': 8000,
                'mutation_rate': 0.01,
                'num_elites': 100,
                'min_securities': 8,
                'max_securities': 20,
            },
            results={
                'best_sharpe': 1.5,
                'portfolio_return': 0.15,
                'portfolio_volatility': 0.10,
                'num_selected': 12,
                'elapsed_seconds': 45.3,
            },
            holdings=[('SPY', 0.3), ('QQQ', 0.4), ('VTI', 0.3)],
            exchange='US',
        )
        self.assertIsInstance(run_id, int)
        self.assertGreater(run_id, 0)

        # Check holdings (returned with ticker symbol via JOIN)
        holdings = db.get_run_holdings(self.conn, run_id)
        self.assertEqual(len(holdings), 3)
        tickers = [h['ticker'] for h in holdings]
        self.assertIn('SPY', tickers)

    def test_params_json_stored(self):
        run_id = db.save_optimisation_run(
            self.conn,
            params={
                'script': 'simple_ga_optimisation',
                'num_generations': 70,
                'total_population_size': 8000,
                'mutation_rate': 0.01,
            },
            results={'best_sharpe': 1.5},
            holdings=[],
            exchange='US',
        )
        row = self.conn.execute(
            "SELECT params_json FROM optimisation_runs WHERE id = ?", (run_id,)
        ).fetchone()
        params = json.loads(row['params_json'])
        self.assertEqual(params['num_generations'], 70)
        self.assertAlmostEqual(params['mutation_rate'], 0.01)

    def test_get_recent_runs(self):
        for i in range(3):
            db.save_optimisation_run(
                self.conn,
                params={'script': 'test', 'best_sharpe': float(i)},
                results={},
                holdings=[],
                exchange='US',
            )
        runs = db.get_recent_runs(self.conn, n=2)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]['id'], 3)

    def test_get_recent_runs_by_script(self):
        db.save_optimisation_run(self.conn, params={'script': 'a'}, results={},
                                 holdings=[], exchange='US')
        db.save_optimisation_run(self.conn, params={'script': 'b'}, results={},
                                 holdings=[], exchange='US')
        runs = db.get_recent_runs(self.conn, script='a')
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]['script'], 'a')

    def test_holdings_fk_enforced(self):
        """Holdings must reference valid tickers via ticker_id."""
        run_id = db.save_optimisation_run(
            self.conn,
            params={'script': 'test'},
            results={},
            holdings=[],
            exchange='US',
        )
        with self.assertRaises(Exception):
            self.conn.execute(
                "INSERT INTO portfolio_holdings (run_id, ticker_id, weight) VALUES (?, ?, ?)",
                (run_id, 99999, 0.5),
            )


class TestBacktest(unittest.TestCase):
    """Test backtest session and results save/retrieve."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')
        # Pre-create tickers for holdings resolution
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        prices = pd.DataFrame({
            'SPY': [100.0, 101.0, 102.0],
            'QQQ': [200.0, 201.0, 202.0],
        }, index=dates)
        db.save_prices(self.conn, prices, exchange='US', asset_type='etf')

    def tearDown(self):
        self.conn.close()

    def test_save_session_and_results(self):
        session_id = db.save_backtest_session(self.conn, {
            'data_source': 'yahoo_finance',
            'num_portfolios': 20,
            'num_days_oos': 252,
            'use_forecast': True,
            'optimiser_params': {'num_children': 100},
        })
        self.assertIsInstance(session_id, int)

        db.save_backtest_result(self.conn, session_id, 'cc_optimised', 0,
                                metrics={
                                    'annualised_return': 0.12,
                                    'annualised_volatility': 0.08,
                                    'sharpe_ratio': 1.5,
                                    'downside_deviation': 0.05,
                                    'max_drawdown': -0.15,
                                    'calmar_ratio': 0.8,
                                    'sortino_ratio': 2.4,
                                },
                                holdings=[('SPY', 0.5), ('QQQ', 0.5)],
                                exchange='US')

        results = db.get_backtest_results(self.conn, session_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['category'], 'cc_optimised')
        self.assertAlmostEqual(results[0]['sharpe_ratio'], 1.5)

        # Check holdings JSON (convenience cache)
        holdings = json.loads(results[0]['holdings_json'])
        self.assertEqual(len(holdings), 2)
        self.assertEqual(holdings[0]['ticker'], 'SPY')

    def test_backtest_holdings_normalised(self):
        """Normalised backtest_holdings table is populated alongside holdings_json."""
        session_id = db.save_backtest_session(self.conn, {
            'num_portfolios': 1,
            'num_days_oos': 252,
        })
        db.save_backtest_result(self.conn, session_id, 'cc_optimised', 0,
                                metrics={'sharpe_ratio': 1.0},
                                holdings=[('SPY', 0.6), ('QQQ', 0.4)],
                                exchange='US')

        result_id = self.conn.execute(
            "SELECT id FROM backtest_results WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        rows = self.conn.execute(
            "SELECT t.symbol, bh.weight "
            "FROM backtest_holdings bh JOIN tickers t ON bh.ticker_id = t.id "
            "WHERE bh.result_id = ? ORDER BY bh.weight DESC",
            (result_id,),
        ).fetchall()
        self.assertEqual(len(rows), 2)
        symbols = {r['symbol'] for r in rows}
        self.assertEqual(symbols, {'SPY', 'QQQ'})

    def test_backtest_unique_constraint(self):
        """Cannot save duplicate (session_id, category, portfolio_index)."""
        session_id = db.save_backtest_session(self.conn, {
            'num_portfolios': 1,
            'num_days_oos': 252,
        })
        db.save_backtest_result(self.conn, session_id, 'cc_optimised', 0,
                                metrics={'sharpe_ratio': 1.0},
                                exchange='US')
        with self.assertRaises(Exception):
            db.save_backtest_result(self.conn, session_id, 'cc_optimised', 0,
                                    metrics={'sharpe_ratio': 2.0},
                                    exchange='US')

    def test_optimiser_params_json_stored(self):
        session_id = db.save_backtest_session(self.conn, {
            'num_portfolios': 20,
            'num_days_oos': 252,
            'optimiser_params': {'num_children': 100, 'method': 'ga'},
        })
        row = self.conn.execute(
            "SELECT optimiser_params_json FROM backtest_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        params = json.loads(row['optimiser_params_json'])
        self.assertEqual(params['num_children'], 100)

    def test_get_recent_backtests(self):
        for i in range(3):
            db.save_backtest_session(self.conn, {
                'num_portfolios': 10 + i,
                'num_days_oos': 252,
            })
        recent = db.get_recent_backtests(self.conn, n=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]['id'], 3)


class TestMetadata(unittest.TestCase):
    """Test data source and metadata functions."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')

    def tearDown(self):
        self.conn.close()

    def test_save_and_get_data_source(self):
        ds_id = db.save_data_source(
            self.conn, source='yahoo_finance',
            num_tickers=100, num_rows=50000,
        )
        self.assertIsInstance(ds_id, int)

        latest = db.get_latest_data_source(self.conn, source='yahoo_finance')
        self.assertIsNotNone(latest)
        self.assertEqual(latest['source'], 'yahoo_finance')
        self.assertEqual(latest['num_tickers'], 100)

    def test_get_latest_data_source_none(self):
        result = db.get_latest_data_source(self.conn)
        self.assertIsNone(result)


class TestTickerNames(unittest.TestCase):
    """Test ticker name storage, backward compat, and backfill."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        self.prices_df = pd.DataFrame({
            'SPY': [100.0, 101.0, 102.0],
            'QQQ': [200.0, 201.0, 202.0],
        }, index=dates)

    def tearDown(self):
        self.conn.close()

    def test_name_stored_when_provided(self):
        names = {'SPY': 'SPDR S&P 500 ETF', 'QQQ': 'Invesco QQQ Trust'}
        db.save_prices(self.conn, self.prices_df, exchange='US', names=names)
        row = self.conn.execute(
            "SELECT name FROM tickers WHERE symbol = 'SPY'"
        ).fetchone()
        self.assertEqual(row['name'], 'SPDR S&P 500 ETF')

    def test_name_null_when_not_provided(self):
        db.save_prices(self.conn, self.prices_df, exchange='US')
        row = self.conn.execute(
            "SELECT name FROM tickers WHERE symbol = 'SPY'"
        ).fetchone()
        self.assertIsNone(row['name'])

    def test_name_backfill_on_existing_ticker(self):
        db.save_prices(self.conn, self.prices_df, exchange='US')
        row = self.conn.execute(
            "SELECT name FROM tickers WHERE symbol = 'SPY'"
        ).fetchone()
        self.assertIsNone(row['name'])

        names = {'SPY': 'SPDR S&P 500 ETF'}
        db.save_prices(self.conn, self.prices_df, exchange='US', names=names)
        row = self.conn.execute(
            "SELECT name FROM tickers WHERE symbol = 'SPY'"
        ).fetchone()
        self.assertEqual(row['name'], 'SPDR S&P 500 ETF')

    def test_name_column_exists_in_schema(self):
        cols = [row[1] for row in
                self.conn.execute("PRAGMA table_info(tickers)").fetchall()]
        self.assertIn('name', cols)


class TestSavePricesEdgeCases(unittest.TestCase):
    def setUp(self):
        self.conn = db.get_connection(':memory:')
        self.dates = pd.date_range('2024-01-01', periods=3, freq='D')

    def tearDown(self):
        self.conn.close()

    def test_save_prices_duplicate_column_names(self):
        df = pd.DataFrame(
            [[100.0, 101.0], [102.0, 103.0], [104.0, 105.0]],
            index=self.dates,
            columns=['SPY', 'SPY'],
        )
        with self.assertRaises(Exception):
            db.save_prices(self.conn, df, exchange='US')


class TestLoadPricesAssetTypeFilter(unittest.TestCase):
    """Test that load_prices can filter by asset_type."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        self.etf_prices = pd.DataFrame(
            {'SPY': [100.0, 101.0, 102.0], 'QQQ': [200.0, 201.0, 202.0]},
            index=dates,
        )
        self.stock_prices = pd.DataFrame(
            {'AAPL': [150.0, 151.0, 152.0], 'MSFT': [300.0, 301.0, 302.0]},
            index=dates,
        )
        db.save_prices(self.conn, self.etf_prices, exchange='US', asset_type='etf')
        db.save_prices(self.conn, self.stock_prices, exchange='US', asset_type='stock')

    def tearDown(self):
        self.conn.close()

    def test_filter_by_stock(self):
        result = db.load_prices(self.conn, exchange='US', asset_type='stock')
        self.assertEqual(sorted(result.columns.tolist()), ['AAPL', 'MSFT'])

    def test_filter_by_etf(self):
        result = db.load_prices(self.conn, exchange='US', asset_type='etf')
        self.assertEqual(sorted(result.columns.tolist()), ['QQQ', 'SPY'])

    def test_no_filter_returns_all(self):
        result = db.load_prices(self.conn, exchange='US')
        self.assertEqual(sorted(result.columns.tolist()),
                         ['AAPL', 'MSFT', 'QQQ', 'SPY'])

    def test_filter_nonexistent_type_returns_empty(self):
        result = db.load_prices(self.conn, exchange='US', asset_type='managed_fund')
        self.assertTrue(result.empty)


class TestGetLatestPricesDate(unittest.TestCase):
    """Test get_latest_prices_date helper."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        etf_prices = pd.DataFrame({'SPY': [100.0, 101, 102, 103, 104]}, index=dates)
        stock_dates = pd.date_range('2024-01-01', periods=3, freq='D')
        stock_prices = pd.DataFrame({'AAPL': [150.0, 151, 152]}, index=stock_dates)
        db.save_prices(self.conn, etf_prices, exchange='US', asset_type='etf')
        db.save_prices(self.conn, stock_prices, exchange='US', asset_type='stock')

    def tearDown(self):
        self.conn.close()

    def test_latest_date_no_filter(self):
        result = db.get_latest_prices_date(self.conn, exchange='US')
        self.assertEqual(result, '2024-01-05')

    def test_latest_date_filter_stock(self):
        result = db.get_latest_prices_date(self.conn, exchange='US', asset_type='stock')
        self.assertEqual(result, '2024-01-03')

    def test_latest_date_filter_etf(self):
        result = db.get_latest_prices_date(self.conn, exchange='US', asset_type='etf')
        self.assertEqual(result, '2024-01-05')

    def test_latest_date_empty_db(self):
        empty_conn = db.get_connection(':memory:')
        result = db.get_latest_prices_date(empty_conn)
        self.assertIsNone(result)
        empty_conn.close()


class TestCountryColumn(unittest.TestCase):
    """Test country metadata storage and filtering."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        self.us_prices = pd.DataFrame(
            {'AAPL': [150.0, 151, 152], 'MSFT': [300.0, 301, 302]},
            index=dates,
        )
        self.jp_prices = pd.DataFrame(
            {'TM': [180.0, 181, 182]},
            index=dates,
        )

    def tearDown(self):
        self.conn.close()

    def test_country_stored_when_provided(self):
        countries = {'AAPL': 'United States', 'MSFT': 'United States'}
        db.save_prices(self.conn, self.us_prices, exchange='US',
                       asset_type='stock', countries=countries)
        row = self.conn.execute(
            "SELECT country FROM tickers WHERE symbol = 'AAPL'"
        ).fetchone()
        self.assertEqual(row['country'], 'United States')

    def test_country_null_when_not_provided(self):
        db.save_prices(self.conn, self.us_prices, exchange='US', asset_type='stock')
        row = self.conn.execute(
            "SELECT country FROM tickers WHERE symbol = 'AAPL'"
        ).fetchone()
        self.assertIsNone(row['country'])

    def test_country_backfill(self):
        db.save_prices(self.conn, self.us_prices, exchange='US', asset_type='stock')
        countries = {'AAPL': 'United States'}
        db.save_prices(self.conn, self.us_prices, exchange='US',
                       asset_type='stock', countries=countries)
        row = self.conn.execute(
            "SELECT country FROM tickers WHERE symbol = 'AAPL'"
        ).fetchone()
        self.assertEqual(row['country'], 'United States')

    def test_exclude_countries_filter(self):
        countries_us = {'AAPL': 'United States', 'MSFT': 'United States'}
        countries_jp = {'TM': 'Japan'}
        db.save_prices(self.conn, self.us_prices, exchange='US',
                       asset_type='stock', countries=countries_us)
        db.save_prices(self.conn, self.jp_prices, exchange='US',
                       asset_type='stock', countries=countries_jp)
        result = db.load_prices(self.conn, exchange='US',
                                exclude_countries=['Japan'])
        self.assertEqual(sorted(result.columns.tolist()), ['AAPL', 'MSFT'])

    def test_exclude_countries_keeps_null_country(self):
        db.save_prices(self.conn, self.us_prices, exchange='US', asset_type='etf')
        countries_jp = {'TM': 'Japan'}
        db.save_prices(self.conn, self.jp_prices, exchange='US',
                       asset_type='stock', countries=countries_jp)
        result = db.load_prices(self.conn, exchange='US',
                                exclude_countries=['Japan'])
        self.assertIn('AAPL', result.columns)
        self.assertNotIn('TM', result.columns)


if __name__ == '__main__':
    unittest.main()
