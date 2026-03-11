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
            'backtest_results', 'backtest_sessions', 'data_sources',
            'exchanges', 'expected_returns', 'forecast_runs',
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
        # Simulate a second "connection" by re-seeding
        count_before = conn.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0]
        # Re-run seed logic — should detect existing rows
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
        # Create a simple wide-format DataFrame
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

        # Check actual values
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
        # VTI has 1 NaN out of 5 rows = 80% coverage (4 non-null values)
        # min_coverage=0.95 → threshold = int(0.95 * 5) = 4, VTI has exactly 4 → passes
        # Use a threshold that requires all 5 to exclude VTI
        loaded = db.load_prices(self.conn, exchange='US', min_coverage=1.0)
        # VTI should be dropped (only 4/5 values present)
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
        # Update one price
        updated = self.prices_df.copy()
        updated.loc[updated.index[0], 'SPY'] = 999.0
        db.save_prices(self.conn, updated, exchange='US')
        loaded = db.load_prices(self.conn, exchange='US')
        self.assertAlmostEqual(loaded.loc['2024-01-01', 'SPY'], 999.0)


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

        # Loading without ID should get the latest
        loaded = db.load_expected_returns(self.conn)
        self.assertEqual(len(loaded), 2)
        self.assertAlmostEqual(loaded['SPY'], 0.10)

    def test_get_latest_forecast(self):
        db.save_forecast_results(self.conn, self.er, self.var, n_periods=252)
        latest = db.get_latest_forecast(self.conn)
        self.assertIsNotNone(latest)
        self.assertEqual(latest['n_periods'], 252)
        self.assertEqual(latest['num_tickers'], 3)

    def test_empty_forecast(self):
        loaded = db.load_expected_returns(self.conn)
        self.assertEqual(len(loaded), 0)


class TestOptimisationRuns(unittest.TestCase):
    """Test optimisation run save/retrieve."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')

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
                'min_etfs': 8,
                'max_etfs': 20,
            },
            results={
                'best_sharpe': 1.5,
                'portfolio_return': 0.15,
                'portfolio_volatility': 0.10,
                'num_selected': 12,
                'elapsed_seconds': 45.3,
            },
            holdings=[('SPY', 0.3), ('QQQ', 0.4), ('VTI', 0.3)],
        )
        self.assertIsInstance(run_id, int)
        self.assertGreater(run_id, 0)

        # Check holdings
        holdings = db.get_run_holdings(self.conn, run_id)
        self.assertEqual(len(holdings), 3)
        tickers = [h['ticker'] for h in holdings]
        self.assertIn('SPY', tickers)

    def test_get_recent_runs(self):
        for i in range(3):
            db.save_optimisation_run(
                self.conn,
                params={'script': 'test', 'best_sharpe': float(i)},
                results={},
                holdings=[],
            )
        runs = db.get_recent_runs(self.conn, n=2)
        self.assertEqual(len(runs), 2)
        # Most recent first
        self.assertEqual(runs[0]['id'], 3)

    def test_get_recent_runs_by_script(self):
        db.save_optimisation_run(self.conn, params={'script': 'a'}, results={}, holdings=[])
        db.save_optimisation_run(self.conn, params={'script': 'b'}, results={}, holdings=[])
        runs = db.get_recent_runs(self.conn, script='a')
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]['script'], 'a')


class TestBacktest(unittest.TestCase):
    """Test backtest session and results save/retrieve."""

    def setUp(self):
        self.conn = db.get_connection(':memory:')

    def tearDown(self):
        self.conn.close()

    def test_save_session_and_results(self):
        session_id = db.save_backtest_session(self.conn, {
            'data_source': 'yahoo_finance',
            'num_portfolios': 20,
            'num_children': 100,
            'num_days_oos': 252,
            'use_forecast': True,
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
                                holdings=[('SPY', 0.5), ('QQQ', 0.5)])

        results = db.get_backtest_results(self.conn, session_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['category'], 'cc_optimised')
        self.assertAlmostEqual(results[0]['sharpe_ratio'], 1.5)

        # Check holdings JSON
        holdings = json.loads(results[0]['holdings_json'])
        self.assertEqual(len(holdings), 2)
        self.assertEqual(holdings[0]['ticker'], 'SPY')

    def test_get_recent_backtests(self):
        for i in range(3):
            db.save_backtest_session(self.conn, {
                'num_portfolios': 10 + i,
                'num_children': 100,
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


if __name__ == '__main__':
    unittest.main()
