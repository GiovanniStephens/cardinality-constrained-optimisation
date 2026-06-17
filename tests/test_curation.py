"""Tests for the data-driven curated ETF universe + volume schema."""

import os
import tempfile
import unittest

import pandas as pd

from src import db
from src.db.schema import _migrate_to_4
from src.db.prices import update_volumes, load_avg_dollar_volume
from src.data_loading import load_curated_universe
from tests.helpers import BaseDBTest

from curate_universe import elect_representatives


class TestVolumeSchema(BaseDBTest):
    """Migration v4: prices.volume column."""

    def test_volume_column_present(self):
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(prices)").fetchall()]
        self.assertIn('volume', cols)

    def test_migration_idempotent(self):
        # Re-running _migrate_to_4 on a table that already has volume is a no-op.
        _migrate_to_4(self.conn)
        _migrate_to_4(self.conn)
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(prices)").fetchall()]
        self.assertEqual(cols.count('volume'), 1)


class TestVolumeStoreAndADV(BaseDBTest):
    """update_volumes writes onto existing rows; load_avg_dollar_volume scores."""

    def setUp(self):
        super().setUp()
        # Two tickers, flat $10 and $20 close so ADV = price * volume is clean.
        dates = pd.bdate_range('2022-01-03', periods=5, freq='B')
        self.prices = pd.DataFrame({'AAA': [10.0] * 5, 'BBB': [20.0] * 5}, index=dates)
        db.save_prices(self.conn, self.prices, exchange='US', asset_type='etf')

    def test_update_and_adv(self):
        vols = pd.DataFrame({'AAA': [100] * 5, 'BBB': [10] * 5}, index=self.prices.index)
        n = update_volumes(self.conn, vols, exchange='US', asset_type='etf')
        self.assertEqual(n, 10)  # 2 tickers x 5 dates
        adv = load_avg_dollar_volume(self.conn, exchange='US', asset_type='etf')
        self.assertAlmostEqual(adv['AAA'], 10.0 * 100, places=6)   # $1,000/day
        self.assertAlmostEqual(adv['BBB'], 20.0 * 10, places=6)    # $200/day
        self.assertGreater(adv['AAA'], adv['BBB'])

    def test_update_does_not_fabricate_close(self):
        # A date with no existing price row must not create a row.
        extra = pd.DataFrame({'AAA': [999]}, index=pd.to_datetime(['2030-01-01']))
        update_volumes(self.conn, extra, exchange='US', asset_type='etf')
        cnt = self.conn.execute(
            "SELECT COUNT(*) FROM prices p JOIN tickers t ON p.ticker_id=t.id "
            "WHERE t.symbol='AAA' AND p.date='2030-01-01'").fetchone()[0]
        self.assertEqual(cnt, 0)

    def test_adv_omits_tickers_without_volume(self):
        # No volumes written -> ADV series is empty.
        adv = load_avg_dollar_volume(self.conn, exchange='US', asset_type='etf')
        self.assertEqual(len(adv), 0)


class TestElectRepresentatives(unittest.TestCase):
    """Cluster -> one liquid representative; forced tickers always retained."""

    def _names(self, tickers):
        return {t: '' for t in tickers}

    def test_most_liquid_wins(self):
        tickers = ['A', 'B', 'C', 'D']
        labels = [1, 1, 2, 2]            # {A,B}, {C,D}
        adv = {'A': 100, 'B': 50, 'C': 10, 'D': 99}
        nobs = {t: 1000 for t in tickers}
        rows = elect_representatives(tickers, labels, adv, nobs, self._names(tickers), set())
        kept = {r['ticker'] for r in rows}
        self.assertEqual(kept, {'A', 'D'})     # highest ADV in each cluster
        a = next(r for r in rows if r['ticker'] == 'A')
        self.assertEqual(a['aliases'], 'B')
        self.assertEqual(a['n_members'], 2)

    def test_forced_beats_liquidity(self):
        # SMH is the LEAST liquid in its cluster but must still be the rep.
        tickers = ['X', 'SMH']
        labels = [1, 1]
        adv = {'X': 1000, 'SMH': 1}
        nobs = {t: 1000 for t in tickers}
        rows = elect_representatives(tickers, labels, adv, nobs,
                                     self._names(tickers), {'SMH'})
        kept = {r['ticker'] for r in rows}
        self.assertEqual(kept, {'SMH'})
        smh = next(r for r in rows if r['ticker'] == 'SMH')
        self.assertEqual(smh['aliases'], 'X')
        self.assertTrue(smh['forced'])

    def test_two_forced_in_one_cluster_both_survive(self):
        tickers = ['SPY', 'SMH', 'Z']
        labels = [1, 1, 1]               # all three correlated into one cluster
        adv = {'SPY': 500, 'SMH': 5, 'Z': 999}
        nobs = {t: 1000 for t in tickers}
        rows = elect_representatives(tickers, labels, adv, nobs,
                                     self._names(tickers), {'SPY', 'SMH'})
        kept = {r['ticker'] for r in rows}
        self.assertIn('SPY', kept)       # most-liquid forced -> cluster rep
        self.assertIn('SMH', kept)       # lost election but promoted to its own row
        self.assertNotIn('Z', kept)      # non-forced, less liquid than SPY -> alias
        smh = next(r for r in rows if r['ticker'] == 'SMH')
        self.assertTrue(smh['forced'])
        # SMH must not also appear as an alias of SPY's row
        spy = next(r for r in rows if r['ticker'] == 'SPY')
        self.assertNotIn('SMH', spy['aliases'].split(','))

    def test_history_breaks_adv_tie(self):
        tickers = ['P', 'Q']
        labels = [1, 1]
        adv = {'P': 100, 'Q': 100}       # equal liquidity
        nobs = {'P': 500, 'Q': 1260}     # Q has more history
        rows = elect_representatives(tickers, labels, adv, nobs, self._names(tickers), set())
        self.assertEqual({r['ticker'] for r in rows}, {'Q'})


class TestLoadCuratedUniverse(unittest.TestCase):

    def test_reads_ticker_column_uppercased(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'curated.csv')
            pd.DataFrame({'ticker': ['spy', 'Smh', 'AGG'], 'adv': [1, 2, 3]}).to_csv(path, index=False)
            tickers = load_curated_universe(path)
            self.assertEqual(tickers, ['SPY', 'SMH', 'AGG'])

    def test_missing_file_raises_with_hint(self):
        with self.assertRaises(FileNotFoundError) as cm:
            load_curated_universe('/nonexistent/curated_universe.csv')
        self.assertIn('curate_universe.py', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
