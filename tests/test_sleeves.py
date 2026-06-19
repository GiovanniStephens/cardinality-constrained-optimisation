"""Tests for the managed-futures (TSMOM) sleeve and its book-level blend.

Synthetic data only — no DB dependency. The most important test is
``test_sleeve_is_causal``: it proves the sleeve return decided/realized before a
given day never reacts to a future price, which is the core no-look-ahead
guarantee the whole experiment rests on.
"""

import unittest

import numpy as np
import pandas as pd

from src.config import TSMOM_BASKET_MULTI
from src.returns import calculate_log_returns
from src.sleeves.trend import (
    compute_tsmom_returns,
    _instrument_position,
    _resolve_basket,
    _aggregate,
)
from src.sleeves.overlay import _basket_candidates
from src.backtest.simulation import (
    run_portfolio,
    evaluate_portfolios,
    get_statistics_with_sleeve,
    evaluate_portfolios_with_sleeve,
)

# Small frozen-spec overrides so synthetic series stay cheap.
_FAST = dict(lookback=20, vol_lookback=10, rebalance_days=5)


def make_synthetic_prices(n_days=500, n_tickers=30, seed=42,
                          start='2018-01-01', daily_drift=0.0002,
                          daily_vol=0.01):
    """Deterministic synthetic GBM prices (mirrors tests.helpers, but kept
    local so these sleeve tests don't import the optimiser package — which
    pulls in the optional `pygad` dependency)."""
    np.random.seed(seed)
    dates = pd.bdate_range(start, periods=n_days, freq='B')
    tickers = [f'S{i}' for i in range(n_tickers)]
    log_rets = np.random.randn(n_days, n_tickers) * daily_vol + daily_drift
    prices = 100 * np.exp(log_rets.cumsum(axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _single_ticker_basket(prices):
    """A basket mapping one slot per column of `prices` (each chain length 1)."""
    return {c.lower(): [c] for c in prices.columns}


class TestTsmomCausality(unittest.TestCase):
    def test_sleeve_is_causal(self):
        """Perturbing the price on day t must not change any sleeve return
        BEFORE day t (positions are decided from strictly-earlier prices)."""
        prices = make_synthetic_prices(n_days=200, n_tickers=5, seed=1)
        basket = _single_ticker_basket(prices)
        base = compute_tsmom_returns(prices, basket=basket, **_FAST)

        t = 120
        perturbed = prices.copy()
        perturbed.iloc[t, 0] *= 1.5
        out = compute_tsmom_returns(perturbed, basket=basket, **_FAST)

        # Everything strictly before the perturbed day is untouched.
        self.assertTrue(np.allclose(base.values[:t], out.values[:t]),
                        "sleeve returns before the perturbed day changed — "
                        "look-ahead leak")
        # The perturbation must propagate somewhere from day t onward
        # (r_t and r_{t+1} contain p_t), else the test proves nothing.
        self.assertFalse(np.allclose(base.values[t:], out.values[t:]))


class TestTsmomSignal(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2018-01-01', periods=160)

    def test_signal_long_on_uptrend(self):
        # Strong trend-to-noise so the monthly-held sign is unambiguous.
        rng = np.random.RandomState(0)
        r = pd.Series(0.002 + 0.001 * rng.randn(160), index=self.dates)
        pos = _instrument_position(r, allow_short=True, target_vol_instr=0.1,
                                   **_FAST)
        post = pos.dropna()
        self.assertGreater((post > 0).mean(), 0.9)

    def test_signal_short_on_downtrend(self):
        rng = np.random.RandomState(0)
        r = pd.Series(-0.002 + 0.001 * rng.randn(160), index=self.dates)
        pos = _instrument_position(r, allow_short=True, target_vol_instr=0.1,
                                   **_FAST)
        post = pos.dropna()
        self.assertGreater((post < 0).mean(), 0.9)

    def test_short_disabled_never_negative(self):
        rng = np.random.RandomState(0)
        r = pd.Series(-0.002 + 0.001 * rng.randn(160), index=self.dates)
        pos = _instrument_position(r, allow_short=False, target_vol_instr=0.1,
                                   **_FAST)
        self.assertTrue((pos.dropna() >= 0).all())


class TestTsmomVolTarget(unittest.TestCase):
    def test_book_vol_near_target(self):
        prices = make_synthetic_prices(n_days=400, n_tickers=5, seed=3,
                                       daily_vol=0.01)
        basket = _single_ticker_basket(prices)
        sleeve = compute_tsmom_returns(
            prices, basket=basket, target_vol_book=0.10,
            lookback=40, vol_lookback=20, rebalance_days=5)
        post = sleeve.iloc[80:]                       # drop warmup
        ann_vol = post.std() * np.sqrt(252)
        # Generous band: the point is it targets ~10%, not 1% or 50%.
        self.assertTrue(0.04 < ann_vol < 0.30,
                        f"sleeve annualised vol {ann_vol:.3f} far from 0.10 target")


class TestSleeveBlend(unittest.TestCase):
    def setUp(self):
        prices = make_synthetic_prices(n_days=120, n_tickers=6, seed=5)
        self.oos = calculate_log_returns(prices)
        self.port = ['S0', 'S1', 'S2']
        self.w = np.array([0.4, 0.3, 0.3])

    def test_blend_math(self):
        rng = np.random.RandomState(7)
        sleeve = pd.Series(rng.randn(len(self.oos)) * 0.005, index=self.oos.index)
        alpha = 0.3
        stats = get_statistics_with_sleeve(self.port, self.w, self.oos,
                                           sleeve, alpha)
        book = np.asarray(run_portfolio(self.port, self.w, self.oos))
        combined = (1 - alpha) * book + alpha * sleeve.values
        self.assertTrue(np.isclose(stats['annualised_return'],
                                   combined.mean() * 252))
        exp_std = combined.std() * np.sqrt(252)
        self.assertTrue(np.isclose(stats['sharpe_ratio'],
                                   combined.mean() * 252 / exp_std))

    def test_alpha_zero_matches_baseline(self):
        """evaluate_portfolios_with_sleeve(alpha=0) == evaluate_portfolios."""
        portfolios = [['S0', 'S1', 'S2'], ['S1', 'S2', 'S3']]
        weights = [np.array([.4, .3, .3]), np.array([.5, .25, .25])]
        rng = np.random.RandomState(11)
        sleeve = pd.Series(rng.randn(len(self.oos)) * 0.01, index=self.oos.index)

        base = evaluate_portfolios(portfolios, weights, self.oos, self.oos,
                                   'cc_x')
        sl0 = evaluate_portfolios_with_sleeve(portfolios, weights, self.oos,
                                              self.oos, 'cc_x_trend0', sleeve,
                                              0.0)
        for pb, ps in zip(base.portfolios, sl0.portfolios):
            for k in pb.metrics:
                self.assertTrue(np.isclose(pb.metrics[k], ps.metrics[k]),
                                f"metric {k} drifted at alpha=0")
            self.assertTrue(np.isclose(pb.is_sharpe, ps.is_sharpe))

    def test_blend_alignment_reindexes_and_fills(self):
        """Sleeve with extra dates + a missing OOS date → reindex onto OOS and
        treat the missing date as flat (0.0), not a shift."""
        rng = np.random.RandomState(13)
        extra = pd.bdate_range(self.oos.index[-1] + pd.Timedelta(days=1),
                               periods=10)
        full_index = self.oos.index.append(extra)
        sleeve = pd.Series(rng.randn(len(full_index)) * 0.01, index=full_index)
        sleeve = sleeve.drop(self.oos.index[3])      # a missing OOS date

        alpha = 0.25
        stats = get_statistics_with_sleeve(self.port, self.w, self.oos,
                                           sleeve, alpha)
        book = np.asarray(run_portfolio(self.port, self.w, self.oos))
        aligned = sleeve.reindex(self.oos.index).fillna(0.0).values
        combined = (1 - alpha) * book + alpha * aligned
        self.assertTrue(np.isclose(stats['annualised_return'],
                                   combined.mean() * 252))


class TestMultiMarketBasket(unittest.TestCase):
    """The nested, cluster-balanced multi-market basket (TSMOM_BASKET_MULTI)."""

    _AGG = dict(lookback=20, vol_lookback=10, target_vol_instr=0.1,
                allow_short=True, rebalance_days=5)

    def _leg_pnl(self, log_returns, tk):
        r = log_returns[tk]
        pos = _instrument_position(r, allow_short=True, target_vol_instr=0.1,
                                   **_FAST)
        return pos * r

    def test_cluster_balance(self):
        """Two-level weighting caps each CLUSTER at 1/N_clusters of the book,
        regardless of leg count — a 3-leg equity cluster and a 1-leg bond
        cluster each contribute 50%, NOT 3/4 vs 1/4 (which a flat mean gives).
        This is the orthogonality fix: equities can't dominate by leg count.
        """
        prices = make_synthetic_prices(n_days=200, n_tickers=4, seed=21)
        prices.columns = ['EA', 'EB', 'EC', 'BX']
        log_returns = calculate_log_returns(prices)

        nested = {'equity': {'a': 'EA', 'b': 'EB', 'c': 'EC'},
                  'bond':   {'x': 'BX'}}
        combined = _aggregate(nested, log_returns, **self._AGG)

        eq = pd.concat([self._leg_pnl(log_returns, t)
                        for t in ('EA', 'EB', 'EC')], axis=1).mean(axis=1)
        bx = self._leg_pnl(log_returns, 'BX')
        expected = 0.5 * eq + 0.5 * bx           # equal cluster weight
        self.assertTrue(np.allclose(combined.iloc[50:].values,
                                    expected.iloc[50:].values),
                        "nested aggregate is not cluster-balanced 50/50")

        # And it genuinely differs from the flat 4-leg mean (bond at 1/4),
        # proving the cluster weighting changes the result when counts differ.
        flat = {'a': 'EA', 'b': 'EB', 'c': 'EC', 'x': 'BX'}
        flat_combined = _aggregate(flat, log_returns, **self._AGG)
        self.assertFalse(np.allclose(combined.iloc[50:].values,
                                     flat_combined.iloc[50:].values),
                         "cluster-balanced aggregate matched the flat mean")

    def test_flat_equals_single_leg_nested(self):
        """A nested basket with exactly one leg per cluster must produce the
        IDENTICAL sleeve to the equivalent flat basket — pins the equivalence
        and guards the flat path (the committed 5-ETF results) against drift."""
        prices = make_synthetic_prices(n_days=220, n_tickers=2, seed=22)
        flat = {'eq': ['S0'], 'bd': ['S1']}
        nested = {'c_eq': {'eq': ['S0']}, 'c_bd': {'bd': ['S1']}}
        a = compute_tsmom_returns(prices, basket=flat, **_FAST)
        b = compute_tsmom_returns(prices, basket=nested, **_FAST)
        self.assertTrue(np.allclose(a.values, b.values),
                        "one-leg-per-cluster nested != equivalent flat basket")

    def test_sleeve_is_causal_nested(self):
        """Causality must hold under the nested cluster-balanced path too:
        perturbing a leg's price on day t changes no sleeve return before t."""
        prices = make_synthetic_prices(n_days=200, n_tickers=6, seed=23)
        nested = {'eq': {'a': ['S0'], 'b': ['S1']},
                  'bd': {'c': ['S2'], 'd': ['S3']}}
        base = compute_tsmom_returns(prices, basket=nested, **_FAST)

        t = 120
        perturbed = prices.copy()
        perturbed.iloc[t, 0] *= 1.5              # perturb leg 'a' (S0)
        out = compute_tsmom_returns(perturbed, basket=nested, **_FAST)
        self.assertTrue(np.allclose(base.values[:t], out.values[:t]),
                        "nested sleeve leaked: a return before t reacted")
        self.assertFalse(np.allclose(base.values[t:], out.values[t:]))

    def test_nested_resolution_drops_empty_clusters(self):
        """A cluster whose every leg is missing from `prices` is dropped; the
        surviving clusters remain (and re-share the weight by count)."""
        prices = make_synthetic_prices(n_days=200, n_tickers=5, seed=24)
        nested = {'equity': {'a': ['S0']},
                  'missing': {'z': ['NOPE']},     # no usable ticker
                  'bond':   {'b': ['S1']}}
        resolved = _resolve_basket(prices, nested, min_obs=30)
        self.assertNotIn('missing', resolved)
        self.assertEqual(resolved,
                         {'equity': {'a': 'S0'}, 'bond': {'b': 'S1'}})

    def test_basket_candidates_flattens_nested(self):
        """`_basket_candidates` flattens the nested basket to every fallback
        ticker, so the DB load fetches the full multi-market set."""
        cands = _basket_candidates(TSMOM_BASKET_MULTI)
        for tk in ('SPY', 'QQQ', 'IWM', 'EEM', 'EFA', 'SHY', 'IEF', 'TLT',
                   'AGG', 'LQD', 'HYG', 'GLD', 'IAU', 'USO', 'DBA', 'VNQ',
                   'IYR', 'SCHH', 'VNQI'):
            self.assertIn(tk, cands)

    def test_multi_basket_screens_fx_and_broad_commodity(self):
        """Thin FX wrappers and broad-commodity baskets are deliberately NOT in
        the multi-market basket (tradeability + signal quality). Encoding it
        here trips a test if a future edit silently re-adds them."""
        cands = set(_basket_candidates(TSMOM_BASKET_MULTI))
        for tk in ('UUP', 'FXE', 'FXY', 'FXB', 'FXA', 'FXF',  # thin FX
                   'DBC', 'GSG', 'DJP'):                       # broad commodity
            self.assertNotIn(tk, cands)


if __name__ == '__main__':
    unittest.main()
