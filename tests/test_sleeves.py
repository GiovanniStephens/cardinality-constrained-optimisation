"""Tests for the managed-futures (TSMOM) sleeve and its book-level blend.

Synthetic data only — no DB dependency. The most important test is
``test_sleeve_is_causal``: it proves the sleeve return decided/realized before a
given day never reacts to a future price, which is the core no-look-ahead
guarantee the whole experiment rests on.
"""

import unittest

import numpy as np
import pandas as pd

from src.returns import calculate_log_returns
from src.sleeves.trend import compute_tsmom_returns, _instrument_position
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


if __name__ == '__main__':
    unittest.main()
