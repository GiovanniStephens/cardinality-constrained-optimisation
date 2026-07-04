"""Tests for src/leverage.py — safe-leverage sizing primitives."""

import unittest

import numpy as np
from scipy import stats

from src.leverage import (
    annualised_moments,
    cornish_fisher_var,
    cvar,
    first_passage_breach_prob,
    historical_var,
    kelly_leverage,
    levered_expectations,
    liquidation_drop,
    max_leverage_for_drawdown,
    parametric_var,
    safe_leverage,
    stationary_bootstrap,
    stress_report,
    var_cap_leverage,
    vol_target_leverage,
)


def _synthetic_returns(n=2520, daily_mu=0.0003, daily_vol=0.006, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(daily_mu, daily_vol, n)


class TestClosedForms(unittest.TestCase):

    def test_liquidation_drop_exact_values(self):
        # d* = (1 - mL) / (L(1-m)) at m=0.25
        self.assertAlmostEqual(liquidation_drop(1.5), 0.5556, places=4)
        self.assertAlmostEqual(liquidation_drop(2.0), 1 / 3, places=6)
        self.assertAlmostEqual(liquidation_drop(1.25), 0.7333, places=4)
        self.assertAlmostEqual(liquidation_drop(3.0), 1 / 9, places=6)

    def test_liquidation_drop_unlevered_is_full_loss(self):
        self.assertEqual(liquidation_drop(1.0), 1.0)
        self.assertEqual(liquidation_drop(0.8), 1.0)

    def test_max_leverage_for_drawdown_inverts_identity(self):
        for lev in (1.25, 1.5, 2.0, 3.0):
            d = liquidation_drop(lev)
            # buffer=1.0: exact inversion
            self.assertAlmostEqual(
                max_leverage_for_drawdown(d, buffer=1.0), lev, places=6)

    def test_max_leverage_buffer_reduces_leverage(self):
        base = max_leverage_for_drawdown(0.34, buffer=1.0)
        buffered = max_leverage_for_drawdown(0.34, buffer=1.25)
        self.assertLess(buffered, base)

    def test_kelly_formula_exact(self):
        # (mu - r_b) / sigma^2 = (0.08 - 0.05) / 0.01 = 3.0
        self.assertAlmostEqual(kelly_leverage(0.08, 0.10, 0.05), 3.0, places=9)

    def test_kelly_negative_when_mu_below_borrow(self):
        self.assertLess(kelly_leverage(0.04, 0.10, 0.055), 0.0)

    def test_vol_target_leverage_and_cap(self):
        self.assertAlmostEqual(vol_target_leverage(0.10, 0.05), 2.0)  # hits cap
        self.assertAlmostEqual(vol_target_leverage(0.10, 0.08, cap=2.0), 1.25)

    def test_var_cap_leverage_linear_scaling(self):
        self.assertAlmostEqual(var_cap_leverage(0.10, 0.15), 1.5)

    def test_levered_expectations_net_of_financing(self):
        out = levered_expectations(mu=0.08, sigma=0.05, leverage=1.5,
                                   r_borrow=0.05)
        self.assertAlmostEqual(out['return'], 1.5 * 0.08 - 0.5 * 0.05, places=9)
        self.assertAlmostEqual(out['vol'], 0.075, places=9)
        # unlevered Sharpe 1.6; levered net Sharpe must be lower (spread drag)
        self.assertLess(out['sharpe'], 0.08 / 0.05)


class TestVarFamily(unittest.TestCase):

    def test_parametric_var_matches_normal_quantile(self):
        # zero drift, sigma=0.16 annual, 1-day, 99%
        v = parametric_var(0.0, 0.16, 1, 0.99)
        expected = stats.norm.ppf(0.99) * 0.16 * np.sqrt(1 / 252)
        self.assertAlmostEqual(v, expected, places=12)

    def test_cornish_fisher_reduces_to_parametric_when_gaussian(self):
        v_cf = cornish_fisher_var(0.05, 0.12, 0.0, 0.0, 21, 0.99)
        v_n = parametric_var(0.05, 0.12, 21, 0.99)
        self.assertAlmostEqual(v_cf, v_n, places=12)

    def test_cornish_fisher_negative_skew_raises_var(self):
        v_neg = cornish_fisher_var(0.0, 0.12, -1.0, 3.0, 1, 0.99)
        v_n = parametric_var(0.0, 0.12, 1, 0.99)
        self.assertGreater(v_neg, v_n)

    def test_historical_var_and_cvar_ordering(self):
        r = _synthetic_returns()
        v = historical_var(r, 21, 0.99)
        c = cvar(r, 21, 0.99)
        self.assertGreaterEqual(c, v)  # expected shortfall >= VaR

    def test_annualised_moments_recovers_inputs(self):
        r = _synthetic_returns(n=100_000, daily_mu=0.0004, daily_vol=0.01,
                               seed=11)
        mu, sigma, skew, kurt = annualised_moments(r)
        self.assertAlmostEqual(mu, 0.0004 * 252, delta=0.01)
        self.assertAlmostEqual(sigma, 0.01 * np.sqrt(252), delta=0.005)
        self.assertAlmostEqual(skew, 0.0, delta=0.05)
        self.assertAlmostEqual(kurt, 0.0, delta=0.1)


class TestBootstrapEngine(unittest.TestCase):

    def test_bootstrap_shape_and_determinism(self):
        r = _synthetic_returns(n=500)
        a = stationary_bootstrap(r, 50, 100, seed=42)
        b = stationary_bootstrap(r, 50, 100, seed=42)
        self.assertEqual(a.shape, (50, 100))
        np.testing.assert_array_equal(a, b)

    def test_bootstrap_preserves_moments(self):
        r = _synthetic_returns(n=2520)
        sims = stationary_bootstrap(r, 2000, 252, seed=1)
        self.assertAlmostEqual(sims.mean(), r.mean(), delta=5e-5)
        self.assertAlmostEqual(sims.std(), r.std(), delta=5e-4)

    def test_breach_prob_zero_for_unlevered(self):
        r = _synthetic_returns()
        self.assertEqual(first_passage_breach_prob(r, 1.0), 0.0)

    def test_breach_prob_zero_for_low_vol_modest_leverage(self):
        # ~1.6% annual vol book at 1.5x: needs a 56% drop — impossible here.
        r = _synthetic_returns(daily_mu=0.0004, daily_vol=0.001)
        p = first_passage_breach_prob(r, 1.5, n_paths=2000, seed=3)
        self.assertEqual(p, 0.0)

    def test_breach_prob_high_for_wild_vol_high_leverage(self):
        # ~95% annual vol at 3x: liquidation drop is only 11% — most paths
        # touch it within a year (a few escape upward early, so not ~1.0).
        r = _synthetic_returns(daily_mu=0.0, daily_vol=0.06, seed=5)
        p = first_passage_breach_prob(r, 3.0, n_paths=2000, seed=5)
        self.assertGreater(p, 0.8)

    def test_breach_prob_monotone_in_leverage(self):
        r = _synthetic_returns(daily_mu=0.0002, daily_vol=0.012, seed=9)
        sims = stationary_bootstrap(r, 5000, 252, seed=9)
        probs = [first_passage_breach_prob(r, lev, sims=sims)
                 for lev in (1.2, 1.5, 2.0, 2.5, 3.0)]
        self.assertEqual(probs, sorted(probs))

    def test_safe_leverage_respects_grid_bounds_and_target(self):
        r = _synthetic_returns(daily_mu=0.0003, daily_vol=0.004, seed=13)
        l_safe, curve = safe_leverage(r, p_breach_max=0.01, n_paths=2000,
                                      seed=13)
        self.assertGreaterEqual(l_safe, 1.0)
        self.assertLessEqual(l_safe, 2.0)
        # every grid point at or below l_safe must satisfy the target
        for lev, p in curve:
            if lev <= l_safe:
                self.assertLessEqual(p, 0.01)

    def test_safe_leverage_returns_floor_when_everything_breaches(self):
        r = _synthetic_returns(daily_mu=-0.002, daily_vol=0.08, seed=17)
        l_safe, _ = safe_leverage(r, p_breach_max=0.001, n_paths=1000, seed=17)
        self.assertEqual(l_safe, 1.0)


class TestMinReturnNormalisation(unittest.TestCase):
    """--min-return <= 0 must disable the floor in BOTH backends: the C++ GA
    needs a negative sentinel, the Python SLSQP paths need None."""

    def test_zero_and_negative_disable_floor(self):
        from run_rebalance import normalise_min_return
        for off in (0, 0.0, -1, None):
            cpp, py = normalise_min_return(off)
            self.assertLess(cpp, 0.0)
            self.assertIsNone(py)

    def test_positive_floor_passes_through(self):
        from run_rebalance import normalise_min_return
        cpp, py = normalise_min_return(0.12)
        self.assertEqual(cpp, 0.12)
        self.assertEqual(py, 0.12)


class TestStressLayer(unittest.TestCase):

    def test_stress_report_flags_survival_correctly(self):
        r = _synthetic_returns(daily_vol=0.002)  # tiny historical drawdown
        rep = stress_report(r, leverage=1.5)
        by_name = {row['scenario']: row for row in rep}
        # d*(1.5) = 55.6%: survives COVID-34%, dies in GFC-56.8%
        self.assertTrue(by_name['COVID-2020 speed (-34%)']['survives'])
        self.assertFalse(by_name['GFC-2008 (-56.8%)']['survives'])

    def test_stress_report_2x_dies_in_covid(self):
        r = _synthetic_returns(daily_vol=0.002)
        rep = stress_report(r, leverage=2.0)  # d* = 33.3%
        by_name = {row['scenario']: row for row in rep}
        self.assertFalse(by_name['COVID-2020 speed (-34%)']['survives'])


class TestCliSmoke(unittest.TestCase):

    def test_help_flag(self):
        import os
        import subprocess
        import sys
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, os.path.join(repo, 'run_leverage_analysis.py'),
             '--help'],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0)
        self.assertIn('usage', result.stdout.lower())


if __name__ == '__main__':
    unittest.main()
