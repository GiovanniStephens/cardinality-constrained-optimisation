"""Safe-leverage sizing for a margin-financed portfolio.

Sizes broker-margin leverage on a daily log-return stream (the deployable book)
using several independent methodologies, so a recommendation is the MINIMUM of
all caps rather than the output of any single model:

- **Kelly with financing** (upper sanity bound only): L* = (mu - r_borrow)/sigma^2.
  Half-Kelly keeps ~75% of the growth at ~half the vol; >2x Kelly guarantees
  underperforming cash (MacLean-Thorp-Ziemba). mu-estimation error dominates.
- **Liquidation-drop identity** (Reg-T mechanics): a book drop d triggers broker
  auto-liquidation when equity < m * assets, at d* = (1 - m*L) / (L * (1 - m)).
  At m=0.25: L=1.5 survives a 56% drop, L=2.0 only 33%.
- **VaR / CVaR family**: parametric-normal, Cornish-Fisher (skew/kurtosis
  adjusted), and historical, over configurable horizons; caps scale linearly in L.
- **Stationary block-bootstrap first-passage simulation** (the core engine,
  Politis-Romano 1994): resample the book's daily returns in random-length blocks
  (preserving autocorrelation and vol clustering), simulate levered equity paths
  with daily financing accrual, and measure the probability that a path EVER
  breaches the maintenance line -- because IB liquidates in real time, terminal
  VaR is not the binding statistic; first passage is.

All functions are pure and operate on numpy arrays / floats; the CLI wrapper
(run_leverage_analysis.py) handles data loading and reporting. Sizing inputs
should be HAIRCUT estimates (mu haircut, sigma inflation), never raw in-sample
moments: the max-Sharpe portfolio is the estimation-error maximiser (Michaud
1989), and levering it levers exactly that error.
"""

import numpy as np
from scipy import stats

from src.config import TRADING_DAYS_PER_YEAR


# ─── Moments ──────────────────────────────────────────────────────────────────

def annualised_moments(returns):
    """Annualised mean/vol plus daily skew and excess kurtosis.

    :param returns: array-like of daily log returns.
    :return: (mu, sigma, skew, excess_kurtosis) — mu/sigma annualised,
        skew/kurtosis on the daily series (scale them per-horizon downstream).
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        raise ValueError("need at least 2 return observations")
    mu = float(r.mean()) * TRADING_DAYS_PER_YEAR
    sigma = float(r.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    skew = float(stats.skew(r))
    ex_kurt = float(stats.kurtosis(r))  # Fisher: 0 for a normal
    return mu, sigma, skew, ex_kurt


# ─── VaR family (positive numbers = loss fractions) ──────────────────────────

def parametric_var(mu, sigma, horizon_days, confidence=0.99):
    """Normal VaR of the unlevered book over a horizon, as a positive loss.

    :param mu: annualised mean return.
    :param sigma: annualised volatility.
    """
    h = horizon_days / TRADING_DAYS_PER_YEAR
    z = stats.norm.ppf(confidence)
    return float(z * sigma * np.sqrt(h) - mu * h)


def cornish_fisher_var(mu, sigma, skew, ex_kurt, horizon_days, confidence=0.99):
    """Modified (Cornish-Fisher) VaR adjusting the normal quantile for
    skewness and excess kurtosis. Reduces to parametric_var when both are 0.

    Daily skew/kurtosis are scaled to the horizon under iid aggregation
    (S_h = S/sqrt(n), K_h = K/n), which keeps 1-day VaR fully fat-tailed while
    longer horizons converge toward the normal (CLT).
    """
    n = max(1, int(horizon_days))
    s = skew / np.sqrt(n)
    k = ex_kurt / n
    # Expand at the LOWER (loss) tail quantile: z is negative, and negative
    # skew / excess kurtosis push z_cf further negative (larger loss).
    z = stats.norm.ppf(1.0 - confidence)
    z_cf = (z
            + (z ** 2 - 1) * s / 6
            + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    h = horizon_days / TRADING_DAYS_PER_YEAR
    return float(-(mu * h + z_cf * sigma * np.sqrt(h)))


def historical_var(returns, horizon_days, confidence=0.99):
    """Historical VaR from overlapping horizon windows of the daily series,
    as a positive simple-return loss fraction."""
    r = np.asarray(returns, dtype=float)
    n = int(horizon_days)
    if r.size < n:
        raise ValueError("series shorter than the horizon")
    window_sums = np.convolve(r, np.ones(n), mode='valid')  # overlapping log sums
    losses = 1.0 - np.exp(window_sums)                      # positive = loss
    return float(np.quantile(losses, confidence))


def cvar(returns, horizon_days, confidence=0.99):
    """Historical CVaR (expected shortfall): mean loss beyond the VaR quantile."""
    r = np.asarray(returns, dtype=float)
    n = int(horizon_days)
    if r.size < n:
        raise ValueError("series shorter than the horizon")
    window_sums = np.convolve(r, np.ones(n), mode='valid')
    losses = 1.0 - np.exp(window_sums)
    var = np.quantile(losses, confidence)
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else float(var)


# ─── Closed-form leverage caps ────────────────────────────────────────────────

def liquidation_drop(leverage, maintenance=0.25):
    """Book drop d* that triggers broker auto-liquidation at gross leverage L.

    Equity after a drop d is E*(1 - L*d) on assets E*L*(1-d); liquidation when
    equity < maintenance * assets:  d* = (1 - m*L) / (L * (1 - m)).
    L <= 1 (no loan) can never be liquidated -> 1.0 (a 100% loss bound).
    """
    if leverage <= 1.0:
        return 1.0
    d = (1.0 - maintenance * leverage) / (leverage * (1.0 - maintenance))
    return float(min(1.0, max(0.0, d)))


def max_leverage_for_drawdown(d_stress, maintenance=0.25, buffer=1.25):
    """Largest L whose liquidation drop d*(L) still exceeds buffer * d_stress.

    Inverts the identity: L = 1 / (d*(1-m) + m) at d = min(1, buffer*d_stress).
    """
    d = min(1.0, abs(d_stress) * buffer)
    return float(1.0 / (d * (1.0 - maintenance) + maintenance))


def kelly_leverage(mu, sigma, r_borrow):
    """Growth-optimal (full-Kelly) leverage with financing:
    L* = (mu - r_borrow) / sigma^2. Negative when mu < r_borrow (don't lever).
    Use a FRACTION of this (quarter/half) — the plug-in estimate overshoots
    whenever mu is overestimated, and overbetting past 2x Kelly guarantees
    underperforming cash.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float((mu - r_borrow) / sigma ** 2)


def vol_target_leverage(sigma_target, sigma_hat, cap=2.0):
    """Constant-risk leverage: L = min(sigma_target / sigma_hat, cap)."""
    if sigma_hat <= 0:
        raise ValueError("sigma_hat must be positive")
    return float(min(sigma_target / sigma_hat, cap))


def var_cap_leverage(var_unlevered, budget):
    """Max L such that L * VaR (or CVaR) of the unlevered book stays within a
    loss budget — tail risk scales linearly in L for a fixed distribution."""
    if var_unlevered <= 0:
        return float('inf')
    return float(budget / var_unlevered)


# ─── Stationary bootstrap + first-passage engine ─────────────────────────────

def stationary_bootstrap(returns, n_paths, horizon_days, avg_block=10, seed=None):
    """Politis-Romano (1994) stationary bootstrap of a daily return series.

    Blocks have geometric random length (mean *avg_block*) and wrap circularly,
    preserving short-range autocorrelation and volatility clustering that an
    iid bootstrap destroys (and which understate ruin probability).

    :return: ndarray of shape (n_paths, horizon_days) of resampled returns.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 2:
        raise ValueError("need at least 2 return observations")
    rng = np.random.default_rng(seed)
    horizon = int(horizon_days)

    idx = np.empty((n_paths, horizon), dtype=np.int64)
    # Vectorised over paths, sequential over time (a scan): at each step a path
    # either continues its block (index+1, circular) or restarts uniformly.
    restart = rng.random((n_paths, horizon)) < (1.0 / avg_block)
    restart[:, 0] = True
    fresh = rng.integers(0, n, size=(n_paths, horizon))
    idx[:, 0] = fresh[:, 0]
    for t in range(1, horizon):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(restart[:, t], fresh[:, t], cont)
    return r[idx]


def first_passage_breach_prob(returns, leverage, maintenance=0.25,
                              r_borrow=0.055, n_paths=20_000,
                              horizon_days=252, avg_block=10, seed=None,
                              sims=None):
    """P(a levered equity path EVER breaches the maintenance line) over a horizon.

    Retail margin model (not constant-L LETF resets): the loan is fixed in
    dollars and accrues interest daily; assets follow the bootstrapped book.
    Breach at time t when equity_t < m * assets_t, equivalently when the
    cumulative log return falls below the (interest-rising) barrier:

        barrier_t = ln( (L-1) / (L * (1-m)) ) + r_borrow * t / 252

    :param sims: optional pre-generated bootstrap array (n_paths, horizon) to
        share draws across leverage levels (guarantees monotonicity in L).
    """
    if leverage <= 1.0:
        return 0.0
    if sims is None:
        sims = stationary_bootstrap(returns, n_paths, horizon_days,
                                    avg_block=avg_block, seed=seed)
    horizon = sims.shape[1]
    cum = np.cumsum(sims, axis=1)
    t = np.arange(1, horizon + 1)
    barrier = (np.log((leverage - 1.0) / (leverage * (1.0 - maintenance)))
               + r_borrow * t / TRADING_DAYS_PER_YEAR)
    breached = (cum < barrier).any(axis=1)
    return float(breached.mean())


def safe_leverage(returns, p_breach_max=0.01, maintenance=0.25, r_borrow=0.055,
                  horizon_days=252, l_min=1.0, l_max=2.0, l_step=0.05,
                  n_paths=20_000, avg_block=10, seed=None):
    """Largest L on a grid with first-passage breach probability <= p_breach_max.

    A single set of bootstrap draws is shared across all leverage levels, so the
    breach curve is monotone in L by construction.

    :return: (L_safe, curve) where curve is a list of (L, p_breach) tuples.
    """
    sims = stationary_bootstrap(returns, n_paths, horizon_days,
                                avg_block=avg_block, seed=seed)
    curve = []
    l_safe = 1.0
    grid = np.round(np.arange(l_min, l_max + 1e-9, l_step), 6)
    for lev in grid:
        p = first_passage_breach_prob(returns, float(lev),
                                      maintenance=maintenance,
                                      r_borrow=r_borrow, sims=sims)
        curve.append((float(lev), p))
        if p <= p_breach_max:
            l_safe = float(lev)
    return l_safe, curve


# ─── Stress layer ─────────────────────────────────────────────────────────────

# Book-level shock fractions applied directly to the levered book. The equity
# benchmarks behind them: 2008 GFC -56.8% (407d), 2020 COVID -34% (~23 trading
# days, fastest ever), 2022 both-down year (60/40 -17.5%). A diversified
# low-vol book falls less than pure equities, so these are conservative when
# applied to the whole book.
DEFAULT_STRESS_SHOCKS = {
    '2022-style both-down': 0.175,
    'book -20% shock': 0.20,
    'COVID-2020 speed (-34%)': 0.34,
    'GFC-2008 (-56.8%)': 0.568,
}


def stress_report(returns, leverage, maintenance=0.25, gap=0.10, shocks=None):
    """Survival check of gross leverage L against drawdown stresses.

    Financing accrual is deliberately excluded here — these are instantaneous
    drop checks against d*(L); the first-passage engine handles interest paths.

    Returns a list of dicts: scenario name, book drop, the liquidation drop
    d*(L), and whether the levered book survives (drop < d*). Includes the
    book's own worst historical drawdown, that drawdown compounded with an
    overnight gap, and the standard shock set.
    """
    from src.metrics import maximum_drawdown
    r = list(np.asarray(returns, dtype=float))
    worst_dd = abs(maximum_drawdown(r))          # negative -> positive fraction
    d_star = liquidation_drop(leverage, maintenance)

    scenarios = {'worst historical drawdown': worst_dd,
                 f'worst drawdown + {gap:.0%} overnight gap':
                     1.0 - (1.0 - worst_dd) * (1.0 - gap)}
    scenarios.update(shocks if shocks is not None else DEFAULT_STRESS_SHOCKS)

    out = []
    for name, drop in sorted(scenarios.items(), key=lambda kv: kv[1]):
        out.append({
            'scenario': name,
            'book_drop': float(drop),
            'liquidation_drop': d_star,
            'survives': bool(drop < d_star),
        })
    return out


def levered_expectations(mu, sigma, leverage, r_borrow):
    """Net expected return, vol, Sharpe and log-growth of the levered book.

    return_net = L*mu - (L-1)*r_borrow
    sharpe_net = return_net / (L*sigma)
    growth     = return_net - 0.5*(L*sigma)^2      (the quadratic variance tax)
    """
    ret = leverage * mu - (leverage - 1.0) * r_borrow
    vol = leverage * sigma
    sharpe = ret / vol if vol > 0 else 0.0
    growth = ret - 0.5 * vol ** 2
    return {'return': float(ret), 'vol': float(vol),
            'sharpe': float(sharpe), 'log_growth': float(growth)}
