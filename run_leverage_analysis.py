"""Determine safe margin leverage for a deployable portfolio.

Loads a saved rebalance book from the database (holdings + weights), rebuilds
its daily return stream (equity book blended with the synthetic managed-futures
sleeve), and sizes broker-margin leverage with several independent
methodologies (src.leverage). The recommendation is the MINIMUM of all caps:

    half-Kelly (financing-aware)  |  vol-target  |  stressed-drawdown identity
    CVaR budget  |  first-passage P(liquidation) < target  |  hard ceiling 2.0

Sizing runs on HAIRCUT inputs (mu halved, sigma inflated) — the max-Sharpe book
is the estimation-error maximiser, and in-sample moments of an optimised
portfolio flatter both return and vol. IB mechanics assumed: Reg-T 25%
maintenance, real-time auto-liquidation (no margin calls), floating USD
financing. The tool is allowed to answer "L = 1.0 — don't lever".

Usage:
    python run_leverage_analysis.py                       # latest cc_copulae run
    python run_leverage_analysis.py --run-id 68 --seed 42
    python run_leverage_analysis.py --borrow-rate 0.08    # financing stress
"""

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from src.logging_config import setup_logging
from src import config

setup_logging()
logger = logging.getLogger(__name__)

HARD_CAP = 2.0  # Reg-T initial margin is 2:1 at open; also the practitioner ceiling.


def parse_args():
    p = argparse.ArgumentParser(description='Safe margin-leverage analysis')
    p.add_argument('--run-id', type=int, default=None,
                   help='optimisation_runs.id of the book to analyse '
                        '(default: latest rebalance_cc_copulae run).')
    p.add_argument('--portfolio-value', type=float, default=100_000,
                   help='Equity (NLV) for the dollar sheet (default: %(default)s)')
    p.add_argument('--sleeve-alpha', type=float, default=0.25,
                   help='Managed-futures sleeve fraction blended into the book '
                        'return stream (default: %(default)s).')
    p.add_argument('--no-sleeve', action='store_true',
                   help='Analyse the equity book alone (alpha = 0).')
    p.add_argument('--maintenance', type=float,
                   default=config.REBALANCE_MAINTENANCE_MARGIN,
                   help='Maintenance margin fraction (default: %(default)s, Reg-T).')
    p.add_argument('--borrow-rate', type=float, default=config.REBALANCE_BORROW_RATE,
                   help='Annual margin financing rate (default: %(default)s).')
    p.add_argument('--p-breach', type=float, default=0.01,
                   help='Max acceptable P(maintenance liquidation) over the '
                        'horizon (default: %(default)s).')
    p.add_argument('--horizon-days', type=int, default=252,
                   help='Simulation horizon in trading days (default: %(default)s).')
    p.add_argument('--mu-haircut', type=float, default=0.5,
                   help='Multiplier on in-sample mean return (default: %(default)s).')
    p.add_argument('--sigma-inflation', type=float, default=1.5,
                   help='Multiplier on in-sample vol (default: %(default)s); the '
                        'worst rolling 63-day vol is used if higher.')
    p.add_argument('--no-vol-floor', action='store_true',
                   help='Drop the worst-63d-vol floor on sigma (sensitivity '
                        'analysis only — sizes on sigma * inflation alone).')
    p.add_argument('--vol-target', type=float, default=0.10,
                   help='Annualised vol target for the vol-target cap '
                        '(default: %(default)s).')
    p.add_argument('--n-paths', type=int, default=20_000,
                   help='Bootstrap paths (default: %(default)s).')
    p.add_argument('--avg-block', type=int, default=10,
                   help='Mean bootstrap block length in days (default: %(default)s).')
    p.add_argument('--seed', type=int, default=42,
                   help='Bootstrap RNG seed (default: %(default)s).')
    return p.parse_args()


# ── Book reconstruction ───────────────────────────────────────────────────────

def load_book(conn, run_id):
    """Return (run_id, script, tickers, weights) for a saved optimisation run."""
    if run_id is None:
        row = conn.execute(
            "SELECT id, script FROM optimisation_runs "
            "WHERE script = 'rebalance_cc_copulae' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            logger.error("No rebalance_cc_copulae run found — pass --run-id.")
            sys.exit(1)
        run_id = row['id']
    script = conn.execute(
        "SELECT script FROM optimisation_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if script is None:
        logger.error("Run id %d not found.", run_id)
        sys.exit(1)
    rows = conn.execute(
        "SELECT t.symbol, h.weight FROM portfolio_holdings h "
        "JOIN tickers t ON t.id = h.ticker_id WHERE h.run_id = ? "
        "ORDER BY h.weight DESC", (run_id,)
    ).fetchall()
    if not rows:
        logger.error("Run id %d has no holdings.", run_id)
        sys.exit(1)
    return run_id, script['script'], [r['symbol'] for r in rows], \
        np.array([r['weight'] for r in rows], dtype=float)


def book_return_series(conn, tickers, weights, sleeve_alpha):
    """Daily log returns of the deployable book over the maximal common window.

    Equity book: buy-and-hold with weight drift (run_portfolio). Sleeve: the
    synthetic full-history TSMOM stream (real MF ETFs are too young), blended
    (1-alpha)*book + alpha*sleeve — matching the backtest convention.
    """
    from src import db
    from src.returns import calculate_log_returns
    from src.backtest.simulation import run_portfolio

    prices = db.load_prices(conn, exchange='US', tickers=list(tickers),
                            min_coverage=0.0, exclude_flagged=False)
    prices.index = pd.to_datetime(prices.index)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logger.error("No price data for %s — cannot rebuild the book.", missing)
        sys.exit(1)
    # Maximal common window: from the first date on which EVERY holding trades
    # (bfill would fabricate flat early prices and understate vol).
    start = max(prices[t].first_valid_index() for t in tickers)
    prices = prices.loc[start:, list(tickers)].ffill().dropna()
    log_returns = calculate_log_returns(prices)

    book = np.array(run_portfolio(list(tickers), np.asarray(weights),
                                  log_returns), dtype=float)
    index = log_returns.index
    if sleeve_alpha > 0:
        from src.sleeves.overlay import get_cached_sleeve_series
        sleeve = get_cached_sleeve_series(conn)
        sleeve = sleeve.reindex(index).fillna(0.0).to_numpy()
        book = (1.0 - sleeve_alpha) * book + sleeve_alpha * sleeve
    return pd.Series(book, index=index)


def stressed_series(returns, mu_eff, sigma_eff):
    """Rescale the historical daily series to the haircut moments, preserving
    its shape (skew, clustering): demean, scale vol, add back the haircut mean."""
    r = np.asarray(returns, dtype=float)
    sigma_raw = r.std(ddof=1)
    scale = (sigma_eff / np.sqrt(config.TRADING_DAYS_PER_YEAR)) / sigma_raw
    return (r - r.mean()) * scale + mu_eff / config.TRADING_DAYS_PER_YEAR


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    from src import db
    from src import leverage as lev

    conn = db.get_connection()
    run_id, script, tickers, weights = load_book(conn, args.run_id)
    alpha = 0.0 if args.no_sleeve else args.sleeve_alpha
    logger.info("Analysing run %d (%s): %d holdings, sleeve alpha %.0f%%",
                run_id, script, len(tickers), alpha * 100)

    series = book_return_series(conn, tickers, weights, alpha)
    conn.close()

    mu, sigma, skew, ex_kurt = lev.annualised_moments(series)
    # Haircut inputs: halve mu; inflate sigma, floored at the worst realised
    # 63-day vol (annualised) — whichever is more pessimistic.
    worst_vol = float(series.rolling(63).std().max()) * \
        np.sqrt(config.TRADING_DAYS_PER_YEAR)
    mu_eff = mu * args.mu_haircut
    sigma_eff = sigma * args.sigma_inflation
    if not args.no_vol_floor:
        sigma_eff = max(sigma_eff, worst_vol)
    r_b, m = args.borrow_rate, args.maintenance

    stressed = stressed_series(series, mu_eff, sigma_eff)

    print()
    print("=" * 78)
    print(f"LEVERAGE ANALYSIS — run {run_id} ({script}), "
          f"{len(tickers)} holdings + {alpha:.0%} MF sleeve")
    print(f"Window: {series.index[0].date()} to {series.index[-1].date()} "
          f"({len(series)} trading days)")
    print("=" * 78)
    print(f"{'':24}{'raw (in-sample)':>18}{'sized on (haircut)':>20}")
    print(f"{'annual return':<24}{mu:>17.2%}{mu_eff:>19.2%}")
    print(f"{'annual vol':<24}{sigma:>17.2%}{sigma_eff:>19.2%}")
    print(f"{'Sharpe':<24}{mu / sigma:>17.2f}{mu_eff / sigma_eff:>19.2f}")
    print(f"daily skew {skew:+.2f}, excess kurtosis {ex_kurt:+.2f}; "
          f"worst rolling 63d vol {worst_vol:.2%}")
    print(f"financing {r_b:.2%} floating, maintenance {m:.0%}, "
          f"P(liquidation) target {args.p_breach:.1%}/{args.horizon_days}d")

    # ── VaR table (on the stressed distribution) ─────────────────────────────
    print()
    print(f"{'VaR (loss, stressed)':<24}{'95% 1d':>10}{'99% 1d':>10}"
          f"{'95% 21d':>10}{'99% 21d':>10}")
    rows = [
        ('parametric', lambda h, c: lev.parametric_var(mu_eff, sigma_eff, h, c)),
        ('Cornish-Fisher', lambda h, c: lev.cornish_fisher_var(
            mu_eff, sigma_eff, skew, ex_kurt, h, c)),
        ('historical', lambda h, c: lev.historical_var(stressed, h, c)),
        ('CVaR (hist.)', lambda h, c: lev.cvar(stressed, h, c)),
    ]
    for name, fn in rows:
        vals = [fn(h, c) for h, c in ((1, 0.95), (1, 0.99), (21, 0.95), (21, 0.99))]
        print(f"{name:<24}" + ''.join(f"{v:>10.2%}" for v in vals))

    # ── Leverage caps ────────────────────────────────────────────────────────
    # True half-Kelly, even if < 1 — the final max(1.0, min(caps)) floors the
    # recommendation, but the table should show the honest number.
    kelly_full = lev.kelly_leverage(mu_eff, sigma_eff, r_b)
    kelly_half = kelly_full * 0.5
    vol_cap = lev.vol_target_leverage(args.vol_target, sigma_eff, cap=HARD_CAP)
    from src.metrics import maximum_drawdown
    worst_dd = abs(maximum_drawdown(list(series)))
    dd_cap = lev.max_leverage_for_drawdown(max(worst_dd, 0.34), m, buffer=1.25)
    cvar_99_21 = lev.cvar(stressed, 21, 0.99)
    cvar_cap = lev.max_leverage_for_drawdown(cvar_99_21, m, buffer=1.25)
    fp_cap, curve = lev.safe_leverage(
        stressed, p_breach_max=args.p_breach, maintenance=m, r_borrow=r_b,
        horizon_days=args.horizon_days, l_max=HARD_CAP,
        n_paths=args.n_paths, avg_block=args.avg_block, seed=args.seed)

    caps = {
        f'half-Kelly (full {kelly_full:.2f})': kelly_half,
        f'vol-target {args.vol_target:.0%}': vol_cap,
        f'drawdown identity (worst {max(worst_dd, 0.34):.0%} x1.25)': dd_cap,
        f'CVaR 99%/21d ({cvar_99_21:.1%} x1.25)': cvar_cap,
        f'first-passage P<{args.p_breach:.0%}/1y': fp_cap,
        'hard ceiling (Reg-T / practitioner)': HARD_CAP,
    }
    recommended = max(1.0, min(caps.values()))
    binding = min(caps, key=lambda k: caps[k])

    print()
    print(f"{'Leverage cap':<44}{'max L':>8}")
    print("-" * 52)
    for name, val in sorted(caps.items(), key=lambda kv: kv[1]):
        mark = '  << binds' if name == binding else ''
        print(f"{name:<44}{val:>8.2f}{mark}")
    print("-" * 52)
    print(f"{'RECOMMENDED GROSS LEVERAGE':<44}{recommended:>8.2f}")
    if kelly_full <= 1.0:
        print("NOTE: full Kelly <= 1.0 — haircut return does not cover the "
              "borrow rate; the honest answer is DO NOT LEVER.")

    # ── P(breach) curve ──────────────────────────────────────────────────────
    print()
    print("P(maintenance breach within 1y) vs leverage (stressed bootstrap):")
    for l_val, p in curve:
        if round(l_val * 100) % 10 == 0:  # print at 0.1 steps
            note = (f'   (d* = {lev.liquidation_drop(l_val, m):.0%} drop)'
                    if l_val > 1 else '   (no loan)')
            print(f"  L={l_val:.1f}: {p:>7.2%}{note}")

    # ── Stress table at the recommended L ────────────────────────────────────
    print()
    print(f"Stress survival at L = {recommended:.2f} "
          f"(liquidation drop d* = {lev.liquidation_drop(recommended, m):.1%}):")
    for row in lev.stress_report(series, recommended, maintenance=m):
        status = 'SURVIVES' if row['survives'] else '** LIQUIDATED **'
        print(f"  {row['scenario']:<42}{row['book_drop']:>7.1%}  {status}")
    rate_kelly = lev.kelly_leverage(mu_eff, sigma_eff, r_b + 0.02)
    print(f"  financing +200bp: full Kelly {kelly_full:.2f} -> {rate_kelly:.2f}"
          + ('  ** lever turns negative-carry **' if rate_kelly <= 1.0 else ''))

    # ── Dollar sheet ─────────────────────────────────────────────────────────
    pv = args.portfolio_value
    loan = (recommended - 1.0) * pv
    exp = lev.levered_expectations(mu_eff, sigma_eff, recommended, r_b)
    base = lev.levered_expectations(mu_eff, sigma_eff, 1.0, r_b)
    print()
    print(f"Dollar sheet at L = {recommended:.2f} on ${pv:,.0f} equity:")
    print(f"  positions ${recommended * pv:,.0f} = equity ${pv:,.0f} "
          f"+ margin loan ${loan:,.0f}")
    print(f"  annual financing cost ~${loan * r_b:,.0f} at {r_b:.2%}")
    print(f"  {'':16}{'unlevered':>12}{'levered':>12}")
    print(f"  {'exp. return':<16}{base['return']:>12.2%}{exp['return']:>12.2%}")
    print(f"  {'vol':<16}{base['vol']:>12.2%}{exp['vol']:>12.2%}")
    print(f"  {'Sharpe (net)':<16}{base['sharpe']:>12.2f}{exp['sharpe']:>12.2f}")
    print(f"  {'log growth':<16}{base['log_growth']:>12.2%}{exp['log_growth']:>12.2%}")

    print()
    print("WARNING: these sizes lever ESTIMATED moments of an optimised (noise-"
          "fit) book; IB liquidates in real time on maintenance breach — gaps "
          "can execute at the low. Treat the recommendation as a ceiling.")
    print(json.dumps({'run_id': run_id, 'recommended_leverage': recommended,
                      'binding_cap': binding}, indent=None))


if __name__ == '__main__':
    main()
