"""Produce a rebalance portfolio by comparing the top deployable methods.

Runs GA selection once (C++ island GA) and Monte-Carlo selection once on the
latest training window, then weights each candidate method — applying the
country/sector group constraints to every SLSQP path — and prints a
side-by-side comparison with dollar allocations.

Methods compared (see CLAUDE.md "Strategy Taxonomy & Empirical Verdicts"):
    cc_copulae         GA selection + Gaussian copula     (deployable pick — tightest OOS)
    mc_optimised       MC selection + max-Sharpe SLSQP    (best OOS mean-robustness)
    cc_optimised       GA selection + max-Sharpe SLSQP
    cc_equal_weight    GA selection + 1/N                 (robust baseline)
    mc_random_weights  MC selection + random weights      (reference floor)

The reported Sharpe is in-sample and biased upward. The printed OOS column
applies a 50% haircut and a Harvey-Liu multiple-testing correction; treat any
in-sample Sharpe above 1.5 as a red flag, not a win.

The search universe is liquidity-filtered (US-listed ETFs above a dollar-volume
floor; see src.liquidity), and the deployable book is reported as the equity
portfolio plus a managed-futures trend-sleeve capital split, spread equally over
a basket of CTA ETFs (DBMF/KMLM/CTA by default; see --sleeve-etfs).

Usage:
    python run_rebalance.py [--portfolio-value 100000] [--time-budget 600]
                            [--min-etfs 10] [--max-etfs 15] [--min-adv 1e6]
                            [--sleeve-alpha 0.25] [--no-sleeve] [--no-gpu] [--seed N]
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from src.logging_config import setup_logging
from src import config

setup_logging()
logger = logging.getLogger(__name__)

BINARY_PATH = os.path.join(os.path.dirname(__file__), 'cpp', 'optimisation')

# Candidate methods, in display order. (label, selection_source, weight_mode)
METHODS = [
    ('cc_copulae',        'cc', 'copulae'),
    ('mc_optimised',      'mc', 'optimal'),
    ('cc_optimised',      'cc', 'optimal'),
    ('cc_equal_weight',   'cc', 'equal'),
    ('mc_random_weights', 'mc', 'random'),
]

DEPLOYABLE_RECOMMENDATION = 'cc_copulae'


def normalise_min_return(min_return):
    """Map the CLI --min-return onto per-backend semantics.

    Values <= 0 (or None) disable the floor everywhere — but the two backends
    spell "off" differently: the C++ GA gate is `minReturn >= 0 && ret <
    minReturn` (needs a NEGATIVE sentinel), while the Python SLSQP/selection
    paths bind on any non-None value (need None).

    :return: (min_return_cpp, min_return_py)
    """
    if min_return is None or min_return <= 0:
        return -1.0, None
    return float(min_return), float(min_return)


def parse_args():
    p = argparse.ArgumentParser(description='Rebalance: compare top methods')
    p.add_argument('--portfolio-value', type=float, default=100_000,
                   help='Portfolio value for dollar allocations (default: %(default)s)')
    p.add_argument('--min-etfs', type=int, default=10,
                   help='Minimum holdings (default: %(default)s)')
    p.add_argument('--max-etfs', type=int, default=15,
                   help='Maximum holdings (default: %(default)s)')
    p.add_argument('--lookback-days', type=int, default=config.REBALANCE_LOOKBACK_DAYS,
                   help='Calendar days of history for the live allocation '
                        '(default: %(default)s = 2y admission; research/backtest '
                        'stays on the 5y standard).')
    p.add_argument('--time-budget', type=float, default=600,
                   help='GA time budget in seconds (default: %(default)s)')
    p.add_argument('--min-return', type=float, default=config.ISLAND_GA_MIN_RETURN,
                   help='Minimum annualised return floor; <= 0 disables it for '
                        'pure max-Sharpe (default: %(default)s). Without cheap '
                        'leverage the floor picks the growth point on the '
                        'frontier — the validated production config.')
    p.add_argument('--pop-size', type=int, default=10_000,
                   help='GA population per island (default: %(default)s)')
    p.add_argument('--generations', type=int, default=10_000,
                   help='GA generations per island (default: %(default)s)')
    p.add_argument('--no-gpu', action='store_true', help='Disable Metal GPU')
    p.add_argument('--seed', type=int, default=-1,
                   help='Random seed, -1 for random (default: %(default)s)')
    p.add_argument('--no-save', action='store_true',
                   help='Do not persist runs to the database')
    p.add_argument('--asset-type', default='etf',
                   help="Asset-type filter: 'etf' (default, excludes single "
                        "stocks) or 'all' to include equities.")
    p.add_argument('--min-weight', type=float, default=config.GA_MIN_WEIGHT,
                   help='Minimum weight per holding (default: %(default)s)')
    p.add_argument('--max-weight', type=float, default=0.25,
                   help='Maximum weight per holding (default: %(default)s)')
    p.add_argument('--must-have', default=','.join(config.REBALANCE_MUST_HAVE),
                   help="Comma-separated tickers forced into every portfolio "
                        "(default: %(default)s). Empty string to force none.")
    p.add_argument('--curated', action='store_true',
                   help="Restrict the search universe to the curated allow-list "
                        "(data/curated_universe.csv, built by curate_universe.py).")
    p.add_argument('--min-adv', type=float, default=config.REBALANCE_MIN_ADV_USD,
                   help='Minimum average daily dollar volume per holding '
                        '(default: %(default)s; 0 disables the ADV floor, '
                        'leaving only the foreign-listing filter).')
    p.add_argument('--no-liquidity-filter', action='store_true',
                   help='Disable the liquidity/tradeability filter entirely '
                        '(allows foreign listings and sub-ADV names).')
    p.add_argument('--sleeve-alpha', type=float, default=0.25,
                   help='Managed-futures capital fraction of the deployable book '
                        '(default: %(default)s), split equally across --sleeve-etfs.')
    p.add_argument('--sleeve-etfs', default=','.join(config.REBALANCE_SLEEVE_ETFS),
                   help='Comma-separated managed-futures ETFs that share the sleeve '
                        'allocation equally (default: %(default)s).')
    p.add_argument('--no-sleeve', action='store_true',
                   help='Report the equity book only, without the managed-futures split.')
    return p.parse_args()


# ── Selection: C++ island GA ────────────────────────────────────────────────

def run_cpp_ga(binary_data_path, args, min_return_cpp):
    """Invoke the C++ island GA and return its parsed JSON result.

    Parameterised on cardinality / min-return (unlike run_optimisation.py which
    bakes these into module constants), so the GA respects the 10-15 holding
    constraint for the rebalance. *min_return_cpp* must already be normalised
    (negative = no floor; see normalise_min_return).
    """
    cmd = [
        BINARY_PATH, '--binary', '--data', binary_data_path,
        '--mode', 'ga',
        '--pop-size', str(args.pop_size),
        '--generations', str(args.generations),
        '--min-etfs', str(args.min_etfs),
        '--max-etfs', str(args.max_etfs),
        '--num-elites', '100',
        '--migration-interval', '10',
        '--migration-rate', '0.1',
        '--min-return', str(min_return_cpp),
        '--time-budget', str(args.time_budget),
        '--seed', str(args.seed),
        '--mutation-initial', '0.008',
        '--mutation-final', '0.002',
        '--stagnation-restart', '500',
        '--top-k', '50',
    ]
    if not args.no_gpu:
        cmd.append('--gpu')

    logger.info("Running C++ GA: %s", ' '.join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=args.time_budget + 60)
    elapsed = time.time() - start
    if proc.returncode != 0:
        logger.error("C++ GA exited %d. stderr tail:\n%s",
                     proc.returncode, proc.stderr[-1000:])
        return None, elapsed
    try:
        result = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.error("Failed to parse C++ JSON. stdout head:\n%s",
                     proc.stdout[:500])
        return None, elapsed
    logger.info("C++ GA finished in %.1fs", elapsed)
    return result, elapsed


# ── Helpers ─────────────────────────────────────────────────────────────────

def canonical(prices, tickers):
    """Return *tickers* present in *prices*, in price-column order.

    Canonical ordering keeps the weight vector, expected returns, covariance,
    and group-constraint indices all aligned for every method.
    """
    wanted = set(tickers)
    return [t for t in prices.columns if t in wanted]


def selection_vector(prices, tickers):
    cols = list(prices.columns)
    sel = np.zeros(len(cols), dtype=int)
    idx = {t: i for i, t in enumerate(cols)}
    for t in tickers:
        if t in idx:
            sel[idx[t]] = 1
    return sel


def force_must_haves(tickers, must_haves, max_etfs):
    """Guarantee *must_haves* are in the selection, capped to *max_etfs*.

    Must-haves are placed first (always kept); the candidate's own picks fill the
    remaining slots. Already-present must-haves are not duplicated.
    """
    if not must_haves:
        return list(tickers)
    merged = list(must_haves) + [t for t in tickers if t not in must_haves]
    return merged[:max_etfs]


def portfolio_metrics(prices, tickers, weights):
    """In-sample annualised return, volatility, and Sharpe for a portfolio."""
    from src.returns import calculate_log_returns, calculate_expected_returns
    from src.covariance import calculate_covariance_matrix
    from src.weights import calculate_portfolio_variance

    log_returns = calculate_log_returns(prices[tickers])
    er = calculate_expected_returns(log_returns).values
    cov = calculate_covariance_matrix(log_returns)
    w = np.asarray(weights)
    ret = float(np.dot(w, er))
    vol = float(np.sqrt(calculate_portfolio_variance(w, cov)))
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def compute_weights(mode, prices, tickers, gc, gm, min_weight, max_weight,
                    min_return=None):
    """Return weights aligned to *tickers* (canonical order) for a weight mode.

    ``min_return`` (annualised) adds a floor on the portfolio's expected return
    in the SLSQP modes, pushing the book toward higher-return holdings.
    """
    from src.weights import optimise_weights

    n = len(tickers)
    if mode == 'equal':
        return np.ones(n) / n
    if mode == 'random':
        from src.backtest.simulation import get_random_weights
        return get_random_weights(list(tickers))

    if mode == 'optimal':
        sel = selection_vector(prices, tickers)
        res = optimise_weights(
            sel, prices,
            min_weight=min_weight, max_weight=max_weight, min_return=min_return,
            group_constraints=gc, group_membership=gm,
            selected_tickers=tickers if gc else None,
        )
        return np.asarray(res.x)

    if mode == 'copulae':
        from src.returns import calculate_log_returns, calculate_expected_returns
        from src.covariance import estimate_corr_using_copulas
        log_returns = calculate_log_returns(prices[tickers])
        er = calculate_expected_returns(log_returns).values
        corr = estimate_corr_using_copulas(log_returns)
        std = log_returns.std().values * math.sqrt(config.TRADING_DAYS_PER_YEAR)
        D = np.diag(std)
        cov = D @ corr @ D
        res = optimise_weights(
            expected_returns=er, cov_matrix=cov,
            min_weight=min_weight, max_weight=max_weight, min_return=min_return,
            group_constraints=gc, group_membership=gm,
            selected_tickers=tickers if gc else None,
        )
        return np.asarray(res.x)

    raise ValueError(f"Unknown weight mode: {mode}")


# ── Category caps (asset-class limits via the crude name classifier) ─────────

def build_category_caps(prices, conn):
    """Per-ticker cap-group map + the category-cap constraints, from the crude
    name classifier (src.categorise). Returns (group_of, caps, excluded) where
    caps is an {group: (min, max)} dict for the synthetic 'asset_class' dimension
    and excluded is the list of tickers in REBALANCE_EXCLUDE_CATEGORIES.
    """
    from src.categorise import classify_etf
    from src import db
    ex = db._get_exchange_id(conn, 'US')
    names = {r['symbol']: (r['name'] or '') for r in conn.execute(
        "SELECT symbol, name FROM tickers WHERE exchange_id = ?", (ex,))}
    label = {t: classify_etf(names.get(t, '')) for t in prices.columns}
    group_of = {t: config.REBALANCE_CAP_GROUP.get(label[t], 'Unknown')
                for t in prices.columns}
    excluded = [t for t in prices.columns
                if label[t] in config.REBALANCE_EXCLUDE_CATEGORIES]
    caps = {g: (0.0, c) for g, c in config.REBALANCE_CATEGORY_CAPS.items()}
    return group_of, caps, excluded


def pick_cc_selection(ga_result, prices, gc, gm, min_w, max_w, min_return=None,
                      must_haves=(), max_etfs=None):
    """Choose the GA selection that maximises max-Sharpe *while satisfying the
    category caps AND the return floor*, scanning the C++ GA's top-K candidates.
    This makes the constraints actually bind on GA-selected portfolios instead of
    silently falling back when the single best candidate can't meet them. Falls
    back to the GA's best raw pick if none are compliant.

    *must_haves* are forced into every candidate before weighting (capped to
    *max_etfs*), so the chosen cc selection always contains them.
    """
    from src.group_constraints import check_constraints
    cols = set(prices.columns)
    cap = max_etfs or len(cols)
    best_compliant = best_any = None
    for sol in (ga_result or {}).get('top_solutions') or []:
        cand = [t for t in (sol.get('tickers') or []) if t in cols]
        tk = canonical(prices, force_must_haves(cand, must_haves, cap))
        if len(tk) < 2:
            continue
        try:
            w = compute_weights('optimal', prices, tk, gc, gm, min_w, max_w, min_return)
        except Exception:  # noqa: BLE001
            continue
        ret, _, sh = portfolio_metrics(prices, tk, w)
        if best_any is None or sh > best_any[0]:
            best_any = (sh, tk)
        caps_ok = check_constraints(tk, w, gm, gc)[0] if (gc and gm) else True
        ret_ok = (min_return is None) or (ret >= min_return - 1e-3)
        if caps_ok and ret_ok and (best_compliant is None or sh > best_compliant[0]):
            best_compliant = (sh, tk)
    chosen = best_compliant or best_any
    if chosen:
        return chosen[1]
    raw = (ga_result or {}).get('selected_tickers')
    return canonical(prices, raw) if raw else None


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)

    if not os.path.isfile(BINARY_PATH):
        logger.error("C++ binary not found at %s — run 'make build-cpp'", BINARY_PATH)
        sys.exit(1)

    # ── Load latest training window (DB-first, 95% coverage filter) ──────────
    from src.data_loading import load_training_data
    from src.config import ETF_PRICES_CSV
    asset_type = None if args.asset_type == 'all' else args.asset_type
    logger.info("Loading training data (lookback=%d days, asset_type=%s)...",
                args.lookback_days, asset_type or 'all')
    # min_history flags are advisory here: the production admission bar is the
    # coverage test over this (shorter) window, not the 5y research standard.
    prices = load_training_data(exchange='US', csv_fallback=ETF_PRICES_CSV,
                                lookback_days=args.lookback_days,
                                asset_type=asset_type,
                                allow_min_history_flags=True)
    if args.curated:
        from src.data_loading import load_curated_universe
        curated = set(load_curated_universe())
        kept = [c for c in prices.columns if c in curated]
        logger.info("Curated universe: restricting %d -> %d tickers",
                    prices.shape[1], len(kept))
        prices = prices[kept]
    logger.info("Loaded %d dates x %d tickers (%s to %s)",
                prices.shape[0], prices.shape[1],
                prices.index[0].date(), prices.index[-1].date())

    # Open the DB early — both the liquidity filter and the group constraints
    # below need a connection.
    from src import db
    conn = db.get_connection()

    # ── Liquidity / tradeability filter ──────────────────────────────────────
    # Drop foreign (dot-suffix) listings and US ETFs below the ADV floor so the
    # book only holds IB-tradeable names. Must-haves are protected from the filter.
    if not args.no_liquidity_filter:
        from src.liquidity import filter_by_liquidity
        _must = [t.strip().upper() for t in (args.must_have or '').split(',') if t.strip()]
        prices = filter_by_liquidity(prices, conn, args.min_adv, protect=_must)

    # ── Must-have (forced) holdings ──────────────────────────────────────────
    must_haves = [t.strip().upper() for t in (args.must_have or '').split(',') if t.strip()]
    missing = [t for t in must_haves if t not in prices.columns]
    if missing:
        logger.warning("Must-have(s) not in the active universe — skipping: %s "
                       "(check `make health-check`).", missing)
        must_haves = [t for t in must_haves if t in prices.columns]
    if len(must_haves) > args.max_etfs:
        logger.error("More must-haves (%d) than the holdings cap (%d). Raise "
                     "--max-etfs or drop a must-have.", len(must_haves), args.max_etfs)
        sys.exit(1)
    if must_haves:
        logger.info("Must-have holdings (forced into every portfolio): %s", must_haves)

    # ── Category caps + group constraints ────────────────────────────────────
    from src.group_constraints import load_membership, check_constraints

    # Asset-class caps from the crude classifier; drop any excluded categories
    # (but never drop a must-have, even if its category is excluded).
    group_of, cat_caps, excluded_syms = build_category_caps(prices, conn)
    excluded_syms = [t for t in excluded_syms if t not in must_haves]
    if excluded_syms:
        prices = prices.drop(columns=excluded_syms)
        logger.info("Dropped %d tickers in excluded categories %s",
                    len(excluded_syms), config.REBALANCE_EXCLUDE_CATEGORIES)

    gc = {**config.GROUP_CONSTRAINTS, 'asset_class': cat_caps}
    gm = load_membership(conn, list(prices.columns), exchange='US')
    for t in prices.columns:
        gm.setdefault(t, {})['asset_class'] = group_of[t]
    logger.info("Category caps active: %s",
                {g: hi for g, (lo, hi) in cat_caps.items()})

    # ── Selection: GA (cc_*) and Monte Carlo (mc_*) ──────────────────────────
    from src.returns import calculate_log_returns
    from src.binary_io import write_binary_data
    from src.optimisers.monte_carlo import parallel_monte_carlo, monte_carlo_search

    # Return floor: <= 0 disables it (C++ wants -1, Python wants None).
    min_return_cpp, min_return_py = normalise_min_return(args.min_return)
    if min_return_py is None:
        logger.info("Return floor disabled — pure max-Sharpe optimisation.")

    # GA selection via C++ binary
    log_returns_full = calculate_log_returns(prices)
    tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    tmp.close()
    cc_tickers = None
    try:
        write_binary_data(log_returns_full, tmp.name)
        ga_result, ga_elapsed = run_cpp_ga(tmp.name, args, min_return_cpp)
    finally:
        os.unlink(tmp.name)

    if ga_result is not None:
        cc_tickers = pick_cc_selection(ga_result, prices, gc, gm,
                                       args.min_weight, args.max_weight,
                                       min_return_py, must_haves, args.max_etfs)
        if cc_tickers:
            logger.info("GA selected (cap-aware) %d tickers: %s",
                        len(cc_tickers), cc_tickers)
    if not cc_tickers:
        logger.warning("GA selection failed — cc_* methods will be skipped.")

    # Monte Carlo selection
    logger.info("Running Monte Carlo selection (%d trials)...", config.MC_NUM_TRIALS)
    mc_start = time.time()
    if args.seed >= 0:
        mc_sol, _ = monte_carlo_search(prices, config.MC_NUM_TRIALS,
                                       args.min_etfs, args.max_etfs)
    else:
        mc_sol, _ = parallel_monte_carlo(prices, config.MC_NUM_TRIALS,
                                         os.cpu_count(), args.min_etfs, args.max_etfs)
    mc_elapsed = time.time() - mc_start
    mc_tickers = list(prices.columns[mc_sol == 1]) if mc_sol is not None else None
    if mc_tickers:
        mc_tickers = canonical(prices, force_must_haves(mc_tickers, must_haves,
                                                        args.max_etfs))
        logger.info("MC selected %d tickers: %s", len(mc_tickers), mc_tickers)
    else:
        logger.warning("MC selection failed — mc_* methods will be skipped.")

    selections = {'cc': cc_tickers, 'mc': mc_tickers}
    elapsed_by_source = {'cc': ga_elapsed if ga_result else None, 'mc': mc_elapsed}

    # ── Build each method's portfolio ────────────────────────────────────────
    portfolios = []  # (label, tickers, weights, ret, vol, sharpe, violations, source)
    for label, source, mode in METHODS:
        tickers = selections.get(source)
        if not tickers:
            logger.warning("Skipping %s (no %s selection).", label, source)
            continue
        try:
            weights = compute_weights(mode, prices, tickers, gc, gm,
                                      args.min_weight, args.max_weight,
                                      min_return_py)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Weighting failed for %s: %s", label, exc)
            continue
        ret, vol, sharpe = portfolio_metrics(prices, tickers, weights)
        violations = []
        if gc and gm:
            _, violations = check_constraints(tickers, weights, gm, gc)
        portfolios.append((label, tickers, weights, ret, vol, sharpe,
                           violations, source))

    if not portfolios:
        logger.error("No portfolios produced. Aborting.")
        conn.close()
        sys.exit(1)

    # ── Report ───────────────────────────────────────────────────────────────
    K = len(portfolios)
    T = prices.shape[0]
    hl_factor = math.sqrt(max(0.0, 1 - 2 * math.log(K) / T)) if K > 1 else 1.0

    print()
    print("=" * 78)
    print(f"REBALANCE COMPARISON  —  {K} methods, {prices.shape[1]} ticker universe")
    print(f"Window: {prices.index[0].date()} to {prices.index[-1].date()} "
          f"({T} trading days)")
    print("=" * 78)
    print(f"{'Method':<18}{'Holds':>6}{'IS Sharpe':>11}{'Return':>9}"
          f"{'Vol':>8}{'OOS est':>9}  Notes")
    print("-" * 78)
    for label, tickers, weights, ret, vol, sharpe, violations, _ in portfolios:
        # OOS estimate: 50% haircut, then Harvey-Liu multiple-testing correction.
        oos = 0.5 * sharpe * hl_factor
        notes = []
        if sharpe > 1.5:
            notes.append("IS>1.5 suspect")
        if min_return_py is not None and ret < min_return_py - 1e-3:
            notes.append(f"return<{min_return_py:.0%} target")
        if violations:
            notes.append(f"{len(violations)} cap breach")
        if label == DEPLOYABLE_RECOMMENDATION:
            notes.append("<< recommended")
        print(f"{label:<18}{len(tickers):>6}{sharpe:>11.3f}{ret:>8.1%}"
              f"{vol:>8.1%}{oos:>9.3f}  {', '.join(notes)}")
    print("-" * 78)
    print(f"OOS est = 0.5 x IS Sharpe x Harvey-Liu factor ({hl_factor:.3f}). "
          f"Research OOS ceiling ~1.0-1.2.")

    # Per-method holdings + dollar allocations.
    pv = args.portfolio_value
    for label, tickers, weights, ret, vol, sharpe, violations, _ in portfolios:
        print()
        print(f"── {label}  (IS Sharpe {sharpe:.3f}, {len(tickers)} holdings) ──")
        print(f"{'Ticker':<10}{'Weight':>9}{f'  ${pv:,.0f}':>16}")
        for t, w in sorted(zip(tickers, weights), key=lambda x: -x[1]):
            if w > 1e-4:
                star = ' ★' if t in must_haves else ''
                print(f"{t:<10}{w:>8.1%}{w * pv:>15,.0f}{star}")
        for v in violations:
            print(f"  ! group cap: {v}")

    # ── Deployable book: equity + managed-futures (DBMF) capital split ────────
    # The trend sleeve is a synthetic return stream, not tradeable equity weights,
    # so it enters a live book as a capital allocation: (1-alpha) to the equity
    # book, alpha split equally across the managed-futures ETFs (DBMF/KMLM/...),
    # which proxy the validated synthetic TSMOM stream. Held by hand because the
    # funds are too young for the GA's history filter and because a fixed strategic
    # allocation protects the diversifier the in-sample optimiser would underweight.
    sleeve_etfs = [t.strip().upper() for t in (args.sleeve_etfs or '').split(',') if t.strip()]
    alpha = 0.0 if (args.no_sleeve or not sleeve_etfs) else args.sleeve_alpha
    if args.sleeve_alpha and not args.no_sleeve and not sleeve_etfs:
        logger.warning("--sleeve-alpha set but --sleeve-etfs is empty; "
                       "reporting the equity book only.")
    rec = next((p for p in portfolios if p[0] == DEPLOYABLE_RECOMMENDATION), None)
    if rec is None:
        logger.warning("Recommended method %s unavailable; deployable book uses %s.",
                       DEPLOYABLE_RECOMMENDATION, portfolios[0][0])
        rec = portfolios[0]
    rec_label, rec_tickers, rec_weights = rec[0], rec[1], rec[2]
    eq = 1.0 - alpha
    print()
    print("=" * 78)
    header = f"DEPLOYABLE BOOK — {rec_label} equity book"
    if alpha:
        header += f" + {alpha:.0%} managed futures ({'/'.join(sleeve_etfs)})"
    print(header)
    print("=" * 78)
    print(f"{'Ticker':<10}{'Weight':>9}{f'  ${pv:,.0f}':>16}")
    for t, w in sorted(zip(rec_tickers, rec_weights), key=lambda x: -x[1]):
        if w * eq > 1e-4:
            star = ' ★' if t in must_haves else ''
            print(f"{t:<10}{w * eq:>8.1%}{w * eq * pv:>15,.0f}{star}")
    if alpha:
        each = alpha / len(sleeve_etfs)
        for etf in sleeve_etfs:
            print(f"{etf:<10}{each:>8.1%}{each * pv:>15,.0f}  managed futures")
    print("-" * 78)
    if alpha:
        each = alpha / len(sleeve_etfs)
        print(f"Split: {eq:.0%} equity / {alpha:.0%} managed futures "
              f"({', '.join(sleeve_etfs)} equal-weight, {each:.1%} each). Sleeve "
              f"validated vs DBMF (+0.48 corr, sleeve_reality_check.py); CPCV shows a "
              f"small OOS mean lift + variance reduction at alpha=0.25. Multiple funds "
              f"diversify single-manager risk (they are correlated trend-followers).")
    else:
        print("Equity book only (--no-sleeve); managed-futures sleeve omitted.")

    # ── Persist ──────────────────────────────────────────────────────────────
    if not args.no_save:
        from src.portfolio_utils import save_optimisation_result
        for label, tickers, weights, ret, vol, sharpe, _, source in portfolios:
            run_id = save_optimisation_result(
                conn, tickers, np.asarray(weights), prices,
                script_name=f'rebalance_{label}',
                params={
                    'data_source': 'us_db',
                    'method': label,
                    'selection': source,
                    'min_etfs': args.min_etfs,
                    'max_etfs': args.max_etfs,
                    'window_start': str(prices.index[0].date()),
                    'window_end': str(prices.index[-1].date()),
                    'seed': args.seed,
                },
                exchange='US',
                elapsed_seconds=elapsed_by_source.get(source),
            )
            logger.info("Saved %s to DB (run_id=%d)", label, run_id)

    conn.close()
    print()
    print(f"Recommended deployable portfolio: {DEPLOYABLE_RECOMMENDATION} "
          f"(highest OOS mean and tightest distribution among GA-selected methods).")


if __name__ == '__main__':
    main()
