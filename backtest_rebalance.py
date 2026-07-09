"""Quarterly-rebalanced walk-forward backtest of the cc_copulae production config.

At each quarter-end the cc_copulae portfolio is re-optimised on the trailing
~5 years under the full production config — category caps, Leveraged/Inverse
split, the SMH must-have, the return floor, the cardinality cap and per-holding
bounds — then held for the next quarter (buy-and-hold with weight drift). The
out-of-sample daily returns are stitched across all quarters and the realised
annualised Sharpe / vol / return / max-drawdown are reported, with a Deflated
Sharpe and a SPY buy-and-hold benchmark for context.

This is the honest OOS gate: re-optimise → hold a quarter → rebalance → repeat.

Usage:
    python backtest_rebalance.py [--min-return 0.12] [--must-have SMH]
        [--max-etfs 30] [--max-weight 0.12] [--ga-time-budget 60] [--max-windows N]
"""

import argparse
import logging
import os
import tempfile
import types

import numpy as np
import pandas as pd

from src.logging_config import setup_logging
from src import config, db
from src.config import DB_PATH, TRADING_DAYS_PER_YEAR, DATA_MIN_COVERAGE, DATA_FFILL_LIMIT
from src.returns import calculate_log_returns
from src.binary_io import write_binary_data
from src.backtest.windows import slice_window_data
from src.backtest.windows import generate_windows
from src.backtest.simulation import run_portfolio
from src.metrics import maximum_drawdown, compute_method_dsr
from src.group_constraints import load_membership
import run_rebalance as rb

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Quarterly-rebalanced walk-forward of cc_copulae')
    p.add_argument('--min-return', type=float, default=0.12)
    p.add_argument('--must-have', default=','.join(config.REBALANCE_MUST_HAVE))
    p.add_argument('--min-etfs', type=int, default=10)
    p.add_argument('--max-etfs', type=int, default=30)
    p.add_argument('--min-weight', type=float, default=config.GA_MIN_WEIGHT)
    p.add_argument('--max-weight', type=float,
                   default=config.REBALANCE_MAX_WEIGHT)
    p.add_argument('--ga-time-budget', type=float, default=60.0,
                   help='GA seconds per quarter (lower than a single run; many windows)')
    p.add_argument('--train-years', type=int, default=5)
    p.add_argument('--test-days', type=int, default=63, help='~1 quarter')
    p.add_argument('--max-windows', type=int, default=None,
                   help='Cap number of windows (smoke test). Default: all.')
    p.add_argument('--uncap', default='',
                   help="Comma-separated cap-groups to leave UNconstrained "
                        "(e.g. 'Inverse,Leveraged'). Default: none (all caps on).")
    p.add_argument('--pool-lev-inv', type=float, default=None, metavar='CAP',
                   help="Pool Leveraged + Inverse into ONE combined cap at CAP "
                        "(e.g. 0.20) for this run only; config stays split.")
    p.add_argument('--curated', action='store_true',
                   help="Restrict the universe to the curated allow-list "
                        "(data/curated_universe.csv, built by curate_universe.py).")
    p.add_argument('--no-gpu', action='store_true')
    return p.parse_args()


def optimise_window(train, conn, args, must_haves):
    """Run the cc_copulae production config on a trailing-window price slice.
    Returns (tickers, weights) or (None, None) if it can't produce a portfolio."""
    # Point-in-time coverage: keep tickers with >=95% data in THIS window.
    thr = int(DATA_MIN_COVERAGE * len(train))
    train = train.dropna(axis=1, thresh=thr).ffill(limit=DATA_FFILL_LIMIT).dropna(axis=1)
    if train.shape[1] < args.max_etfs:
        return None, None, train

    group_of, cat_caps, excluded = rb.build_category_caps(train, conn)
    uncap = {g.strip() for g in (args.uncap or '').split(',') if g.strip()}
    if uncap:
        cat_caps = {g: v for g, v in cat_caps.items() if g not in uncap}
    if args.pool_lev_inv is not None:
        # Pool Leveraged + Inverse into one combined cap (this run only).
        group_of = {t: ('Leveraged/Inverse' if g in ('Leveraged', 'Inverse') else g)
                    for t, g in group_of.items()}
        cat_caps = {g: v for g, v in cat_caps.items() if g not in ('Leveraged', 'Inverse')}
        cat_caps['Leveraged/Inverse'] = (0.0, args.pool_lev_inv)
    excluded = [t for t in excluded if t not in must_haves]
    if excluded:
        train = train.drop(columns=excluded)
    gc = {**config.GROUP_CONSTRAINTS, 'asset_class': cat_caps}
    gm = load_membership(conn, list(train.columns), exchange='US')
    for t in train.columns:
        gm.setdefault(t, {})['asset_class'] = group_of[t]

    mh = [t for t in must_haves if t in train.columns]

    # The two backends spell "floor off" differently (C++ negative sentinel,
    # Python None) — route both through the same normaliser as run_rebalance.
    min_return_cpp, min_return_py = rb.normalise_min_return(args.min_return)

    ga_args = types.SimpleNamespace(
        pop_size=10000, generations=10000, min_etfs=args.min_etfs,
        max_etfs=args.max_etfs,
        time_budget=args.ga_time_budget, seed=-1, no_gpu=args.no_gpu)

    lr = calculate_log_returns(train)
    tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    tmp.close()
    try:
        write_binary_data(lr, tmp.name)
        ga_result, _ = rb.run_cpp_ga(tmp.name, ga_args, min_return_cpp)
    finally:
        os.unlink(tmp.name)
    if ga_result is None:
        return None, None, train

    cc = rb.pick_cc_selection(ga_result, train, gc, gm, args.min_weight,
                              args.max_weight, min_return_py, mh, args.max_etfs)
    if not cc:
        return None, None, train
    weights, slsqp_ok = rb.compute_weights('copulae', train, cc, gc, gm,
                                           args.min_weight, args.max_weight,
                                           min_return_py)
    if not slsqp_ok:
        logger.warning("SLSQP fell back to 1/N for this quarter's book.")
    return cc, np.asarray(weights), train


def main():
    args = parse_args()
    must_haves = [t.strip().upper() for t in (args.must_have or '').split(',') if t.strip()]

    logger.info("Loading full ETF price history (all dates, active)...")
    conn = db.get_connection(DB_PATH)
    full = db.load_prices(conn, exchange='US', asset_type='etf', min_coverage=0)
    full.index = pd.to_datetime(full.index)
    full = full.sort_index()
    if args.curated:
        from src.data_loading import load_curated_universe
        curated = set(load_curated_universe())
        kept = [c for c in full.columns if c in curated]
        logger.info("Curated universe: restricting %d -> %d ETFs", full.shape[1], len(kept))
        full = full[kept]
    logger.info("Loaded %d dates x %d active ETFs (%s to %s)",
                full.shape[0], full.shape[1], full.index[0].date(), full.index[-1].date())

    windows = generate_windows(full.index,
                               train_days=args.train_years * TRADING_DAYS_PER_YEAR,
                               test_days=args.test_days, step_days=args.test_days)
    if args.max_windows:
        windows = windows[:args.max_windows]
    logger.info("Quarterly walk-forward: %d windows (train %dy, test %dd, step %dd)",
                len(windows), args.train_years, args.test_days, args.test_days)

    oos_daily = []          # stitched daily OOS log returns (cc_copulae)
    spy_daily = []          # SPY benchmark, same windows
    per_q = []              # per-quarter summary rows
    for i, w in enumerate(windows, 1):
        train = full.loc[w.train_start:w.train_end]
        _, oos_lr_all = slice_window_data(w, full)   # OOS log returns (all tickers)
        cc, weights, _ = optimise_window(train, conn, args, must_haves)
        if cc is None:
            logger.warning("[%d/%d] %s: no portfolio (insufficient universe)", i, len(windows), w.label)
            continue
        # keep only holdings that have OOS data this quarter; renormalise
        held = [t for t in cc if t in oos_lr_all.columns]
        wv = np.array([weights[cc.index(t)] for t in held], dtype=float)
        if wv.sum() <= 0:
            continue
        wv = wv / wv.sum()
        q_ret = run_portfolio(held, wv, oos_lr_all[held])
        oos_daily.extend(q_ret)
        ann = float(np.mean(q_ret) * TRADING_DAYS_PER_YEAR) if q_ret else 0.0
        vol = float(np.std(q_ret) * np.sqrt(TRADING_DAYS_PER_YEAR)) if q_ret else 0.0
        per_q.append((w.label, len(held), 'SMH' in held, ann, vol,
                      ann / vol if vol > 0 else 0.0))
        if 'SPY' in oos_lr_all.columns:
            spy_daily.extend(run_portfolio(['SPY'], np.array([1.0]), oos_lr_all[['SPY']]))
        logger.info("[%d/%d] %s: %d holds, SMH=%s, OOS ann %.1f%% vol %.1f%% Sharpe %.2f",
                    i, len(windows), w.label, len(held), 'SMH' in held,
                    ann * 100, vol * 100, (ann / vol if vol > 0 else 0.0))
    conn.close()

    if not oos_daily:
        logger.error("No OOS returns produced.")
        return

    r = np.asarray(oos_daily)
    ann_ret = float(np.mean(r) * TRADING_DAYS_PER_YEAR)
    ann_vol = float(np.std(r) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    mdd = maximum_drawdown(list(r))
    dsr = compute_method_dsr(sharpe, r, num_trials=len(windows))

    print()
    print("=" * 72)
    print("QUARTERLY-REBALANCED WALK-FORWARD — cc_copulae (production config)")
    pool_str = (f"Lev+Inv pooled @{args.pool_lev_inv:.0%}"
                if args.pool_lev_inv is not None else "Lev/Inv split (config)")
    print(f"Config: min_return={args.min_return:.0%}, must-have={must_haves}, "
          f"max_etfs={args.max_etfs}, max_weight={args.max_weight:.0%}, "
          f"uncapped={args.uncap or 'none'}, {pool_str}")
    print("=" * 72)
    print(f"Quarters simulated:        {len(per_q)}")
    print(f"OOS trading days:          {len(r)}")
    print(f"Realised annual return:    {ann_ret:>7.1%}")
    print(f"Realised annual vol:       {ann_vol:>7.1%}")
    print(f"** Realised OOS Sharpe:    {sharpe:>7.2f} **")
    print(f"Max drawdown:              {mdd:>7.1%}")
    print(f"Deflated Sharpe (prob>0):  {dsr['dsr']:>7.2f}")
    print(f"Return skew / exkurt:      {dsr['skewness']:.2f} / {dsr['excess_kurtosis']:.2f}")
    if spy_daily:
        s = np.asarray(spy_daily)
        sv = float(np.std(s) * np.sqrt(TRADING_DAYS_PER_YEAR))
        ss = float(np.mean(s) * TRADING_DAYS_PER_YEAR) / sv if sv > 0 else 0.0
        print(f"  SPY benchmark Sharpe:    {ss:>7.2f}  (ann {np.mean(s)*TRADING_DAYS_PER_YEAR:.1%}, "
              f"vol {sv:.1%})")
    print("-" * 72)
    print("Per-quarter OOS (label, holds, SMH, ann ret, ann vol, Sharpe):")
    for label, n, smh, a, v, sh in per_q:
        print(f"  {label:<16} {n:>3} {'Y' if smh else '-'}  {a:>7.1%} {v:>7.1%} {sh:>6.2f}")


if __name__ == '__main__':
    main()
