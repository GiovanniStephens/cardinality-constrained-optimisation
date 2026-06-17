"""Build a data-driven curated ETF universe by correlation clustering.

Replaces the broad ~15k-US-ETF search space (mostly foreign-listed/illiquid) with
a compact allow-list of distinct, liquid exposures — the highest-leverage PBO
reducer (CLAUDE.md). It also removes the foreign/illiquid tickers that drove the
fat-tail (excess-kurtosis ~67) risk in the walk-forward backtest, since the
candidate pre-filter keeps only US-listed full-history ETFs.

Method (generalises ``build_dedup_map.py`` from near-identical-twin collapse to
genuine clustering with a liquidity-aware representative):

  1. Candidate pre-filter: US-listed (no '.'), asset_type=etf, 5y full history,
     not bad-flagged, exclude China/Russia.
  2. Sample-Pearson correlation of log returns -> distance d = 1 - corr.
  3. Hierarchical (average-linkage) clustering; cut by --threshold (correlation)
     or --n-clusters.
  4. Representative per cluster = highest average dollar volume (most liquid),
     tie-broken by history length. Must-haves (config.REBALANCE_MUST_HAVE) and the
     benchmark (SPY) are always retained as their own representatives.
  5. Write data/curated_universe.csv (+ .txt ticker list) and print a review table.

ADV requires volume — seed it once with `python -m src.db backfill-volume`.

Usage:
    python curate_universe.py [--threshold 0.90] [--n-clusters N] [--benchmark SPY]
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.logging_config import setup_logging
from src import config, db
from src.config import DB_PATH, ETF_PRICES_CSV, DATA_DIR, MIN_HISTORY_DAYS, EXCLUDED_COUNTRIES
from src.data_loading import load_training_data
from src.returns import calculate_log_returns
from src.categorise import classify_etf

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Build a curated ETF universe by correlation clustering')
    p.add_argument('--threshold', type=float, default=0.90,
                   help='Merge clusters whose members correlate >= this (avg linkage). '
                        'Higher = more, smaller clusters = larger universe. (default: %(default)s)')
    p.add_argument('--n-clusters', type=int, default=None,
                   help='Target cluster count (overrides --threshold via maxclust).')
    p.add_argument('--lookback-days', type=int, default=config.DATA_LOOKBACK_DAYS)
    p.add_argument('--min-history', type=int, default=MIN_HISTORY_DAYS,
                   help='Min non-null return rows to keep a candidate (default: %(default)s)')
    p.add_argument('--adv-window', type=int, default=126,
                   help='Trading days for the average-dollar-volume liquidity score.')
    p.add_argument('--benchmark', default='SPY',
                   help='Benchmark ticker always retained (default: %(default)s)')
    p.add_argument('--out', default=os.path.join(DATA_DIR, 'curated_universe.csv'))
    return p.parse_args()


def elect_representatives(tickers, labels, adv, nobs, names, forced):
    """Elect one representative per correlation cluster and return curated rows.

    The most liquid member (highest average dollar volume) wins, tie-broken by
    history length. Any forced ticker (must-haves + benchmark) present in a cluster
    is chosen as that cluster's representative, and every forced ticker is
    guaranteed to survive even if it shared a cluster with another forced ticker.

    :param tickers: list of candidate symbols (index-aligned with `labels`).
    :param labels: cluster label per ticker.
    :param adv: {ticker: average dollar volume}.
    :param nobs: {ticker: history-length proxy} (dict or pandas Series).
    :param names: {ticker: fund name} for asset-class classification.
    :param forced: set of tickers that must be retained as representatives.
    :return: list of row dicts (ticker, cluster_id, asset_class, adv, n_days,
        n_members, aliases, forced).
    """
    def score(t):                       # most liquid wins; tie-break on history length
        return (adv.get(t, 0.0) or 0.0, int(nobs[t]))

    clusters = {}
    for t, lab in zip(tickers, labels):
        clusters.setdefault(int(lab), []).append(t)

    rows = []
    for lab, members in clusters.items():
        forced_members = [m for m in members if m in forced]
        rep = max(forced_members, key=score) if forced_members else max(members, key=score)
        aliases = [m for m in members if m != rep]
        rows.append({'ticker': rep, 'cluster_id': lab,
                     'asset_class': classify_etf(names.get(rep, '')),
                     'adv': adv.get(rep, 0.0) or 0.0, 'n_days': int(nobs[rep]),
                     'n_members': len(members), 'aliases': ','.join(aliases),
                     'forced': rep in forced})

    # Guarantee every forced ticker survives, even if it shared a cluster with
    # another forced ticker and lost the election (promote it to its own row).
    idx = {t: i for i, t in enumerate(tickers)}
    kept = {r['ticker'] for r in rows}
    for t in forced:
        if t in idx and t not in kept:
            lab = int(labels[idx[t]])
            rows.append({'ticker': t, 'cluster_id': lab,
                         'asset_class': classify_etf(names.get(t, '')),
                         'adv': adv.get(t, 0.0) or 0.0, 'n_days': int(nobs[t]),
                         'n_members': 0, 'aliases': '', 'forced': True})
            for r in rows:
                if r['cluster_id'] == lab and t in r['aliases'].split(','):
                    r['aliases'] = ','.join(a for a in r['aliases'].split(',') if a and a != t)
    return rows


def _load_candidates(conn, args):
    """US-listed, full-history ETF price frame (the clustering candidate pool)."""
    prices = load_training_data(exchange='US', csv_fallback=ETF_PRICES_CSV,
                                lookback_days=args.lookback_days, asset_type='etf')
    # US-listed only: foreign listings carry a '.' suffix (.TO/.L/.KS/...).
    prices = prices[[c for c in prices.columns if '.' not in c]]
    # Exclude China/Russia where the metadata says so (mostly NULL, near-noop, but honest).
    ex_id = db._get_exchange_id(conn, 'US')
    bad_country = {r['symbol'] for r in conn.execute(
        "SELECT symbol FROM tickers WHERE exchange_id = ? AND country IN ({})".format(
            ','.join('?' for _ in EXCLUDED_COUNTRIES)), (ex_id, *EXCLUDED_COUNTRIES)).fetchall()}
    if bad_country:
        prices = prices[[c for c in prices.columns if c not in bad_country]]
    return prices


def main():
    args = parse_args()
    must_haves = [t.upper() for t in config.REBALANCE_MUST_HAVE]
    forced = set(must_haves) | {args.benchmark.upper()}

    conn = db.get_connection(DB_PATH)
    logger.info("Loading US-listed full-history ETF candidates...")
    prices = _load_candidates(conn, args)

    returns = calculate_log_returns(prices)
    nobs = returns.ne(0).sum()                          # non-zero return rows ~ real history
    keep = [t for t in prices.columns if nobs[t] >= args.min_history]
    returns = returns[keep]
    tickers = list(returns.columns)
    n = len(tickers)
    logger.info("Candidates after full-history filter: %d ETFs x %d days", n, returns.shape[0])
    missing_forced = [t for t in forced if t not in tickers]
    if missing_forced:
        logger.warning("Forced tickers absent from candidates (skipped): %s", missing_forced)

    # ── Correlation distance ─────────────────────────────────────────────────
    corr = np.clip(returns.corr().values, -1.0, 1.0)
    dist = 1.0 - corr
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist[dist < 0] = 0.0

    # ── Hierarchical clustering ──────────────────────────────────────────────
    Z = linkage(squareform(dist, checks=False), method='average')
    if args.n_clusters:
        labels = fcluster(Z, t=args.n_clusters, criterion='maxclust')
    else:
        labels = fcluster(Z, t=1.0 - args.threshold, criterion='distance')
    n_clusters = len(np.unique(labels))
    logger.info("Clustering -> %d clusters (%s)", n_clusters,
                f"n_clusters={args.n_clusters}" if args.n_clusters
                else f"corr threshold={args.threshold}")

    # ── Liquidity score + ticker names ───────────────────────────────────────
    adv = db.load_avg_dollar_volume(conn, exchange='US', asset_type='etf',
                                    tickers=tickers, window=args.adv_window)
    adv = adv.to_dict()
    ex_id = db._get_exchange_id(conn, 'US')
    names = {r['symbol']: (r['name'] or '') for r in conn.execute(
        "SELECT symbol, name FROM tickers WHERE exchange_id = ?", (ex_id,)).fetchall()}
    conn.close()

    # ── Assemble clusters, elect representatives ─────────────────────────────
    rows = elect_representatives(tickers, labels, adv, nobs, names, forced)

    curated = pd.DataFrame(rows).sort_values(['asset_class', 'adv'],
                                             ascending=[True, False]).reset_index(drop=True)
    curated.to_csv(args.out, index=False)
    txt = os.path.splitext(args.out)[0] + '.txt'
    with open(txt, 'w') as f:
        f.write('\n'.join(curated['ticker']) + '\n')

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("=" * 84)
    print("CURATED ETF UNIVERSE  —  correlation clustering + liquidity")
    print("=" * 84)
    knob = (f"n_clusters={args.n_clusters}" if args.n_clusters
            else f"corr threshold={args.threshold}")
    print(f"Candidates (US-listed, {args.min_history}+ days): {n}")
    print(f"Clusters / curated size: {n_clusters}  ({knob})")
    print(f"Written: {args.out}  +  {txt}")
    print("-" * 84)
    by_class = curated.groupby('asset_class').size().sort_values(ascending=False)
    print("Asset-class spread:")
    for cls, cnt in by_class.items():
        print(f"  {cls:<16} {cnt}")
    print("-" * 84)
    print(f"{'ticker':<10}{'class':<16}{'ADV($/day)':>14}  {'#mem':>4}  name / collapses")
    for _, r in curated.iterrows():
        star = ' *' if r['forced'] else '  '
        advs = f"${r['adv']/1e6:,.0f}M" if r['adv'] else "   n/a"
        nm = names.get(r['ticker'], '')[:30]
        extra = f"  (+{r['n_members']-1} aliases)" if r['n_members'] > 1 else ''
        print(f"{r['ticker']:<10}{r['asset_class']:<16}{advs:>14}  {r['n_members']:>4}{star}{nm}{extra}")

    # illiquid US-listed singletons that survived (no floor applied — flag for review)
    singles = curated[(curated['n_members'] == 1) & (curated['adv'] < 5e6) & (~curated['forced'])]
    if len(singles):
        print("-" * 84)
        print(f"⚠ {len(singles)} illiquid singleton(s) survived (<$5M/day, no floor applied):")
        print("   " + ', '.join(singles['ticker'].tolist()))


if __name__ == '__main__':
    main()
