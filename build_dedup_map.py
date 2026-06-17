"""Build an ETF de-duplication map from pairwise return correlations.

Collapses near-identical ETFs (e.g. the same fund cross-listed on several
exchanges, or two share classes tracking the same index) into clusters and
keeps ONE tradeable representative per cluster — so the optimiser never selects,
and you never trade, two or three practical duplicates.

Pairwise, computed in column blocks: it never materialises the full N x N
correlation matrix (which is rank-deficient and noise-dominated when N > T).
Each pair is judged on its own correlation over its jointly-available
observations, against a strict threshold where redundancy is unambiguous.

Output: a JSON map {representative_ticker: [alias_tickers...]} written to
data/etf_dedup_map.json, plus a human-readable summary for review.

Usage:
    python build_dedup_map.py [--threshold 0.97] [--min-obs 250]
"""

import argparse
import json
import logging
import os

import numpy as np

from src.logging_config import setup_logging
from src import config, db
from src.config import DB_PATH, ETF_PRICES_CSV
from src.data_loading import load_training_data
from src.returns import calculate_log_returns

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Build ETF dedup map (pairwise correlations)')
    p.add_argument('--threshold', type=float, default=0.97,
                   help='Correlation >= this marks a duplicate pair (default: %(default)s)')
    p.add_argument('--min-obs', type=int, default=250,
                   help='Min joint observations to trust a pair (default: %(default)s)')
    p.add_argument('--block', type=int, default=512,
                   help='Column block size for pairwise compute (default: %(default)s)')
    p.add_argument('--lookback-days', type=int, default=config.DATA_LOOKBACK_DAYS)
    p.add_argument('--out', default=os.path.join(os.path.dirname(DB_PATH), 'etf_dedup_map.json'))
    return p.parse_args()


class UnionFind:
    """Disjoint-set for grouping duplicate pairs into clusters."""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    args = parse_args()

    logger.info("Loading ETF universe...")
    prices = load_training_data(exchange='US', csv_fallback=ETF_PRICES_CSV,
                                lookback_days=args.lookback_days, asset_type='etf')
    tickers = list(prices.columns)
    n = len(tickers)
    logger.info("Universe: %d ETFs x %d days", n, prices.shape[0])

    # ── Standardise returns (z-scores), so a dot product == correlation ──────
    returns = calculate_log_returns(prices).values          # T x N (row 0 NaN)
    mask = np.isfinite(returns)
    mean = np.nanmean(returns, axis=0)
    std = np.nanstd(returns, axis=0, ddof=1)
    std = np.where((std == 0) | ~np.isfinite(std), np.nan, std)
    z = (returns - mean) / std
    z[~np.isfinite(z)] = 0.0                                 # missing obs contribute 0
    m = mask.astype(np.float64)

    # ── Pairwise correlations in blocks (never the full N x N at once) ───────
    uf = UnionFind(n)
    edges = 0
    thr, min_obs = args.threshold, args.min_obs
    for start in range(0, n, args.block):
        end = min(start + args.block, n)
        zb = z[:, start:end]                                # T x b
        mb = m[:, start:end]
        num = zb.T @ z                                      # b x N : sum of products
        cnt = mb.T @ m                                      # b x N : joint observations
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = num / np.where(cnt > 1, cnt - 1, np.nan)
        for li in range(end - start):
            gi = start + li
            neigh = np.where((corr[li] >= thr) & (cnt[li] >= min_obs))[0]
            for gj in neigh:
                if gj > gi:                                 # upper triangle only
                    uf.union(gi, int(gj))
                    edges += 1
        logger.info("  cols %d-%d processed (duplicate pairs so far: %d)", start, end, edges)

    # ── Assemble clusters ────────────────────────────────────────────────────
    comp = {}
    for i in range(n):
        comp.setdefault(uf.find(i), []).append(i)
    clusters = [idxs for idxs in comp.values() if len(idxs) > 1]
    collapsed = sum(len(c) - 1 for c in clusters)
    logger.info("Found %d duplicate clusters covering %d tickers; %d collapse away.",
                len(clusters), sum(len(c) for c in clusters), collapsed)

    # ── Pick a representative per cluster: prefer US-listed, then most data ──
    conn = db.get_connection(DB_PATH)
    ex_id = db._get_exchange_id(conn, 'US')
    names = {r['symbol']: r['name'] for r in conn.execute(
        "SELECT symbol, name FROM tickers WHERE exchange_id = ?", (ex_id,)).fetchall()}
    conn.close()
    obs_count = mask.sum(axis=0)

    def is_us_listed(sym):
        return '.' not in sym                               # foreign listings carry .TO/.L/.KS/...

    def rep_key(idx):
        return (is_us_listed(tickers[idx]), int(obs_count[idx]))

    dedup_map = {}
    cluster_detail = []
    for idxs in clusters:
        rep = max(idxs, key=rep_key)
        aliases = [tickers[i] for i in idxs if i != rep]
        dedup_map[tickers[rep]] = aliases
        cluster_detail.append((len(idxs), tickers[rep], aliases))

    # ── Save map + deduped universe ──────────────────────────────────────────
    with open(args.out, 'w') as f:
        json.dump(dedup_map, f, indent=2)
    alias_set = {a for al in dedup_map.values() for a in al}
    deduped = [t for t in tickers if t not in alias_set]
    uni_path = os.path.splitext(args.out)[0] + '_universe.txt'
    with open(uni_path, 'w') as f:
        f.write('\n'.join(deduped) + '\n')

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"ETF DEDUP MAP  —  threshold corr >= {thr}, min joint obs {min_obs}")
    print("=" * 78)
    print(f"Universe:            {n} ETFs")
    print(f"Duplicate clusters:  {len(clusters)} (each = several tickers, one underlying)")
    print(f"Tickers collapsed:   {collapsed}  ->  deduped universe: {len(deduped)} ETFs")
    print(f"Map written to:      {args.out}")
    print(f"Deduped universe:    {uni_path}")
    print("-" * 78)
    print("Largest clusters (kept representative  ←  collapsed aliases):")
    for size, rep, aliases in sorted(cluster_detail, key=lambda x: -x[0])[:20]:
        rep_name = (names.get(rep) or '')[:38]
        print(f"\n  [{size}] {rep:12s} {rep_name}")
        for a in aliases:
            print(f"        ← {a:12s} {(names.get(a) or '')[:38]}")


if __name__ == '__main__':
    main()
