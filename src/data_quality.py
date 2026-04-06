"""
Post-download data quality validation.

Runs checks against price data in the database and flags bad tickers
via the `excluded` column on the tickers table. Flagged tickers are
automatically skipped by load_prices() (exclude_flagged=True default).

Usage:
    python -m src.data_quality                  # validate US exchange
    python -m src.data_quality --exchange US     # explicit exchange
    python -m src.data_quality --dry-run         # preview without writing
"""

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from src import db
from src import config as uc

logger = logging.getLogger(__name__)

# MAD-to-standard-deviation conversion factor (exact for normal distributions).
MAD_TO_STD = 1.4826


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _load_prices_by_ticker(conn, exchange_id):
    """Batch-load all close prices for an exchange in a single query.

    Returns dict mapping ticker_id -> (symbol, list_of_closes) where closes
    are ordered by date.
    """
    rows = conn.execute(
        """SELECT t.id, t.symbol, p.close
           FROM tickers t
           JOIN prices p ON p.ticker_id = t.id
           WHERE t.exchange_id = ?
           ORDER BY t.id, p.date""",
        (exchange_id,),
    ).fetchall()

    symbols = {}
    prices = defaultdict(list)
    for r in rows:
        tid = r['id']
        symbols[tid] = r['symbol']
        prices[tid].append(r['close'])

    return {tid: (symbols[tid], closes) for tid, closes in prices.items()}


def _check_per_ticker(prices_by_ticker, min_rows, check_fn):
    """Apply *check_fn* to each ticker with at least *min_rows* prices.

    :param prices_by_ticker: dict from _load_prices_by_ticker().
    :param min_rows: minimum price rows required; tickers below are skipped.
    :param check_fn: callable(ticker_id, closes_list) -> (tid, reason) or None.
    :return: list of (ticker_id, reason) tuples for flagged tickers.
    """
    flagged = []
    for tid, (_symbol, closes) in prices_by_ticker.items():
        if len(closes) < min_rows:
            continue
        result = check_fn(tid, closes)
        if result is not None:
            flagged.append(result)
    return flagged


# ─── Individual checks ────────────────────────────────────────────────────────

def _check_min_history(conn, exchange_id, min_days):
    """Flag tickers with fewer than min_days price rows."""
    rows = conn.execute(
        """SELECT t.id, t.symbol, COUNT(p.date) AS n_days
           FROM tickers t
           LEFT JOIN prices p ON p.ticker_id = t.id
           WHERE t.exchange_id = ?
           GROUP BY t.id
           HAVING n_days < ?""",
        (exchange_id, min_days),
    ).fetchall()
    return [(r['id'], f"min_history:{r['n_days']}_days") for r in rows]


def _check_stale_prices(conn, exchange_id, max_staleness_days):
    """Flag tickers whose last price is too old relative to the DB max date."""
    row = conn.execute(
        """SELECT MAX(p.date) FROM prices p
           JOIN tickers t ON p.ticker_id = t.id
           WHERE t.exchange_id = ?""",
        (exchange_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return []

    db_max = datetime.strptime(row[0], '%Y-%m-%d')
    cutoff = (db_max - timedelta(days=max_staleness_days)).strftime('%Y-%m-%d')

    rows = conn.execute(
        """SELECT t.id, t.symbol, MAX(p.date) AS last_date
           FROM tickers t
           JOIN prices p ON p.ticker_id = t.id
           WHERE t.exchange_id = ?
           GROUP BY t.id
           HAVING last_date < ?""",
        (exchange_id, cutoff),
    ).fetchall()
    return [(r['id'], f"stale:last_trade_{r['last_date']}") for r in rows]


def _check_zero_variance(prices_by_ticker, min_annual_vol):
    """Flag tickers with annualised volatility below threshold."""

    def _check(tid, closes_list):
        closes = np.array(closes_list, dtype=float)
        closes = closes[closes > 0]
        if len(closes) < 30:
            return None
        log_rets = np.diff(np.log(closes))
        annual_vol = np.std(log_rets) * np.sqrt(uc.TRADING_DAYS_PER_YEAR)
        if annual_vol < min_annual_vol:
            return (tid, f"zero_variance:vol={annual_vol:.6f}")
        return None

    return _check_per_ticker(prices_by_ticker, 30, _check)


def _check_frozen_prices(prices_by_ticker, max_consecutive_same):
    """Flag tickers with long runs of identical consecutive close prices."""

    def _check(tid, closes):
        max_run = _longest_constant_run(closes)
        if max_run >= max_consecutive_same:
            return (tid, f"frozen_price:run={max_run}_days")
        return None

    return _check_per_ticker(prices_by_ticker, max_consecutive_same, _check)


def _longest_constant_run(values):
    """Return the length of the longest run of identical consecutive values."""
    if not values:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return max_run


def _check_extreme_returns(prices_by_ticker, max_extreme_pct):
    """Flag tickers where >max_extreme_pct of days have returns >10x the
    robust standard deviation (MAD-based).

    Uses median absolute deviation instead of regular std to prevent
    the extreme returns themselves from inflating the threshold.
    """

    def _check(tid, closes_list):
        closes = np.array(closes_list, dtype=float)
        closes = closes[closes > 0]
        if len(closes) < 60:
            return None
        log_rets = np.diff(np.log(closes))
        median = np.median(log_rets)
        robust_std = np.median(np.abs(log_rets - median)) * MAD_TO_STD
        if robust_std == 0:
            return None  # zero_variance check handles this
        extreme_count = np.sum(np.abs(log_rets) > 10 * robust_std)
        extreme_pct = extreme_count / len(log_rets)
        if extreme_pct > max_extreme_pct:
            return (tid, f"extreme_returns:{extreme_pct:.1%}_of_days")
        return None

    return _check_per_ticker(prices_by_ticker, 60, _check)


# ─── Main validation entry point ─────────────────────────────────────────────

def validate_universe(conn, exchange='US', dry_run=False):
    """Run all quality checks and flag bad tickers. Returns summary dict.

    If dry_run=True, prints what would be excluded without modifying the DB.
    """
    exchange_id = db._get_exchange_id(conn, exchange)

    # Clear existing exclusions so tickers can be re-evaluated
    if not dry_run:
        conn.execute(
            "UPDATE tickers SET excluded = NULL WHERE exchange_id = ? AND excluded IS NOT NULL",
            (exchange_id,),
        )

    # Batch-load prices once for the three checks that need per-ticker prices.
    prices_by_ticker = _load_prices_by_ticker(conn, exchange_id)

    checks = [
        ('min_history', _check_min_history(conn, exchange_id, uc.MIN_HISTORY_DAYS)),
        ('stale', _check_stale_prices(conn, exchange_id, uc.MAX_STALENESS_DAYS)),
        ('zero_variance', _check_zero_variance(prices_by_ticker, uc.MIN_ANNUAL_VOLATILITY)),
        ('frozen_price', _check_frozen_prices(prices_by_ticker, uc.MAX_CONSECUTIVE_SAME_PRICE)),
        ('extreme_returns', _check_extreme_returns(prices_by_ticker, uc.MAX_EXTREME_RETURN_PCT)),
    ]

    # Deduplicate: first check to flag a ticker wins
    seen = set()
    exclusions = []
    summary = {}
    for check_name, results in checks:
        count = 0
        for ticker_id, reason in results:
            if ticker_id not in seen:
                seen.add(ticker_id)
                exclusions.append((ticker_id, reason))
                count += 1
        summary[check_name] = count
        logger.info("  %s: %d tickers flagged", check_name, count)

    total_tickers = conn.execute(
        "SELECT COUNT(*) FROM tickers WHERE exchange_id = ?", (exchange_id,),
    ).fetchone()[0]
    summary['total_tickers'] = total_tickers
    summary['total_excluded'] = len(exclusions)
    summary['total_active'] = total_tickers - len(exclusions)

    if dry_run:
        logger.info("DRY RUN: would exclude %d/%d tickers", len(exclusions), total_tickers)
    else:
        for ticker_id, reason in exclusions:
            db.set_ticker_excluded(conn, ticker_id, reason)
        conn.commit()
        logger.info("Excluded %d/%d tickers (%d active)",
                     len(exclusions), total_tickers, summary['total_active'])

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    from src.logging_config import setup_logging
    setup_logging()
    parser = argparse.ArgumentParser(
        description='Validate universe data quality and flag bad tickers.')
    parser.add_argument('--exchange', default='US',
                        help='Exchange code to validate (default: US).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview exclusions without modifying the database.')
    args = parser.parse_args()

    conn = db.get_connection()
    summary = validate_universe(conn, exchange=args.exchange, dry_run=args.dry_run)
    conn.close()

    logger.info("Validation summary (%s):", args.exchange)
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)


if __name__ == '__main__':
    from src.logging_config import setup_logging
    setup_logging()
    main()
