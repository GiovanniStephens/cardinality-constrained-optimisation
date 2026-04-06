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
from datetime import datetime, timedelta

import numpy as np

from src import db
from src import config

logger = logging.getLogger(__name__)


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


def _check_zero_variance(conn, exchange_id, min_annual_vol):
    """Flag tickers with annualised volatility below threshold.

    Computes log-return std from raw prices in SQL/Python hybrid.
    """
    # Get all ticker IDs for this exchange
    ticker_rows = conn.execute(
        "SELECT id, symbol FROM tickers WHERE exchange_id = ?",
        (exchange_id,),
    ).fetchall()

    flagged = []
    for tr in ticker_rows:
        prices = conn.execute(
            "SELECT close FROM prices WHERE ticker_id = ? ORDER BY date",
            (tr['id'],),
        ).fetchall()
        if len(prices) < 30:
            continue  # too few rows; min_history check handles this
        closes = np.array([r['close'] for r in prices], dtype=float)
        closes = closes[closes > 0]  # drop zeros
        if len(closes) < 30:
            continue
        log_rets = np.diff(np.log(closes))
        annual_vol = np.std(log_rets) * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        if annual_vol < min_annual_vol:
            flagged.append((tr['id'], f"zero_variance:vol={annual_vol:.6f}"))
    return flagged


def _check_frozen_prices(conn, exchange_id, max_consecutive_same):
    """Flag tickers with long runs of identical consecutive close prices."""
    ticker_rows = conn.execute(
        "SELECT id, symbol FROM tickers WHERE exchange_id = ?",
        (exchange_id,),
    ).fetchall()

    flagged = []
    for tr in ticker_rows:
        prices = conn.execute(
            "SELECT close FROM prices WHERE ticker_id = ? ORDER BY date",
            (tr['id'],),
        ).fetchall()
        if len(prices) < max_consecutive_same:
            continue
        closes = [r['close'] for r in prices]
        max_run = _longest_constant_run(closes)
        if max_run >= max_consecutive_same:
            flagged.append((tr['id'], f"frozen_price:run={max_run}_days"))
    return flagged


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


def _check_extreme_returns(conn, exchange_id, max_extreme_pct):
    """Flag tickers where >max_extreme_pct of days have returns >10x the
    robust standard deviation (MAD-based).

    Uses median absolute deviation instead of regular std to prevent
    the extreme returns themselves from inflating the threshold.
    """
    ticker_rows = conn.execute(
        "SELECT id, symbol FROM tickers WHERE exchange_id = ?",
        (exchange_id,),
    ).fetchall()

    flagged = []
    for tr in ticker_rows:
        prices = conn.execute(
            "SELECT close FROM prices WHERE ticker_id = ? ORDER BY date",
            (tr['id'],),
        ).fetchall()
        if len(prices) < 60:
            continue
        closes = np.array([r['close'] for r in prices], dtype=float)
        closes = closes[closes > 0]
        if len(closes) < 60:
            continue
        log_rets = np.diff(np.log(closes))
        # Robust std: MAD * 1.4826 ≈ std for normal distributions
        mad = np.median(np.abs(log_rets - np.median(log_rets)))
        robust_std = mad * 1.4826
        if robust_std == 0:
            continue  # zero_variance check handles this
        extreme_count = np.sum(np.abs(log_rets) > 10 * robust_std)
        extreme_pct = extreme_count / len(log_rets)
        if extreme_pct > max_extreme_pct:
            flagged.append((tr['id'],
                            f"extreme_returns:{extreme_pct:.1%}_of_days"))
    return flagged


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

    checks = [
        ('min_history', _check_min_history(conn, exchange_id, config.MIN_HISTORY_DAYS)),
        ('stale', _check_stale_prices(conn, exchange_id, config.MAX_STALENESS_DAYS)),
        ('zero_variance', _check_zero_variance(conn, exchange_id, config.MIN_ANNUAL_VOLATILITY)),
        ('frozen_price', _check_frozen_prices(conn, exchange_id, config.MAX_CONSECUTIVE_SAME_PRICE)),
        ('extreme_returns', _check_extreme_returns(conn, exchange_id, config.MAX_EXTREME_RETURN_PCT)),
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

    print(f"\nValidation summary ({args.exchange}):")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
