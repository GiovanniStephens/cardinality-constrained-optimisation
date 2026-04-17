"""
Downloads price data from Yahoo Finance.

Provides core download primitives, sequential download-and-save, and a CLI
entry point. Session management and proxy/Tor setup live in
src.download_session; ticker validation in src.download_validate; universe
building in src.universe; concurrent worker infrastructure in
src.download_workers.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.config import (
    DOWNLOAD_BACKOFF_BASE,
    DOWNLOAD_DEFAULT_BATCH_SIZE,
    DOWNLOAD_DEFAULT_END,
    DOWNLOAD_DEFAULT_START,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_THREADS,
    DOWNLOAD_TIMEOUT,
    TICKER_EXCLUDE_SUFFIXES,
    TICKER_EXCLUDE_NAME_PATTERNS,
)
from src.download_validate import validate_tickers, _retry_with_splitting
from src.exceptions import DownloadError, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price downloading
# ---------------------------------------------------------------------------

def _download_batch(tickers: list[str], start: str, end: str,
                    session: Any = None) -> Optional[pd.DataFrame]:
    """Download a single batch from yfinance. Returns wide DataFrame or None."""
    import src.download_session as _sess
    tickers_str = " ".join(tickers)
    if session is None:
        session = _sess._make_session()
    # Disable yfinance internal threading when using a proxy — curl_cffi
    # sessions are not safe for concurrent use across yfinance's threads.
    # Our external workers already provide download parallelism.
    yf_threads = False if _sess._get_state('_proxy_url') else DOWNLOAD_THREADS
    prices = yf.download(
        tickers_str, interval="1d", group_by="ticker", start=start, end=end,
        threads=yf_threads, timeout=DOWNLOAD_TIMEOUT, session=session,
    )
    # Validate returned structure (P5)
    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        return None
    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices.index = pd.to_datetime(prices.index)
        except (ValueError, TypeError) as e:
            logger.warning("Batch returned non-date index type: %s (%s)", type(prices.index), e)
            return None

    batch_prices = {}
    skipped = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                batch_prices[ticker] = prices["Close"].tolist()
            else:
                batch_prices[ticker] = prices[ticker]["Close"].tolist()
        except (KeyError, TypeError):
            skipped.append(ticker)
    # Log aggregated summary instead of per-ticker warnings (P7)
    if skipped:
        preview = ', '.join(skipped[:10])
        suffix = f'... (+{len(skipped) - 10} more)' if len(skipped) > 10 else ''
        logger.info("Batch: %d/%d tickers had no data: %s%s",
                     len(skipped), len(tickers), preview, suffix)
    if not batch_prices:
        return None
    return pd.DataFrame(batch_prices, index=prices.index)


def _download_batch_with_timeout(tickers: list[str], start: str, end: str,
                                 timeout_seconds: float) -> Optional[pd.DataFrame]:
    """Wrap _download_batch with a timeout. Returns None on timeout."""
    # Don't pass a pre-built session — curl_cffi sessions are not safe to
    # share across threads.  _download_batch will call _make_session()
    # inside the executor thread instead.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_download_batch, tickers, start, end)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeout:
            logger.warning("Batch download timed out after %ds (%d tickers)",
                           timeout_seconds, len(tickers))
            return None


def download_data(
    tickers_df: pd.DataFrame,
    ticker_column: str = "Tickers",
    start: str = DOWNLOAD_DEFAULT_START,
    end: str = DOWNLOAD_DEFAULT_END,
    batch_size: int = DOWNLOAD_DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Downloads closing price data from Yahoo Finance for the given tickers.

    Processes tickers in batches to avoid timeouts with large ticker lists.

    :param tickers_df: DataFrame containing ticker symbols.
    :param ticker_column: name of the column containing ticker symbols.
    :param start: start date for price data.
    :param end: end date for price data.
    :param batch_size: number of tickers to download per batch.
    :returns: daily closing price data as a DataFrame.
    """
    all_tickers = tickers_df[ticker_column].tolist()
    batches = [
        all_tickers[i : i + batch_size]
        for i in range(0, len(all_tickers), batch_size)
    ]

    all_prices = {}
    for batch_num, batch in enumerate(batches, 1):
        logger.info("Downloading batch %d/%d (%d tickers)...",
                     batch_num, len(batches), len(batch))
        batch_df = None
        for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
            try:
                batch_df = _download_batch(batch, start, end)
                break
            except (ConnectionError, OSError, DownloadError) as e:
                if attempt == DOWNLOAD_MAX_RETRIES:
                    raise DownloadError(
                        f"Failed after {DOWNLOAD_MAX_RETRIES} retries: {e}") from e
                logger.warning("Batch %d attempt %d failed: %s",
                               batch_num, attempt, e)
                time.sleep(DOWNLOAD_BACKOFF_BASE ** attempt)
            except Exception as e:
                if attempt == DOWNLOAD_MAX_RETRIES:
                    raise DownloadError(
                        f"Failed after {DOWNLOAD_MAX_RETRIES} retries (unexpected): {e}") from e
                logger.warning("Batch %d attempt %d failed (unexpected): %s",
                               batch_num, attempt, e)
                time.sleep(DOWNLOAD_BACKOFF_BASE ** attempt)
        if batch_df is not None:
            for col in batch_df.columns:
                all_prices[col] = batch_df[col].tolist()

    if not all_prices:
        raise ValueError("No valid price data could be extracted for any ticker.")
    return pd.DataFrame(all_prices)


def download_and_save(
    tickers: list[str], conn: sqlite3.Connection, exchange: str,
    asset_type: str = 'etf',
    start: str = DOWNLOAD_DEFAULT_START, end: str = DOWNLOAD_DEFAULT_END,
    batch_size: int = DOWNLOAD_DEFAULT_BATCH_SIZE,
    null_threshold: float = 0.9,
    names: Optional[dict[str, str]] = None,
    countries: Optional[dict[str, str]] = None,
    max_retries: int = 3,
    on_batch_complete: Optional[Callable[[list[str], int], None]] = None,
    on_batch_failed: Optional[Callable[[list[str], int], None]] = None,
    rate_limit_delay: Optional[float] = None,
    batch_timeout: Optional[float] = None,
    circuit_breaker_threshold: Optional[int] = None,
    max_rate_limit_delay: Optional[float] = None,
    circuit_breaker_max_trips: Optional[int] = None,
    circuit_breaker_cooldown: Optional[float] = None,
    sectors: Optional[dict[str, str]] = None,
    industries: Optional[dict[str, str]] = None,
    category_groups: Optional[dict[str, str]] = None,
    categories: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """
    Download prices in batches and persist each batch to the database immediately.

    This is the preferred path for large universes (1000+ tickers). Each batch is
    saved via upsert so the run can be safely interrupted and resumed.

    :param tickers: list of ticker symbol strings.
    :param conn: open sqlite3 connection (from db.get_connection()).
    :param exchange: DB exchange code ('US', 'NZX', 'ASX').
    :param asset_type: DB asset type ('etf', 'stock', 'fund').
    :param start: start date for price data.
    :param end: end date for price data.
    :param batch_size: number of tickers per yfinance request.
    :param null_threshold: fraction of non-null rows required to keep a ticker.
    :param names: optional dict {symbol: name_string}.
    :param countries: optional dict {symbol: country_string}.
    :param max_retries: retries per batch on download failure.
    :param on_batch_complete: optional callback(saved_tickers: list[str], batch_num: int)
        called after each successful batch save.
    :param on_batch_failed: optional callback(failed_tickers: list[str], batch_num: int)
        called when a batch exhausts all retries.
    :param rate_limit_delay: seconds to sleep between batches (default: 0).
    :param batch_timeout: seconds before a single batch download times out.
    :param circuit_breaker_threshold: consecutive failed batches before tripping.
    :param max_rate_limit_delay: maximum adaptive inter-batch delay (seconds).
    :param circuit_breaker_max_trips: hard abort after this many trips.
    :param circuit_breaker_cooldown: seconds to pause on trip before retrying.
    :returns: dict with keys total_tickers, saved_tickers, failed_batches,
        circuit_breaker_tripped (bool), and circuit_breaker_trip_count (int).
    """
    from src import db, config
    import src.download_session as _sess

    # Apply config defaults
    if rate_limit_delay is None:
        rate_limit_delay = config.PIPELINE_RATE_LIMIT_DELAY
    if batch_timeout is None:
        batch_timeout = config.PIPELINE_BATCH_TIMEOUT
    if circuit_breaker_threshold is None:
        circuit_breaker_threshold = config.PIPELINE_CIRCUIT_BREAKER_THRESHOLD
    if max_rate_limit_delay is None:
        max_rate_limit_delay = config.PIPELINE_MAX_RATE_LIMIT_DELAY
    if circuit_breaker_max_trips is None:
        circuit_breaker_max_trips = config.PIPELINE_CIRCUIT_BREAKER_MAX_TRIPS
    if circuit_breaker_cooldown is None:
        circuit_breaker_cooldown = config.PIPELINE_CIRCUIT_BREAKER_COOLDOWN

    # Deduplicate ticker list (P6)
    seen = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    if len(deduped) < len(tickers):
        logger.info("Removed %d duplicate tickers", len(tickers) - len(deduped))
    tickers = deduped

    batches = [
        tickers[i : i + batch_size]
        for i in range(0, len(tickers), batch_size)
    ]
    total_saved = 0
    failed_batches = []
    consecutive_failures = 0
    circuit_breaker_tripped = False
    circuit_breaker_trip_count = 0
    current_delay = rate_limit_delay
    t_start = time.time()

    use_tqdm = sys.stderr.isatty()
    batch_iter = enumerate(batches, 1)
    if use_tqdm:
        pbar = tqdm(total=len(batches), desc="Downloading", unit="batch",
                    file=sys.stderr)
    else:
        pbar = None

    for batch_num, batch in batch_iter:
        if not use_tqdm:
            logger.info("Batch %d/%d (%d tickers)...",
                        batch_num, len(batches), len(batch))
        batch_df = None
        hit_rate_limit = False
        for attempt in range(1, max_retries + 1):
            try:
                batch_df = _download_batch_with_timeout(
                    batch, start, end, batch_timeout)
                if batch_df is not None and not batch_df.empty:
                    break
                # Empty result likely means Yahoo silently rate-limited
                if attempt < max_retries:
                    hit_rate_limit = True
                    if _sess._get_state('_tor_enabled'):
                        _sess._rotate_tor_circuit()
                    current_delay = min(current_delay * DOWNLOAD_BACKOFF_BASE, max_rate_limit_delay)
                    jittered = current_delay * random.uniform(0.8, 1.2)
                    logger.warning("Batch %d attempt %d/%d: no data returned, "
                                   "likely rate-limited. Backing off %.0fs "
                                   "(adaptive delay now %.0fs)",
                                   batch_num, attempt, max_retries,
                                   jittered, current_delay)
                    time.sleep(jittered)
            except (ConnectionError, OSError, TimeoutError) as e:
                error_msg = str(e).lower()
                is_rate_limit = ('429' in error_msg or 'too many' in error_msg
                                 or 'rate' in error_msg)
                if is_rate_limit:
                    hit_rate_limit = True
                    if _sess._get_state('_tor_enabled'):
                        _sess._rotate_tor_circuit()
                    current_delay = min(current_delay * DOWNLOAD_BACKOFF_BASE, max_rate_limit_delay)
                    jittered = current_delay * random.uniform(0.8, 1.2)
                    logger.warning("Rate limited on batch %d (attempt %d/%d), "
                                   "backing off %.0fs (adaptive delay now %.0fs)",
                                   batch_num, attempt, max_retries,
                                   jittered, current_delay)
                    time.sleep(jittered)
                else:
                    logger.warning("Batch %d attempt %d/%d failed: %s",
                                   batch_num, attempt, max_retries, e)
                    logger.debug("Batch %d traceback:", batch_num, exc_info=True)
                    if attempt < max_retries:
                        jittered = current_delay * random.uniform(0.8, 1.2)
                        time.sleep(jittered)

        if batch_df is None or batch_df.empty:
            # Try sub-batch splitting before giving up
            min_sub = config.PIPELINE_MIN_SUB_BATCH_SIZE
            if len(batch) > min_sub:
                logger.info("Batch %d: full batch failed, attempting sub-batch "
                            "splitting (min=%d, delay=%.0fs)...",
                            batch_num, min_sub, current_delay)
                delay_state = [current_delay, max_rate_limit_delay,
                               rate_limit_delay]
                split_df, still_failed = _retry_with_splitting(
                    list(batch), start, end, batch_timeout,
                    min_sub, delay_state)
                current_delay = delay_state[0]  # propagate back
                if split_df is not None and not split_df.empty:
                    batch_df = split_df
                    if still_failed:
                        failed_batches.append({
                            'batch_num': batch_num,
                            'tickers': still_failed,
                        })
                        if on_batch_failed:
                            on_batch_failed(still_failed, batch_num)
                        logger.info("Batch %d: recovered %d tickers, "
                                    "%d still failed after splitting",
                                    batch_num, len(split_df.columns),
                                    len(still_failed))
                    # Fall through to the save path below
                else:
                    batch_df = None  # splitting recovered nothing

            if batch_df is None or batch_df.empty:
                logger.warning("Batch %d: no data returned, skipping.",
                               batch_num)
                failed_batches.append({
                    'batch_num': batch_num, 'tickers': list(batch),
                })
                consecutive_failures += 1
                if on_batch_failed:
                    on_batch_failed(list(batch), batch_num)
                if pbar:
                    pbar.update(1)

                # Circuit breaker with cooldown-then-retry
                if consecutive_failures >= circuit_breaker_threshold:
                    circuit_breaker_trip_count += 1
                    if circuit_breaker_trip_count >= circuit_breaker_max_trips:
                        logger.error(
                            "Circuit breaker tripped %d times (max %d). "
                            "Aborting download. Use --resume to retry later.",
                            circuit_breaker_trip_count,
                            circuit_breaker_max_trips)
                        circuit_breaker_tripped = True
                        break
                    else:
                        logger.warning(
                            "Circuit breaker trip %d/%d: %d consecutive "
                            "failures. Cooling down %.0fs before retry...",
                            circuit_breaker_trip_count,
                            circuit_breaker_max_trips,
                            consecutive_failures,
                            circuit_breaker_cooldown)
                        time.sleep(circuit_breaker_cooldown)
                        consecutive_failures = 0
                        current_delay = max_rate_limit_delay
                continue

        # Reset circuit breaker on success
        consecutive_failures = 0

        # Adaptive delay: already escalated during retries above;
        # on clean success, halve toward the base delay
        if not hit_rate_limit:
            old = current_delay
            current_delay = max(rate_limit_delay, current_delay * 0.5)
            if current_delay < old:
                logger.info("Clean batch, adaptive delay halved to %.1fs",
                            current_delay)

        # Filter out tickers with too many nulls
        pre_filter_count = len(batch_df.columns)
        threshold = int(len(batch_df) * null_threshold)
        batch_df = batch_df.dropna(axis=1, thresh=threshold)
        dropped = pre_filter_count - len(batch_df.columns)
        if dropped:
            logger.info("Batch %d: dropped %d/%d tickers below %.0f%% coverage",
                        batch_num, dropped, pre_filter_count, null_threshold * 100)
        if batch_df.empty:
            if pbar:
                pbar.update(1)
            continue

        # Build per-batch metadata dicts
        batch_names = None
        if names:
            batch_names = {t: names[t] for t in batch_df.columns if t in names}
        batch_countries = None
        if countries:
            batch_countries = {t: countries[t] for t in batch_df.columns if t in countries}
        batch_sectors = None
        if sectors:
            batch_sectors = {t: sectors[t] for t in batch_df.columns if t in sectors}
        batch_industries = None
        if industries:
            batch_industries = {t: industries[t] for t in batch_df.columns if t in industries}
        batch_cat_groups = None
        if category_groups:
            batch_cat_groups = {t: category_groups[t] for t in batch_df.columns
                                if t in category_groups}
        batch_categories = None
        if categories:
            batch_categories = {t: categories[t] for t in batch_df.columns
                                if t in categories}

        db.save_prices(conn, batch_df, exchange=exchange, asset_type=asset_type,
                       names=batch_names, countries=batch_countries,
                       sectors=batch_sectors, industries=batch_industries,
                       category_groups=batch_cat_groups,
                       categories=batch_categories)
        saved_tickers_list = list(batch_df.columns)
        total_saved += len(saved_tickers_list)

        # Enhanced progress reporting (P8)
        elapsed = time.time() - t_start
        if pbar:
            pbar.update(1)
            pbar.set_postfix(
                saved=total_saved,
                failed=len(failed_batches),
                rate=f"{total_saved / elapsed:.0f}/s" if elapsed > 0 else "N/A",
            )
        else:
            logger.info("Batch %d: saved %d tickers (total: %d)",
                        batch_num, len(saved_tickers_list), total_saved)

        if on_batch_complete:
            on_batch_complete(saved_tickers_list, batch_num)

        if _sess._get_state('_tor_enabled'):
            from src.config import TOR_ROTATE_EVERY_N_BATCHES
            if batch_num % TOR_ROTATE_EVERY_N_BATCHES == 0:
                _sess._rotate_tor_circuit()

        if current_delay > 0 and batch_num < len(batches):
            jittered = current_delay * random.uniform(0.8, 1.2)
            time.sleep(jittered)

    if pbar:
        pbar.close()

    # Summary log (P7)
    if failed_batches:
        total_failed_tickers = sum(len(fb['tickers']) for fb in failed_batches)
        logger.warning("Download summary: %d/%d batches failed (%d tickers). "
                       "Use checkpoint to retry.",
                       len(failed_batches), len(batches), total_failed_tickers)

    return {
        'total_tickers': len(tickers),
        'saved_tickers': total_saved,
        'failed_batches': failed_batches,
        'circuit_breaker_tripped': circuit_breaker_tripped,
        'circuit_breaker_trip_count': circuit_breaker_trip_count,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _add_file_logging(log_dir: str = 'data') -> str:
    """Add a file handler so logs survive terminal close during long runs."""
    from datetime import datetime as dt
    os.makedirs(log_dir, exist_ok=True)
    ts = dt.now().strftime('%Y%m%dT%H%M%S')
    log_path = os.path.join(log_dir, f'download_{ts}.log')
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return log_path


def main() -> None:
    # Register this module under its canonical name so that when the pipeline
    # does `from src.download_data import ...`, it gets THIS module (with our
    # globals) instead of a fresh copy with _proxy_url=None.
    # This is the standard fix for the `python -m` double-import issue.
    import sys as _sys
    if __name__ == '__main__' and 'src.download_data' not in _sys.modules:
        _sys.modules['src.download_data'] = _sys.modules['__main__']

    from src.logging_config import setup_logging
    from src.universe import (
        ALL_ASSET_TYPES, ASSET_TYPE_MAP, build_security_universe,
        filter_unwanted_tickers, load_tickers,
    )

    setup_logging()
    parser = argparse.ArgumentParser(
        description='Build a security universe and download price data.')

    # Universe source arguments
    parser.add_argument('--asset-types', nargs='+',
                        default=ALL_ASSET_TYPES,
                        choices=ALL_ASSET_TYPES,
                        help='Asset types to include (default: all).')
    parser.add_argument('--countries', nargs='+', default=None,
                        help='Countries to filter equities by. '
                             'Defaults to config.INCLUDED_COUNTRIES (27 countries).')
    parser.add_argument('--all-countries', action='store_true',
                        help='Download equities from all countries '
                             '(override the default country filter).')
    parser.add_argument('--market-caps', nargs='+', default=None,
                        help='Market cap categories to include for equities. '
                             'Defaults to config.INCLUDED_MARKET_CAPS (Small Cap+).')
    parser.add_argument('--all-market-caps', action='store_true',
                        help='Download equities of all market cap sizes '
                             '(override the default market cap filter).')
    parser.add_argument('--sectors', nargs='+', default=None,
                        help='Sectors to filter equities by.')
    parser.add_argument('--exchanges', nargs='+', default=None,
                        help='Exchanges to filter equities by. '
                             'Defaults to config.INCLUDED_EXCHANGES (major US).')
    parser.add_argument('--all-exchanges', action='store_true',
                        help='Download equities from all exchanges '
                             '(override the default US exchange filter).')
    parser.add_argument('--from-csv', default=None,
                        help='Load tickers from a local CSV instead of '
                             'FinanceDatabase (e.g. data/ETFs.csv).')
    parser.add_argument('--ticker-column', default='Tickers',
                        help='Column name for tickers in the CSV.')
    parser.add_argument('--asset-type', default='etf',
                        help='Asset type label when loading from CSV '
                             '(ignored when using FinanceDatabase).')
    parser.add_argument('--exchange', default='US',
                        help='Database exchange code (US, NZX, ASX).')
    parser.add_argument('--start', default=DOWNLOAD_DEFAULT_START,
                        help='Start date for price data.')
    parser.add_argument('--end',
                        default=__import__('datetime').date.today().isoformat(),
                        help='End date for price data (default: today).')
    parser.add_argument('--null-threshold', type=float, default=0.9,
                        help='Fraction of non-null values required to keep '
                             'a ticker (0-1).')
    parser.add_argument('--incremental', action='store_true',
                        help='Only download data after the latest date already '
                             'in the database.')

    # Pipeline arguments
    parser.add_argument('--subset', type=int, default=None,
                        help='Download only the first N tickers (for testing).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded and check disk, '
                             'then exit without downloading.')
    parser.add_argument('--stage-only', action='store_true',
                        help='Download into staging DB but do not promote '
                             'to production.')
    parser.add_argument('--promote', default=None, metavar='STAGING_DB',
                        help='Promote a previously staged DB to production.')
    parser.add_argument('--rollback', default=None, metavar='BACKUP_PATH',
                        help='Restore production DB from a backup file.')
    parser.add_argument('--resume', default=None, metavar='CHECKPOINT',
                        help='Resume a previously interrupted download from '
                             'a checkpoint JSON file.')
    parser.add_argument('--rate-limit', type=float, default=None,
                        help='Seconds to wait between batches '
                             '(default: from config).')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip production DB backup before promotion.')
    parser.add_argument('--keep-staging', action='store_true',
                        help='Keep staging DB after promotion (for forensics).')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run data quality checks on existing DB without '
                             'downloading.')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip data quality checks after download.')
    parser.add_argument('--retry-dropped', action='store_true',
                        help='Retry tickers from the known-bad cache in the '
                             'database instead of building a new universe.')
    parser.add_argument('--ignore-cache', action='store_true',
                        help='Ignore the known-bad ticker cache and '
                             'download all tickers.')
    parser.add_argument('--skip-prefilter', action='store_true',
                        help='Skip the regex pre-filter that removes '
                             'warrants, units, preferred shares, rights, '
                             'and SPACs.')
    parser.add_argument('--validate-first', action='store_true',
                        help='Run a quick validation pass to identify valid '
                             'tickers before the full download. Downloads 1 week '
                             'of data per ticker to check availability.')
    parser.add_argument('--skip-validation-cache', action='store_true',
                        help='Ignore cached validation results and re-validate.')
    parser.add_argument('--proxy', default=None,
                        help='SOCKS5 or HTTP proxy URL for downloads '
                             '(e.g. socks5://user:pass@host:port).')
    parser.add_argument('--use-tor', action='store_true',
                        help='Route downloads through Tor SOCKS5 proxy with '
                             'automatic circuit rotation for IP diversity. '
                             'Requires: brew install tor && brew services start tor')
    parser.add_argument('--workers', type=int, default=None,
                        help='Concurrent download workers (default: 1). '
                             'Use with --proxy for best results.')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear the known-bad ticker cache and exit.')
    args = parser.parse_args()

    from src import db, config
    import src.download_session as _sess

    # ── Proxy setup ─────────────────────────────────────────────────────
    if args.proxy or args.use_tor:
        if args.use_tor:
            _sess._proxy_url = config.TOR_SOCKS_PROXY
            _sess._tor_enabled = True
            try:
                _sess._rotate_tor_circuit()
            except Exception as e:
                logger.error("Tor not reachable: %s. Install with: "
                             "brew install tor && brew services start tor", e)
                return
        else:
            _sess._proxy_url = args.proxy
        logger.info("Proxy enabled: %s", _sess._proxy_url)

    # ── Apply default filters for equities ───────────────────────────────
    if args.all_countries:
        args.countries = None
        logger.info("--all-countries: downloading equities from all countries")
    elif args.countries is None and 'equities' in args.asset_types:
        args.countries = config.INCLUDED_COUNTRIES
        logger.info("Defaulting to %d configured countries (use --all-countries to override)",
                     len(args.countries))

    if args.all_market_caps:
        args.market_caps = None
        logger.info("--all-market-caps: downloading equities of all market cap sizes")
    elif args.market_caps is None and 'equities' in args.asset_types:
        args.market_caps = config.INCLUDED_MARKET_CAPS
        logger.info("Defaulting to market caps: %s (use --all-market-caps to override)",
                     ', '.join(args.market_caps))

    if args.all_exchanges:
        args.exchanges = None
        logger.info("--all-exchanges: downloading equities from all exchanges")
    elif args.exchanges is None and 'equities' in args.asset_types:
        args.exchanges = config.INCLUDED_EXCHANGES
        logger.info("Defaulting to US exchanges: %s (use --all-exchanges to override)",
                     ', '.join(args.exchanges))

    # ── Standalone operations (no download needed) ─────────────────────────

    if args.clear_cache:
        conn = db.get_connection()
        db.clear_known_bad_tickers(conn, exchange=args.exchange)
        conn.close()
        return

    if args.rollback:
        from src.pipeline import rollback
        rollback(args.rollback, db.DB_PATH)
        logger.info("Rollback complete.")
        return

    if args.promote:
        from src.pipeline import promote_staging, backup_database
        conn_prod = db.get_connection()
        if not args.no_backup and os.path.exists(db.DB_PATH):
            from datetime import datetime as dt
            ts = dt.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{db.DB_PATH}.backup_{ts}"
            backup_database(conn_prod, backup_path)
        promoted = promote_staging(args.promote, conn_prod, exchange=args.exchange)
        conn_prod.close()
        logger.info("Promoted %d tickers from %s", promoted, args.promote)
        return

    if args.validate_only:
        from src.data_quality import validate_universe
        conn = db.get_connection()
        summary = validate_universe(conn, exchange=args.exchange)
        conn.close()
        logger.info("Validation summary: %s", summary)
        return

    # ── File logging (survives terminal close) ──────────────────────────
    log_path = _add_file_logging()
    logger.info("Logging to %s", log_path)

    # ── Build ticker list ──────────────────────────────────────────────────

    if args.retry_dropped:
        # Load tickers that have failed at least once from the database
        conn = db.get_connection()
        # Use min_failures=1 to retry everything that's ever failed
        known_bad = db.load_known_bad_tickers(conn, exchange=args.exchange,
                                              min_failures=1)
        conn.close()
        if not known_bad:
            logger.error("No known-bad tickers in DB for exchange %s. "
                         "Nothing to retry.", args.exchange)
            return

        tickers_df = pd.DataFrame({
            'Tickers': sorted(known_bad),
            'Name': '',
            'Country': '',
            'AssetType': 'equity',  # default; pipeline uses --asset-type
        })
        logger.info("Retrying %d known-bad tickers from DB cache "
                     "(exchange=%s)", len(tickers_df), args.exchange)

        # Clear the cache so successfully downloaded tickers aren't
        # re-cached; those that fail again will get their count incremented
        conn = db.get_connection()
        db.clear_known_bad_tickers(conn, exchange=args.exchange)
        conn.close()
        logger.info("Cleared known-bad cache (tickers that still fail "
                     "will be re-recorded)")

    elif args.from_csv:
        logger.info("Loading tickers from %s", args.from_csv)
        tickers_df = load_tickers(args.from_csv, args.ticker_column)
    else:
        logger.info("Building security universe from FinanceDatabase...")
        tickers_df = build_security_universe(
            asset_types=args.asset_types,
            countries=args.countries,
            sectors=args.sectors,
            exchanges=args.exchanges,
            market_caps=args.market_caps,
        )
        logger.info("Built universe: %d securities", len(tickers_df))
        for at in tickers_df['AssetType'].unique():
            count = (tickers_df['AssetType'] == at).sum()
            logger.info("  %s: %d", at, count)
        # Log country breakdown for equities
        if 'Country' in tickers_df.columns:
            equities_mask = tickers_df['AssetType'] == 'equity'
            if equities_mask.any():
                country_counts = (tickers_df.loc[equities_mask, 'Country']
                                  .value_counts()
                                  .sort_values(ascending=False))
                logger.info("Equity country breakdown (%d countries):",
                            len(country_counts))
                for country, cnt in country_counts.items():
                    logger.info("    %s: %d", country, cnt)

    # ── Pre-filter unwanted tickers (warrants, units, SPACs, etc.) ─────────

    if not args.skip_prefilter:
        before = len(tickers_df)
        name_col = 'Name' if 'Name' in tickers_df.columns else None
        tickers_df, removed = filter_unwanted_tickers(
            tickers_df, ticker_column=args.ticker_column,
            name_column=name_col or 'Name',
            skip_name_filter=(name_col is None),
        )
        if len(removed):
            logger.info("Pre-filter removed %d unwanted tickers "
                        "(%d remain). Use --skip-prefilter to bypass.",
                        len(removed), len(tickers_df))
            # Log breakdown by removal reason
            suffix_re = re.compile(TICKER_EXCLUDE_SUFFIXES)
            suffix_hits = removed[args.ticker_column].str.contains(
                suffix_re, na=False).sum()
            name_hits = len(removed) - suffix_hits
            if suffix_hits:
                logger.info("  Suffix matches (warrants/units/preferred/rights): %d",
                            suffix_hits)
            if name_hits:
                logger.info("  Name matches (SPACs/shell companies): %d",
                            name_hits)
    else:
        logger.info("--skip-prefilter: skipping ticker pre-filter")

    # ── Validate tickers via quick yfinance check (opt-in) ────────────────

    if args.validate_first:
        ticker_list = tickers_df[args.ticker_column].tolist()
        # Skip tickers already in DB with price data — no need to validate
        conn = db.get_connection()
        existing = db.get_tickers_with_prices(conn, exchange=args.exchange)
        conn.close()
        to_validate = [t for t in ticker_list if t not in existing]
        if len(ticker_list) - len(to_validate) > 0:
            logger.info("Skipping validation for %d tickers already in DB",
                         len(ticker_list) - len(to_validate))
        cache_hours = None if args.skip_validation_cache else config.VALIDATION_CACHE_HOURS
        valid, invalid, unvalidated = validate_tickers(
            to_validate,
            cache_dir=str(Path(config.DATA_DIR)),
            max_cache_hours=cache_hours,
        )
        if invalid:
            # Remove invalid tickers from universe
            tickers_df = tickers_df[~tickers_df[args.ticker_column].isin(invalid)]
            logger.info("Validation removed %d invalid tickers (%d valid, "
                         "%d unvalidated, %d remain)",
                         len(invalid), len(valid), len(unvalidated), len(tickers_df))
            # Pre-seed known-bad cache so future runs skip these even
            # without --validate-first
            conn = db.get_connection()
            db.save_known_bad_tickers(conn, list(invalid), exchange=args.exchange)
            db.save_known_bad_tickers(conn, list(invalid), exchange=args.exchange)
            # Called twice → failure_count=2 → auto-filtered by known-bad cache
            conn.close()
            logger.info("Pre-seeded %d invalid tickers into known-bad cache",
                         len(invalid))
        else:
            logger.info("Validation: all tickers appear valid or unvalidated")

    # ── Filter known-bad tickers ────────────────────────────────────────────

    if not args.ignore_cache:
        conn = db.get_connection()
        known_bad = db.load_known_bad_tickers(conn, exchange=args.exchange)
        conn.close()
        if known_bad:
            before = len(tickers_df)
            tickers_df = tickers_df[~tickers_df[args.ticker_column].isin(known_bad)]
            removed = before - len(tickers_df)
            if removed:
                logger.info("Filtered %d known-bad tickers from DB cache "
                            "(%d remain, %d in cache). "
                            "Use --ignore-cache to override.",
                            removed, len(tickers_df), len(known_bad))
    else:
        logger.info("--ignore-cache: skipping known-bad ticker filter")

    # ── Determine start date ───────────────────────────────────────────────

    start = args.start
    if args.incremental:
        conn = db.get_connection()
        latest = db.get_latest_prices_date(conn, exchange=args.exchange)
        conn.close()
        if latest:
            from datetime import datetime as dt_cls, timedelta
            next_day = (dt_cls.strptime(latest, '%Y-%m-%d')
                        + timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info("Incremental mode: latest DB date is %s, downloading from %s",
                         latest, next_day)
            start = next_day
        else:
            logger.info("Incremental mode: no existing data, full download from %s", start)

    # ── Dry run ────────────────────────────────────────────────────────────

    if args.dry_run:
        from src.pipeline import preflight_check
        num_tickers = len(tickers_df)
        if args.subset:
            num_tickers = min(num_tickers, args.subset)

        from datetime import datetime as dt_cls
        start_dt = dt_cls.strptime(start, '%Y-%m-%d')
        end_dt = dt_cls.strptime(args.end, '%Y-%m-%d')
        est_days = int((end_dt - start_dt).days * 252 / 365)

        data_dir = os.path.dirname(db.DB_PATH)
        staging_path = os.path.join(data_dir, 'staging_dryrun.db')
        preflight = preflight_check(db.DB_PATH, staging_path,
                                    num_tickers, est_days)

        if args.validate_first:
            logger.info("DRY RUN — would validate %d tickers across %d date "
                         "windows before download",
                         num_tickers, len(config.VALIDATION_WINDOWS))
        logger.info("DRY RUN — would download %d tickers, %s to %s",
                     num_tickers, start, args.end)
        logger.info("  Estimated staging DB: %.2f GB", preflight['estimated_staging_gb'])
        logger.info("  Production DB: %.2f GB", preflight['prod_size_gb'])
        logger.info("  Total space needed: %.2f GB", preflight['estimated_total_gb'])
        logger.info("  Disk available: %.2f GB", preflight['available_gb'])
        if preflight['ok']:
            logger.info("  Preflight: PASS")
        else:
            for w in preflight['warnings']:
                logger.warning("  Preflight: FAIL — %s", w)
        return

    # ── Resume from checkpoint ─────────────────────────────────────────────

    checkpoint_path = None
    staging_db_path = None
    if args.resume:
        from src.pipeline import load_checkpoint
        checkpoint = load_checkpoint(args.resume)
        if not checkpoint:
            logger.error("Checkpoint file not found or empty: %s", args.resume)
            return
        checkpoint_path = args.resume
        staging_db_path = checkpoint.get('staging_db')

    # ── Run pipeline per asset type ────────────────────────────────────────

    from src.pipeline import run_pipeline

    logger.info("Downloading prices for %d tickers...", len(tickers_df))

    if 'AssetType' in tickers_df.columns:
        all_manifests = []
        groups = list(tickers_df.groupby('AssetType'))
        for idx, (fd_type, group) in enumerate(groups):
            db_type = ASSET_TYPE_MAP.get(fd_type, fd_type)
            ticker_list = group[args.ticker_column].tolist()
            names = None
            if 'Name' in group.columns:
                names = dict(zip(group[args.ticker_column], group['Name']))
            countries = None
            if 'Country' in group.columns:
                countries = dict(zip(group[args.ticker_column], group['Country']))
            sectors = None
            if 'sector' in group.columns:
                sectors = dict(zip(group[args.ticker_column], group['sector']))
                sectors = {k: v for k, v in sectors.items() if pd.notna(v)}
            industries_map = None
            if 'industry' in group.columns:
                industries_map = dict(zip(group[args.ticker_column], group['industry']))
                industries_map = {k: v for k, v in industries_map.items() if pd.notna(v)}
            cat_groups = None
            if 'category_group' in group.columns:
                cat_groups = dict(zip(group[args.ticker_column], group['category_group']))
                cat_groups = {k: v for k, v in cat_groups.items() if pd.notna(v)}
            cat_map = None
            if 'category' in group.columns:
                cat_map = dict(zip(group[args.ticker_column], group['category']))
                cat_map = {k: v for k, v in cat_map.items() if pd.notna(v)}
            logger.info("Running pipeline for %d %s tickers...",
                        len(ticker_list), db_type)
            manifest = run_pipeline(
                ticker_list, exchange=args.exchange, asset_type=db_type,
                start=start, end=args.end, null_threshold=args.null_threshold,
                names=names, countries=countries,
                sectors=sectors or None, industries=industries_map or None,
                category_groups=cat_groups or None, categories=cat_map or None,
                subset=args.subset,
                stage_only=args.stage_only,
                skip_validation=args.skip_validation,
                keep_staging=args.keep_staging,
                no_backup=args.no_backup,
                checkpoint_path=checkpoint_path,
                staging_db_path=staging_db_path,
                rate_limit_delay=args.rate_limit,
                n_workers=args.workers,
            )
            all_manifests.append((db_type, manifest))

            # Cooldown between asset type pipelines to avoid rate limiting
            if idx < len(groups) - 1:
                cooldown = config.PIPELINE_INTER_TYPE_COOLDOWN
                logger.info("Cooling down %.0fs before next asset type...",
                            cooldown)
                time.sleep(cooldown)

        for db_type, manifest in all_manifests:
            dl = manifest.get('download_result', {})
            logger.info("  %s: status=%s, saved=%s, failed_batches=%s",
                         db_type, manifest['status'],
                         dl.get('saved_tickers', 'N/A'),
                         len(dl.get('failed_batches', [])))
    else:
        ticker_list = tickers_df[args.ticker_column].tolist()
        names = None
        if 'Name' in tickers_df.columns:
            names = dict(zip(tickers_df[args.ticker_column], tickers_df['Name']))
        manifest = run_pipeline(
            ticker_list, exchange=args.exchange, asset_type=args.asset_type,
            start=start, end=args.end, null_threshold=args.null_threshold,
            names=names,
            subset=args.subset,
            stage_only=args.stage_only,
            skip_validation=args.skip_validation,
            keep_staging=args.keep_staging,
            no_backup=args.no_backup,
            checkpoint_path=checkpoint_path,
            staging_db_path=staging_db_path,
            rate_limit_delay=args.rate_limit,
            n_workers=args.workers,
        )
        dl = manifest.get('download_result', {})
        logger.info("Pipeline: status=%s, saved=%s, failed_batches=%s",
                     manifest['status'],
                     dl.get('saved_tickers', 'N/A'),
                     len(dl.get('failed_batches', [])))

    # ── Final summary ─────────────────────────────────────────────────────
    _log_final_summary(all_manifests if 'AssetType' in tickers_df.columns
                       else [('all', manifest)], log_path)


def _log_final_summary(manifests: list[tuple[str, dict[str, Any]]],
                       log_path: str) -> None:
    """Print a clear final summary with key stats and file locations."""
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)

    total_saved = 0
    total_failed_batches = 0
    total_failed_tickers = 0
    for db_type, m in manifests:
        dl = m.get('download_result', {})
        saved = dl.get('saved_tickers', 0)
        failed = dl.get('failed_batches', [])
        failed_ticker_count = sum(len(fb['tickers']) for fb in failed)
        duration = m.get('duration_seconds', 0)
        total_saved += saved
        total_failed_batches += len(failed)
        total_failed_tickers += failed_ticker_count

        minutes, secs = divmod(int(duration), 60)
        hours, minutes = divmod(minutes, 60)

        logger.info("  %s: status=%s, saved=%d, failed=%d tickers (%d batches), "
                     "duration=%dh%02dm%02ds",
                     db_type, m['status'], saved,
                     failed_ticker_count, len(failed),
                     hours, minutes, secs)

        # Log validation summary if present
        validation = m.get('validation')
        if validation:
            logger.info("    validation: %d active, %d excluded out of %d total",
                        validation.get('total_active', 0),
                        validation.get('total_excluded', 0),
                        validation.get('total_tickers', 0))

        # Log manifest path
        run_id = m.get('run_id', 'unknown')
        logger.info("    manifest: data/manifest_%s.json", run_id)

        # Log failed batch details
        if failed:
            logger.info("    failed batches:")
            for fb in failed:
                tickers_preview = ', '.join(fb['tickers'][:5])
                suffix = (f' ... +{len(fb["tickers"]) - 5} more'
                          if len(fb['tickers']) > 5 else '')
                logger.info("      batch %d (%d tickers): %s%s",
                            fb['batch_num'], len(fb['tickers']),
                            tickers_preview, suffix)

    logger.info("-" * 60)
    logger.info("Totals: %d saved, %d failed tickers", total_saved, total_failed_tickers)
    logger.info("Log file: %s", log_path)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Backward-compatible re-exports (moved to separate modules)
# ---------------------------------------------------------------------------
# Use __getattr__ for lazy imports to avoid circular import with download_workers.
_UNIVERSE_ATTRS = {
    'ASSET_TYPE_MAP', 'ALL_ASSET_TYPES',
    'get_equities', 'get_etfs', 'get_funds', 'get_cryptos', 'get_currencies',
    'build_security_universe', 'filter_unwanted_tickers', 'load_tickers',
}
_WORKER_ATTRS = {
    'concurrent_download_and_save',
    '_partition_tickers', '_reset_yf_singleton',
    '_subprocess_worker', '_worker_download',
    '_db_writer', '_concurrent_thread_download',
    '_concurrent_subprocess_download',
}
_SESSION_ATTRS = {
    '_proxy_url', '_tor_enabled', '_proxy_session_counter',
    '_proxy_session_counter_lock',
    'set_proxy_state', '_make_session', '_rotate_tor_circuit',
}


def __getattr__(name: str) -> Any:
    if name in _UNIVERSE_ATTRS:
        from src import universe
        return getattr(universe, name)
    if name in _WORKER_ATTRS:
        from src import download_workers
        return getattr(download_workers, name)
    if name in _SESSION_ATTRS:
        import src.download_session as _sess
        return getattr(_sess, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == '__main__':
    main()
