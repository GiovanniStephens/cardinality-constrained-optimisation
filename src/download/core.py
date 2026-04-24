"""
Downloads price data from Yahoo Finance.

Provides core download primitives, sequential download-and-save, and a CLI
entry point. Session management and proxy/Tor setup live in
src.download.session; ticker validation in src.download.validate; universe
building in src.universe; concurrent worker infrastructure in
src.download.workers.
"""

from __future__ import annotations

import logging
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
from .validate import validate_tickers, _retry_with_splitting
from src.exceptions import DownloadError, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _deduplicate_tickers(tickers: list[str]) -> list[str]:
    """Return a deduplicated ticker list preserving original order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    if len(deduped) < len(tickers):
        logger.info("Removed %d duplicate tickers", len(tickers) - len(deduped))
    return deduped


def _build_batch_metadata(
    columns,
    *, names=None, countries=None, sectors=None,
    industries=None, category_groups=None, categories=None,
) -> dict:
    """Filter metadata dicts to only include tickers present in *columns*."""
    metadata: dict = {}
    for key, source in [
        ('names', names), ('countries', countries),
        ('sectors', sectors), ('industries', industries),
        ('category_groups', category_groups), ('categories', categories),
    ]:
        if source:
            metadata[key] = {t: source[t] for t in columns if t in source}
    return metadata


# ---------------------------------------------------------------------------
# Price downloading
# ---------------------------------------------------------------------------

_RATE_LIMIT_SIGNALS = ('too many requests', 'rate limit', 'yfratelimit', '429')


def _classify_empty_batch() -> str:
    """Inspect yfinance._ERRORS to distinguish rate-limits from genuine no-data.

    Returns 'rate_limit' if any per-ticker error contains a throttling signal,
    'no_data' otherwise (all errors are data-level, or no per-ticker info at
    all — empty _ERRORS means yfinance didn't log a rate-limit, which for a
    fresh-IP request is a reliable signal the data genuinely does not exist).
    """
    try:
        errors = dict(getattr(yf.shared, '_ERRORS', {}))
    except Exception:
        return 'no_data'
    for err in errors.values():
        low = (err or '').lower()
        if any(sig in low for sig in _RATE_LIMIT_SIGNALS):
            return 'rate_limit'
    return 'no_data'


def _download_batch(tickers: list[str], start: str, end: str,
                    session: Any = None) -> Optional[pd.DataFrame]:
    """Download a single batch from yfinance. Returns wide DataFrame or None."""
    from . import session as _sess
    tickers_str = " ".join(tickers)
    if session is None:
        session = _sess._make_session()
    # Disable yfinance internal threading when using a proxy — curl_cffi
    # sessions are not safe for concurrent use across yfinance's threads.
    # Our external workers already provide download parallelism.
    yf_threads = False if _sess._get_state('_proxy_url') else DOWNLOAD_THREADS
    # Clear yfinance's per-ticker error dict so _classify_empty_batch()
    # only sees errors from this call.
    try:
        yf.shared._ERRORS.clear()
    except Exception:
        pass
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
            # With group_by="ticker", yfinance returns a MultiIndex DataFrame
            # with the ticker as the top-level column — regardless of whether
            # the batch has one ticker or many. Access via prices[ticker]["Close"]
            # in both cases.
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
    from . import session as _sess

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

    tickers = _deduplicate_tickers(tickers)

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
        data_level_empty = False  # classifier said 'no_data' (don't feed circuit breaker)
        for attempt in range(1, max_retries + 1):
            try:
                batch_df = _download_batch_with_timeout(
                    batch, start, end, batch_timeout)
                if batch_df is not None and not batch_df.empty:
                    break
                # Empty result: rate-limited OR legitimately no data.
                classification = _classify_empty_batch()
                if classification == 'no_data':
                    # All tickers errored data-level (delisted, no timezone).
                    # No point retrying; a new IP wouldn't help.
                    data_level_empty = True
                    logger.info("Batch %d: no data (data-level, not rate-limit); "
                                "skipping without backoff.", batch_num)
                    break
                # Rate-limited (or indeterminate — conservative backoff).
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
                is_rate_limit = _sess.is_rate_limit_error(e)
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
                # Data-level empties (delisted tickers) are clean forward
                # progress, not failures. Reset the counter so 10 rate-limits
                # scattered across 500 no_data skips don't trip what's meant
                # to catch a rate-limit cascade.
                if data_level_empty:
                    consecutive_failures = 0
                else:
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

        metadata = _build_batch_metadata(
            batch_df.columns, names=names, countries=countries,
            sectors=sectors, industries=industries,
            category_groups=category_groups, categories=categories)

        db.save_prices(conn, batch_df, exchange=exchange, asset_type=asset_type,
                       **metadata)
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
