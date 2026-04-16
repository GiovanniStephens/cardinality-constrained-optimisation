"""
Builds a security universe from FinanceDatabase and downloads price data
from Yahoo Finance.

Supports equities, ETFs, and funds. Can also load tickers from a local CSV
for working with previously scraped lists.
"""

import argparse
import logging
import os
import queue
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

from curl_cffi.requests import Session as CffiSession
import financedatabase as fd
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.config import (DOWNLOAD_THREADS, DOWNLOAD_TIMEOUT,
                        TICKER_EXCLUDE_SUFFIXES, TICKER_EXCLUDE_NAME_PATTERNS)

logger = logging.getLogger(__name__)

# Realistic browser User-Agent strings for rotation.  Yahoo fingerprints
# by UA, so cycling across these makes batches look like distinct clients.
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
]


_proxy_url = None   # Set from CLI: --proxy or --use-tor
_tor_enabled = False  # Tor-specific: enables NEWNYM circuit rotation
_proxy_session_counter = int(time.time()) % 100_000  # Timestamp-seeded to avoid burned ranges
_proxy_session_counter_lock = threading.Lock()


def _make_session():
    """Create a curl_cffi Session with a random User-Agent header.

    For rotating residential proxies: if the proxy URL contains a username
    ending in digits (e.g. ``mdgihswf-11``), the trailing number is replaced
    with a per-session counter so each batch gets a distinct exit IP.
    """
    global _proxy_session_counter
    session = CffiSession()
    session.headers['User-Agent'] = random.choice(_USER_AGENTS)
    if _proxy_url:
        url = _proxy_url
        # Rotate proxy username suffix for residential proxies
        # e.g. http://user-11:pass@host → http://user-42:pass@host
        if re.match(r'https?://[^:]*-\d+:', url):
            with _proxy_session_counter_lock:
                _proxy_session_counter += 1
                counter = _proxy_session_counter
            url = re.sub(r'(-)\d+:', rf'\g<1>{counter}:', url, count=1)
        session.proxies = {'http': url, 'https': url}
    return session


def _rotate_tor_circuit():
    """Request a new Tor circuit (new exit IP) via the ControlPort."""
    try:
        from stem import Signal
        from stem.control import Controller
        from src.config import TOR_CONTROL_PORT, TOR_CONTROL_PASSWORD
        with Controller.from_port(port=TOR_CONTROL_PORT) as controller:
            if TOR_CONTROL_PASSWORD:
                controller.authenticate(password=TOR_CONTROL_PASSWORD)
            else:
                controller.authenticate()
            controller.signal(Signal.NEWNYM)
    except Exception as e:
        logger.warning("Tor circuit rotation failed: %s", e)

# Maps FinanceDatabase asset type labels to DB-canonical names.
# DB migration uses 'stock' for equities (see db._migrate_ticker_list).
ASSET_TYPE_MAP = {
    'equity': 'stock', 'etf': 'etf', 'fund': 'fund',
    'crypto': 'crypto', 'currency': 'currency',
}


# ---------------------------------------------------------------------------
# Universe building (from FinanceDatabase)
# ---------------------------------------------------------------------------

def _filter_by_exchange(df, exchanges):
    """Post-filter a FinanceDatabase result by exchange column.

    The library's select() does not accept 'exchange' as a keyword, so we
    filter the returned DataFrame manually.
    """
    if exchanges and 'exchange' in df.columns:
        if isinstance(exchanges, str):
            exchanges = [exchanges]
        df = df[df['exchange'].isin(exchanges)]
    return df


def get_equities(countries=None, sectors=None, industries=None,
                 exchanges=None, market_caps=None) -> pd.DataFrame:
    """
    Retrieves equity tickers from FinanceDatabase.

    :param countries: country or list of countries to filter by.
    :param sectors: sector or list of sectors to filter by.
    :param industries: industry or list of industries to filter by.
    :param exchanges: exchange or list of exchanges to filter by (post-filter).
    :param market_caps: market cap category or list to filter by (post-filter).
    :returns: DataFrame with ticker symbols and metadata.
    """
    equities = fd.Equities()
    kwargs = {}
    if countries:
        kwargs['country'] = countries
    if sectors:
        kwargs['sector'] = sectors
    if industries:
        kwargs['industry'] = industries
    df = _filter_by_exchange(equities.select(**kwargs), exchanges)
    if market_caps and 'market_cap' in df.columns:
        if isinstance(market_caps, str):
            market_caps = [market_caps]
        df = df[df['market_cap'].isin(market_caps)]
    return df


def get_etfs(category_groups=None, categories=None, families=None,
             exchanges=None) -> pd.DataFrame:
    """
    Retrieves ETF tickers from FinanceDatabase.

    :param category_groups: category group or list to filter by.
    :param categories: category or list to filter by.
    :param families: ETF family/provider or list to filter by.
    :param exchanges: exchange or list of exchanges to filter by.
    :returns: DataFrame with ticker symbols and metadata.
    """
    etfs = fd.ETFs()
    kwargs = {}
    if category_groups:
        kwargs['category_group'] = category_groups
    if categories:
        kwargs['category'] = categories
    if families:
        kwargs['family'] = families
    return _filter_by_exchange(etfs.select(**kwargs), exchanges)


def get_funds(category_groups=None, categories=None, families=None,
              exchanges=None) -> pd.DataFrame:
    """
    Retrieves fund tickers from FinanceDatabase.

    :param category_groups: category group or list to filter by.
    :param categories: category or list to filter by.
    :param families: fund family/provider or list to filter by.
    :param exchanges: exchange or list of exchanges to filter by.
    :returns: DataFrame with ticker symbols and metadata.
    """
    funds = fd.Funds()
    kwargs = {}
    if category_groups:
        kwargs['category_group'] = category_groups
    if categories:
        kwargs['category'] = categories
    if families:
        kwargs['family'] = families
    return _filter_by_exchange(funds.select(**kwargs), exchanges)


def get_cryptos() -> pd.DataFrame:
    """Retrieves cryptocurrency tickers from FinanceDatabase."""
    return fd.Cryptos().select()


def get_currencies() -> pd.DataFrame:
    """Retrieves currency pair tickers from FinanceDatabase."""
    return fd.Currencies().select()


ALL_ASSET_TYPES = ['equities', 'etfs', 'funds', 'cryptos', 'currencies']


def build_security_universe(asset_types=None, countries=None, sectors=None,
                            industries=None, exchanges=None,
                            market_caps=None, etf_categories=None,
                            etf_category_groups=None) -> pd.DataFrame:
    """
    Builds a combined universe of securities from multiple asset types.

    :param asset_types: list of asset types to include. Defaults to ALL_ASSET_TYPES.
    :param countries: country filter (applies to equities).
    :param sectors: sector filter (applies to equities).
    :param industries: industry filter (applies to equities).
    :param exchanges: exchange filter (applies to equities only).
    :param market_caps: market cap filter (applies to equities).
    :param etf_categories: category filter (applies to ETFs).
    :param etf_category_groups: category group filter (applies to ETFs).
    :returns: DataFrame with columns ['Tickers', 'Name', 'Country', 'AssetType'].
    """
    if asset_types is None:
        asset_types = ALL_ASSET_TYPES

    all_securities = []

    def _append(df, asset_type):
        if df.empty:
            return
        row = {
            'Tickers': df.index,
            'Name': df['name'] if 'name' in df.columns else '',
            'Country': df['country'] if 'country' in df.columns else '',
            'AssetType': asset_type,
        }
        # Carry through sector/industry (equities) and category_group/category (ETFs)
        for col in ('sector', 'industry', 'category_group', 'category'):
            if col in df.columns:
                row[col] = df[col]
        all_securities.append(pd.DataFrame(row))

    if 'equities' in asset_types:
        _append(get_equities(countries=countries, sectors=sectors,
                             industries=industries, exchanges=exchanges,
                             market_caps=market_caps), 'equity')

    if 'etfs' in asset_types:
        _append(get_etfs(category_groups=etf_category_groups,
                         categories=etf_categories), 'etf')

    if 'funds' in asset_types:
        _append(get_funds(), 'fund')

    if 'cryptos' in asset_types:
        _append(get_cryptos(), 'crypto')

    if 'currencies' in asset_types:
        _append(get_currencies(), 'currency')

    if not all_securities:
        return pd.DataFrame(columns=['Tickers', 'Name', 'Country', 'AssetType'])

    combined = pd.concat(all_securities, ignore_index=True)
    combined = combined.drop_duplicates(subset='Tickers')
    return combined


def filter_unwanted_tickers(df, ticker_column='Tickers', name_column='Name',
                            skip_suffix_filter=False, skip_name_filter=False):
    """Remove warrants, units, preferred shares, rights, and SPACs.

    Applies two tiers of regex filtering (zero API calls):
      1a. Ticker suffix patterns (e.g. -WT, -UN, -PA, -RT)
      1b. Name patterns (e.g. "Acquisition Corp", "Blank Check")

    :param df: DataFrame with at least a ticker column.
    :param ticker_column: column containing ticker symbols.
    :param name_column: column containing security names.
    :param skip_suffix_filter: if True, skip suffix regex filtering.
    :param skip_name_filter: if True, skip name pattern filtering.
    :returns: (filtered_df, removed_df) — both DataFrames.
    """
    if df.empty:
        return df.copy(), df.iloc[:0].copy()

    remove_mask = pd.Series(False, index=df.index)

    # Tier 1a: suffix regex on ticker symbols
    if not skip_suffix_filter and ticker_column in df.columns:
        suffix_re = re.compile(TICKER_EXCLUDE_SUFFIXES)
        remove_mask |= df[ticker_column].str.contains(suffix_re, na=False)

    # Tier 1b: name patterns for SPACs/shell companies
    if not skip_name_filter and name_column in df.columns:
        combined_pattern = '|'.join(TICKER_EXCLUDE_NAME_PATTERNS)
        name_re = re.compile(combined_pattern, re.IGNORECASE)
        remove_mask |= df[name_column].str.contains(name_re, na=False)

    removed = df[remove_mask].copy()
    filtered = df[~remove_mask].copy()
    return filtered, removed


# ---------------------------------------------------------------------------
# Ticker loading from CSV (for previously scraped lists)
# ---------------------------------------------------------------------------

def load_tickers(filename: str, ticker_column: str = 'Tickers') -> pd.DataFrame:
    """
    Loads the list of tickers from a local CSV file.

    :param filename: path to CSV file containing tickers.
    :param ticker_column: name of the column containing ticker symbols.
    :returns: DataFrame with at least the ticker column.
    """
    tickers = pd.read_csv(filename)
    if tickers.empty:
        raise ValueError(f"Ticker list file '{filename}' is empty.")
    if ticker_column not in tickers.columns:
        raise ValueError(
            f"Column '{ticker_column}' not found in {filename}. "
            f"Available columns: {list(tickers.columns)}"
        )
    return tickers


# ---------------------------------------------------------------------------
# Price downloading
# ---------------------------------------------------------------------------

def _download_batch(tickers, start, end, session=None):
    """Download a single batch from yfinance. Returns wide DataFrame or None."""
    tickers_str = " ".join(tickers)
    if session is None:
        session = _make_session()
    # Disable yfinance internal threading when using a proxy — curl_cffi
    # sessions are not safe for concurrent use across yfinance's threads.
    # Our external workers already provide download parallelism.
    yf_threads = False if _proxy_url else DOWNLOAD_THREADS
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
        except Exception:
            logger.warning("Batch returned non-date index type: %s", type(prices.index))
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


def _download_batch_with_timeout(tickers, start, end, timeout_seconds):
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



def validate_tickers(tickers, batch_size=None, delay=None, max_retries=None,
                     timeout=None, validation_windows=None,
                     cache_dir=None, max_cache_hours=None):
    """Quick validation pass: download 1 week of data per ticker to identify
    which tickers yfinance can actually serve.

    Uses two date windows (mid-2019, mid-2024) to catch both older and
    newer listings. A ticker with data in EITHER window is valid.

    Failed batches (possible rate-limit) are treated conservatively:
    all their tickers remain as 'unvalidated' and proceed to full download.

    :param tickers: list of ticker symbols to validate.
    :param batch_size: tickers per validation batch (default: config).
    :param delay: seconds between batches (default: config).
    :param max_retries: retries for failed batches (default: config).
    :param timeout: seconds per batch download (default: config).
    :param validation_windows: list of (start, end) date pairs (default: config).
    :param cache_dir: directory for cache file (default: config.DATA_DIR).
    :param max_cache_hours: use cached results if fresher than this; None
        disables caching (default: config).
    :returns: (valid_set, invalid_set, unvalidated_set)
        valid: confirmed to have data on yfinance
        invalid: in a successful batch but returned no data
        unvalidated: in a failed batch (kept for full download)
    """
    import json
    from datetime import datetime as dt_cls
    from src import config as cfg

    batch_size = batch_size or cfg.VALIDATION_BATCH_SIZE
    delay = delay if delay is not None else cfg.VALIDATION_DELAY
    max_retries = max_retries if max_retries is not None else cfg.VALIDATION_MAX_RETRIES
    timeout = timeout or cfg.VALIDATION_TIMEOUT
    validation_windows = validation_windows or cfg.VALIDATION_WINDOWS
    cache_dir = cache_dir or cfg.DATA_DIR
    if max_cache_hours is None:
        max_cache_hours = cfg.VALIDATION_CACHE_HOURS

    cache_path = os.path.join(cache_dir, cfg.VALIDATION_CACHE_FILE)
    tickers_set = set(tickers)

    # ── Helper: save cache incrementally ────────────────────────────────
    save_interval = 10  # save every N batches
    batches_since_save = 0

    def _save_cache(v, inv, unv, force=False):
        nonlocal batches_since_save
        batches_since_save += 1
        if not force and batches_since_save < save_interval:
            return
        batches_since_save = 0
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_data = {
                'timestamp': dt_cls.now().isoformat(),
                'valid': sorted(v),
                'invalid': sorted(inv - v),
                'unvalidated': sorted(unv - v),
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except OSError:
            pass

    # ── Load progress from cache (resume or fresh hit) ───────────────────
    valid = set()
    invalid = set()
    unvalidated = set()

    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            cache_ts = dt_cls.fromisoformat(cached['timestamp'])
            age_hours = (dt_cls.now() - cache_ts).total_seconds() / 3600

            cached_valid = set(cached.get('valid', []))
            cached_invalid = set(cached.get('invalid', []))
            cached_unvalidated = set(cached.get('unvalidated', []))
            categorised = cached_valid | cached_invalid | cached_unvalidated
            unchecked = tickers_set - categorised

            # Fresh + complete cache → return immediately
            if max_cache_hours and age_hours < max_cache_hours and not unchecked:
                valid = cached_valid & tickers_set
                invalid = cached_invalid & tickers_set
                unvalidated = tickers_set - valid - invalid
                logger.info("Validation cache hit (%.1fh old): %d valid, "
                            "%d invalid, %d unvalidated",
                            age_hours, len(valid), len(invalid), len(unvalidated))
                return valid, invalid, unvalidated

            # Stale complete cache → re-validate from scratch
            if not unchecked and (not max_cache_hours or age_hours >= max_cache_hours):
                logger.info("Validation cache stale (%.1fh old), re-validating",
                            age_hours)
            else:
                # Partial cache → resume from where we left off
                valid = cached_valid
                invalid = cached_invalid
                unvalidated = cached_unvalidated
                logger.info("Resuming validation: %d valid, %d invalid, "
                            "%d unvalidated from cache, %d unchecked remain",
                            len(valid), len(invalid), len(unvalidated),
                            len(unchecked))
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Validation cache corrupt, starting fresh")

    # ── Validate across windows ──────────────────────────────────────────

    for window_idx, (win_start, win_end) in enumerate(validation_windows, 1):
        # Only skip tickers confirmed valid or unvalidated (failed batch);
        # re-check invalid tickers — they may exist in a different window
        skip = valid | unvalidated
        remaining = [t for t in tickers if t not in skip]
        if not remaining:
            break

        logger.info("Validation window %d/%d (%s to %s): %d tickers to check",
                     window_idx, len(validation_windows), win_start, win_end,
                     len(remaining))

        batches = [remaining[i:i + batch_size]
                   for i in range(0, len(remaining), batch_size)]

        for batch_idx, batch in enumerate(tqdm(batches, desc=f"Validating (window {window_idx})",
                                                unit="batch")):
            result = _download_batch_with_timeout(batch, win_start, win_end, timeout)

            if result is not None and not result.empty:
                # Successful batch — categorise tickers
                returned_tickers = set(result.columns)
                valid |= returned_tickers
                # Tickers in this batch with no data → invalid (for this window)
                batch_invalid = set(batch) - returned_tickers
                invalid |= batch_invalid
                # Remove from unvalidated if previously there
                unvalidated -= returned_tickers
            elif result is not None and result.empty:
                # Batch succeeded but empty — all tickers invalid
                invalid |= set(batch)
            else:
                # Batch failed (None) — retry once
                retried = False
                for retry in range(max_retries):
                    if _tor_enabled:
                        _rotate_tor_circuit()
                    time.sleep(delay)
                    result = _download_batch_with_timeout(batch, win_start, win_end, timeout)
                    if result is not None and not result.empty:
                        returned_tickers = set(result.columns)
                        valid |= returned_tickers
                        invalid |= set(batch) - returned_tickers
                        unvalidated -= returned_tickers
                        retried = True
                        break
                    elif result is not None and result.empty:
                        invalid |= set(batch)
                        retried = True
                        break
                if not retried:
                    # All retries failed — conservatively keep as unvalidated
                    unvalidated |= set(batch)

            _save_cache(valid, invalid, unvalidated)

            if _tor_enabled:
                from src.config import TOR_ROTATE_EVERY_N_BATCHES
                if (batch_idx + 1) % TOR_ROTATE_EVERY_N_BATCHES == 0:
                    _rotate_tor_circuit()

            if batch_idx < len(batches) - 1:
                time.sleep(delay)

    # Tickers valid in any window should not be in invalid
    invalid -= valid
    unvalidated -= valid

    # ── Save final cache ─────────────────────────────────────────────────
    _save_cache(valid, invalid, unvalidated, force=True)
    logger.info("Validation cache saved to %s", cache_path)

    logger.info("Validation complete: %d valid, %d invalid, %d unvalidated",
                 len(valid), len(invalid), len(unvalidated))
    return valid, invalid, unvalidated


def _retry_with_splitting(tickers, start, end, timeout_seconds,
                          min_batch_size, delay_state):
    """Split a failed batch into halves and retry. Returns (df_or_None, failed_tickers).

    delay_state is a mutable list [current_delay, max_delay, base_delay]
    so adaptive backoff propagates across recursive calls.
    """
    current_delay, max_delay, base_delay = delay_state

    # Try the full list with 2 quick retries
    for attempt in range(1, 3):
        df = _download_batch_with_timeout(tickers, start, end, timeout_seconds)
        if df is not None and not df.empty:
            # Success — halve delay
            delay_state[0] = max(base_delay, current_delay * 0.5)
            return df, []
        if attempt < 2:
            # Failed — escalate delay
            current_delay = min(current_delay * 2, max_delay)
            delay_state[0] = current_delay
            jittered = current_delay * random.uniform(0.8, 1.2)
            time.sleep(jittered)

    # Can't split further — return as failed
    if len(tickers) <= min_batch_size:
        return None, list(tickers)

    # Split in half and recurse
    mid = len(tickers) // 2
    left, right = tickers[:mid], tickers[mid:]

    jittered = delay_state[0] * random.uniform(0.8, 1.2)
    time.sleep(jittered)
    df_left, failed_left = _retry_with_splitting(
        left, start, end, timeout_seconds, min_batch_size, delay_state)

    jittered = delay_state[0] * random.uniform(0.8, 1.2)
    time.sleep(jittered)
    df_right, failed_right = _retry_with_splitting(
        right, start, end, timeout_seconds, min_batch_size, delay_state)

    # Combine successful results
    parts = [d for d in (df_left, df_right) if d is not None and not d.empty]
    combined = pd.concat(parts, axis=1) if parts else None
    return combined, failed_left + failed_right


def download_data(
    tickers_df: pd.DataFrame,
    ticker_column: str = "Tickers",
    start: str = "2014-04-30",
    end: str = "2025-04-30",
    batch_size: int = 500,
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
        for attempt in range(1, 4):
            try:
                batch_df = _download_batch(batch, start, end)
                break
            except Exception as e:
                if attempt == 3:
                    raise ConnectionError(
                        f"Failed after 3 retries: {e}") from e
                logger.warning("Batch %d attempt %d failed: %s",
                               batch_num, attempt, e)
                time.sleep(2 ** attempt)
        if batch_df is not None:
            for col in batch_df.columns:
                all_prices[col] = batch_df[col].tolist()

    if not all_prices:
        raise ValueError("No valid price data could be extracted for any ticker.")
    return pd.DataFrame(all_prices)


def download_and_save(
    tickers, conn, exchange, asset_type='etf',
    start='2014-04-30', end='2025-04-30',
    batch_size=500, null_threshold=0.9,
    names=None, countries=None, max_retries=3,
    on_batch_complete=None, on_batch_failed=None,
    rate_limit_delay=None, batch_timeout=None,
    circuit_breaker_threshold=None, max_rate_limit_delay=None,
    circuit_breaker_max_trips=None, circuit_breaker_cooldown=None,
    sectors=None, industries=None, category_groups=None, categories=None,
):
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
                    if _tor_enabled:
                        _rotate_tor_circuit()
                    current_delay = min(current_delay * 2, max_rate_limit_delay)
                    jittered = current_delay * random.uniform(0.8, 1.2)
                    logger.warning("Batch %d attempt %d/%d: no data returned, "
                                   "likely rate-limited. Backing off %.0fs "
                                   "(adaptive delay now %.0fs)",
                                   batch_num, attempt, max_retries,
                                   jittered, current_delay)
                    time.sleep(jittered)
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = ('429' in error_msg or 'too many' in error_msg
                                 or 'rate' in error_msg)
                if is_rate_limit:
                    hit_rate_limit = True
                    if _tor_enabled:
                        _rotate_tor_circuit()
                    current_delay = min(current_delay * 2, max_rate_limit_delay)
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

        if _tor_enabled:
            from src.config import TOR_ROTATE_EVERY_N_BATCHES
            if batch_num % TOR_ROTATE_EVERY_N_BATCHES == 0:
                _rotate_tor_circuit()

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
# Concurrent download (multi-worker)
# ---------------------------------------------------------------------------

def _partition_tickers(tickers, n_workers):
    """Split tickers into N contiguous sublists.

    Contiguous (not round-robin) so each worker's retries are isolated.
    If there are fewer tickers than workers, returns one partition per ticker.
    """
    n = min(n_workers, len(tickers))
    if n <= 0:
        return []
    base, extra = divmod(len(tickers), n)
    partitions = []
    start = 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        partitions.append(tickers[start:start + size])
        start += size
    return partitions


def _worker_download(worker_id, tickers, start, end, batch_size,
                     null_threshold, result_queue, names, countries,
                     sectors, industries, category_groups, categories,
                     max_retries, rate_limit_delay, batch_timeout,
                     circuit_breaker_threshold, max_rate_limit_delay,
                     circuit_breaker_max_trips, circuit_breaker_cooldown):
    """Per-worker download loop. Puts results onto result_queue.

    Always sends a sentinel (None) when done, even on exception, so the
    DB writer thread never hangs waiting for a worker that crashed.
    """
    from src import config

    try:
        batches = [
            tickers[i:i + batch_size]
            for i in range(0, len(tickers), batch_size)
        ]

        consecutive_failures = 0
        current_delay = rate_limit_delay
        circuit_breaker_trip_count = 0

        for batch_num, batch in enumerate(batches, 1):
            batch_df = None
            hit_rate_limit = False

            for attempt in range(1, max_retries + 1):
                try:
                    batch_df = _download_batch_with_timeout(
                        batch, start, end, batch_timeout)
                    if batch_df is not None and not batch_df.empty:
                        break
                    if attempt < max_retries:
                        hit_rate_limit = True
                        if _tor_enabled:
                            _rotate_tor_circuit()
                        current_delay = min(current_delay * 2, max_rate_limit_delay)
                        jittered = current_delay * random.uniform(0.8, 1.2)
                        logger.warning("Worker %d batch %d attempt %d/%d: no data, "
                                       "backing off %.0fs",
                                       worker_id, batch_num, attempt, max_retries,
                                       jittered)
                        time.sleep(jittered)
                except Exception as e:
                    error_msg = str(e).lower()
                    is_rate_limit = ('429' in error_msg or 'too many' in error_msg
                                     or 'rate' in error_msg)
                    if is_rate_limit:
                        hit_rate_limit = True
                        if _tor_enabled:
                            _rotate_tor_circuit()
                        current_delay = min(current_delay * 2, max_rate_limit_delay)
                        jittered = current_delay * random.uniform(0.8, 1.2)
                        logger.warning("Worker %d rate limited batch %d "
                                       "(attempt %d/%d), backing off %.0fs",
                                       worker_id, batch_num, attempt,
                                       max_retries, jittered)
                        time.sleep(jittered)
                    else:
                        logger.warning("Worker %d batch %d attempt %d/%d failed: %s",
                                       worker_id, batch_num, attempt, max_retries, e)
                        if attempt < max_retries:
                            jittered = current_delay * random.uniform(0.8, 1.2)
                            time.sleep(jittered)

            if batch_df is None or batch_df.empty:
                # Try sub-batch splitting before giving up
                min_sub = config.PIPELINE_MIN_SUB_BATCH_SIZE
                if len(batch) > min_sub:
                    delay_state = [current_delay, max_rate_limit_delay,
                                   rate_limit_delay]
                    split_df, still_failed = _retry_with_splitting(
                        list(batch), start, end, batch_timeout,
                        min_sub, delay_state)
                    current_delay = delay_state[0]
                    if split_df is not None and not split_df.empty:
                        batch_df = split_df
                        if still_failed:
                            result_queue.put(('failed', still_failed, batch_num))
                    else:
                        batch_df = None

                if batch_df is None or batch_df.empty:
                    result_queue.put(('failed', list(batch), batch_num))
                    consecutive_failures += 1

                    if consecutive_failures >= circuit_breaker_threshold:
                        circuit_breaker_trip_count += 1
                        if circuit_breaker_trip_count >= circuit_breaker_max_trips:
                            logger.error("Worker %d: circuit breaker tripped %d "
                                         "times, aborting.",
                                         worker_id, circuit_breaker_trip_count)
                            break
                        else:
                            logger.warning("Worker %d: circuit breaker trip %d/%d, "
                                           "cooling down %.0fs",
                                           worker_id, circuit_breaker_trip_count,
                                           circuit_breaker_max_trips,
                                           circuit_breaker_cooldown)
                            time.sleep(circuit_breaker_cooldown)
                            consecutive_failures = 0
                            current_delay = max_rate_limit_delay
                    continue

            # Reset on success
            consecutive_failures = 0
            if not hit_rate_limit:
                current_delay = max(rate_limit_delay, current_delay * 0.5)

            # Filter out tickers with too many nulls
            threshold = int(len(batch_df) * null_threshold)
            batch_df = batch_df.dropna(axis=1, thresh=threshold)
            if batch_df.empty:
                continue

            # Build per-batch metadata
            metadata = {}
            if names:
                metadata['names'] = {t: names[t] for t in batch_df.columns
                                     if t in names}
            if countries:
                metadata['countries'] = {t: countries[t] for t in batch_df.columns
                                         if t in countries}
            if sectors:
                metadata['sectors'] = {t: sectors[t] for t in batch_df.columns
                                       if t in sectors}
            if industries:
                metadata['industries'] = {t: industries[t]
                                          for t in batch_df.columns
                                          if t in industries}
            if category_groups:
                metadata['category_groups'] = {t: category_groups[t]
                                               for t in batch_df.columns
                                               if t in category_groups}
            if categories:
                metadata['categories'] = {t: categories[t]
                                          for t in batch_df.columns
                                          if t in categories}

            saved_list = list(batch_df.columns)
            result_queue.put(('data', batch_df, metadata, saved_list, batch_num))

            if _tor_enabled:
                from src.config import TOR_ROTATE_EVERY_N_BATCHES
                if batch_num % TOR_ROTATE_EVERY_N_BATCHES == 0:
                    _rotate_tor_circuit()

            if current_delay > 0 and batch_num < len(batches):
                jittered = current_delay * random.uniform(0.8, 1.2)
                time.sleep(jittered)
    finally:
        # Always send sentinel so the DB writer thread never hangs
        result_queue.put(None)


def _db_writer(result_queue, conn, exchange, asset_type,
               on_batch_complete, on_batch_failed, pbar, n_workers):
    """Single-threaded consumer that writes results to the database.

    Returns (total_saved, failed_batches_list).
    """
    from src import db

    total_saved = 0
    failed_batches = []
    sentinels_received = 0
    t_start = time.time()

    while sentinels_received < n_workers:
        item = result_queue.get()
        if item is None:
            sentinels_received += 1
            continue

        msg_type = item[0]
        if msg_type == 'data':
            _, batch_df, metadata, saved_list, batch_num = item
            db.save_prices(
                conn, batch_df, exchange=exchange, asset_type=asset_type,
                names=metadata.get('names'),
                countries=metadata.get('countries'),
                sectors=metadata.get('sectors'),
                industries=metadata.get('industries'),
                category_groups=metadata.get('category_groups'),
                categories=metadata.get('categories'),
            )
            total_saved += len(saved_list)
            if on_batch_complete:
                on_batch_complete(saved_list, batch_num)
            if pbar:
                pbar.update(1)
                elapsed = time.time() - t_start
                pbar.set_postfix(
                    saved=total_saved,
                    failed=len(failed_batches),
                    rate=f"{total_saved / elapsed:.0f}/s" if elapsed > 0
                         else "N/A",
                )

        elif msg_type == 'failed':
            _, failed_tickers, batch_num = item
            failed_batches.append({
                'batch_num': batch_num,
                'tickers': failed_tickers,
            })
            if on_batch_failed:
                on_batch_failed(failed_tickers, batch_num)
            if pbar:
                pbar.update(1)

    return total_saved, failed_batches


def concurrent_download_and_save(
    tickers, conn, exchange, n_workers, asset_type='etf',
    start='2014-04-30', end='2025-04-30',
    batch_size=500, null_threshold=0.9,
    names=None, countries=None, max_retries=3,
    on_batch_complete=None, on_batch_failed=None,
    rate_limit_delay=None, batch_timeout=None,
    circuit_breaker_threshold=None, max_rate_limit_delay=None,
    circuit_breaker_max_trips=None, circuit_breaker_cooldown=None,
    sectors=None, industries=None, category_groups=None, categories=None,
):
    """Download prices concurrently with N workers and a single DB writer thread.

    Same interface as download_and_save() plus n_workers parameter.
    Workers put results on a bounded queue; a single DB writer thread
    consumes them to avoid SQLite WAL contention.

    The DB writer opens its own connection to the same database (extracted
    from the caller's connection) because SQLite connections cannot be shared
    across threads.
    """
    from src import db, config

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

    # Deduplicate
    seen = set()
    deduped = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    if len(deduped) < len(tickers):
        logger.info("Removed %d duplicate tickers", len(tickers) - len(deduped))
    tickers = deduped

    partitions = _partition_tickers(tickers, n_workers)
    actual_workers = len(partitions)

    # Count total batches across all workers for progress bar
    total_batches = sum(
        (len(p) + batch_size - 1) // batch_size for p in partitions
    )

    result_q = queue.Queue(maxsize=actual_workers * 2)

    use_tqdm = sys.stderr.isatty()
    pbar = None
    if use_tqdm:
        pbar = tqdm(total=total_batches, desc="Downloading", unit="batch",
                    file=sys.stderr)

    # Extract DB path from caller's connection — the writer thread needs
    # its own connection because SQLite objects are thread-bound.
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]

    # Start DB writer thread
    writer_result = [None]  # mutable container for thread return value

    def _writer_thread():
        writer_conn = db.get_connection(db_path)
        try:
            writer_result[0] = _db_writer(
                result_q, writer_conn, exchange, asset_type,
                on_batch_complete, on_batch_failed, pbar, actual_workers)
        except Exception:
            logger.error("DB writer thread crashed", exc_info=True)
        finally:
            writer_conn.close()

    writer = threading.Thread(target=_writer_thread, daemon=True)
    writer.start()

    # Launch workers
    common_kwargs = dict(
        start=start, end=end, batch_size=batch_size,
        null_threshold=null_threshold, result_queue=result_q,
        names=names, countries=countries,
        sectors=sectors, industries=industries,
        category_groups=category_groups, categories=categories,
        max_retries=max_retries, rate_limit_delay=rate_limit_delay,
        batch_timeout=batch_timeout,
        circuit_breaker_threshold=circuit_breaker_threshold,
        max_rate_limit_delay=max_rate_limit_delay,
        circuit_breaker_max_trips=circuit_breaker_max_trips,
        circuit_breaker_cooldown=circuit_breaker_cooldown,
    )

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = []
        for wid, partition in enumerate(partitions):
            f = pool.submit(_worker_download, wid, partition, **common_kwargs)
            futures.append(f)

        # Wait for all workers, propagate exceptions as warnings
        for f in futures:
            try:
                f.result()
            except Exception as e:
                logger.error("Worker raised exception: %s", e)

    writer.join()

    if pbar:
        pbar.close()

    if writer_result[0] is None:
        logger.error("DB writer thread failed — no results recorded")
        return {
            'total_tickers': len(tickers),
            'saved_tickers': 0,
            'failed_batches': [],
            'circuit_breaker_tripped': False,
            'circuit_breaker_trip_count': 0,
        }

    total_saved, failed_batches = writer_result[0]

    if failed_batches:
        total_failed_tickers = sum(len(fb['tickers']) for fb in failed_batches)
        logger.warning("Download summary: %d batches failed (%d tickers).",
                       len(failed_batches), total_failed_tickers)

    return {
        'total_tickers': len(tickers),
        'saved_tickers': total_saved,
        'failed_batches': failed_batches,
        'circuit_breaker_tripped': False,
        'circuit_breaker_trip_count': 0,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _add_file_logging(log_dir='data'):
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


def main():
    from src.logging_config import setup_logging
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
    parser.add_argument('--start', default='2014-04-30',
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

    # ── Proxy setup ─────────────────────────────────────────────────────
    if args.proxy or args.use_tor:
        global _proxy_url, _tor_enabled
        if args.use_tor:
            _proxy_url = config.TOR_SOCKS_PROXY
            _tor_enabled = True
            try:
                _rotate_tor_circuit()
            except Exception as e:
                logger.error("Tor not reachable: %s. Install with: "
                             "brew install tor && brew services start tor", e)
                return
        else:
            _proxy_url = args.proxy
        logger.info("Proxy enabled: %s", _proxy_url)

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


def _log_final_summary(manifests, log_path):
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


if __name__ == '__main__':
    main()
