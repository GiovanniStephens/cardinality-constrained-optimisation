"""
Ticker validation and batch retry-with-splitting for downloads.

Provides a quick validation pass that test-downloads 1 week of data per
ticker to identify which tickers yfinance can actually serve, with cache
management and multi-window support. Also provides recursive sub-batch
splitting for failed batches.
"""

import logging
import os
import random
import time

import pandas as pd
from tqdm import tqdm

from . import core as _core
from . import session as _sess

logger = logging.getLogger(__name__)


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

    # -- Helper: save cache incrementally ------------------------------------
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

    # -- Load progress from cache (resume or fresh hit) ----------------------
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

            # Fresh + complete cache -> return immediately
            if max_cache_hours and age_hours < max_cache_hours and not unchecked:
                valid = cached_valid & tickers_set
                invalid = cached_invalid & tickers_set
                unvalidated = tickers_set - valid - invalid
                logger.info("Validation cache hit (%.1fh old): %d valid, "
                            "%d invalid, %d unvalidated",
                            age_hours, len(valid), len(invalid), len(unvalidated))
                return valid, invalid, unvalidated

            # Stale complete cache -> re-validate from scratch
            if not unchecked and (not max_cache_hours or age_hours >= max_cache_hours):
                logger.info("Validation cache stale (%.1fh old), re-validating",
                            age_hours)
            else:
                # Partial cache -> resume from where we left off
                valid = cached_valid
                invalid = cached_invalid
                unvalidated = cached_unvalidated
                logger.info("Resuming validation: %d valid, %d invalid, "
                            "%d unvalidated from cache, %d unchecked remain",
                            len(valid), len(invalid), len(unvalidated),
                            len(unchecked))
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Validation cache corrupt, starting fresh")

    # -- Validate across windows ---------------------------------------------

    for window_idx, (win_start, win_end) in enumerate(validation_windows, 1):
        # Only skip tickers confirmed valid or unvalidated (failed batch);
        # re-check invalid tickers -- they may exist in a different window
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
            result = _core._download_batch_with_timeout(batch, win_start, win_end, timeout)

            if result is not None and not result.empty:
                # Successful batch -- categorise tickers
                returned_tickers = set(result.columns)
                valid |= returned_tickers
                # Tickers in this batch with no data -> invalid (for this window)
                batch_invalid = set(batch) - returned_tickers
                invalid |= batch_invalid
                # Remove from unvalidated if previously there
                unvalidated -= returned_tickers
            elif result is not None and result.empty:
                # Batch succeeded but empty -- all tickers invalid
                invalid |= set(batch)
            else:
                # Batch failed (None) -- retry once
                retried = False
                for retry in range(max_retries):
                    if _sess._get_state('_tor_enabled'):
                        _sess._rotate_tor_circuit()
                    time.sleep(delay)
                    result = _core._download_batch_with_timeout(batch, win_start, win_end, timeout)
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
                    # All retries failed -- conservatively keep as unvalidated
                    unvalidated |= set(batch)

            _save_cache(valid, invalid, unvalidated)

            if _sess._get_state('_tor_enabled'):
                from src.config import TOR_ROTATE_EVERY_N_BATCHES
                if (batch_idx + 1) % TOR_ROTATE_EVERY_N_BATCHES == 0:
                    _sess._rotate_tor_circuit()

            if batch_idx < len(batches) - 1:
                time.sleep(delay)

    # Tickers valid in any window should not be in invalid
    invalid -= valid
    unvalidated -= valid

    # -- Save final cache ----------------------------------------------------
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
        df = _core._download_batch_with_timeout(tickers, start, end, timeout_seconds)
        if df is not None and not df.empty:
            # Success -- halve delay
            delay_state[0] = max(base_delay, current_delay * 0.5)
            return df, []
        if attempt < 2:
            # Failed -- escalate delay
            current_delay = min(current_delay * 2, max_delay)
            delay_state[0] = current_delay
            jittered = current_delay * random.uniform(0.8, 1.2)
            time.sleep(jittered)

    # Can't split further -- return as failed
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
