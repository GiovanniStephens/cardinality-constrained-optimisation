"""
Multi-worker concurrent download infrastructure for price data.

Provides thread-based and subprocess-based concurrent download paths,
a single-threaded DB writer consumer, and partitioning utilities.

Functions are accessed through ``src.download.core`` via the module object
(not direct name bindings) so that unittest.mock.patch on
``src.download.core._download_batch`` etc. is respected at call time.
"""

import logging
import multiprocessing
import queue
import random
import sys
import threading
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# Import the module, NOT individual names.  This ensures every call goes
# through the module's current attribute (which may be a mock in tests).
from . import core as _dd
from . import session as _sess
from . import validate as _val

logger = logging.getLogger(__name__)


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


def _reset_yf_singleton():
    """Destroy the YfData singleton so the next yf.download() creates a fresh
    instance with a new session, crumb, and cookie.

    Simply clearing crumb/cookie isn't enough — residual state (cached
    cookies in the session, internal flags) can cause the new session to
    inherit the old IP's rate-limit status. Deleting the instance forces
    a completely clean slate.

    Safe to call in a subprocess where no other threads share the singleton.
    """
    try:
        import yfinance.data as yd
        with yd.SingletonMeta._lock:
            if yd.YfData in yd.SingletonMeta._instances:
                del yd.SingletonMeta._instances[yd.YfData]
    except Exception:
        pass


def _batch_download_loop(
    worker_id, tickers, start, end, batch_size, null_threshold,
    result_queue, names, countries, sectors, industries,
    category_groups, categories, max_retries, rate_limit_delay,
    batch_timeout, circuit_breaker_threshold, max_rate_limit_delay,
    circuit_breaker_max_trips, circuit_breaker_cooldown,
    session_rotate_interval, tor_enabled,
    download_fn, make_session_fn, rotate_tor_fn, build_metadata_fn,
    is_rate_limit_fn, enable_sub_batch_splitting=False,
    retry_splitting_fn=None, log=None,
    classify_empty_fn=None,
):
    """Core batch-download-with-retry loop shared by thread and subprocess workers.

    Parameterised by injected dependencies so it works in both thread
    (module-level imports) and subprocess (absolute-import) contexts.
    """
    if log is None:
        log = logger

    batches = [
        tickers[i:i + batch_size]
        for i in range(0, len(tickers), batch_size)
    ]

    session = make_session_fn()
    session_age = 0
    consecutive_failures = 0
    current_delay = rate_limit_delay
    circuit_breaker_trip_count = 0

    for batch_num, batch in enumerate(batches, 1):
        batch_df = None
        hit_rate_limit = False
        data_level_empty = False  # set when classifier says 'no_data'

        if session_age >= session_rotate_interval:
            session = make_session_fn()
            _reset_yf_singleton()
            session_age = 0
            log.info("Worker %d: rotated session at batch %d",
                     worker_id, batch_num)

        for attempt in range(1, max_retries + 1):
            try:
                batch_df = download_fn(batch, start, end, session=session)
                if batch_df is not None and not batch_df.empty:
                    break
                # Empty result: classify as rate-limit vs data-level.
                if classify_empty_fn is not None and classify_empty_fn() == 'no_data':
                    data_level_empty = True
                    log.info("Worker %d batch %d: no data (data-level); "
                             "skipping without retry.", worker_id, batch_num)
                    break
                if attempt < max_retries:
                    hit_rate_limit = True
                    session = make_session_fn()
                    _reset_yf_singleton()
                    session_age = 0
                    if tor_enabled and rotate_tor_fn:
                        rotate_tor_fn()
                    jittered = rate_limit_delay * random.uniform(0.8, 1.2)
                    log.warning(
                        "Worker %d batch %d attempt %d/%d: no data, "
                        "rotating session, retrying in %.0fs",
                        worker_id, batch_num, attempt, max_retries, jittered)
                    time.sleep(jittered)
            except Exception as e:
                if is_rate_limit_fn(e):
                    hit_rate_limit = True
                    session = make_session_fn()
                    _reset_yf_singleton()
                    session_age = 0
                    if tor_enabled and rotate_tor_fn:
                        rotate_tor_fn()
                    current_delay = min(current_delay * 2, max_rate_limit_delay)
                    jittered = current_delay * random.uniform(0.8, 1.2)
                    log.warning(
                        "Worker %d rate limited batch %d (attempt %d/%d), "
                        "rotating session, backing off %.0fs",
                        worker_id, batch_num, attempt, max_retries, jittered)
                    time.sleep(jittered)
                else:
                    log.warning("Worker %d batch %d attempt %d/%d failed: %s",
                                worker_id, batch_num, attempt, max_retries, e)
                    if attempt < max_retries:
                        jittered = current_delay * random.uniform(0.8, 1.2)
                        time.sleep(jittered)

        session_age += 1

        if batch_df is None or batch_df.empty:
            # Try sub-batch splitting before giving up (thread path only)
            if enable_sub_batch_splitting and retry_splitting_fn:
                from src import config as _cfg
                min_sub = _cfg.PIPELINE_MIN_SUB_BATCH_SIZE
                if len(batch) > min_sub:
                    delay_state = [current_delay, max_rate_limit_delay,
                                   rate_limit_delay]
                    split_df, still_failed = retry_splitting_fn(
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
                # Data-level empties (delisted tickers) must not feed the
                # circuit breaker — otherwise a run of dead tickers looks
                # like a rate-limit cascade.
                if not data_level_empty:
                    consecutive_failures += 1

                if consecutive_failures >= circuit_breaker_threshold:
                    circuit_breaker_trip_count += 1
                    if circuit_breaker_trip_count >= circuit_breaker_max_trips:
                        log.error("Worker %d: circuit breaker tripped %d "
                                  "times, aborting.",
                                  worker_id, circuit_breaker_trip_count)
                        break
                    else:
                        log.warning("Worker %d: circuit breaker trip %d/%d, "
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

        metadata = build_metadata_fn(
            batch_df.columns, names=names, countries=countries,
            sectors=sectors, industries=industries,
            category_groups=category_groups, categories=categories)

        saved_list = list(batch_df.columns)
        result_queue.put(('data', batch_df, metadata, saved_list, batch_num))

        if tor_enabled and rotate_tor_fn:
            from src.config import TOR_ROTATE_EVERY_N_BATCHES
            if batch_num % TOR_ROTATE_EVERY_N_BATCHES == 0:
                rotate_tor_fn()

        if current_delay > 0 and batch_num < len(batches):
            jittered = current_delay * random.uniform(0.8, 1.2)
            time.sleep(jittered)


def _subprocess_worker(worker_id, tickers, proxy_url, proxy_counter_start,
                       result_queue, start, end, batch_size, null_threshold,
                       names, countries, sectors, industries,
                       category_groups, categories, max_retries,
                       rate_limit_delay, batch_timeout,
                       circuit_breaker_threshold, max_rate_limit_delay,
                       circuit_breaker_max_trips, circuit_breaker_cooldown,
                       tor_enabled, session_rotate_interval):
    """Top-level worker function for multiprocessing.Process.

    Runs in a child process with its own yfinance singleton, so crumb/cookie
    state is fully isolated from other workers.
    """
    import src.download.core as _dd_sub
    import src.download.session as _sess_sub

    _sess_sub.set_proxy_state(proxy_url, tor_enabled, proxy_counter_start)

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [%(levelname)s] Worker {worker_id}: %(message)s',
    )
    sub_logger = logging.getLogger(__name__)

    try:
        _batch_download_loop(
            worker_id=worker_id, tickers=tickers, start=start, end=end,
            batch_size=batch_size, null_threshold=null_threshold,
            result_queue=result_queue, names=names, countries=countries,
            sectors=sectors, industries=industries,
            category_groups=category_groups, categories=categories,
            max_retries=max_retries, rate_limit_delay=rate_limit_delay,
            batch_timeout=batch_timeout,
            circuit_breaker_threshold=circuit_breaker_threshold,
            max_rate_limit_delay=max_rate_limit_delay,
            circuit_breaker_max_trips=circuit_breaker_max_trips,
            circuit_breaker_cooldown=circuit_breaker_cooldown,
            session_rotate_interval=session_rotate_interval,
            tor_enabled=tor_enabled,
            download_fn=_dd_sub._download_batch,
            make_session_fn=_sess_sub._make_session,
            rotate_tor_fn=_sess_sub._rotate_tor_circuit,
            build_metadata_fn=_dd_sub._build_batch_metadata,
            is_rate_limit_fn=_sess_sub.is_rate_limit_error,
            classify_empty_fn=_dd_sub._classify_empty_batch,
            log=sub_logger,
        )
    finally:
        result_queue.put(None)


def _worker_download(worker_id, tickers, start, end, batch_size,
                     null_threshold, result_queue, names, countries,
                     sectors, industries, category_groups, categories,
                     max_retries, rate_limit_delay, batch_timeout,
                     circuit_breaker_threshold, max_rate_limit_delay,
                     circuit_breaker_max_trips, circuit_breaker_cooldown):
    """Per-worker download loop (thread-based). Puts results onto result_queue.

    Always sends a sentinel (None) when done, even on exception, so the
    DB writer thread never hangs waiting for a worker that crashed.
    """
    from src import config

    try:
        _batch_download_loop(
            worker_id=worker_id, tickers=tickers, start=start, end=end,
            batch_size=batch_size, null_threshold=null_threshold,
            result_queue=result_queue, names=names, countries=countries,
            sectors=sectors, industries=industries,
            category_groups=category_groups, categories=categories,
            max_retries=max_retries, rate_limit_delay=rate_limit_delay,
            batch_timeout=batch_timeout,
            circuit_breaker_threshold=circuit_breaker_threshold,
            max_rate_limit_delay=max_rate_limit_delay,
            circuit_breaker_max_trips=circuit_breaker_max_trips,
            circuit_breaker_cooldown=circuit_breaker_cooldown,
            session_rotate_interval=config.PIPELINE_SESSION_ROTATE_INTERVAL,
            tor_enabled=_sess._get_state('_tor_enabled'),
            download_fn=_dd._download_batch,
            make_session_fn=_sess._make_session,
            rotate_tor_fn=_sess._rotate_tor_circuit,
            build_metadata_fn=_dd._build_batch_metadata,
            is_rate_limit_fn=_sess.is_rate_limit_error,
            classify_empty_fn=_dd._classify_empty_batch,
            enable_sub_batch_splitting=True,
            retry_splitting_fn=_val._retry_with_splitting,
        )
    finally:
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

    When a proxy is configured and n_workers > 1, uses subprocess-based workers
    instead of threads. This is necessary because yfinance's YfData class is a
    process-wide singleton — threads all share the same crumb/cookie, causing
    cross-contamination when each thread has a different proxy IP. Subprocesses
    get their own Python interpreter with an isolated singleton.

    Without a proxy (or with a single worker), uses the faster thread-based path.
    """
    from src import config

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

    tickers = _dd._deduplicate_tickers(tickers)

    use_subprocesses = _sess._proxy_url is not None and n_workers > 1
    if use_subprocesses:
        return _concurrent_subprocess_download(
            tickers, conn, exchange, n_workers, asset_type,
            start, end, batch_size, null_threshold,
            names, countries, max_retries,
            on_batch_complete, on_batch_failed,
            rate_limit_delay, batch_timeout,
            circuit_breaker_threshold, max_rate_limit_delay,
            circuit_breaker_max_trips, circuit_breaker_cooldown,
            sectors, industries, category_groups, categories,
        )

    return _concurrent_thread_download(
        tickers, conn, exchange, n_workers, asset_type,
        start, end, batch_size, null_threshold,
        names, countries, max_retries,
        on_batch_complete, on_batch_failed,
        rate_limit_delay, batch_timeout,
        circuit_breaker_threshold, max_rate_limit_delay,
        circuit_breaker_max_trips, circuit_breaker_cooldown,
        sectors, industries, category_groups, categories,
    )


def _concurrent_thread_download(
    tickers, conn, exchange, n_workers, asset_type,
    start, end, batch_size, null_threshold,
    names, countries, max_retries,
    on_batch_complete, on_batch_failed,
    rate_limit_delay, batch_timeout,
    circuit_breaker_threshold, max_rate_limit_delay,
    circuit_breaker_max_trips, circuit_breaker_cooldown,
    sectors, industries, category_groups, categories,
):
    """Thread-based concurrent download (original path)."""
    from src import db

    partitions = _partition_tickers(tickers, n_workers)
    actual_workers = len(partitions)

    total_batches = sum(
        (len(p) + batch_size - 1) // batch_size for p in partitions
    )

    result_q = queue.Queue(maxsize=actual_workers * 2)

    use_tqdm = sys.stderr.isatty()
    pbar = None
    if use_tqdm:
        pbar = tqdm(total=total_batches, desc="Downloading", unit="batch",
                    file=sys.stderr)

    db_path = conn.execute("PRAGMA database_list").fetchone()[2]

    writer_result = [None]

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

    stagger_delay = 5

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = []
        for wid, partition in enumerate(partitions):
            if wid > 0 and _sess._proxy_url:
                time.sleep(stagger_delay)
            f = pool.submit(_worker_download, wid, partition, **common_kwargs)
            futures.append(f)

        for f in futures:
            try:
                f.result()
            except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError) as e:
                logger.warning("Worker cancelled or timed out: %s", e)
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


def _concurrent_subprocess_download(
    tickers, conn, exchange, n_workers, asset_type,
    start, end, batch_size, null_threshold,
    names, countries, max_retries,
    on_batch_complete, on_batch_failed,
    rate_limit_delay, batch_timeout,
    circuit_breaker_threshold, max_rate_limit_delay,
    circuit_breaker_max_trips, circuit_breaker_cooldown,
    sectors, industries, category_groups, categories,
):
    """Subprocess-based concurrent download for proxy isolation.

    Each subprocess gets its own Python interpreter with a fresh yfinance
    singleton, so crumb/cookie state is fully isolated per worker.
    """
    from src import db, config

    partitions = _partition_tickers(tickers, n_workers)
    actual_workers = len(partitions)

    logger.info("Using subprocess workers for proxy isolation (%d workers)",
                actual_workers)

    total_batches = sum(
        (len(p) + batch_size - 1) // batch_size for p in partitions
    )

    # multiprocessing.Queue for cross-process IPC (pickle-based)
    result_q = multiprocessing.Queue(maxsize=actual_workers * 2)

    use_tqdm = sys.stderr.isatty()
    pbar = None
    if use_tqdm:
        pbar = tqdm(total=total_batches, desc="Downloading", unit="batch",
                    file=sys.stderr)

    db_path = conn.execute("PRAGMA database_list").fetchone()[2]

    # DB writer runs in a thread in the main process — single writer,
    # no SQLite contention
    writer_result = [None]

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

    stagger_delay = config.PIPELINE_SUBPROCESS_STAGGER
    session_rotate_interval = config.PIPELINE_SESSION_ROTATE_INTERVAL

    processes = []
    try:
        for wid, partition in enumerate(partitions):
            if wid > 0:
                time.sleep(stagger_delay)

            # Each worker gets a non-overlapping counter range.
            # Use random base to avoid reusing burned session IDs from
            # previous runs (time-based seeds overlap across restarts).
            counter_start = wid * 100_000 + random.randint(0, 99_999)

            p = multiprocessing.Process(
                target=_subprocess_worker,
                args=(wid, partition, _sess._proxy_url, counter_start, result_q),
                kwargs=dict(
                    start=start, end=end, batch_size=batch_size,
                    null_threshold=null_threshold,
                    names=names, countries=countries,
                    sectors=sectors, industries=industries,
                    category_groups=category_groups, categories=categories,
                    max_retries=max_retries, rate_limit_delay=rate_limit_delay,
                    batch_timeout=batch_timeout,
                    circuit_breaker_threshold=circuit_breaker_threshold,
                    max_rate_limit_delay=max_rate_limit_delay,
                    circuit_breaker_max_trips=circuit_breaker_max_trips,
                    circuit_breaker_cooldown=circuit_breaker_cooldown,
                    tor_enabled=_sess._tor_enabled,
                    session_rotate_interval=session_rotate_interval,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)
            logger.info("Launched subprocess worker %d (pid %d) with %d tickers",
                        wid, p.pid, len(partition))

        # Wait for all subprocesses
        for p in processes:
            p.join()

        # Inject extra sentinels for any worker that died without sending one
        for p in processes:
            if p.exitcode != 0 and p.exitcode is not None:
                logger.error("Worker pid %d exited with code %d",
                             p.pid, p.exitcode)
                result_q.put(None)

    except KeyboardInterrupt:
        logger.warning("Interrupted — terminating workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        # Inject sentinels for dead workers so DB writer finishes
        for p in processes:
            result_q.put(None)

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
