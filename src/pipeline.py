"""
Pipeline orchestrator for robust, staged data ingestion.

Coordinates: preflight disk check -> staging DB creation -> batched download
with checkpointing -> data quality validation -> promotion to production
(with backup) -> cleanup.
"""

import json
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone

from src import config, db
from src.data_quality import validate_universe
from src.download_data import concurrent_download_and_save, download_and_save

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    """Load checkpoint JSON. Returns dict or empty dict if file missing."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_checkpoint(path, state):
    """Atomically write checkpoint JSON (write to .tmp then rename)."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def filter_completed(tickers, checkpoint):
    """Remove already-completed tickers from the download list."""
    completed = set(checkpoint.get('completed_tickers', []))
    if not completed:
        return tickers
    remaining = [t for t in tickers if t not in completed]
    logger.info("Checkpoint: %d already completed, %d remaining",
                len(completed), len(remaining))
    return remaining


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def preflight_check(prod_db_path, staging_db_path, num_tickers, num_trading_days):
    """
    Check disk space and estimate storage needs.

    Returns dict with ok, available_gb, estimated_staging_gb, warnings.
    """
    estimated_rows = num_tickers * num_trading_days
    estimated_bytes = estimated_rows * config.PIPELINE_BYTES_PER_ROW

    prod_size = os.path.getsize(prod_db_path) if os.path.exists(prod_db_path) else 0
    # Need space for: staging DB + backup of production DB
    total_needed = estimated_bytes + prod_size

    target_dir = os.path.dirname(staging_db_path) or '.'
    stat = shutil.disk_usage(target_dir)
    available = stat.free

    ok = available > total_needed * config.PIPELINE_DISK_HEADROOM
    warnings = []
    if not ok:
        warnings.append(
            f"Need ~{total_needed / (1024**3):.1f} GB "
            f"(staging {estimated_bytes / (1024**3):.1f} GB + "
            f"backup {prod_size / (1024**3):.1f} GB) "
            f"but only {available / (1024**3):.1f} GB free"
        )

    result = {
        'ok': ok,
        'available_gb': round(available / (1024**3), 2),
        'estimated_staging_gb': round(estimated_bytes / (1024**3), 2),
        'estimated_total_gb': round(total_needed / (1024**3), 2),
        'prod_size_gb': round(prod_size / (1024**3), 2),
        'warnings': warnings,
    }
    return result


# ---------------------------------------------------------------------------
# Backup and rollback
# ---------------------------------------------------------------------------

def backup_database(source_conn, backup_path):
    """
    Create a consistent backup using sqlite3.Connection.backup().

    Handles WAL mode correctly. Returns the backup path.
    """
    dest = sqlite3.connect(backup_path)
    try:
        source_conn.backup(dest)
    finally:
        dest.close()
    size_mb = os.path.getsize(backup_path) / (1024 * 1024)
    logger.info("Backup created: %s (%.1f MB)", backup_path, size_mb)
    return backup_path


def rollback(backup_path, prod_db_path):
    """Restore production DB from a backup file."""
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"Backup not found: {backup_path}")
    # Remove WAL/SHM files if present (they belong to the old DB state)
    for suffix in ('-wal', '-shm'):
        wal = prod_db_path + suffix
        if os.path.exists(wal):
            os.remove(wal)
    shutil.copy2(backup_path, prod_db_path)
    logger.info("Rolled back %s from %s", prod_db_path, backup_path)


# ---------------------------------------------------------------------------
# Promotion (staging -> production)
# ---------------------------------------------------------------------------

def promote_staging(staging_db_path, conn_prod, exchange, chunk_size=200):
    """
    Copy validated data from staging DB into production DB.

    Reads tickers from staging in chunks to avoid loading the entire dataset
    into memory at once. Uses save_prices() which does INSERT OR REPLACE.
    """
    conn_staging = db.get_connection(staging_db_path)
    try:
        # Get all ticker symbols in the staging DB for this exchange
        exchange_id = db._get_exchange_id(conn_staging, exchange)
        rows = conn_staging.execute(
            "SELECT symbol, asset_type, name, country FROM tickers WHERE exchange_id = ?",
            (exchange_id,),
        ).fetchall()

        if not rows:
            logger.warning("No tickers found in staging DB for exchange %s", exchange)
            return 0

        symbols = [r['symbol'] for r in rows]
        asset_types = {r['symbol']: r['asset_type'] for r in rows}
        names_map = {r['symbol']: r['name'] for r in rows if r['name']}
        countries_map = {r['symbol']: r['country'] for r in rows if r['country']}

        # Group by asset_type to match save_prices() expectations
        by_type = {}
        for sym in symbols:
            at = asset_types[sym]
            by_type.setdefault(at, []).append(sym)

        total_promoted = 0
        for asset_type, type_symbols in by_type.items():
            # Load and promote in chunks
            for i in range(0, len(type_symbols), chunk_size):
                chunk = type_symbols[i:i + chunk_size]
                prices_df = db.load_prices(
                    conn_staging, exchange=exchange, tickers=chunk,
                    exclude_flagged=False, min_coverage=0, ffill_limit=None,
                )
                if prices_df.empty:
                    continue

                chunk_names = {s: names_map[s] for s in chunk if s in names_map}
                chunk_countries = {s: countries_map[s] for s in chunk if s in countries_map}

                db.save_prices(
                    conn_prod, prices_df, exchange=exchange,
                    asset_type=asset_type,
                    names=chunk_names or None,
                    countries=chunk_countries or None,
                    source='staged_promotion',
                )
                total_promoted += prices_df.shape[1]
                logger.info("Promoted %d/%d tickers (%s)",
                            total_promoted, len(symbols), asset_type)

        # Promotion integrity check (P9)
        exchange_id_prod = db._get_exchange_id(conn_prod, exchange)
        prod_count = conn_prod.execute(
            "SELECT COUNT(DISTINCT symbol) FROM tickers WHERE exchange_id = ?",
            (exchange_id_prod,),
        ).fetchone()[0]
        if prod_count < len(symbols):
            logger.warning(
                "Promotion integrity: production has %d tickers but "
                "staging had %d for exchange %s",
                prod_count, len(symbols), exchange)

        logger.info("Promotion complete: %d tickers transferred", total_promoted)
        return total_promoted
    finally:
        conn_staging.close()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(path, manifest):
    """Write the run manifest to a JSON file."""
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Manifest written to %s", path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _run_preflight(prod_db_path, staging_db_path, tickers, start, end, manifest):
    """Run disk-space preflight check. Returns True if OK, False to abort."""
    from datetime import datetime as dt
    start_dt = dt.strptime(start, '%Y-%m-%d')
    end_dt = dt.strptime(end, '%Y-%m-%d')
    est_trading_days = int((end_dt - start_dt).days * 252 / 365)

    preflight = preflight_check(prod_db_path, staging_db_path,
                                len(tickers), est_trading_days)
    manifest['preflight'] = preflight
    logger.info("Preflight: %.1f GB available, ~%.1f GB needed (staging + backup)",
                preflight['available_gb'], preflight['estimated_total_gb'])

    if not preflight['ok']:
        for w in preflight['warnings']:
            logger.error("PREFLIGHT FAILED: %s", w)
        manifest['status'] = 'preflight_failed'
    return preflight['ok']


def _run_download(tickers, conn_staging, checkpoint, checkpoint_path,
                  manifest, n_workers=1, **download_kwargs):
    """Download into staging DB with checkpointing. Returns download result or None on interrupt."""
    def _on_batch_complete(saved_tickers_list, batch_num):
        checkpoint['completed_tickers'].extend(saved_tickers_list)
        checkpoint['last_batch'] = batch_num
        checkpoint['updated_at'] = datetime.now(timezone.utc).isoformat()
        save_checkpoint(checkpoint_path, checkpoint)

    def _on_batch_failed(failed_tickers_list, batch_num):
        checkpoint['failed_tickers'].extend(failed_tickers_list)
        checkpoint['updated_at'] = datetime.now(timezone.utc).isoformat()
        save_checkpoint(checkpoint_path, checkpoint)

    try:
        if n_workers > 1:
            result = concurrent_download_and_save(
                tickers, conn_staging,
                n_workers=n_workers,
                on_batch_complete=_on_batch_complete,
                on_batch_failed=_on_batch_failed,
                **download_kwargs,
            )
        else:
            result = download_and_save(
                tickers, conn_staging,
                on_batch_complete=_on_batch_complete,
                on_batch_failed=_on_batch_failed,
                **download_kwargs,
            )
    except KeyboardInterrupt:
        logger.warning("Download interrupted. Checkpoint saved at %s",
                        checkpoint_path)
        manifest['status'] = 'interrupted'
        manifest['download_result'] = {
            'completed_tickers': len(checkpoint.get('completed_tickers', [])),
        }
        return None

    manifest['download_result'] = result
    logger.info("Download complete: %d/%d saved, %d failed batches",
                result['saved_tickers'], result['total_tickers'],
                len(result['failed_batches']))
    return result


def _save_dropped_tickers(result, data_dir, asset_type, run_id, manifest,
                          prod_db_path, exchange):
    """Cache failed tickers in the production DB."""
    if not result['failed_batches']:
        return
    all_failed = []
    for fb in result['failed_batches']:
        all_failed.extend(fb['tickers'])
    if all_failed:
        prod_conn = db.get_connection(prod_db_path)
        db.save_known_bad_tickers(prod_conn, all_failed, exchange=exchange)
        prod_conn.close()
        manifest['failed_ticker_count'] = len(all_failed)
        logger.info("Cached %d failed tickers in DB (use --retry-dropped "
                    "or --clear-cache to manage)", len(all_failed))


def _run_validation(conn_staging, exchange, manifest):
    """Validate staging DB data quality. Returns True if OK."""
    logger.info("Running data quality validation on staging DB...")
    validation = validate_universe(conn_staging, exchange=exchange)
    manifest['validation'] = validation

    exclusion_rate = (validation['total_excluded'] / validation['total_tickers']
                      if validation['total_tickers'] > 0 else 0)
    logger.info("Validation: %d/%d excluded (%.0f%%), %d active",
                 validation['total_excluded'], validation['total_tickers'],
                 exclusion_rate * 100, validation['total_active'])

    if validation['total_active'] == 0:
        logger.error("Validation failed: no active tickers remain")
        manifest['status'] = 'validation_failed'
        return False
    return True


def _promote_and_cleanup(staging_db_path, prod_db_path, exchange, run_id,
                         manifest, no_backup, keep_staging, checkpoint_path):
    """Backup production, promote staging, and clean up."""
    conn_prod = db.get_connection(prod_db_path)

    backup_path = None
    if not no_backup and os.path.exists(prod_db_path):
        backup_path = f"{prod_db_path}.backup_{run_id}"
        backup_database(conn_prod, backup_path)
        manifest['backup_path'] = backup_path

    try:
        promoted = promote_staging(staging_db_path, conn_prod, exchange=exchange)
        manifest['promoted_tickers'] = promoted
        manifest['status'] = 'promoted'
        logger.info("Promotion successful: %d tickers", promoted)
    except Exception as e:
        logger.error("Promotion failed: %s", e)
        manifest['status'] = 'promotion_failed'
        manifest['promotion_error'] = str(e)
        if backup_path:
            logger.info("Backup available at %s for rollback", backup_path)
        conn_prod.close()
        raise
    finally:
        conn_prod.close()

    # Cleanup
    if not keep_staging and os.path.exists(staging_db_path):
        os.remove(staging_db_path)
        for suffix in ('-wal', '-shm'):
            p = staging_db_path + suffix
            if os.path.exists(p):
                os.remove(p)
        logger.info("Staging DB removed: %s", staging_db_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return promoted


def run_pipeline(
    tickers,
    exchange,
    asset_type='etf',
    start='2014-04-30',
    end='2025-04-30',
    batch_size=None,
    null_threshold=0.9,
    names=None,
    countries=None,
    sectors=None,
    industries=None,
    category_groups=None,
    categories=None,
    max_retries=None,
    subset=None,
    stage_only=False,
    skip_validation=False,
    keep_staging=False,
    no_backup=False,
    checkpoint_path=None,
    staging_db_path=None,
    rate_limit_delay=None,
    prod_db_path=None,
    n_workers=None,
):
    """
    Full staged pipeline: preflight -> stage -> validate -> promote.

    Returns a manifest dict with full run details.
    """
    t_start = time.time()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    # Apply defaults from config
    if batch_size is None:
        from src.download_data import _proxy_url
        if _proxy_url and n_workers and n_workers > 1:
            batch_size = config.PIPELINE_PROXY_BATCH_SIZE
        else:
            batch_size = config.PIPELINE_BATCH_SIZE
    if max_retries is None:
        max_retries = config.PIPELINE_MAX_RETRIES
    if rate_limit_delay is None:
        rate_limit_delay = config.PIPELINE_RATE_LIMIT_DELAY
    if prod_db_path is None:
        prod_db_path = db.DB_PATH
    if n_workers is None:
        n_workers = config.PIPELINE_DEFAULT_WORKERS
    n_workers = max(1, min(n_workers, config.PIPELINE_MAX_WORKERS))

    data_dir = os.path.dirname(prod_db_path)
    if staging_db_path is None:
        staging_db_path = os.path.join(data_dir, f'staging_{run_id}.db')
    if checkpoint_path is None:
        checkpoint_path = os.path.join(data_dir, f'checkpoint_{run_id}.json')

    if subset is not None:
        tickers = tickers[:subset]
        logger.info("Subset mode: using first %d tickers", len(tickers))

    manifest = {
        'run_id': run_id,
        'status': 'started',
        'exchange': exchange,
        'asset_type': asset_type,
        'start_date': start,
        'end_date': end,
        'total_tickers': len(tickers),
        'staging_db': staging_db_path,
        'checkpoint': checkpoint_path,
        'prod_db': prod_db_path,
    }

    def _write_and_return():
        manifest_path = os.path.join(data_dir, f'manifest_{run_id}.json')
        write_manifest(manifest_path, manifest)
        return manifest

    # ── Preflight ─────────────────────────────────────────────────────────
    if not _run_preflight(prod_db_path, staging_db_path, tickers, start,
                          end, manifest):
        return _write_and_return()

    # ── Resume from checkpoint ────────────────────────────────────────────
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        logger.info("Resuming from checkpoint: %d tickers already completed",
                     len(checkpoint.get('completed_tickers', [])))
        tickers = filter_completed(tickers, checkpoint)
        if not tickers:
            logger.info("All tickers already completed per checkpoint")
            manifest['status'] = 'already_complete'
            return manifest
    else:
        checkpoint = {
            'run_id': run_id,
            'exchange': exchange,
            'staging_db': staging_db_path,
            'completed_tickers': [],
            'failed_tickers': [],
        }

    # ── Download into staging DB ──────────────────────────────────────────
    logger.info("Creating staging DB: %s", staging_db_path)
    conn_staging = db.get_connection(staging_db_path)

    result = _run_download(
        tickers, conn_staging, checkpoint, checkpoint_path, manifest,
        n_workers=n_workers,
        exchange=exchange, asset_type=asset_type,
        start=start, end=end, batch_size=batch_size,
        null_threshold=null_threshold, names=names, countries=countries,
        sectors=sectors, industries=industries,
        category_groups=category_groups, categories=categories,
        max_retries=max_retries, rate_limit_delay=rate_limit_delay,
    )
    if result is None:
        conn_staging.close()
        return _write_and_return()

    _save_dropped_tickers(result, data_dir, asset_type, run_id, manifest,
                          prod_db_path, exchange)

    if result.get('circuit_breaker_tripped'):
        logger.error("Circuit breaker tripped. Checkpoint saved at %s. "
                     "Use --resume to retry after the issue is resolved.",
                     checkpoint_path)
        conn_staging.close()
        manifest['status'] = 'circuit_breaker_tripped'
        return _write_and_return()

    # ── Validate ──────────────────────────────────────────────────────────
    if not skip_validation:
        if not _run_validation(conn_staging, exchange, manifest):
            conn_staging.close()
            return _write_and_return()

    conn_staging.close()

    if stage_only:
        logger.info("Stage-only mode: staging DB at %s", staging_db_path)
        manifest['status'] = 'staged'
        return _write_and_return()

    # ── Promote to production ─────────────────────────────────────────────
    try:
        _promote_and_cleanup(staging_db_path, prod_db_path, exchange, run_id,
                             manifest, no_backup, keep_staging,
                             checkpoint_path)
    except Exception:
        return _write_and_return()

    # ── Final manifest ────────────────────────────────────────────────────
    manifest['duration_seconds'] = round(time.time() - t_start, 1)
    disk_after = shutil.disk_usage(data_dir)
    manifest['disk_free_gb_after'] = round(disk_after.free / (1024**3), 2)
    return _write_and_return()
