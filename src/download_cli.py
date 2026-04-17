"""CLI entry point for building a security universe and downloading price data.

Extracted from download_data.py to separate CLI concerns from download logic.
"""

import argparse
import logging
import os

from src.download_data import (
    ASSET_TYPE_MAP,
    build_security_universe,
    load_tickers,
)

logger = logging.getLogger(__name__)


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
                        default=['etfs'],
                        choices=['equities', 'etfs', 'funds'],
                        help='Asset types to include.')
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
    parser.add_argument('--output', default='data/Prices.csv',
                        help='Output CSV file path for prices.')
    parser.add_argument('--universe-output', default='data/Securities.csv',
                        help='Output CSV for the security universe list.')
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
    args = parser.parse_args()

    from src import db, config

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

    if args.from_csv:
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
        tickers_df.to_csv(args.universe_output, index=False)
        logger.info("Saved %d securities to %s", len(tickers_df), args.universe_output)
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
        for fd_type, group in tickers_df.groupby('AssetType'):
            db_type = ASSET_TYPE_MAP.get(fd_type, fd_type)
            ticker_list = group[args.ticker_column].tolist()
            names = None
            if 'Name' in group.columns:
                names = dict(zip(group[args.ticker_column], group['Name']))
            countries = None
            if 'Country' in group.columns:
                countries = dict(zip(group[args.ticker_column], group['Country']))
            logger.info("Running pipeline for %d %s tickers...",
                        len(ticker_list), db_type)
            manifest = run_pipeline(
                ticker_list, exchange=args.exchange, asset_type=db_type,
                start=start, end=args.end, null_threshold=args.null_threshold,
                names=names, countries=countries,
                subset=args.subset,
                stage_only=args.stage_only,
                skip_validation=args.skip_validation,
                keep_staging=args.keep_staging,
                no_backup=args.no_backup,
                checkpoint_path=checkpoint_path,
                staging_db_path=staging_db_path,
                rate_limit_delay=args.rate_limit,
            )
            all_manifests.append((db_type, manifest))

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
        )
        dl = manifest.get('download_result', {})
        logger.info("Pipeline: status=%s, saved=%s, failed_batches=%s",
                     manifest['status'],
                     dl.get('saved_tickers', 'N/A'),
                     len(dl.get('failed_batches', [])))

    # Step 4: Export to CSV for backward compatibility (only after promotion)
    if not args.stage_only:
        conn = db.get_connection()
        prices_df = db.load_prices(conn, exchange=args.exchange)
        conn.close()
        if not prices_df.empty:
            prices_df.to_csv(args.output)
            logger.info("Saved %d rows x %d columns to %s",
                        len(prices_df), len(prices_df.columns), args.output)
        else:
            logger.warning("No prices in database to export.")

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
