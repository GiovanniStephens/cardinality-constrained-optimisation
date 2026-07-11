import os
import sys

from src.logging_config import setup_logging
setup_logging()

from src.db.connection import get_connection, DB_PATH
from src.db.tickers import backfill_metadata
from src.db.migrations import migrate_csvs

import logging
logger = logging.getLogger(__name__)

if len(sys.argv) > 1 and sys.argv[1] == 'migrate':
    conn = get_connection()
    migrate_csvs(conn)
    conn.close()
elif len(sys.argv) > 1 and sys.argv[1] == 'backfill':
    conn = get_connection()
    backfill_metadata(conn)
    conn.close()
elif len(sys.argv) > 1 and sys.argv[1] == 'backfill-names':
    import argparse
    from src.db.name_backfill import backfill_names, parse_directory
    p = argparse.ArgumentParser(prog='python -m src.db backfill-names')
    p.add_argument('--from-files', nargs='+', default=None, metavar='FILE',
                   help='Parse local symbol-directory files instead of '
                        'downloading from nasdaqtrader.com.')
    a = p.parse_args(sys.argv[2:])
    names = None
    if a.from_files:
        names = {}
        for path in a.from_files:
            with open(path) as f:
                for sym, name in parse_directory(f.read()).items():
                    names.setdefault(sym, name)
    conn = get_connection()
    backfill_names(conn, names=names)
    conn.close()
elif len(sys.argv) > 1 and sys.argv[1] == 'backfill-volume':
    import argparse
    from src.db.volume_backfill import backfill_volume
    p = argparse.ArgumentParser(prog='python -m src.db backfill-volume')
    p.add_argument('--period', default='9mo',
                   help='yfinance period to fetch (default: %(default)s)')
    p.add_argument('--batch-size', type=int, default=60)
    p.add_argument('--sleep', type=float, default=0.5,
                   help='seconds between batches (default: %(default)s)')
    p.add_argument('--asset-type', default='etf')
    a = p.parse_args(sys.argv[2:])
    conn = get_connection()
    backfill_volume(conn, asset_type=a.asset_type, period=a.period,
                    batch_size=a.batch_size, sleep=a.sleep)
    conn.close()
elif len(sys.argv) > 1 and sys.argv[1] == 'purge-phantom-rows':
    import argparse
    from datetime import datetime as dt
    from src.db.prices import purge_phantom_rows
    p = argparse.ArgumentParser(
        prog='python -m src.db purge-phantom-rows',
        description='Delete phantom weekend/NYSE-holiday price rows for '
                    'US-listed (dot-less) symbols. July 2026 incident: '
                    'promotion materialised unlimited-ffill rows across '
                    'mixed-calendar chunks.')
    p.add_argument('--dry-run', action='store_true',
                   help='report what would be deleted without touching the DB')
    p.add_argument('--no-backup', action='store_true',
                   help='skip the pre-purge database backup')
    p.add_argument('--exchange', default='US')
    a = p.parse_args(sys.argv[2:])
    conn = get_connection()
    if not a.dry_run and not a.no_backup:
        from src.pipeline import backup_database
        ts = dt.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{DB_PATH}.backup_{ts}"
        backup_database(conn, backup_path)
        logger.info("Pre-purge backup: %s", backup_path)
    purge_phantom_rows(conn, exchange=a.exchange, dry_run=a.dry_run)
    conn.close()
else:
    # Create empty database with schema
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    logger.info("Database created at: %s", DB_PATH)
    logger.info("Tables (%d):", len(tables))
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t['name']}").fetchone()[0]
        logger.info("  %s: %d rows", t['name'], count)
    conn.close()
