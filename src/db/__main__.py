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
