import os
import sys

from src.logging_config import setup_logging
setup_logging()

from src.db.connection import get_connection, DB_PATH
from src.db.migrations import migrate_csvs

import logging
logger = logging.getLogger(__name__)

if len(sys.argv) > 1 and sys.argv[1] == 'migrate':
    conn = get_connection()
    migrate_csvs(conn)
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
