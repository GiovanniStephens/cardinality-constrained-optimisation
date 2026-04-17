"""Database connection management."""

import logging
import sqlite3
from datetime import datetime, timezone

from src.config import DB_PATH
from src.db.schema import SCHEMA_SQL, DEFAULT_EXCHANGES, _apply_migrations

logger = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path=None):
    """Open a database connection, create tables if needed, seed exchanges."""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    _apply_migrations(conn)
    # Seed exchanges if empty
    count = conn.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0]
    if count == 0:
        conn.executemany(
            "INSERT INTO exchanges (code, name, country) VALUES (?, ?, ?)",
            DEFAULT_EXCHANGES,
        )
        conn.commit()
    return conn


def _get_exchange_id(conn, code):
    """Look up exchange id by code. Raises ValueError if not found."""
    row = conn.execute(
        "SELECT id FROM exchanges WHERE code = ?", (code,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown exchange code: {code!r}")
    return row[0]
