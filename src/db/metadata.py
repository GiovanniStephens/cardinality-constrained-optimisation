"""Data source and metadata functions."""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from src.db.connection import _now

logger = logging.getLogger(__name__)


def _save_data_source_no_commit(conn: sqlite3.Connection, source: str,
                                exchange_id: Optional[int] = None,
                                date_range_start: Optional[str] = None,
                                date_range_end: Optional[str] = None,
                                num_tickers: Optional[int] = None,
                                num_rows: Optional[int] = None,
                                notes: Optional[str] = None) -> int:
    """Insert a data_source row without committing (for use inside transactions)."""
    now = _now()
    cur = conn.execute(
        """INSERT INTO data_sources (
            exchange_id, source, downloaded_at,
            date_range_start, date_range_end,
            num_tickers, num_rows, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (exchange_id, source, now, date_range_start, date_range_end,
         num_tickers, num_rows, notes),
    )
    return cur.lastrowid


def save_data_source(conn: sqlite3.Connection, source: str,
                     exchange_id: Optional[int] = None,
                     date_range_start: Optional[str] = None,
                     date_range_end: Optional[str] = None,
                     num_tickers: Optional[int] = None,
                     num_rows: Optional[int] = None,
                     notes: Optional[str] = None) -> int:
    """Record a data download event. Returns data_source id."""
    with conn:
        return _save_data_source_no_commit(
            conn, source, exchange_id, date_range_start, date_range_end,
            num_tickers, num_rows, notes,
        )


def get_latest_data_source(conn: sqlite3.Connection, source: Optional[str] = None) -> Optional[sqlite3.Row]:
    """Get the most recent data source entry."""
    if source:
        return conn.execute(
            "SELECT * FROM data_sources WHERE source = ? ORDER BY id DESC LIMIT 1",
            (source,),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM data_sources ORDER BY id DESC LIMIT 1"
    ).fetchone()


def get_latest_forecast(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    """Get the most recent forecast run."""
    return conn.execute(
        "SELECT * FROM forecast_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
