"""Data source and metadata functions."""

import logging

from src.db.connection import _now

logger = logging.getLogger(__name__)


def _save_data_source_no_commit(conn, source, exchange_id=None,
                                date_range_start=None, date_range_end=None,
                                num_tickers=None, num_rows=None, notes=None):
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


def save_data_source(conn, source, exchange_id=None, date_range_start=None,
                     date_range_end=None, num_tickers=None, num_rows=None,
                     notes=None):
    """Record a data download event. Returns data_source id."""
    with conn:
        return _save_data_source_no_commit(
            conn, source, exchange_id, date_range_start, date_range_end,
            num_tickers, num_rows, notes,
        )


def get_latest_data_source(conn, source=None):
    """Get the most recent data source entry."""
    if source:
        return conn.execute(
            "SELECT * FROM data_sources WHERE source = ? ORDER BY id DESC LIMIT 1",
            (source,),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM data_sources ORDER BY id DESC LIMIT 1"
    ).fetchone()


def get_latest_forecast(conn):
    """Get the most recent forecast run."""
    return conn.execute(
        "SELECT * FROM forecast_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
