"""Backfill missing ticker names from the Nasdaq Trader symbol directory.

The Nasdaq-directory ETF ingest (July 2026) writes symbols and prices but no
names, and FinanceDatabase has never heard of post-2021 launches — leaving
``tickers.name`` NULL. That breaks the crude name classifier
(:func:`src.categorise.classify_etf`): unnamed funds all land in the capped
``Unknown`` bucket, which made the rebalance category caps infeasible against
GA selections (July 2026 incident — SLSQP could never converge).

This module fetches the nightly symbol directory files
(``nasdaqtrader.com/dynamic/SymDir/{nasdaqlisted,otherlisted}.txt``), parses
Symbol → Security Name, and fills in names for rows where ``name`` is NULL or
empty. Existing names are never overwritten. Re-runnable; run it after any
symbol-directory ingest:

    python -m src.db backfill-names
    python -m src.db backfill-names --from-files a.txt b.txt   # offline re-run
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

NASDAQ_DIRECTORY_URLS = (
    'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt',
    'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt',
)


def parse_directory(text: str) -> dict[str, str]:
    """Parse a pipe-delimited symbol directory into {symbol: security_name}.

    Both files put the symbol in column 0 and the security name in column 1
    (``nasdaqlisted.txt``: Symbol; ``otherlisted.txt``: ACT Symbol). The header
    row and the trailing "File Creation Time" row are skipped.
    """
    names: dict[str, str] = {}
    for line in text.splitlines()[1:]:
        parts = line.split('|')
        if len(parts) < 2:
            continue
        sym = parts[0].strip()
        if not sym or sym.startswith('File Creation'):
            continue
        name = parts[1].strip()
        if name:
            names.setdefault(sym, name)
    return names


def fetch_directory_names(urls: Iterable[str] = NASDAQ_DIRECTORY_URLS,
                          timeout: int = 30) -> dict[str, str]:
    """Download and parse the symbol directory files. Later files never
    overwrite earlier symbols (nasdaqlisted takes precedence)."""
    import urllib.request
    names: dict[str, str] = {}
    for url in urls:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode('utf-8', errors='replace')
        parsed = parse_directory(text)
        for sym, name in parsed.items():
            names.setdefault(sym, name)
        logger.info("Parsed %d names from %s", len(parsed), url)
    return names


def backfill_names(conn: sqlite3.Connection,
                   names: Optional[dict[str, str]] = None,
                   exchange: str = 'US') -> dict:
    """Fill in ``tickers.name`` where NULL/empty from a {symbol: name} map.

    Never overwrites an existing name. Returns stats:
    {missing, matched, updated}.
    """
    if names is None:
        names = fetch_directory_names()
    from src.db.connection import _get_exchange_id
    exchange_id = _get_exchange_id(conn, exchange)
    rows = conn.execute(
        "SELECT id, symbol FROM tickers WHERE exchange_id = ? "
        "AND (name IS NULL OR name = '') AND symbol NOT LIKE '%.%'",
        (exchange_id,)).fetchall()
    updated = 0
    for r in rows:
        name = names.get(r['symbol'])
        if name:
            conn.execute("UPDATE tickers SET name = ? WHERE id = ?",
                         (name, r['id']))
            updated += 1
    conn.commit()
    stats = {'missing': len(rows), 'matched': updated, 'updated': updated}
    logger.info("Name backfill: %d tickers missing a name, %d filled from "
                "the directory", len(rows), updated)
    return stats
