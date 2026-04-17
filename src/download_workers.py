"""Backward-compatibility stub — real implementation in src.download.workers."""
from __future__ import annotations
from typing import Any

from src.download.workers import (
    concurrent_download_and_save,
    _partition_tickers,
    _reset_yf_singleton,
    _subprocess_worker,
    _worker_download,
    _db_writer,
    _concurrent_thread_download,
    _concurrent_subprocess_download,
)


def __getattr__(name: str) -> Any:
    import src.download.workers as _mod
    return getattr(_mod, name)
