"""Backward-compatibility stub — real implementation in src.download.cli."""
from __future__ import annotations
from typing import Any

from src.download.cli import main, _add_file_logging, _log_final_summary


def __getattr__(name: str) -> Any:
    import src.download.cli as _mod
    return getattr(_mod, name)
