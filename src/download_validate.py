"""Backward-compatibility stub — real implementation in src.download.validate."""
from __future__ import annotations
from typing import Any

from src.download.validate import validate_tickers, _retry_with_splitting

# Re-export _sess so test patches on ``src.download_validate._sess`` still work
from src.download import session as _sess


def __getattr__(name: str) -> Any:
    import src.download.validate as _mod
    return getattr(_mod, name)
