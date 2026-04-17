"""Backward-compatibility stub — real implementation in src.download.session."""
from __future__ import annotations
from typing import Any

from src.download.session import (
    is_rate_limit_error,
    set_proxy_state,
    _get_state,
    _make_session,
    _rotate_tor_circuit,
    _proxy_url,
    _tor_enabled,
    _proxy_session_counter,
    _proxy_session_counter_lock,
)


def __getattr__(name: str) -> Any:
    import src.download.session as _mod
    return getattr(_mod, name)
