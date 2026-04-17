"""Backward-compatibility stub -- real implementation in src.download.core."""
from __future__ import annotations
from typing import Any

# Re-export eagerly loaded names so they're importable without __getattr__
from src.download.core import (
    download_data,
    download_and_save,
    _download_batch,
    _download_batch_with_timeout,
)


def __getattr__(name: str) -> Any:
    # Try core module first (download functions)
    import src.download.core as _core
    try:
        return getattr(_core, name)
    except AttributeError:
        pass
    # Try session module (proxy state, session creation, Tor)
    import src.download.session as _session
    try:
        return getattr(_session, name)
    except AttributeError:
        pass
    # Try validate module
    import src.download.validate as _validate
    try:
        return getattr(_validate, name)
    except AttributeError:
        pass
    # Try workers module
    import src.download.workers as _workers
    try:
        return getattr(_workers, name)
    except AttributeError:
        pass
    # Try CLI module
    import src.download.cli as _cli
    try:
        return getattr(_cli, name)
    except AttributeError:
        pass
    # Try universe module (proxied in the original download_data)
    import src.universe as _universe
    try:
        return getattr(_universe, name)
    except AttributeError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == '__main__':
    from src.download.cli import main
    main()
