"""
Download package for Yahoo Finance price data.

Re-exports the public API from sub-modules for convenient access.
"""

from src.download.core import (
    download_data,
    download_and_save,
    _download_batch,
    _download_batch_with_timeout,
)
from src.download.cli import main, _add_file_logging, _log_final_summary
from src.download.session import (
    set_proxy_state,
    is_rate_limit_error,
    _make_session,
    _rotate_tor_circuit,
)
from src.download.validate import validate_tickers, _retry_with_splitting
from src.download.workers import concurrent_download_and_save

# Re-export universe symbols that were previously proxied via download_data.__getattr__
from src.universe import (
    ASSET_TYPE_MAP,
    ALL_ASSET_TYPES,
    get_equities,
    get_etfs,
    get_funds,
    get_cryptos,
    get_currencies,
    build_security_universe,
    filter_unwanted_tickers,
    load_tickers,
)

__all__ = [
    # core
    'download_data', 'download_and_save',
    # cli
    'main',
    # session
    'set_proxy_state', 'is_rate_limit_error',
    # validate
    'validate_tickers',
    # workers
    'concurrent_download_and_save',
    # universe (re-exports)
    'ASSET_TYPE_MAP', 'ALL_ASSET_TYPES',
    'get_equities', 'get_etfs', 'get_funds', 'get_cryptos', 'get_currencies',
    'build_security_universe', 'filter_unwanted_tickers', 'load_tickers',
]
