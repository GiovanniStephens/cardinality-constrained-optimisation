"""
Session management, proxy rotation, and Tor circuit control for downloads.

Provides process-local proxy state, curl_cffi session creation with Chrome
TLS impersonation, and Tor NEWNYM circuit rotation.
"""

import logging
import random
import re
import threading

from curl_cffi.requests import Session as CffiSession

logger = logging.getLogger(__name__)

_proxy_url = None   # Set from CLI: --proxy or --use-tor
_tor_enabled = False  # Tor-specific: enables NEWNYM circuit rotation
_proxy_session_counter = random.randint(0, 999_999)  # Random seed to avoid reusing burned session IDs
_proxy_session_counter_lock = threading.Lock()


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception indicates Yahoo Finance rate limiting."""
    error_msg = str(error).lower()
    return '429' in error_msg or 'too many' in error_msg or 'rate' in error_msg


def set_proxy_state(proxy_url, tor_enabled, counter_start):
    """Set process-local proxy globals.

    Called by subprocess workers (in download_workers) after fork to
    initialise this module's state in the child process.
    """
    global _proxy_url, _tor_enabled, _proxy_session_counter
    _proxy_url = proxy_url
    _tor_enabled = tor_enabled
    _proxy_session_counter = counter_start


def _get_state(name):
    """Return the current value of a session state variable.

    Checks ``src.download.core``'s own ``__dict__`` first (tests may set
    attributes there directly), then falls back to this module's global.
    """
    import sys
    core = sys.modules.get('src.download.core')
    if core is not None and name in core.__dict__:
        return core.__dict__[name]
    return globals()[name]


def _make_session():
    """Create a curl_cffi Session with Chrome TLS impersonation.

    Chrome impersonation is critical — Yahoo's bot detection blocks requests
    with non-browser TLS fingerprints, even from residential proxy IPs.

    For rotating residential proxies: if the proxy URL contains a username
    ending in digits (e.g. ``mdgihswf-11``), the trailing number is replaced
    with a per-session counter so each batch gets a distinct exit IP.
    """
    global _proxy_session_counter
    session = CffiSession(impersonate='chrome')
    proxy = _get_state('_proxy_url')
    if proxy:
        url = proxy
        # Rotate proxy username suffix for residential proxies
        # e.g. http://user-11:pass@host → http://user-42:pass@host
        if re.match(r'https?://[^:]*-\d+:', url):
            with _proxy_session_counter_lock:
                _proxy_session_counter += 1
                counter = _proxy_session_counter
            url = re.sub(r'(-)\d+:', rf'\g<1>{counter}:', url, count=1)
        session.proxies = {'http': url, 'https': url}
    return session


def _rotate_tor_circuit():
    """Request a new Tor circuit (new exit IP) via the ControlPort."""
    try:
        from stem import Signal
        from stem.control import Controller
        from src.config import TOR_CONTROL_PORT, TOR_CONTROL_PASSWORD
        with Controller.from_port(port=TOR_CONTROL_PORT) as controller:
            if TOR_CONTROL_PASSWORD:
                controller.authenticate(password=TOR_CONTROL_PASSWORD)
            else:
                controller.authenticate()
            controller.signal(Signal.NEWNYM)
    except Exception as e:
        logger.warning("Tor circuit rotation failed: %s", e)
