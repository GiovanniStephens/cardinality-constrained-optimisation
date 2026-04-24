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

    When a proxy is configured, also disables yfinance's on-disk cookie
    cache. yfinance persists a Yahoo session cookie to
    ~/Library/Caches/py-yfinance/cookies.db and reloads it on every
    new YfData instance — which defeats IP rotation because Yahoo
    ties our "client identity" to the cached cookie (and therefore the
    crumb token derived from it). With the cache disabled, every yf
    call fetches a fresh cookie through whatever proxy IP is active,
    so each request looks like a distinct client to Yahoo.
    """
    global _proxy_url, _tor_enabled, _proxy_session_counter
    _proxy_url = proxy_url
    _tor_enabled = tor_enabled
    _proxy_session_counter = counter_start

    if proxy_url:
        try:
            import yfinance.data as _yd
            _yd.YfData._load_cookie_curlCffi = lambda self: False
            _yd.YfData._save_cookie_curlCffi = lambda self: False
        except Exception:
            pass


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
    """Create a curl_cffi Session with a rotated browser TLS impersonation.

    Browser impersonation is critical — Yahoo's Akamai blocks requests with
    non-browser TLS fingerprints. But a SINGLE fixed target (``impersonate='chrome'``)
    is itself a fingerprint: Akamai cross-correlates JA3/JA4 across our
    residential IPs and sees "many IPs, one client". Rotating through a pool
    of Chrome / Safari / Edge versions breaks that signal.

    Each call also:
      - rotates the Webshare residential-proxy username suffix (if present)
        so the exit IP differs from the previous session's
      - sets a varied Accept-Language + Referer so headers aren't byte-identical
      - warms up the session with a GET to finance.yahoo.com so Akamai sees
        browser-like site-first traffic rather than pure API-hammering
    """
    global _proxy_session_counter
    from src import config as _cfg

    # Pick TLS impersonation target pseudorandomly per session
    target = random.choice(_cfg.IMPERSONATE_TARGETS) if _cfg.IMPERSONATE_TARGETS else 'chrome'
    try:
        session = CffiSession(impersonate=target)
    except Exception as e:
        logger.debug("Impersonate target %r rejected (%s); falling back to 'chrome'", target, e)
        session = CffiSession(impersonate='chrome')

    # Vary a couple of browser-signal headers per session
    try:
        session.headers['Accept-Language'] = random.choice(_cfg.ACCEPT_LANGUAGE_POOL)
        session.headers['Referer'] = 'https://finance.yahoo.com/'
    except Exception:
        pass

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

    # Warm up: issue one GET to finance.yahoo.com so the session picks up
    # legitimate Akamai cookies and looks like a browser that loaded the
    # site before hitting the API. Failures are swallowed — the warmup is
    # belt-and-braces, not a hard requirement.
    if _cfg.WARMUP_URL:
        try:
            session.get(_cfg.WARMUP_URL, timeout=10)
        except Exception as e:
            logger.debug("Warmup request failed (%s); continuing anyway", e)

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
