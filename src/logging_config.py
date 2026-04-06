"""
Centralised logging configuration for the portfolio optimisation pipeline.

Usage in entry points:
    from src.logging_config import setup_logging
    setup_logging()

Environment variables:
    LOG_LEVEL   — DEBUG, INFO (default), WARNING, ERROR, CRITICAL
    LOG_FORMAT  — 'text' (default) or 'json'
"""

import functools
import json as _json
import logging
import os
import sys
import time


_configured = False


class _JsonFormatter(logging.Formatter):
    """Outputs each log record as a single JSON line."""

    def format(self, record):
        entry = {
            'ts': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'msg': record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry['exception'] = self.formatException(record.exc_info)
        return _json.dumps(entry)


_TEXT_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'


def setup_logging(level=None):
    """
    Configure the root logger. Safe to call multiple times (idempotent).

    :param level: optional override (e.g. logging.DEBUG). If None, reads
                  LOG_LEVEL env var, defaulting to INFO.
    """
    global _configured
    if _configured:
        return
    _configured = True

    if level is None:
        env_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, env_level, logging.INFO)

    use_json = os.environ.get('LOG_FORMAT', 'text').lower() == 'json'

    handler = logging.StreamHandler(sys.stderr)
    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Quieten noisy third-party loggers
    for name in ('urllib3', 'matplotlib', 'yfinance', 'peewee'):
        logging.getLogger(name).setLevel(logging.WARNING)


def log_timing(func):
    """Decorator that logs function entry and elapsed time at INFO level."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info("Starting %s", func.__qualname__)
        start = time.time()
        try:
            result = func(*args, **kwargs)
            logger.info("Completed %s in %.1fs", func.__qualname__,
                        time.time() - start)
            return result
        except Exception:
            logger.warning("Failed %s after %.1fs", func.__qualname__,
                           time.time() - start, exc_info=True)
            raise

    return wrapper
