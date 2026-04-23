"""Tests for src/logging_config.py — logging setup, JSON formatter, timing decorator."""

import json
import logging
import os
import unittest


class TestSetupLogging(unittest.TestCase):
    """Tests for setup_logging() idempotency and configuration."""

    def setUp(self):
        import src.logging_config as lc
        self._module = lc
        self._orig_configured = lc._configured
        self._orig_handlers = logging.root.handlers[:]
        self._orig_level = logging.root.level
        # Reset so each test starts clean
        lc._configured = False
        logging.root.handlers = []

    def tearDown(self):
        self._module._configured = self._orig_configured
        logging.root.handlers = self._orig_handlers
        logging.root.level = self._orig_level

    def test_configures_root_logger(self):
        self._module.setup_logging()
        self.assertTrue(len(logging.root.handlers) > 0)
        self.assertTrue(self._module._configured)

    def test_idempotent(self):
        self._module.setup_logging()
        count1 = len(logging.root.handlers)
        self._module.setup_logging()
        count2 = len(logging.root.handlers)
        self.assertEqual(count1, count2)

    def test_respects_level_override(self):
        self._module.setup_logging(level=logging.DEBUG)
        self.assertEqual(logging.root.level, logging.DEBUG)

    def test_env_log_level(self):
        old = os.environ.get('LOG_LEVEL')
        try:
            os.environ['LOG_LEVEL'] = 'WARNING'
            self._module.setup_logging()
            self.assertEqual(logging.root.level, logging.WARNING)
        finally:
            if old is None:
                os.environ.pop('LOG_LEVEL', None)
            else:
                os.environ['LOG_LEVEL'] = old

    def test_json_format(self):
        old = os.environ.get('LOG_FORMAT')
        try:
            os.environ['LOG_FORMAT'] = 'json'
            self._module.setup_logging()
            handler = logging.root.handlers[0]
            self.assertIsInstance(handler.formatter, self._module._JsonFormatter)
        finally:
            if old is None:
                os.environ.pop('LOG_FORMAT', None)
            else:
                os.environ['LOG_FORMAT'] = old


class TestJsonFormatter(unittest.TestCase):
    """Tests for _JsonFormatter output structure."""

    def setUp(self):
        from src.logging_config import _JsonFormatter
        self.formatter = _JsonFormatter()

    def test_format_basic_record(self):
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='hello world', args=(), exc_info=None,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        self.assertEqual(data['level'], 'INFO')
        self.assertEqual(data['msg'], 'hello world')
        self.assertIn('ts', data)
        self.assertIn('logger', data)

    def test_format_with_exception(self):
        try:
            raise ValueError('test error')
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name='test', level=logging.ERROR, pathname='', lineno=0,
            msg='failure', args=(), exc_info=exc_info,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        self.assertIn('exception', data)
        self.assertIn('ValueError', data['exception'])


class TestLogTiming(unittest.TestCase):
    """Tests for the log_timing decorator."""

    def setUp(self):
        from src.logging_config import log_timing
        self.log_timing = log_timing

    def test_preserves_return_value(self):
        @self.log_timing
        def add(a, b):
            return a + b
        self.assertEqual(add(2, 3), 5)

    def test_logs_completion(self):
        @self.log_timing
        def noop():
            pass
        with self.assertLogs(level='INFO') as cm:
            noop()
        messages = ' '.join(cm.output)
        self.assertIn('noop', messages)

    def test_logs_failure_and_reraises(self):
        @self.log_timing
        def fail():
            raise RuntimeError('boom')

        with self.assertLogs(level='WARNING'):
            with self.assertRaises(RuntimeError):
                fail()


if __name__ == '__main__':
    unittest.main()
