"""Tests for src.download_session — session management, proxy, and Tor circuit control."""

import logging
import unittest
from unittest.mock import MagicMock, patch

import src.download.session as sess


class TestSetProxyState(unittest.TestCase):
    """Test set_proxy_state correctly sets module globals."""

    def setUp(self):
        # Save originals to restore after test
        self._orig_proxy = sess._proxy_url
        self._orig_tor = sess._tor_enabled
        self._orig_counter = sess._proxy_session_counter

    def tearDown(self):
        sess._proxy_url = self._orig_proxy
        sess._tor_enabled = self._orig_tor
        sess._proxy_session_counter = self._orig_counter

    def test_set_proxy_state(self):
        sess.set_proxy_state('socks5://127.0.0.1:9050', True, 42)
        self.assertEqual(sess._proxy_url, 'socks5://127.0.0.1:9050')
        self.assertTrue(sess._tor_enabled)
        self.assertEqual(sess._proxy_session_counter, 42)

    def test_set_proxy_state_none(self):
        sess.set_proxy_state(None, False, 0)
        self.assertIsNone(sess._proxy_url)
        self.assertFalse(sess._tor_enabled)
        self.assertEqual(sess._proxy_session_counter, 0)


class TestMakeSession(unittest.TestCase):
    """Test _make_session with and without proxy."""

    def setUp(self):
        self._orig_proxy = sess._proxy_url
        self._orig_tor = sess._tor_enabled
        self._orig_counter = sess._proxy_session_counter

    def tearDown(self):
        sess._proxy_url = self._orig_proxy
        sess._tor_enabled = self._orig_tor
        sess._proxy_session_counter = self._orig_counter

    @patch('src.download.session.CffiSession')
    def test_make_session_no_proxy(self, mock_session_cls):
        """No proxy set: session created with Chrome impersonation, no proxies."""
        sess._proxy_url = None
        mock_instance = MagicMock(spec=[])  # empty spec: no auto-attributes
        mock_session_cls.return_value = mock_instance

        result = sess._make_session()

        mock_session_cls.assert_called_once_with(impersonate='chrome')
        self.assertIs(result, mock_instance)
        # proxies should not have been set on the instance
        self.assertFalse(hasattr(mock_instance, 'proxies'))

    @patch('src.download.session.CffiSession')
    def test_make_session_with_proxy(self, mock_session_cls):
        """Proxy set: session gets proxies dict."""
        sess._proxy_url = 'socks5://127.0.0.1:9050'
        mock_instance = MagicMock()
        # Start without proxies attribute to verify it gets set
        del mock_instance.proxies
        mock_session_cls.return_value = mock_instance

        result = sess._make_session()

        mock_session_cls.assert_called_once_with(impersonate='chrome')
        self.assertEqual(mock_instance.proxies, {
            'http': 'socks5://127.0.0.1:9050',
            'https': 'socks5://127.0.0.1:9050',
        })

    @patch('src.download.session.CffiSession')
    def test_make_session_proxy_rotation(self, mock_session_cls):
        """Residential proxy URL with digit suffix gets counter rotated."""
        sess._proxy_url = 'http://user-11:pass@proxy.example.com:8080'
        sess._proxy_session_counter = 99
        mock_instance = MagicMock()
        del mock_instance.proxies
        mock_session_cls.return_value = mock_instance

        sess._make_session()

        # Counter should have been incremented to 100
        expected_url = 'http://user-100:pass@proxy.example.com:8080'
        self.assertEqual(mock_instance.proxies['http'], expected_url)
        self.assertEqual(mock_instance.proxies['https'], expected_url)
        self.assertEqual(sess._proxy_session_counter, 100)


class TestRotateTorCircuit(unittest.TestCase):
    """Test _rotate_tor_circuit sends NEWNYM and handles failures."""

    @patch('src.download.session.logger')
    def test_rotate_tor_circuit_success(self, mock_logger):
        """Successful circuit rotation sends NEWNYM signal."""
        mock_controller = MagicMock()
        mock_signal_mod = MagicMock()
        mock_control_mod = MagicMock()
        mock_control_mod.Controller.from_port.return_value = mock_controller

        with patch.dict('sys.modules', {
            'stem': mock_signal_mod,
            'stem.control': mock_control_mod,
        }), \
             patch('src.config.TOR_CONTROL_PORT', 9051), \
             patch('src.config.TOR_CONTROL_PASSWORD', 'testpass'):

            sess._rotate_tor_circuit()

            mock_control_mod.Controller.from_port.assert_called_once_with(port=9051)
            mock_controller.__enter__().authenticate.assert_called_once_with(
                password='testpass')
            mock_controller.__enter__().signal.assert_called_once_with(
                mock_signal_mod.Signal.NEWNYM)

    @patch('src.download.session.logger')
    def test_rotate_tor_circuit_failure(self, mock_logger):
        """Connection refused logs warning, doesn't raise."""
        mock_signal_mod = MagicMock()
        mock_control_mod = MagicMock()
        mock_control_mod.Controller.from_port.side_effect = ConnectionRefusedError(
            "Connection refused")

        with patch.dict('sys.modules', {
            'stem': mock_signal_mod,
            'stem.control': mock_control_mod,
        }), \
             patch('src.config.TOR_CONTROL_PORT', 9051), \
             patch('src.config.TOR_CONTROL_PASSWORD', ''):

            # Should not raise
            sess._rotate_tor_circuit()

            mock_logger.warning.assert_called_once()
            self.assertIn('Tor circuit rotation failed',
                          mock_logger.warning.call_args[0][0])


class TestGetState(unittest.TestCase):
    """Test _get_state fallback logic."""

    def setUp(self):
        self._orig_proxy = sess._proxy_url

    def tearDown(self):
        sess._proxy_url = self._orig_proxy
        # Clean up any mock attributes on download.core
        import sys
        core = sys.modules.get('src.download.core')
        if core is not None and '_proxy_url' in core.__dict__:
            del core.__dict__['_proxy_url']
        dd = sys.modules.get('src.download_data')
        if dd is not None and '_proxy_url' in dd.__dict__:
            del dd.__dict__['_proxy_url']

    def test_get_state_fallback(self):
        """_get_state reads from src.download.core.__dict__ first,
        then falls back to module globals."""
        import sys
        import src.download.core as core

        # Set the module global
        sess._proxy_url = 'http://module-global.example.com'

        # Without core override, should read module global
        # First, ensure core doesn't have _proxy_url set
        if '_proxy_url' in core.__dict__:
            del core.__dict__['_proxy_url']

        result = sess._get_state('_proxy_url')
        self.assertEqual(result, 'http://module-global.example.com')

        # Now set it on core — should prefer that
        core._proxy_url = 'http://download-core-override.example.com'
        result = sess._get_state('_proxy_url')
        self.assertEqual(result, 'http://download-core-override.example.com')


if __name__ == '__main__':
    unittest.main()
