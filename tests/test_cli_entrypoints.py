"""P2.2 — CLI entry point smoke tests for __main__.py modules."""

import subprocess
import sys
import unittest


class TestDbCli(unittest.TestCase):
    """Smoke tests for python -m src.db entry point."""

    def test_help_does_not_crash(self):
        """Running db module with no args should not crash (creates DB)."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.db'],
            capture_output=True, text=True, timeout=30,
        )
        # Exit code 0 means success
        self.assertEqual(result.returncode, 0, result.stderr)


class TestDownloadCli(unittest.TestCase):
    """Smoke tests for python -m src.download entry point."""

    def test_help_flag(self):
        """--help should print usage and exit 0."""
        result = subprocess.run(
            [sys.executable, '-m', 'src.download', '--help'],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('usage', result.stdout.lower())


if __name__ == '__main__':
    unittest.main()
