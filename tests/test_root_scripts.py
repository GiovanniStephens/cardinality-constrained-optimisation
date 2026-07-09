"""Import smoke tests for the root-level CLI scripts.

The root scripts are outside the src/ package and coverage-exempt, so nothing
else catches drift between them (July 2026: backtest_rebalance.py crashed with
TypeError after run_rebalance.run_cpp_ga grew a third required argument, and
the breakage sat undetected because no test imported the file). Importing each
script exercises its module-level code and every cross-script attribute access
at definition time — cheap insurance against signature/import drift.

Scripts must remain import-safe (all work behind ``if __name__ == '__main__'``).
"""

import importlib
import unittest

ROOT_SCRIPTS = [
    'run_rebalance',            # production quarterly rebalance
    'backtest_rebalance',       # honest OOS gate for the production config
    'run_benchmark',
    'run_throughput_benchmark',
    'run_leverage_analysis',
    'run_sleeve_experiment',
    'sleeve_reality_check',
    'curate_universe',
    'build_dedup_map',
]


class TestRootScriptsImport(unittest.TestCase):
    def test_all_root_scripts_import(self):
        for name in ROOT_SCRIPTS:
            with self.subTest(script=name):
                importlib.import_module(name)


if __name__ == '__main__':
    unittest.main()
