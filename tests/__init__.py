import os
import unittest

requires_integration = unittest.skipUnless(
    os.environ.get('RUN_INTEGRATION'),
    'Set RUN_INTEGRATION=1 to run',
)

requires_network = unittest.skipUnless(
    os.environ.get('RUN_NETWORK'),
    'Set RUN_NETWORK=1 to run (hits Yahoo Finance / FinanceDatabase)',
)
