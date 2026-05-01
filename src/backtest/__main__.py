import argparse

from src.logging_config import setup_logging
from src.backtest.runner import main, main_cpcv

setup_logging()

parser = argparse.ArgumentParser(description='Backtest portfolio strategies.')
parser.add_argument('--mode', choices=('walkforward', 'cpcv'),
                    default='walkforward',
                    help='walkforward (default): rolling 5y train + Nmo OOS '
                         'windows. cpcv: Combinatorially Purged CV per '
                         'López de Prado (2018).')
parser.add_argument('--n-groups', type=int, default=None,
                    help='CPCV: number of contiguous groups (default from '
                         'config.CPCV_N_GROUPS).')
parser.add_argument('--k-test', type=int, default=None,
                    help='CPCV: groups per test set (default from '
                         'config.CPCV_K_TEST_GROUPS).')
parser.add_argument('--purge-days', type=int, default=None,
                    help='CPCV: trading days to purge around test groups.')
parser.add_argument('--embargo-days', type=int, default=None,
                    help='CPCV: trading days to embargo after test groups.')
args = parser.parse_args()

if args.mode == 'cpcv':
    main_cpcv(
        n_groups=args.n_groups,
        k_test_groups=args.k_test,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
    )
else:
    main()
