"""Run the managed-futures (TSMOM) sleeve backtest experiment.

Enables the sleeve A/B arms — which are kept OFF by default in
``config.BACKTEST_RUN_SLEEVE_STRATEGIES`` so normal backtests are unaffected —
and runs the existing walk-forward or CPCV pipeline. Each sleeve arm appears
beside its base method (e.g. ``cc_copulae`` vs ``cc_copulae_trend25``), so the
existing DSR / PBO / 95%-CI reporting compares them head-to-head on identical
windows/splits.

    python run_sleeve_experiment.py                  # walk-forward (fast pre-check)
    python run_sleeve_experiment.py --mode cpcv      # CPCV (trusted verdict, ~10h)
    python run_sleeve_experiment.py --with-forecasts # also include ARIMA/GARCH arms

By default the ARIMA/GARCH forecast strategy family is DISABLED: it needs
``pmdarima``, and the project's own notes find forecasting hurts OOS in this
universe. It is orthogonal to the sleeve question. Pass ``--with-forecasts`` to
include it.
"""

import argparse

from src.logging_config import setup_logging
from src.backtest import runner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('walkforward', 'cpcv'),
                    default='walkforward')
    ap.add_argument('--with-forecasts', action='store_true',
                    help='include ARIMA/GARCH forecast strategies (needs pmdarima)')
    ap.add_argument('--n-groups', type=int, default=None)
    ap.add_argument('--k-test', type=int, default=None)
    ap.add_argument('--purge-days', type=int, default=None)
    ap.add_argument('--embargo-days', type=int, default=None)
    args = ap.parse_args()

    setup_logging()
    # Enable the sleeve arms for this run only (config default stays False).
    runner.BACKTEST_RUN_SLEEVE_STRATEGIES = True
    if not args.with_forecasts:
        runner.BACKTEST_RUN_FORECAST_STRATEGIES = False

    if args.mode == 'cpcv':
        runner.main_cpcv(
            n_groups=args.n_groups, k_test_groups=args.k_test,
            purge_days=args.purge_days, embargo_days=args.embargo_days)
    else:
        runner.main()


if __name__ == '__main__':
    main()
