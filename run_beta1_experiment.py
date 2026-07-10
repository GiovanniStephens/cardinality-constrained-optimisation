"""Run the beta-1 / information-ratio backtest experiment.

"Can we beat the market while holding the market's own risk level?" (Crack,
*Scientific Investments* — factor tilting at beta 1). Enables the ``cc_beta1``
arm — kept OFF by default in ``config.BACKTEST_RUN_BETA1_STRATEGY`` so normal
backtests are unaffected — and runs the existing walk-forward or CPCV
pipeline. The arm is max-Sharpe SLSQP on the GA baskets with portfolio beta
pinned to 1.0 vs SPY (train-slice betas, per-basket feasibility clamp); the
information ratio vs SPY is reported for EVERY method, so cc_beta1 is judged
against the same-window IR of the whole strategy family and the bench_spy
IR ≈ 0 sanity check.

    python run_beta1_experiment.py                  # walk-forward (fast pre-check)
    python run_beta1_experiment.py --mode cpcv      # CPCV (trusted verdict, ~10h)
    python run_beta1_experiment.py --with-forecasts # also include ARIMA/GARCH arms

Run ``python beta1_reality_check.py`` first — it answers the cheap questions
(is beta 1 even reachable on GA baskets, what does the pin cost in-sample)
in minutes, before paying for a backtest. Per the repo's Lesson 5, the
walk-forward IR will be regime-inflated; CPCV is the verdict.
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
    # Enable the beta-1 arm for this run only (config default stays False).
    runner.BACKTEST_RUN_BETA1_STRATEGY = True
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
