"""Stage-0 diagnostic for the beta-1 / information-ratio experiment.

NON-BLOCKING plumbing + feasibility gate (not the experiment verdict — that
is the CPCV run via ``run_beta1_experiment.py``). Answers the cheap questions
in minutes, before paying hours for a backtest:

1. Is beta = 1.0 vs SPY even REACHABLE on realistic GA baskets? The GA is
   beta-unaware and historically drifts low-beta, so the pin may have to be
   clamped to each basket's reachable interval.
2. Does the SLSQP equality actually bind (achieved beta == target), or do
   baskets fall into the 1/N fallback?
3. What does the pin cost in-sample — IR and Sharpe of the pinned book vs the
   unpinned max-Sharpe book on the same baskets? (In-sample, so inflated in
   LEVEL; the pinned-vs-unpinned CONTRAST is the informative part.)

Run:  python beta1_reality_check.py            (needs cpp/optimisation built)
"""

import json
import logging
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from src import db
from src.backtest import simulation
from src.config import (
    BACKTEST_BETA1_TARGET,
    BACKTEST_IR_BENCHMARK,
    BACKTEST_MAX_WEIGHT_FLOOR,
    CPP_BINARY_PATH,
    DATA_FFILL_LIMIT,
    DATA_MIN_COVERAGE,
    ISLAND_GA_MIGRATION_INTERVAL,
    ISLAND_GA_MIGRATION_RATE,
    ISLAND_GA_MIN_SECURITIES,
    ISLAND_GA_MAX_SECURITIES,
    ISLAND_GA_MUTATION_RATE_FINAL,
    ISLAND_GA_MUTATION_RATE_INITIAL,
    ISLAND_GA_NUM_ELITES,
    RISK_FREE_RATE,
)
from src.binary_io import write_binary_data
from src.returns import (
    calculate_asset_betas,
    calculate_expected_returns,
    calculate_log_returns,
)
from src.weights import reachable_beta_interval

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TOP_K = 30          # baskets to diagnose
POP_SIZE = 2000     # reduced GA fidelity — this is a plumbing check,
GENERATIONS = 200   # not a search-quality benchmark


def _load_prices():
    """Full-history US prices, cleaned the way the backtest runner cleans."""
    conn = db.get_connection()
    try:
        data = db.load_prices(conn, exchange='US')
    finally:
        conn.close()
    if data.empty:
        raise SystemExit("No price data in the DB — run `make refresh` first.")
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data.dropna(axis=1, thresh=int(DATA_MIN_COVERAGE * len(data)))
    return data.ffill(limit=DATA_FFILL_LIMIT)


def _run_ga_top_k(log_returns, top_k=TOP_K):
    """One reduced-fidelity C++ GA call; returns the top-k ticker baskets.

    Modelled on ``simulation.create_portfolio`` but with ``--top-k`` > 1 and
    a smaller population/generation budget.
    """
    if not os.path.exists(CPP_BINARY_PATH):
        raise SystemExit(
            f"{CPP_BINARY_PATH} not found — build it with `make build-cpp`.")
    fd, bin_path = tempfile.mkstemp(suffix='.bin', prefix='beta1_ga_')
    os.close(fd)
    try:
        write_binary_data(log_returns, bin_path)
        cmd = [
            CPP_BINARY_PATH, '--binary', '--data', bin_path,
            '--mode', 'ga',
            '--pop-size', str(POP_SIZE),
            '--generations', str(GENERATIONS),
            '--num-islands', '1',
            '--num-elites', str(ISLAND_GA_NUM_ELITES),
            '--migration-interval', str(ISLAND_GA_MIGRATION_INTERVAL),
            '--migration-rate', str(ISLAND_GA_MIGRATION_RATE),
            '--mutation-initial', str(ISLAND_GA_MUTATION_RATE_INITIAL),
            '--mutation-final', str(ISLAND_GA_MUTATION_RATE_FINAL),
            '--min-etfs', str(ISLAND_GA_MIN_SECURITIES),
            '--max-etfs', str(ISLAND_GA_MAX_SECURITIES),
            '--risk-free-rate', str(RISK_FREE_RATE),
            '--top-k', str(top_k),
            '--seed', '-1',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"cpp/optimisation failed (rc={result.returncode}): "
                f"{result.stderr[-500:]}")
        out = json.loads(result.stdout)
        baskets = [sol['tickers'] for sol in out.get('top_solutions') or []
                   if len(sol.get('tickers') or []) >= 2]
        if not baskets:
            raise SystemExit(
                f"GA returned no baskets: {result.stdout[:500]}")
        return baskets
    finally:
        try:
            os.unlink(bin_path)
        except OSError:
            pass


def main():
    logger.info("Loading full-history prices...")
    prices = _load_prices()
    logger.info("Price frame: %d rows x %d columns", *prices.shape)
    if BACKTEST_IR_BENCHMARK not in prices.columns:
        raise SystemExit(
            f"{BACKTEST_IR_BENCHMARK} missing from the cleaned price frame — "
            "the IR benchmark must be present (check the bad-ticker cache).")

    log_returns = calculate_log_returns(prices)
    # The simulation weight helpers read these module globals (the same
    # fixture pattern the mode-dispatch tests use).
    simulation._backtest_log_returns = log_returns.transpose()
    simulation._backtest_expected_returns = calculate_expected_returns(
        log_returns)
    betas = calculate_asset_betas(
        log_returns, log_returns[BACKTEST_IR_BENCHMARK])
    spy = log_returns[BACKTEST_IR_BENCHMARK]

    logger.info("Running one reduced-fidelity GA (pop=%d, gens=%d, top-k=%d)"
                "...", POP_SIZE, GENERATIONS, TOP_K)
    baskets = _run_ga_top_k(log_returns)
    logger.info("GA returned %d baskets\n", len(baskets))

    rows = []
    for i, basket in enumerate(baskets):
        b = betas.loc[basket].values.astype(float)
        max_w = max(1 / (len(basket) - 1), BACKTEST_MAX_WEIGHT_FLOOR)
        reach_lo, reach_hi = reachable_beta_interval(b, max_w)
        target = float(np.clip(BACKTEST_BETA1_TARGET, reach_lo, reach_hi))
        clamped = target != BACKTEST_BETA1_TARGET

        w_pin = simulation._beta1_weights(
            basket, asset_betas=b, target_beta=target)
        achieved = float(np.dot(b, w_pin))
        pinned = simulation.get_statistics(basket, w_pin, log_returns,
                                           benchmark_returns=spy)

        w_free = simulation._max_sharpe_weights(basket)
        free = simulation.get_statistics(basket, w_free, log_returns,
                                         benchmark_returns=spy)
        free_beta = float(np.dot(b, w_free))

        rows.append({
            'n': len(basket), 'reach_lo': reach_lo, 'reach_hi': reach_hi,
            'clamped': clamped, 'target': target, 'achieved': achieved,
            'pin_ok': abs(achieved - target) < 1e-3,
            'ir_pinned': pinned['information_ratio'],
            'sharpe_pinned': pinned['sharpe_ratio'],
            'ir_free': free['information_ratio'],
            'sharpe_free': free['sharpe_ratio'],
            'beta_free': free_beta,
        })
        logger.info(
            "  basket %2d (n=%2d)  reach=[%+.2f, %+.2f]%s  achieved=%+.3f%s"
            "  IS: IR %+.2f (free %+.2f)  Sharpe %.2f (free %.2f, "
            "free beta %+.2f)",
            i, len(basket), reach_lo, reach_hi,
            "  CLAMPED->%.2f" % target if clamped else "",
            achieved, "" if rows[-1]['pin_ok'] else "  OFF-TARGET(1/N?)",
            rows[-1]['ir_pinned'], rows[-1]['ir_free'],
            rows[-1]['sharpe_pinned'], rows[-1]['sharpe_free'], free_beta)

    df = pd.DataFrame(rows)
    n = len(df)
    feasible = int((~df['clamped']).sum())
    bound = int(df['pin_ok'].sum())
    logger.info("\n%s", "=" * 66)
    logger.info("STAGE-0 SUMMARY (%d GA baskets, full history, in-sample)", n)
    logger.info("  beta %.2f reachable unclamped : %d/%d (%.0f%%)",
                BACKTEST_BETA1_TARGET, feasible, n, 100 * feasible / n)
    logger.info("  pin bound by SLSQP           : %d/%d", bound, n)
    logger.info("  achieved beta                : mean %.3f  min %.3f  max %.3f",
                df['achieved'].mean(), df['achieved'].min(),
                df['achieved'].max())
    logger.info("  unpinned max-Sharpe beta     : mean %+.3f (the drift the "
                "pin corrects)", df['beta_free'].mean())
    logger.info("  IS IR pinned                 : median %+.3f  [min %+.3f, "
                "max %+.3f]", df['ir_pinned'].median(),
                df['ir_pinned'].min(), df['ir_pinned'].max())
    logger.info("  IS IR unpinned               : median %+.3f",
                df['ir_free'].median())
    logger.info("  IS Sharpe pinned vs free     : median %.3f vs %.3f "
                "(the in-sample cost of the pin)",
                df['sharpe_pinned'].median(), df['sharpe_free'].median())

    ok_feas = feasible >= 0.7 * n
    ok_bind = bound == n
    ok_ir = df['ir_pinned'].median() > 0
    verdict = "PASS" if (ok_feas and ok_bind and ok_ir) else "WEAK/CHECK"
    logger.info("  verdict: %s  (feasibility%s, binding%s, median IS IR>0%s)"
                " — plumbing gate only; the experiment verdict is the CPCV "
                "run.", verdict,
                "+" if ok_feas else "-", "+" if ok_bind else "-",
                "+" if ok_ir else "-")


if __name__ == '__main__':
    main()
