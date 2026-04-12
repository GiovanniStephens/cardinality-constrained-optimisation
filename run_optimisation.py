"""Run C++ Island GA optimisation with Metal GPU on DB price data.

Usage:
    python run_optimisation.py [--time-budget SECONDS] [--no-gpu]
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

BINARY_PATH = os.path.join(os.path.dirname(__file__), 'cpp', 'optimisation')

# GA parameters (match ISLAND_GA config defaults)
DEFAULT_POP_SIZE = 10000      # per island (C++ convention)
DEFAULT_GENERATIONS = 10000
DEFAULT_MIN_ETFS = 15
DEFAULT_MAX_ETFS = 25
DEFAULT_MIN_RETURN = 0.17
DEFAULT_NUM_ELITES = 100
DEFAULT_MIGRATION_INTERVAL = 10
DEFAULT_MIGRATION_RATE = 0.1
DEFAULT_TIME_BUDGET = 600


def parse_args():
    p = argparse.ArgumentParser(description='Run C++ Island GA portfolio optimisation')
    p.add_argument('--time-budget', type=float, default=DEFAULT_TIME_BUDGET,
                   help='Time budget in seconds (default: %(default)s)')
    p.add_argument('--no-gpu', action='store_true',
                   help='Disable Metal GPU acceleration')
    p.add_argument('--pop-size', type=int, default=DEFAULT_POP_SIZE,
                   help='Population per island (default: %(default)s)')
    p.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS,
                   help='Generations per island (default: %(default)s)')
    p.add_argument('--seed', type=int, default=-1,
                   help='Random seed, -1 for random (default: %(default)s)')
    p.add_argument('--lookback-days', type=int, default=None,
                   help='Calendar days of history to use (default: all available)')
    return p.parse_args()


def run_cpp_ga(binary_data_path, args):
    """Invoke C++ binary and parse results."""
    cmd = [
        BINARY_PATH,
        '--binary',
        '--data', binary_data_path,
        '--mode', 'ga',
        '--pop-size', str(args.pop_size),
        '--generations', str(args.generations),
        '--min-etfs', str(DEFAULT_MIN_ETFS),
        '--max-etfs', str(DEFAULT_MAX_ETFS),
        '--num-elites', str(DEFAULT_NUM_ELITES),
        '--migration-interval', str(DEFAULT_MIGRATION_INTERVAL),
        '--migration-rate', str(DEFAULT_MIGRATION_RATE),
        '--min-return', str(DEFAULT_MIN_RETURN),
        '--time-budget', str(args.time_budget),
        '--seed', str(args.seed),
        '--mutation-initial', '0.008',
        '--mutation-final', '0.002',
        '--stagnation-restart', '500',
        '--top-k', '100',
    ]
    if not args.no_gpu:
        cmd.append('--gpu')

    logger.info("Running: %s", ' '.join(cmd))

    stderr_pattern = re.compile(
        r'Island\s+(\d+):\s+Generation\s+(\d+):\s+'
        r'Best fitness\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
    )

    best_so_far = float('-inf')
    start_time = time.time()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    stdout_data = []

    def read_stdout():
        for line in proc.stdout:
            stdout_data.append(line)

    stdout_thread = threading.Thread(target=read_stdout)
    stdout_thread.start()

    last_log_time = start_time
    for line in proc.stderr:
        match = stderr_pattern.match(line.strip())
        if match:
            fitness = float(match.group(3))
            if fitness > best_so_far:
                best_so_far = fitness
            now = time.time()
            if now - last_log_time >= 5.0:
                logger.info("  %.1fs elapsed — best Sharpe so far: %.4f",
                            now - start_time, best_so_far)
                last_log_time = now
        else:
            # Pass through non-convergence stderr (loading messages, etc.)
            sys.stderr.write(line)

    proc.wait(timeout=args.time_budget + 30)
    stdout_thread.join(timeout=5)
    elapsed = time.time() - start_time

    stdout_text = ''.join(stdout_data)
    result_json = None
    try:
        result_json = json.loads(stdout_text)
    except (json.JSONDecodeError, ValueError):
        logger.error("Failed to parse C++ JSON output")
        if stdout_text.strip():
            logger.error("Raw output: %s", stdout_text[:500])

    logger.info("C++ GA completed in %.1fs — best equal-weight Sharpe: %.4f",
                elapsed, best_so_far)

    return result_json, best_so_far, elapsed


def slsqp_refine(result_json, data, group_constraints=None,
                 group_membership=None):
    """Run SLSQP weight refinement on top-K solutions from C++ output."""
    from src.weights import optimise_weights

    top_solutions = result_json.get('top_solutions', [])
    if not top_solutions:
        logger.warning("No top_solutions in C++ output")
        return None, None, None

    logger.info("  SLSQP refinement on %d candidate portfolios...", len(top_solutions))
    all_tickers = list(data.columns)
    best_fitness = float('-inf')
    best_etfs = None
    best_weights = None

    n_total = len(top_solutions)
    for i, sol in enumerate(top_solutions):
        if n_total > 50 and (i + 1) % 100 == 0:
            logger.info("  SLSQP progress: %d/%d candidates evaluated (best=%.4f)",
                        i + 1, n_total, best_fitness if best_fitness > float('-inf') else 0.0)
        tickers = sol.get('tickers', [])
        if len(tickers) < 2:
            continue
        selection = np.zeros(len(all_tickers), dtype=int)
        for t in tickers:
            if t in all_tickers:
                selection[all_tickers.index(t)] = 1
        if np.sum(selection) < 2:
            continue
        try:
            opt = optimise_weights(
                selection, data,
                group_constraints=group_constraints,
                group_membership=group_membership,
                selected_tickers=tickers if group_constraints else None,
            )
            if opt.success and -opt.fun > best_fitness:
                best_fitness = -opt.fun
                best_etfs = tickers
                best_weights = opt.x
                logger.info("  SLSQP solution %d/%d: Sharpe=%.4f (%d holdings)",
                            i + 1, n_total, best_fitness, len(tickers))
        except Exception as e:
            logger.debug("SLSQP failed for solution %d: %s", i + 1, e)

    return best_fitness, best_etfs, best_weights


def main():
    args = parse_args()

    if not os.path.isfile(BINARY_PATH):
        logger.error("C++ binary not found at %s — run 'make build-cpp'", BINARY_PATH)
        sys.exit(1)

    # Load price data from DB
    from src.data_loading import load_prices
    lookback = args.lookback_days
    label = f'{lookback} day' if lookback else 'full history'
    logger.info("Loading price data from DB (%s)...", label)
    data = load_prices(exchange='US', last_n_days=lookback)
    if data.empty:
        logger.error("No price data available")
        sys.exit(1)
    logger.info("Loaded %d dates x %d tickers (%s to %s)",
                data.shape[0], data.shape[1],
                data.index[0].date(), data.index[-1].date())

    # Write binary data for C++
    from src.returns import calculate_log_returns
    from src.binary_io import write_binary_data

    log_returns = calculate_log_returns(data)

    tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    tmp.close()
    try:
        logger.info("Writing binary data (%d x %d)...", *log_returns.shape)
        write_binary_data(log_returns, tmp.name)

        # Run C++ GA
        result_json, best_ew_sharpe, ga_elapsed = run_cpp_ga(tmp.name, args)

        if result_json is None:
            logger.error("C++ binary produced no output")
            sys.exit(1)

        # Load group constraints if configured
        from src.config import GROUP_CONSTRAINTS
        gc = GROUP_CONSTRAINTS
        gm = None
        if gc:
            from src.group_constraints import load_membership
            from src import db as _db
            _conn = _db.get_connection()
            gm = load_membership(_conn, list(data.columns), exchange='US')
            _conn.close()
            logger.info("Group constraints active: %s", list(gc.keys()))

        # SLSQP weight refinement
        logger.info("Running SLSQP weight refinement on top solutions...")
        best_fitness, best_etfs, best_weights = slsqp_refine(
            result_json, data, group_constraints=gc, group_membership=gm,
        )

        if best_etfs is None:
            # Fall back to C++ best solution with equal weights
            best_etfs = result_json.get('selected_tickers', [])
            if best_etfs:
                n = len(best_etfs)
                best_weights = np.full(n, 1.0 / n)
                best_fitness = best_ew_sharpe
                logger.warning("SLSQP refinement failed — using equal weights")
            else:
                logger.error("No valid solution found")
                sys.exit(1)

        # Check group constraints on final portfolio
        if gc and gm and best_etfs:
            from src.group_constraints import check_constraints
            valid, violations = check_constraints(
                best_etfs, best_weights, gm, gc,
            )
            if valid:
                logger.info("Group constraints: all satisfied")
            else:
                for v in violations:
                    logger.warning("Group constraint violation: %s", v)

        # Print results
        logger.info("")
        logger.info("=" * 60)
        logger.info("PORTFOLIO OPTIMISATION RESULTS")
        logger.info("=" * 60)
        logger.info("Equal-weight Sharpe (C++ GA): %.4f", best_ew_sharpe)
        logger.info("Optimised Sharpe (SLSQP):     %.4f", best_fitness)
        logger.info("Holdings: %d", len(best_etfs))
        logger.info("")
        logger.info("%-10s %8s %10s", "Ticker", "Weight%", "$(20,000)")
        logger.info("-" * 30)
        for ticker, weight in sorted(zip(best_etfs, best_weights),
                                     key=lambda x: -x[1]):
            if weight > 1e-4:
                logger.info("%-10s %7.1f%% %10.2f",
                            ticker, weight * 100, weight * 20000)

        # Save to DB
        from src import db
        from src.portfolio_utils import save_optimisation_result

        conn = db.get_connection()
        run_id = save_optimisation_result(
            conn, best_etfs, best_weights, data,
            script_name='cpp_island_ga',
            params={
                'data_source': 'us_db',
                'pop_size': args.pop_size,
                'generations': args.generations,
                'min_etfs': DEFAULT_MIN_ETFS,
                'max_etfs': DEFAULT_MAX_ETFS,
                'min_return': DEFAULT_MIN_RETURN,
                'time_budget': args.time_budget,
                'gpu': not args.no_gpu,
                'seed': args.seed,
            },
            exchange='US',
            elapsed_seconds=ga_elapsed,
        )
        conn.close()
        logger.info("Run saved to database (id=%d)", run_id)

    finally:
        os.unlink(tmp.name)


if __name__ == '__main__':
    main()
