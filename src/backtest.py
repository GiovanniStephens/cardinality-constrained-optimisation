"""Forward-walk backtesting with hypothesis testing.

Orchestrates portfolio creation (GA, MC, random), weight optimisation,
and OOS evaluation across rolling windows. See CLAUDE.md "Sharpe Ratio
Overfitting" for details on IS vs OOS Sharpe degradation.
"""
import logging
import multiprocessing as mp
import time
import uuid
from multiprocessing import cpu_count
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.returns import calculate_log_returns, calculate_expected_returns
from src.covariance import calculate_covariance_matrix
from src.weights import optimise_weights
from src.metrics import warn_if_sharpe_suspicious
from src.config import (
    BACKTEST_NUM_PORTFOLIOS,
    BACKTEST_NUM_CHILDREN,
    BACKTEST_MC_TRIALS,
    BACKTEST_FORECAST_WINDOWS,
    BACKTEST_MAX_WEIGHT_FLOOR,
    BACKTEST_MIN_METHODS_FOR_STATS,
    TRADING_DAYS_PER_YEAR,
    GA_MIN_SECURITIES,
    GA_MAX_SECURITIES,
    GA_MIN_WEIGHT,
    GA_MAX_WEIGHT,
    GA_NUM_GENERATIONS,
    NZ_ETF_PRICES_CSV,
)

from src.backtest_types import WindowSpec, PortfolioResult, MethodResults, WindowResult
from src.backtest_windows import generate_windows, slice_window_data, aggregate_cross_window
from src.backtest_statistics import difference_of_means_hypothesis_test, paired_t_test, friedman_test
from src.backtest_simulation import METRIC_NAMES, get_random_weights, run_portfolio, get_statistics  # noqa: F401 — public API

logger = logging.getLogger(__name__)

NUM_JOBS = cpu_count()


# ─── Module-level state for multiprocessing ─────────────────────────────────
#
# These globals are set before spawning multiprocessing.Pool workers via the
# _init_worker() and _init_weight_worker() initializer functions.  On macOS
# (spawn start method), each worker process re-imports this module, so the
# initializer copies data into these globals in the child process's address
# space.  On Linux (fork), they are inherited from the parent.
#
# This pattern avoids serialising large DataFrames through Pool.map() args
# (which would pickle them per task).  Instead, each worker reads from its
# own copy of these module globals.
#
# The lifecycle is:
#   1. Parent sets globals or passes them via Pool(initializer=..., initargs=...)
#   2. Workers read globals during task execution
#   3. Pool closes → workers terminate → globals are garbage collected
#
# Thread safety: each Pool creates isolated worker processes, so there are
# no race conditions.  However, these globals must NOT be modified after the
# Pool is created — only read.
_backtest_data = None
_use_forecast = False
_backtest_log_returns = None       # Transposed log returns: tickers × dates
_backtest_expected_returns = None  # Series of annualised expected returns


# ─── Pool Worker Helpers ─────────────────────────────────────────────────────


def optimal_weights(portfolio, use_copulae=False):
    """
    Finds the optimal weights (allocations) for the input portfolio.

    Reads from module-level globals set by _init_weight_worker().

    :portfolio: The input portfolio. List of ticker strings.
    :use_copulae: Whether to use copulae or not.
    :return: A list of weights for the input portfolio.
    """
    if len(portfolio) < 2:
        raise ValueError("Portfolio must contain at least 2 assets.")
    missing = set(portfolio) - set(_backtest_log_returns.index)
    if missing:
        raise KeyError(f"Tickers not found in data: {missing}")
    subset = _backtest_log_returns.loc[portfolio, :].transpose()
    er = _backtest_expected_returns.loc[subset.columns].values
    max_weight = max(1 / (len(portfolio) - 1), BACKTEST_MAX_WEIGHT_FLOOR)

    if use_copulae:
        from src.covariance import estimate_corr_using_copulas
        corr = estimate_corr_using_copulas(subset)
        D = np.diag(subset.std().values * np.sqrt(TRADING_DAYS_PER_YEAR))
        cov = np.matmul(np.matmul(D, corr), D)
    else:
        cov = calculate_covariance_matrix(subset).values

    result = optimise_weights(
        expected_returns=er, cov_matrix=cov,
        max_weight=max_weight,
        initial_weights=get_random_weights(portfolio),
    )
    if not result.success:
        logger.warning("Weight optimization did not converge: %s", result.message)
    return result['x']


def _random_selection(num_tickers, min_k, max_k, ticker_names):
    """Generate a random portfolio by selecting min_k..max_k tickers."""
    k = np.random.randint(min_k, max_k + 1)
    chosen = np.random.choice(num_tickers, k, replace=False)
    return [ticker_names[i] for i in chosen]


def _init_worker(training_data, use_forecast):
    """Pool initializer — sets module globals in each worker process."""
    global _backtest_data, _use_forecast
    _backtest_data = training_data
    _use_forecast = use_forecast


def _init_weight_worker(log_returns_T, expected_returns):
    """Pool initializer for weight computation workers."""
    global _backtest_log_returns, _backtest_expected_returns
    _backtest_log_returns = log_returns_T
    _backtest_expected_returns = expected_returns


def _compute_weights_for_portfolio(args):
    """Top-level function for Pool.map — computes weights for a single portfolio.

    Must be top-level (not a lambda/closure) for macOS spawn-based multiprocessing.
    """
    portfolio, mode = args
    if mode == 'random':
        return get_random_weights(portfolio)
    elif mode == 'copulae':
        return optimal_weights(portfolio, use_copulae=True)
    else:  # 'optimal'
        return optimal_weights(portfolio, use_copulae=False)


def create_portfolio(num_children):
    """
    Creates a cardinality-constrained portfolio with the training data.

    :num_children: The number of children in the GA to create.
    :return: A list of tickers.
    """
    from src.optimisers.pygad_ga import PygadOptimiser
    opt = PygadOptimiser(
        num_children=num_children,
        num_generations=GA_NUM_GENERATIONS,
        min_securities=GA_MIN_SECURITIES,
        max_securities=GA_MAX_SECURITIES,
        min_weight=GA_MIN_WEIGHT,
        max_weight=GA_MAX_WEIGHT,
        target_return=None,
        use_forecasts=_use_forecast,
    )
    result = opt.optimise(_backtest_data)
    return result.selected_tickers


# ─── Portfolio Creation ──────────────────────────────────────────────────────


def create_random_portfolios(columns, num_portfolios, min_securities,
                             max_securities):
    """Create random portfolios by selecting random subsets of tickers.

    :param columns: Index or list of available ticker names.
    :param num_portfolios: number of portfolios to create.
    :param min_securities: minimum number of securities per portfolio.
    :param max_securities: maximum number of securities per portfolio.
    :return: list of portfolios (each a list of ticker strings).
    """
    return [
        list(_random_selection(len(columns), min_securities, max_securities,
                               columns))
        for _ in range(num_portfolios)
    ]


def evaluate_portfolios(portfolios, weights_list, oos_log_returns,
                        train_log_returns, category):
    """Evaluate portfolios OOS and return a MethodResults object.

    :param portfolios: list of portfolios (each a list of ticker strings).
    :param weights_list: list of weight arrays.
    :param oos_log_returns: OOS log returns DataFrame.
    :param train_log_returns: training log returns for IS diagnostic.
    :param category: category name string.
    :return: MethodResults object with all portfolio results.
    """
    prs = _evaluate_oos(portfolios, weights_list, oos_log_returns,
                        train_log_returns)
    return MethodResults(category=category, portfolios=prs)


def _create_ga_portfolios(train_prices, num_portfolios, num_children,
                           use_forecast):
    """Create portfolios via GA using a worker pool."""
    label = "forecast" if use_forecast else "no forecast"
    logger.info("  Creating %d GA portfolios (%s)...", num_portfolios, label)
    start = time.time()
    with mp.Pool(
        processes=NUM_JOBS,
        initializer=_init_worker,
        initargs=(train_prices, use_forecast),
    ) as pool:
        portfolios = pool.map(create_portfolio, [num_children] * num_portfolios)
    logger.info("  GA (%s) done in %.1fs", label, time.time() - start)
    return portfolios


def _create_mc_portfolios(train_prices, num_portfolios, mc_trials):
    """Create portfolios via Monte Carlo search."""
    from src.optimisers import monte_carlo as mc

    logger.info("  Creating %d MC portfolios (%d trials)...",
                num_portfolios, mc_trials)
    start = time.time()
    portfolios = []
    for _ in tqdm(range(num_portfolios), desc="  MC portfolios", leave=False):
        solution, _ = mc.monte_carlo_search(
            train_prices, mc_trials,
            min_num_etfs=GA_MIN_SECURITIES,
            max_num_etfs=GA_MAX_SECURITIES,
        )
        if solution is not None:
            portfolios.append(list(train_prices.columns[solution == 1]))
        else:
            portfolios.append(list(_random_selection(
                train_prices.shape[1], GA_MIN_SECURITIES, GA_MAX_SECURITIES,
                train_prices.columns)))
    logger.info("  MC done in %.1fs", time.time() - start)
    return portfolios


def _optimise_all_weights(categories, log_returns_T, expected_returns):
    """Compute weights for all (portfolio, mode) pairs in parallel."""
    weight_tasks = []
    task_metadata = []
    for cat_name, portfolios, mode in categories:
        for i, p in enumerate(portfolios):
            weight_tasks.append((p, mode))
            task_metadata.append((cat_name, i))

    with mp.Pool(
        processes=NUM_JOBS,
        initializer=_init_weight_worker,
        initargs=(log_returns_T, expected_returns),
    ) as pool:
        all_weights = pool.map(_compute_weights_for_portfolio, weight_tasks)

    category_weights = {cat_name: [] for cat_name, _, _ in categories}
    for (cat_name, _idx), w in zip(task_metadata, all_weights):
        category_weights[cat_name].append(w)
    return category_weights


# ─── OOS Evaluation ──────────────────────────────────────────────────────────


def _evaluate_oos(portfolios, weights_list, oos_log_returns,
                  train_log_returns):
    """Evaluate portfolios out-of-sample and compute IS Sharpe diagnostic."""
    results = []
    for p, w in zip(portfolios, weights_list):
        metrics = get_statistics(p, w, oos_log_returns)
        try:
            is_stats = get_statistics(p, w, train_log_returns)
            is_sr = is_stats['sharpe_ratio']
        except Exception:
            logger.debug("IS Sharpe computation failed for portfolio", exc_info=True)
            is_sr = None
        results.append(PortfolioResult(
            portfolio=p, weights=w, metrics=metrics, is_sharpe=is_sr,
        ))
    return results


def _log_window_summary(window, result):
    """Log per-window IS vs OOS Sharpe comparison."""
    logger.info("  Window %s results (%.1fs):",
                window.label, result.elapsed_seconds)
    for cat, mr in sorted(result.method_results.items()):
        is_sharpes = [p.is_sharpe for p in mr.portfolios
                      if p.is_sharpe is not None]
        if is_sharpes:
            mean_is = np.mean(is_sharpes)
            mean_oos = mr.mean_sharpe
            degradation = (((mean_is - mean_oos) / mean_is * 100)
                           if mean_is > 0 else float('nan'))
            logger.info(
                "    %-25s  IS_sharpe=%.4f  OOS_sharpe=%.4f  "
                "degradation=%.0f%%",
                cat, mean_is, mean_oos, degradation,
            )
            warn_if_sharpe_suspicious(
                mean_is, f"Window {window.label} {cat} IS", logger)
        else:
            logger.info("    %-25s  OOS_sharpe=%.4f  std=%.4f",
                         cat, mr.mean_sharpe, mr.sharpe_ratios.std())


def evaluate_window(
    window: WindowSpec,
    full_prices: pd.DataFrame,
    conn,
    num_portfolios: int = BACKTEST_NUM_PORTFOLIOS,
    num_children: int = BACKTEST_NUM_CHILDREN,
    mc_trials: int = BACKTEST_MC_TRIALS,
    use_forecast: bool = False,
) -> WindowResult:
    """
    Run the full backtest for a single rolling window.

    Orchestrates portfolio creation (GA, MC, random), weight optimisation,
    and OOS evaluation.
    """
    window_start = time.time()
    result = WindowResult(window=window)

    # ── Slice data ────────────────────────────────────────────────────────
    train_prices, oos_log_returns = slice_window_data(window, full_prices)

    # ── Prepare optimisation state for weight workers ─────────────────────
    global _backtest_log_returns, _backtest_expected_returns
    log_returns_train = calculate_log_returns(train_prices)
    _backtest_log_returns = log_returns_train.transpose()
    _backtest_expected_returns = calculate_expected_returns(log_returns_train)

    # ── Create portfolios ─────────────────────────────────────────────────
    ga_portfolios = _create_ga_portfolios(
        train_prices, num_portfolios, num_children, use_forecast=False)
    forecast_portfolios = (
        _create_ga_portfolios(
            train_prices, num_portfolios, num_children, use_forecast=True)
        if use_forecast else []
    )
    random_portfolios = [
        list(_random_selection(train_prices.shape[1], GA_MIN_SECURITIES,
                               GA_MAX_SECURITIES, train_prices.columns))
        for _ in range(num_portfolios)
    ]
    mc_portfolios = _create_mc_portfolios(
        train_prices, num_portfolios, mc_trials)

    # ── Optimise weights in parallel ──────────────────────────────────────
    logger.info("  Optimising weights and running OOS evaluation...")
    categories = [
        ('cc_optimised',      ga_portfolios,     'optimal'),
        ('cc_copulae',        ga_portfolios,     'copulae'),
        ('cc_random_weights', ga_portfolios,     'random'),
        ('mc_optimised',      mc_portfolios,     'optimal'),
        ('mc_random_weights', mc_portfolios,     'random'),
        ('random_optimised',  random_portfolios, 'optimal'),
        ('random_random',     random_portfolios, 'random'),
    ]
    cat_weights = _optimise_all_weights(
        categories, _backtest_log_returns, _backtest_expected_returns)

    # ── Evaluate OOS ──────────────────────────────────────────────────────
    train_log_returns = calculate_log_returns(train_prices)
    for cat_name, portfolios, _mode in categories:
        prs = _evaluate_oos(
            portfolios, cat_weights[cat_name],
            oos_log_returns, train_log_returns)
        result.method_results[cat_name] = MethodResults(
            category=cat_name, portfolios=prs)

    if use_forecast and forecast_portfolios:
        fc_weights = _optimise_all_weights(
            [('cc_forecast', forecast_portfolios, 'optimal')],
            _backtest_log_returns, _backtest_expected_returns)
        prs = _evaluate_oos(
            forecast_portfolios, fc_weights['cc_forecast'],
            oos_log_returns, train_log_returns)
        result.method_results['cc_forecast'] = MethodResults(
            category='cc_forecast', portfolios=prs)

    result.elapsed_seconds = time.time() - window_start
    _log_window_summary(window, result)
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    from src.logging_config import setup_logging
    setup_logging()

    from src import db
    from src.data_loading import load_training_data

    bt_start = time.time()

    conn = db.get_connection()
    data = load_training_data(
        exchange='US', csv_fallback=NZ_ETF_PRICES_CSV, lookback_days=None)

    # ── Generate rolling windows ──────────────────────────────────────────
    windows = generate_windows(data.index)
    logger.info("Generated %d rolling windows:", len(windows))
    for w in windows:
        logger.info("  %s: train %s to %s, test %s to %s",
                     w.label,
                     w.train_start.date(), w.train_end.date(),
                     w.test_start.date(), w.test_end.date())

    forecast_labels = set(BACKTEST_FORECAST_WINDOWS)

    # ── Evaluate each window ──────────────────────────────────────────────
    run_group = str(uuid.uuid4())
    all_results: List[WindowResult] = []

    for window in windows:
        use_forecast = window.label in forecast_labels
        logger.info("=" * 60)
        logger.info("Evaluating window: %s (forecast=%s)", window.label, use_forecast)

        wr = evaluate_window(
            window=window,
            full_prices=data,
            conn=conn,
            use_forecast=use_forecast,
        )
        all_results.append(wr)

        # Save per-window results to DB
        session_id = db.save_backtest_session(conn, {
            'data_source': 'yahoo_finance',
            'num_portfolios': BACKTEST_NUM_PORTFOLIOS,
            'num_days_oos': len(data.loc[window.test_start:window.test_end]),
            'use_forecast': use_forecast,
            'optimiser_params': {
                'num_children': BACKTEST_NUM_CHILDREN,
                'mc_trials_per_portfolio': BACKTEST_MC_TRIALS,
            },
            'elapsed_seconds': wr.elapsed_seconds,
            'window_train_start': str(window.train_start.date()),
            'window_train_end': str(window.train_end.date()),
            'window_test_start': str(window.test_start.date()),
            'window_test_end': str(window.test_end.date()),
            'window_label': window.label,
            'run_group_id': run_group,
        })
        for cat, mr in wr.method_results.items():
            for i, pr in enumerate(mr.portfolios):
                db.save_backtest_result(conn, session_id, cat, i,
                    metrics=pr.metrics,
                    holdings=list(zip(pr.portfolio, pr.weights)))
        logger.info("  Saved to DB (session id=%d)", session_id)

    # ── Cross-window aggregation ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CROSS-WINDOW SUMMARY")
    summary_df = aggregate_cross_window(all_results)
    logger.info("\n%s", summary_df.to_string())

    # ── Within-window hypothesis tests ────────────────────────────────────
    logger.info("WITHIN-WINDOW HYPOTHESIS TESTS (per window):")
    comparisons = [
        ('CC optimised vs Random random', 'cc_optimised', 'random_random'),
        ('MC optimised vs Random random', 'mc_optimised', 'random_random'),
        ('CC optimised vs MC optimised',  'cc_optimised', 'mc_optimised'),
        ('CC copulae vs CC optimised',    'cc_optimised', 'cc_copulae'),
    ]
    for wr in all_results:
        logger.info("  Window %s:", wr.window.label)
        for label, cat_a, cat_b in comparisons:
            if cat_a in wr.method_results and cat_b in wr.method_results:
                s1 = wr.method_results[cat_a].sharpe_ratios.tolist()
                s2 = wr.method_results[cat_b].sharpe_ratios.tolist()
                try:
                    t = difference_of_means_hypothesis_test(s1, s2)
                    logger.info("    %-40s  t=%.4f", label, t)
                except ValueError as e:
                    logger.warning("    %-40s  %s", label, e)

    # ── Cross-window paired tests ─────────────────────────────────────────
    logger.info("CROSS-WINDOW PAIRED TESTS (positive t = second > first):")
    all_categories = set()
    for wr in all_results:
        all_categories.update(wr.method_results.keys())
    # Only test categories present in all windows
    core_categories = [c for c in sorted(all_categories)
                       if all(c in wr.method_results for wr in all_results)]

    paired_comparisons = [
        ('CC optimised vs Random random', 'cc_optimised', 'random_random'),
        ('MC optimised vs Random random', 'mc_optimised', 'random_random'),
        ('CC optimised vs MC optimised',  'cc_optimised', 'mc_optimised'),
        ('CC copulae vs CC optimised',    'cc_optimised', 'cc_copulae'),
    ]
    for label, cat_a, cat_b in paired_comparisons:
        if cat_a in core_categories and cat_b in core_categories:
            a_sharpes = {wr.window.label: wr.method_results[cat_a].mean_sharpe
                         for wr in all_results}
            b_sharpes = {wr.window.label: wr.method_results[cat_b].mean_sharpe
                         for wr in all_results}
            try:
                t_stat, p_val = paired_t_test(a_sharpes, b_sharpes)
                logger.info("  %-40s  t=%.4f  p=%.4f", label, t_stat, p_val)
            except ValueError as e:
                logger.warning("  %-40s  %s", label, e)

    # ── Friedman omnibus test ─────────────────────────────────────────────
    if len(core_categories) >= BACKTEST_MIN_METHODS_FOR_STATS and len(all_results) >= 2:
        try:
            chi2, p_val = friedman_test(all_results, core_categories)
            logger.info("FRIEDMAN TEST: chi2=%.4f  p=%.4f", chi2, p_val)
        except ValueError as e:
            logger.warning("Friedman test skipped: %s", e)

    bt_elapsed = time.time() - bt_start
    logger.info("Full rolling backtest completed in %.1fs (%d windows)",
                bt_elapsed, len(windows))
    conn.close()


if __name__ == '__main__':
    main()
