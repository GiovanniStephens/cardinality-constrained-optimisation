"""Forward-walk backtesting orchestrator.

Evaluates portfolios created via GA, Monte Carlo, and random selection
across rolling train/test windows. Computes OOS performance metrics and
runs statistical tests to compare methods.
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

from src.data_loading import load_data
from src.returns import calculate_log_returns, calculate_expected_returns
from src.metrics import warn_if_sharpe_suspicious
from src.config import (
    BACKTEST_NUM_PORTFOLIOS,
    BACKTEST_NUM_CHILDREN,
    BACKTEST_NUM_DAYS_OOS,
    BACKTEST_MC_TRIALS,
    BACKTEST_TRAIN_YEARS,
    BACKTEST_TEST_DAYS,
    BACKTEST_STEP_DAYS,
    BACKTEST_FORECAST_WINDOWS,
    TRADING_DAYS_PER_YEAR,
    DATA_MIN_COVERAGE,
    DATA_FFILL_LIMIT,
    GA_MIN_SECURITIES,
    GA_MAX_SECURITIES,
    NZ_ETF_PRICES_CSV,
)

from .types import WindowSpec, WindowResult
from .windows import generate_windows, slice_window_data
from .simulation import (
    METRIC_NAMES,
    get_random_weights,
    get_statistics,
    create_portfolio,
    create_random_portfolios,
    evaluate_portfolios,
    _random_selection,
    _init_worker,
    _init_weight_worker,
    _compute_weights_for_portfolio,
    _backtest_log_returns,
    _backtest_expected_returns,
)
from .statistics import (
    difference_of_means_hypothesis_test,
    paired_t_test,
    friedman_test,
    aggregate_cross_window,
)

logger = logging.getLogger(__name__)

NUM_JOBS = cpu_count()


# ---- Per-Window Evaluation ---------------------------------------------------


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

    Creates portfolios via GA, MC, and random selection, computes weights
    (optimal, copula, random), and evaluates OOS performance.

    OVERFITTING AWARENESS: The GA optimises on training data, producing
    in-sample (IS) Sharpe ratios that are biased upward due to selection
    bias. The OOS Sharpe ratios from the test period are the real measure
    of portfolio quality. Typical IS -> OOS degradation is 30-50%.
    See CLAUDE.md "Sharpe Ratio Overfitting" section.

    :param window: WindowSpec defining train/test boundaries.
    :param full_prices: complete price DataFrame (will be sliced).
    :param conn: sqlite3 connection (for forecast loading).
    :param num_portfolios: portfolios per method.
    :param num_children: GA population size.
    :param mc_trials: Monte Carlo trials per portfolio.
    :param use_forecast: whether to also run forecast-based GA.
    :return: WindowResult with all method results.
    """
    from src.backtest import simulation
    from src.optimisers import monte_carlo as mc

    window_start = time.time()
    result = WindowResult(window=window)

    # -- Slice data ------------------------------------------------------------
    train_prices, oos_log_returns = slice_window_data(window, full_prices)

    logger.info(
        "  Window %s: train=%d rows, test=%d rows, %d tickers",
        window.label, len(train_prices),
        len(full_prices.loc[window.test_start:window.test_end]),
        train_prices.shape[1],
    )

    # -- Prepare optimisation state for weight optimisation --------------------
    log_returns_train = calculate_log_returns(train_prices)
    simulation._backtest_log_returns = log_returns_train.transpose()
    simulation._backtest_expected_returns = calculate_expected_returns(
        log_returns_train)

    # -- Create GA portfolios (no forecast) ------------------------------------
    logger.info("  Creating %d GA portfolios (no forecast)...", num_portfolios)
    start = time.time()
    with mp.Pool(
        processes=NUM_JOBS,
        initializer=_init_worker,
        initargs=(train_prices, False),
    ) as pool:
        ga_portfolios = pool.map(create_portfolio, [num_children] * num_portfolios)
    logger.info("  GA (no forecast) done in %.1fs", time.time() - start)

    # -- Create GA portfolios (with forecast) if requested ---------------------
    forecast_portfolios = []
    if use_forecast:
        logger.info("  Creating %d GA portfolios (with forecast)...", num_portfolios)
        start = time.time()
        with mp.Pool(
            processes=NUM_JOBS,
            initializer=_init_worker,
            initargs=(train_prices, True),
        ) as pool:
            forecast_portfolios = pool.map(
                create_portfolio, [num_children] * num_portfolios
            )
        logger.info("  GA (forecast) done in %.1fs", time.time() - start)

    # -- Create random portfolios ----------------------------------------------
    random_portfolios = create_random_portfolios(
        train_prices.columns, num_portfolios)

    # -- Create MC portfolios --------------------------------------------------
    logger.info("  Creating %d MC portfolios (%d trials)...", num_portfolios, mc_trials)
    start = time.time()
    mc_portfolios = []
    for _ in tqdm(range(num_portfolios), desc="  MC portfolios", leave=False):
        solution, _ = mc.monte_carlo_search(
            train_prices, mc_trials,
            min_num_etfs=GA_MIN_SECURITIES,
            max_num_etfs=GA_MAX_SECURITIES,
        )
        if solution is not None:
            mc_portfolios.append(list(train_prices.columns[solution == 1]))
        else:
            mc_portfolios.append(list(_random_selection(
                train_prices.shape[1], GA_MIN_SECURITIES, GA_MAX_SECURITIES,
                train_prices.columns)))
    logger.info("  MC done in %.1fs", time.time() - start)

    # -- Compute in-sample Sharpe for each portfolio (overfitting diagnostic) --
    train_log_returns = calculate_log_returns(train_prices)

    # -- Compute weights in parallel -------------------------------------------
    logger.info("  Optimising weights and running OOS evaluation...")

    # Build all (portfolio, mode) work items
    weight_tasks = []
    task_metadata = []  # (category_name, portfolio_index)

    categories = [
        ('cc_optimised',      ga_portfolios,     'optimal'),
        ('cc_copulae',        ga_portfolios,     'copulae'),
        ('cc_random_weights', ga_portfolios,     'random'),
        ('mc_optimised',      mc_portfolios,     'optimal'),
        ('mc_random_weights', mc_portfolios,     'random'),
        ('random_optimised',  random_portfolios, 'optimal'),
        ('random_random',     random_portfolios, 'random'),
    ]

    for cat_name, portfolios, mode in categories:
        for i, p in enumerate(portfolios):
            weight_tasks.append((p, mode))
            task_metadata.append((cat_name, i))

    # Parallel weight computation (SLSQP releases GIL during Fortran calls)
    with mp.Pool(
        processes=NUM_JOBS,
        initializer=_init_weight_worker,
        initargs=(simulation._backtest_log_returns,
                  simulation._backtest_expected_returns),
    ) as pool:
        all_weights = pool.map(_compute_weights_for_portfolio, weight_tasks)

    # Reassemble results by category
    category_weights = {}
    category_portfolios = {}
    for (cat_name, portfolios, mode), _ in zip(categories, range(len(categories))):
        category_weights[cat_name] = []
        category_portfolios[cat_name] = portfolios

    for (cat_name, idx), w in zip(task_metadata, all_weights):
        category_weights[cat_name].append(w)

    for cat_name, portfolios, mode in categories:
        result.method_results[cat_name] = evaluate_portfolios(
            category_portfolios[cat_name], category_weights[cat_name],
            oos_log_returns, train_log_returns, cat_name)

    if use_forecast and forecast_portfolios:
        with mp.Pool(
            processes=NUM_JOBS,
            initializer=_init_weight_worker,
            initargs=(simulation._backtest_log_returns,
                      simulation._backtest_expected_returns),
        ) as pool:
            forecast_weights = pool.map(
                _compute_weights_for_portfolio,
                [(p, 'optimal') for p in forecast_portfolios],
            )
        result.method_results['cc_forecast'] = evaluate_portfolios(
            forecast_portfolios, forecast_weights,
            oos_log_returns, train_log_returns, 'cc_forecast')

    result.elapsed_seconds = time.time() - window_start

    # Log per-window summary with IS vs OOS comparison
    logger.info("  Window %s results (%.1fs):", window.label, result.elapsed_seconds)
    for cat, mr in sorted(result.method_results.items()):
        is_sharpes = [p.is_sharpe for p in mr.portfolios if p.is_sharpe is not None]
        if is_sharpes:
            mean_is = np.mean(is_sharpes)
            mean_oos = mr.mean_sharpe
            degradation = ((mean_is - mean_oos) / mean_is * 100) if mean_is > 0 else float('nan')
            logger.info(
                "    %-25s  IS_sharpe=%.4f  OOS_sharpe=%.4f  degradation=%.0f%%",
                cat, mean_is, mean_oos, degradation,
            )
            warn_if_sharpe_suspicious(mean_is, f"Window {window.label} {cat} IS", logger)
        else:
            logger.info("    %-25s  OOS_sharpe=%.4f  std=%.4f",
                         cat, mr.mean_sharpe, mr.sharpe_ratios.std())

    return result


# ---- Main --------------------------------------------------------------------


def main():
    from src import db

    bt_start = time.time()

    # -- Load prices from DB (CSV fallback) ------------------------------------
    conn = db.get_connection()
    data = db.load_prices(conn, exchange='US')
    if data.empty:
        logger.info("No data in DB, falling back to CSV")
        data = load_data(NZ_ETF_PRICES_CSV)
    else:
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        data = data.dropna(axis=1, thresh=int(DATA_MIN_COVERAGE * len(data)))
        data = data.ffill(limit=DATA_FFILL_LIMIT)
    logger.info("Loaded price data: %d rows x %d columns", *data.shape)

    # -- Generate rolling windows ----------------------------------------------
    windows = generate_windows(data.index)
    logger.info("Generated %d rolling windows:", len(windows))
    for w in windows:
        logger.info("  %s: train %s to %s, test %s to %s",
                     w.label,
                     w.train_start.date(), w.train_end.date(),
                     w.test_start.date(), w.test_end.date())

    forecast_labels = set(BACKTEST_FORECAST_WINDOWS)

    # -- Evaluate each window --------------------------------------------------
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

    _report_results(all_results)

    bt_elapsed = time.time() - bt_start
    logger.info("Full rolling backtest completed in %.1fs (%d windows)",
                bt_elapsed, len(windows))
    conn.close()


def _report_results(all_results: List[WindowResult]) -> None:
    """Cross-window aggregation, hypothesis tests, and Friedman omnibus test."""
    # -- Aggregation -----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("CROSS-WINDOW SUMMARY")
    summary_df = aggregate_cross_window(all_results)
    logger.info("\n%s", summary_df.to_string())

    comparisons = [
        ('CC optimised vs Random random', 'cc_optimised', 'random_random'),
        ('MC optimised vs Random random', 'mc_optimised', 'random_random'),
        ('CC optimised vs MC optimised',  'cc_optimised', 'mc_optimised'),
        ('CC copulae vs CC optimised',    'cc_optimised', 'cc_copulae'),
    ]

    # -- Within-window hypothesis tests ----------------------------------------
    logger.info("WITHIN-WINDOW HYPOTHESIS TESTS (per window):")
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

    # -- Cross-window paired tests ---------------------------------------------
    logger.info("CROSS-WINDOW PAIRED TESTS (positive t = second > first):")
    all_categories = set()
    for wr in all_results:
        all_categories.update(wr.method_results.keys())
    core_categories = [c for c in sorted(all_categories)
                       if all(c in wr.method_results for wr in all_results)]

    for label, cat_a, cat_b in comparisons:
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

    # -- Friedman omnibus test -------------------------------------------------
    if len(core_categories) >= 3 and len(all_results) >= 2:
        try:
            chi2, p_val = friedman_test(all_results, core_categories)
            logger.info("FRIEDMAN TEST: chi2=%.4f  p=%.4f", chi2, p_val)
        except ValueError as e:
            logger.warning("Friedman test skipped: %s", e)
