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
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loading import load_data
from src.returns import calculate_log_returns, calculate_expected_returns
from src.metrics import (
    compute_method_dsr,
    effective_trials_for_method,
    warn_if_sharpe_suspicious,
)
from src.config import (
    BACKTEST_NUM_PORTFOLIOS,
    BACKTEST_NUM_CHILDREN,
    BACKTEST_NUM_DAYS_OOS,
    BACKTEST_MC_TRIALS,
    BACKTEST_TRAIN_YEARS,
    BACKTEST_TEST_DAYS,
    BACKTEST_STEP_DAYS,
    BACKTEST_RUN_FORECAST_STRATEGIES,
    BACKTEST_RUN_FORECAST_COPULA_STRATEGIES,
    BACKTEST_RUN_SLEEVE_STRATEGIES,
    BACKTEST_SLEEVE_BASE_METHODS,
    TSMOM_ALPHAS,
    ISLAND_GA_NUM_GENERATIONS,
    TRADING_DAYS_PER_YEAR,
    DATA_MIN_COVERAGE,
    DATA_FFILL_LIMIT,
    GA_MIN_SECURITIES,
    GA_MAX_SECURITIES,
    NZ_ETF_PRICES_CSV,
    CPCV_N_GROUPS,
    CPCV_K_TEST_GROUPS,
    CPCV_PURGE_DAYS,
    CPCV_EMBARGO_DAYS,
)


def _method_type(category: str) -> str:
    """Map a category name (e.g. 'cc_optimised') to a method type for DSR.

    Selection method ⇒ trials count: 'cc' = GA, 'mc' = Monte Carlo,
    'random' = pure random, 'bench' = fixed market benchmark (no
    selection trials, DSR not meaningful).
    """
    if category.startswith('cc_'):
        return 'cc'
    if category.startswith('mc_'):
        return 'mc'
    if category.startswith('random_'):
        return 'random'
    if category.startswith('bench_'):
        return 'bench'
    raise ValueError(f"Cannot infer method type from category: {category!r}")


def _dsr_for_method(method_results, train_log_returns, num_portfolios: int):
    """Compute DSR for the best in-sample portfolio of a method.

    Returns a dict per :func:`compute_method_dsr`, or None if the method
    has no portfolios with valid is_sharpe.
    """
    portfolios_with_is = [p for p in method_results.portfolios
                          if p.is_sharpe is not None]
    if not portfolios_with_is:
        return None
    best = max(portfolios_with_is, key=lambda p: p.is_sharpe)
    # Compute the best portfolio's training-period daily returns:
    # weighted sum of selected tickers' log returns.
    selected = best.portfolio
    cols = [c for c in selected if c in train_log_returns.columns]
    if not cols:
        return None
    weights = np.asarray(best.weights, dtype=np.float64)
    # Align weights to cols in case any tickers were dropped.
    weight_map = dict(zip(selected, weights))
    aligned = np.array([weight_map[c] for c in cols], dtype=np.float64)
    daily_rets = (train_log_returns[cols].values @ aligned).astype(np.float64)
    method = _method_type(method_results.category)
    if method == 'bench':
        # Fixed-allocation benchmarks have a single deterministic portfolio
        # (num_trials=1); the DSR formulation isn't meaningful.
        return None
    num_trials = effective_trials_for_method(
        method=method,
        num_portfolios=num_portfolios,
        ga_pop=BACKTEST_NUM_CHILDREN,
        ga_generations=ISLAND_GA_NUM_GENERATIONS,
        mc_trials_per_portfolio=BACKTEST_MC_TRIALS,
    )
    return compute_method_dsr(
        observed_sr=best.is_sharpe,
        portfolio_returns=daily_rets,
        num_trials=num_trials,
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
    evaluate_portfolios_with_sleeve,
    benchmark_portfolio,
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
) -> WindowResult:
    """Walk-forward wrapper around :func:`evaluate_split`.

    Slices ``full_prices`` by the window and delegates the actual
    optimisation pipeline to ``evaluate_split``. CPCV calls
    ``evaluate_split`` directly with its purged splits.
    """
    train_prices, oos_log_returns = slice_window_data(window, full_prices)
    return evaluate_split(
        train_prices=train_prices,
        oos_log_returns=oos_log_returns,
        conn=conn,
        label=window.label,
        num_portfolios=num_portfolios,
        num_children=num_children,
        mc_trials=mc_trials,
        window=window,
    )


def evaluate_split(
    train_prices: pd.DataFrame,
    oos_log_returns: pd.DataFrame,
    conn,
    label: str,
    num_portfolios: int = BACKTEST_NUM_PORTFOLIOS,
    num_children: int = BACKTEST_NUM_CHILDREN,
    mc_trials: int = BACKTEST_MC_TRIALS,
    window: Optional[WindowSpec] = None,
) -> WindowResult:
    """
    Run the full backtest pipeline on a single train/test split.

    Generic over how the split was constructed (walk-forward window or
    CPCV combination). Creates portfolios via GA, MC, and random selection,
    computes weights (optimal, copula, random), evaluates OOS performance,
    and computes per-method DSR for overfitting gating.

    OVERFITTING AWARENESS: The GA optimises on training data, producing
    in-sample (IS) Sharpe ratios that are biased upward due to selection
    bias. The OOS Sharpe ratios from the test period are the real measure
    of portfolio quality. Typical IS -> OOS degradation is 30-50%.
    See CLAUDE.md "Sharpe Ratio Overfitting" section.

    :param train_prices: training-period price DataFrame.
    :param oos_log_returns: pre-computed OOS log returns DataFrame.
    :param conn: sqlite3 connection (for forecast loading).
    :param label: human-readable label for this split (used in logs).
    :param num_portfolios: portfolios per method.
    :param num_children: GA population size.
    :param mc_trials: Monte Carlo trials per portfolio.
    :param window: optional WindowSpec (used by walk-forward; CPCV may
        synthesise one from the train/test boundaries or pass None).
    :return: WindowResult with all method results.
    """
    from src.backtest import simulation
    from src.optimisers import monte_carlo as mc

    window_start = time.time()
    if window is None:
        window = WindowSpec(
            train_start=train_prices.index.min(),
            train_end=train_prices.index.max(),
            test_start=oos_log_returns.index.min(),
            test_end=oos_log_returns.index.max(),
            label=label,
        )
    result = WindowResult(window=window)

    logger.info(
        "  Split %s: train=%d rows, test=%d rows, %d tickers",
        label, len(train_prices), len(oos_log_returns),
        train_prices.shape[1],
    )

    # -- Prepare optimisation state for weight optimisation --------------------
    log_returns_train = calculate_log_returns(train_prices)
    simulation._backtest_log_returns = log_returns_train.transpose()
    simulation._backtest_expected_returns = calculate_expected_returns(
        log_returns_train)

    # -- Create GA portfolios --------------------------------------------------
    logger.info("  Creating %d GA portfolios...", num_portfolios)
    start = time.time()
    with mp.Pool(
        processes=NUM_JOBS,
        initializer=_init_worker,
        initargs=(train_prices,),
    ) as pool:
        ga_portfolios = pool.map(create_portfolio, [num_children] * num_portfolios)
    logger.info("  GA done in %.1fs", time.time() - start)

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

    # Annualised historical variance per ticker — feeds the cc_ccc_baseline
    # strategy through the D×R×D covariance path. Computed once so workers
    # can be passed a small portfolio-sliced Series rather than rebuilding
    # it per task.
    hist_var_series = log_returns_train.var() * TRADING_DAYS_PER_YEAR

    # -- Forecast strategies (lazy per-window ARIMA + GARCH) -------------------
    # Fits forecasts only for the union of GA-selected tickers, using the
    # *training-window only* — never the OOS suffix — so each window's
    # results are leakage-free. Cached by (ticker, train_end), so this is
    # a one-shot cost per window per process.
    arima_er_series = None
    garch_var_series = None
    if BACKTEST_RUN_FORECAST_STRATEGIES:
        from src.backtest import forecast_cache
        selected_union = set()
        for ps in (ga_portfolios, mc_portfolios):
            for p in ps:
                selected_union.update(p)
        # Drop tickers that aren't in the training data (defensive — should
        # be empty given the GA only picks from train_prices columns).
        selected_union = [
            t for t in selected_union if t in train_prices.columns
        ]
        if selected_union:
            logger.info("  Warming forecast cache for %d unique tickers...",
                        len(selected_union))
            t0 = time.time()
            forecast_cache.warm_cache_for_window(
                tickers=selected_union,
                train_prices=train_prices,
                train_log_returns=log_returns_train,
                train_end=window.train_end,
                n_periods=BACKTEST_NUM_DAYS_OOS,
                n_workers=NUM_JOBS,
            )
            logger.info("  Forecast cache warmed in %.1fs", time.time() - t0)
            arima_er_series = forecast_cache.arima_er_series_for_window(
                selected_union, window.train_end)
            garch_var_series = forecast_cache.garch_var_series_for_window(
                selected_union, window.train_end)

    # -- Compute weights in parallel -------------------------------------------
    logger.info("  Optimising weights and running OOS evaluation...")

    # Build all (portfolio, mode, kwargs_dict) work items
    weight_tasks = []
    task_metadata = []  # (category_name, portfolio_index)

    categories = [
        ('cc_optimised',           ga_portfolios,     'optimal'),
        ('cc_copulae',             ga_portfolios,     'copulae'),
        ('cc_random_weights',      ga_portfolios,     'random'),
        ('cc_ccc_baseline',        ga_portfolios,     'optimal_ccc'),
        ('cc_equal_weight',        ga_portfolios,     'equal'),
        ('cc_min_variance',        ga_portfolios,     'min_variance'),
        ('cc_inverse_vol',         ga_portfolios,     'inverse_vol'),
        ('cc_risk_parity',         ga_portfolios,     'risk_parity'),
        ('cc_max_diversification', ga_portfolios,     'max_diversification'),
        ('mc_optimised',           mc_portfolios,     'optimal'),
        ('mc_random_weights',      mc_portfolios,     'random'),
        ('random_optimised',       random_portfolios, 'optimal'),
        ('random_random',          random_portfolios, 'random'),
    ]

    if arima_er_series is not None and garch_var_series is not None:
        # Fast forecast strategies (no copula). Always run when forecasts
        # are enabled.
        categories.extend([
            ('cc_arima_er',            ga_portfolios, 'optimal_arima_er'),
            ('cc_garch_var',           ga_portfolios, 'optimal_garch'),
            ('cc_arima_garch',         ga_portfolios, 'optimal_arima_garch'),
        ])
        # Slow forecast+copula strategies. Gated behind their own flag —
        # individual portfolio fits can take 200s+ when n=20 due to
        # super-cubic TCopula scaling combined with the GARCH covariance.
        if BACKTEST_RUN_FORECAST_COPULA_STRATEGIES:
            categories.extend([
                ('cc_garch_copula',        ga_portfolios, 'optimal_garch_copula'),
                ('cc_arima_garch_copula',  ga_portfolios,
                 'optimal_arima_garch_copula'),
            ])

    def _kwargs_for(mode, portfolio):
        """Build the per-task kwargs dict for the given mode + portfolio."""
        portfolio = list(portfolio)
        if mode == 'optimal_ccc':
            return {'var': hist_var_series.loc[portfolio]}
        if mode == 'optimal_arima_er':
            return {'er': arima_er_series.loc[portfolio].values}
        if mode == 'optimal_garch':
            return {'var': garch_var_series.loc[portfolio]}
        if mode == 'optimal_garch_copula':
            return {'var': garch_var_series.loc[portfolio]}
        if mode == 'optimal_arima_garch':
            return {
                'er': arima_er_series.loc[portfolio].values,
                'var': garch_var_series.loc[portfolio],
            }
        if mode == 'optimal_arima_garch_copula':
            return {
                'er': arima_er_series.loc[portfolio].values,
                'var': garch_var_series.loc[portfolio],
            }
        return {}

    for cat_name, portfolios, mode in categories:
        for i, p in enumerate(portfolios):
            weight_tasks.append((p, mode, _kwargs_for(mode, p)))
            task_metadata.append((cat_name, i))

    # Parallel weight computation. chunksize=1 forces dynamic dispatch so
    # the (much slower) cc_copulae tasks get spread across workers instead
    # of stranding one worker with all of them while others idle.
    with mp.Pool(
        processes=NUM_JOBS,
        initializer=_init_weight_worker,
        initargs=(simulation._backtest_log_returns,
                  simulation._backtest_expected_returns),
    ) as pool:
        all_weights = pool.map(_compute_weights_for_portfolio, weight_tasks,
                               chunksize=1)

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

    # -- Managed-futures sleeve overlay arms (research experiment) -------------
    # Blend a precomputed, full-history TSMOM sleeve into the best-travelling
    # base methods at fixed alpha levels. Each arm REUSES its base method's
    # portfolios + weights (only the OOS/IS evaluation is blended), so this adds
    # no weight optimisation. The sleeve is computed once on full DB history then
    # sliced per window inside evaluate_portfolios_with_sleeve (causal — see
    # src/sleeves/). Gated; default off → base-arm numbers are untouched.
    # Note: arm names start cc_/mc_ so _method_type and the DSR/PBO machinery
    # accept them. The DSR moment-correction reconstructs un-blended book returns
    # (a minor diagnostic approximation); the CPCV PBO + CIs are the verdict.
    if BACKTEST_RUN_SLEEVE_STRATEGIES:
        from src.sleeves.overlay import get_cached_sleeve_series
        try:
            sleeve_series = get_cached_sleeve_series(conn)
        except Exception:
            logger.exception("Sleeve series unavailable; skipping sleeve arms")
            sleeve_series = None
        if sleeve_series is not None:
            for base_cat in BACKTEST_SLEEVE_BASE_METHODS:
                if base_cat not in category_portfolios:
                    continue
                for a in TSMOM_ALPHAS:
                    arm = f"{base_cat}_trend{int(round(a * 100))}"
                    result.method_results[arm] = evaluate_portfolios_with_sleeve(
                        category_portfolios[base_cat], category_weights[base_cat],
                        oos_log_returns, train_log_returns, arm, sleeve_series, a)

    # -- Market-benchmark strategies -------------------------------------------
    # Fixed-allocation portfolios that bypass the GA + SLSQP pipeline. Skipped
    # for windows where the required tickers are missing from training or OOS
    # data (e.g. early windows before AGG had history).
    for bench_cat in ('bench_spy', 'bench_6040'):
        bench_tickers, bench_weights = benchmark_portfolio(
            bench_cat, train_log_returns, oos_log_returns)
        if bench_tickers is None:
            logger.info("  Skipping %s — required tickers missing for window",
                        bench_cat)
            continue
        result.method_results[bench_cat] = evaluate_portfolios(
            [bench_tickers], [bench_weights],
            oos_log_returns, train_log_returns, bench_cat)

    result.elapsed_seconds = time.time() - window_start

    # Log per-window summary with IS vs OOS comparison + DSR gating.
    # DSR < 0.5 ⇒ best IS Sharpe is below the expected max under the null
    # given M trials, i.e. indistinguishable from "best of M random
    # strategies". DSR ≥ 0.95 ⇒ statistically significant after the
    # multiple-testing correction.
    logger.info("  Window %s results (%.1fs):", label, result.elapsed_seconds)
    result.dsr_per_method = {}
    for cat, mr in sorted(result.method_results.items()):
        is_sharpes = [p.is_sharpe for p in mr.portfolios if p.is_sharpe is not None]
        if is_sharpes:
            mean_is = np.mean(is_sharpes)
            mean_oos = mr.mean_sharpe
            degradation = ((mean_is - mean_oos) / mean_is * 100) if mean_is > 0 else float('nan')
            try:
                dsr_info = _dsr_for_method(
                    mr, train_log_returns,
                    num_portfolios=len(mr.portfolios))
            except Exception:
                logger.exception("DSR computation failed for %s", cat)
                dsr_info = None
            if dsr_info is None:
                logger.info(
                    "    %-25s  IS_sharpe=%.4f  OOS_sharpe=%.4f  degradation=%.0f%%",
                    cat, mean_is, mean_oos, degradation)
            else:
                result.dsr_per_method[cat] = dsr_info
                gate = ('PASS' if dsr_info['dsr'] >= 0.95 else
                        'WEAK' if dsr_info['dsr'] >= 0.5 else 'FAIL')
                logger.info(
                    "    %-25s  IS_sharpe=%.4f  OOS_sharpe=%.4f  "
                    "degradation=%.0f%%  DSR=%.3f [%s] (M=%.1e, n=%d)",
                    cat, mean_is, mean_oos, degradation,
                    dsr_info['dsr'], gate,
                    dsr_info['num_trials'], dsr_info['num_obs'])
            warn_if_sharpe_suspicious(mean_is, f"Window {label} {cat} IS", logger)
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

    # -- Evaluate each window --------------------------------------------------
    run_group = str(uuid.uuid4())
    all_results: List[WindowResult] = []

    for window in windows:
        logger.info("=" * 60)
        logger.info("Evaluating window: %s", window.label)

        wr = evaluate_window(
            window=window,
            full_prices=data,
            conn=conn,
        )
        all_results.append(wr)

        # Save per-window results to DB
        session_id = db.save_backtest_session(conn, {
            'data_source': 'yahoo_finance',
            'num_portfolios': BACKTEST_NUM_PORTFOLIOS,
            'num_days_oos': len(data.loc[window.test_start:window.test_end]),
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
        ('CC optimised vs Random random',  'cc_optimised', 'random_random'),
        ('MC optimised vs Random random',  'mc_optimised', 'random_random'),
        ('CC optimised vs MC optimised',   'cc_optimised', 'mc_optimised'),
        ('CC copulae vs CC optimised',     'cc_optimised', 'cc_copulae'),
        ('CC equal-weight vs CC optimised', 'cc_optimised', 'cc_equal_weight'),
        ('CC min-variance vs CC optimised', 'cc_optimised', 'cc_min_variance'),
        ('CC ccc-baseline vs CC optimised', 'cc_optimised', 'cc_ccc_baseline'),
        ('CC optimised vs SPY',            'cc_optimised', 'bench_spy'),
        ('CC optimised vs 60/40',          'cc_optimised', 'bench_6040'),
    ]
    # Base vs +trend25 sleeve overlay (only fire when the sleeve arms exist).
    for base in BACKTEST_SLEEVE_BASE_METHODS:
        comparisons.append(
            (f'{base} +trend25 vs base', base, f'{base}_trend25'))

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


# ---- CPCV main ---------------------------------------------------------------


def main_cpcv(n_groups: Optional[int] = None,
              k_test_groups: Optional[int] = None,
              purge_days: Optional[int] = None,
              embargo_days: Optional[int] = None) -> None:
    """Combinatorially Purged Cross-Validation backtest.

    Generates ``C(n_groups, k_test_groups)`` purged train/test splits over
    the full price history, runs the full optimisation pipeline on each,
    and aggregates: per-method OOS Sharpe distribution (mean + 95% CI) and
    method-level PBO (López de Prado 2018). Defaults come from
    ``src.config.CPCV_*``.
    """
    from src import db
    from src.backtest.cpcv import (
        compute_pbo,
        generate_cpcv_splits,
        summarise_method_across_splits,
    )

    if n_groups is None:
        n_groups = CPCV_N_GROUPS
    if k_test_groups is None:
        k_test_groups = CPCV_K_TEST_GROUPS
    if purge_days is None:
        purge_days = CPCV_PURGE_DAYS
    if embargo_days is None:
        embargo_days = CPCV_EMBARGO_DAYS

    bt_start = time.time()

    # Load and clean (mirror main()).
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

    splits = generate_cpcv_splits(
        data.index, n_groups=n_groups, k_test_groups=k_test_groups,
        purge_days=purge_days, embargo_days=embargo_days,
    )
    logger.info(
        "Generated %d CPCV splits (n_groups=%d, k_test=%d, "
        "purge=%dd, embargo=%dd)",
        len(splits), n_groups, k_test_groups, purge_days, embargo_days,
    )

    run_group = str(uuid.uuid4())
    all_results: List[WindowResult] = []
    for idx, split in enumerate(splits, 1):
        logger.info("=" * 60)
        logger.info("CPCV split %d/%d: %s (train=%d, test=%d)",
                    idx, len(splits), split.label,
                    len(split.train_dates), len(split.test_dates))
        train_prices = data.loc[split.train_dates]
        oos_prices = data.loc[split.test_dates]
        # CPCV test_dates are the union of k_test groups — possibly
        # non-contiguous in time, and not necessarily following the
        # train end. The walk-forward boundary-prepending trick would
        # break sort order. Instead, compute log returns directly on
        # the (sorted) test prices. The first day of each test group
        # gets a zero return; for k_test=2, that's a 2-row distortion
        # in ~574 rows of test data — acceptable noise.
        oos_log_returns = calculate_log_returns(oos_prices)

        wr = evaluate_split(
            train_prices=train_prices,
            oos_log_returns=oos_log_returns,
            conn=conn,
            label=split.label,
        )
        all_results.append(wr)

        # Persist per-split results to the DB just like main().
        try:
            session_id = db.save_backtest_session(conn, {
                'data_source': 'yahoo_finance',
                'num_portfolios': BACKTEST_NUM_PORTFOLIOS,
                'num_days_oos': len(oos_log_returns),
                'optimiser_params': {
                    'num_children': BACKTEST_NUM_CHILDREN,
                    'mc_trials_per_portfolio': BACKTEST_MC_TRIALS,
                    'cpcv': True,
                    'n_groups': n_groups,
                    'k_test_groups': k_test_groups,
                    'purge_days': purge_days,
                    'embargo_days': embargo_days,
                },
                'elapsed_seconds': wr.elapsed_seconds,
                'window_train_start': str(train_prices.index.min().date())
                                      if len(train_prices) else '',
                'window_train_end': str(train_prices.index.max().date())
                                    if len(train_prices) else '',
                'window_test_start': str(oos_prices.index.min().date())
                                     if len(oos_prices) else '',
                'window_test_end': str(oos_prices.index.max().date())
                                   if len(oos_prices) else '',
                'window_label': 'cpcv:' + split.label,
                'run_group_id': run_group,
            })
            for cat, mr in wr.method_results.items():
                for i, pr in enumerate(mr.portfolios):
                    db.save_backtest_result(
                        conn, session_id, cat, i,
                        metrics=pr.metrics,
                        holdings=list(zip(pr.portfolio, pr.weights)))
            logger.info("  Saved to DB (session id=%d)", session_id)
        except Exception:
            logger.exception("Failed to persist CPCV split results to DB")

    _report_cpcv_results(all_results)

    bt_elapsed = time.time() - bt_start
    logger.info("Full CPCV backtest completed in %.1fs (%d splits)",
                bt_elapsed, len(splits))
    conn.close()


def _report_cpcv_results(all_results: List[WindowResult]) -> None:
    """Aggregate CPCV splits into per-method OOS distribution + PBO."""
    from src.backtest.cpcv import (
        compute_pbo,
        summarise_method_across_splits,
    )

    if not all_results:
        logger.warning("No CPCV results to report.")
        return

    methods = sorted({m for wr in all_results for m in wr.method_results})
    n_splits = len(all_results)
    n_methods = len(methods)

    # IS-best and OOS-of-IS-best matrices, [n_splits, n_methods].
    is_best = np.zeros((n_splits, n_methods))
    oos_at_is_best = np.zeros((n_splits, n_methods))

    per_method_is = {m: [] for m in methods}
    per_method_oos = {m: [] for m in methods}

    for i, wr in enumerate(all_results):
        for j, m in enumerate(methods):
            mr = wr.method_results.get(m)
            if mr is None or not mr.portfolios:
                continue
            is_sharpes = [p.is_sharpe for p in mr.portfolios
                          if p.is_sharpe is not None]
            oos_sharpes = mr.sharpe_ratios.tolist()
            if not is_sharpes or not oos_sharpes:
                continue
            best_idx = int(np.argmax(is_sharpes))
            is_best[i, j] = is_sharpes[best_idx]
            # Match the IS-best portfolio's OOS Sharpe (best portfolio's OOS).
            if best_idx < len(oos_sharpes):
                oos_at_is_best[i, j] = oos_sharpes[best_idx]
            per_method_is[m].append(is_sharpes[best_idx])
            per_method_oos[m].append(mr.mean_sharpe)

    # Method-level PBO.
    logger.info("=" * 60)
    logger.info("CPCV CROSS-SPLIT SUMMARY")
    if n_methods >= 2:
        try:
            pbo = compute_pbo(is_best, oos_at_is_best)
            verdict = ('OK (robust)' if pbo < 0.3 else
                       'WEAK' if pbo < 0.5 else
                       'OVERFIT')
            logger.info(
                "Method-level PBO across %d methods × %d splits: "
                "PBO=%.3f [%s]", n_methods, n_splits, pbo, verdict)
        except ValueError as e:
            logger.warning("PBO computation failed: %s", e)

        # When sleeve arms are present, also report PBO over the base methods
        # only — the sleeve arms inflate the trial count K, so the with/without
        # comparison shows whether they help or hurt strategy-family robustness.
        sleeve_cols = [j for j, m in enumerate(methods) if '_trend' in m]
        keep = [j for j in range(n_methods) if j not in set(sleeve_cols)]
        if sleeve_cols and len(keep) >= 2:
            try:
                pbo_base = compute_pbo(is_best[:, keep], oos_at_is_best[:, keep])
                logger.info(
                    "Method-level PBO EXCLUDING %d sleeve arms "
                    "(%d base methods): PBO=%.3f",
                    len(sleeve_cols), len(keep), pbo_base)
            except ValueError as e:
                logger.warning("Base-only PBO computation failed: %s", e)

    # Per-method OOS distributions with 95% CI.
    logger.info("Per-method OOS Sharpe distribution (across %d splits):",
                n_splits)
    for m in methods:
        if not per_method_oos[m]:
            continue
        s = summarise_method_across_splits(
            m, per_method_is[m], per_method_oos[m])
        logger.info(
            "  %-25s  OOS_mean=%+.4f  std=%.4f  95%%CI=[%+.4f, %+.4f]  "
            "(n=%d)",
            m, s.mean_oos, s.std_oos,
            s.ci95_oos_low, s.ci95_oos_high, len(per_method_oos[m]))
