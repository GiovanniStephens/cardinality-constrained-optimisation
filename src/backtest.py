import logging
import multiprocessing as mp
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel
from tqdm import tqdm

from src.optimisers.pygad_ga import PygadOptimiser
from src.portfolio_utils import (
    calculate_log_returns,
    load_prices_csv,
    maximum_drawdown,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
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
    BACKTEST_FORECAST_WINDOWS,
    TRADING_DAYS_PER_YEAR,
    DATA_MIN_COVERAGE,
    DATA_FFILL_LIMIT,
    GA_MIN_SECURITIES,
    GA_MAX_SECURITIES,
)

logger = logging.getLogger(__name__)

NUM_JOBS = cpu_count()

METRIC_NAMES = [
    'annualised_return', 'annualised_volatility', 'sharpe_ratio',
    'downside_deviation', 'max_drawdown', 'calmar_ratio', 'sortino_ratio',
]

# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class WindowSpec:
    """Defines a single train/test window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    label: str


@dataclass
class PortfolioResult:
    """Result for a single portfolio within a single window and method."""
    portfolio: List[str]
    weights: np.ndarray
    metrics: Dict[str, float]
    is_sharpe: Optional[float] = None  # in-sample Sharpe (biased upward)


@dataclass
class MethodResults:
    """All portfolio results for one method in one window."""
    category: str
    portfolios: List[PortfolioResult] = field(default_factory=list)

    @property
    def sharpe_ratios(self) -> np.ndarray:
        return np.array([p.metrics['sharpe_ratio'] for p in self.portfolios])

    @property
    def mean_sharpe(self) -> float:
        return float(self.sharpe_ratios.mean())


@dataclass
class WindowResult:
    """All method results for one window."""
    window: WindowSpec
    method_results: Dict[str, MethodResults] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# ─── Window Generation ────────────────────────────────────────────────────────


def generate_windows(
    date_index: pd.DatetimeIndex,
    train_days: int = BACKTEST_TRAIN_YEARS * TRADING_DAYS_PER_YEAR,
    test_days: int = BACKTEST_TEST_DAYS,
    step_days: int = BACKTEST_STEP_DAYS,
) -> List[WindowSpec]:
    """
    Generate non-overlapping rolling forward-walk windows from a date index.

    :param date_index: sorted DatetimeIndex of trading days.
    :param train_days: number of trading days for training.
    :param test_days: number of trading days for OOS testing.
    :param step_days: step size in trading days between windows.
    :return: list of WindowSpec objects.
    """
    dates = date_index.sort_values()
    n = len(dates)
    min_required = train_days + test_days
    if n < min_required:
        raise ValueError(
            f"Need at least {min_required} trading days, got {n}"
        )

    windows = []
    start = 0
    while start + min_required <= n:
        train_start = dates[start]
        train_end = dates[start + train_days - 1]
        test_start = dates[start + train_days]
        test_end_idx = min(start + train_days + test_days - 1, n - 1)
        test_end = dates[test_end_idx]

        label = f"{train_start.year}-{train_end.year}/{test_start.year}"
        windows.append(WindowSpec(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            label=label,
        ))
        start += step_days

    return windows


# ─── Portfolio Simulation ─────────────────────────────────────────────────────


def get_random_weights(portfolio):
    """
    Creates a set of random weighting that sum to 1.

    :portfolio: Input portfolio to get the length.
    :return: A set of random weights equal in length to the portfolio.
    """
    if not portfolio:
        raise ValueError("Cannot generate weights for an empty portfolio.")
    random_weights = np.random.random(len(portfolio))
    random_weights /= np.sum(random_weights)
    return random_weights


def optimal_weights(portfolio, optimiser, use_copulae=False):
    """
    Finds the optimal weights (allocations) for the
    input portfolio using a PygadOptimiser instance.

    :portfolio: The input portfolio. List of ticker strings.
    :optimiser: PygadOptimiser instance with prepared inputs.
    :use_copulae: Whether to use copulae or not.
    :return: A list of weights for the input portfolio.
    """
    if len(portfolio) < 2:
        raise ValueError("Portfolio must contain at least 2 assets.")
    missing = set(portfolio) - set(optimiser._data.index)
    if missing:
        raise KeyError(f"Tickers not found in data: {missing}")
    random_weights = get_random_weights(portfolio)
    max_w = max(1 / (len(portfolio) - 1), 0.3)
    result = optimiser._optimize_weights(
        optimiser._data.loc[portfolio, :].transpose(),
        random_weights,
        use_copulae=use_copulae,
        max_weight=max_w,
        min_weight=0.0,
    )
    if not result.success:
        logger.warning("Weight optimization did not converge: %s", result.message)
    return result['x']


def run_portfolio(portfolio, weights, oos_log_returns):
    """
    Buy-and-hold simulation over the OOS period with natural weight drift.

    :param portfolio: list of ticker strings.
    :param weights: initial weight allocations (same order as portfolio).
    :param oos_log_returns: DataFrame with dates as rows, tickers as columns.
                            Only the OOS period — no offset needed.
    :return: list of daily portfolio log returns.
    """
    subset = oos_log_returns[portfolio]
    portfolio_returns = []
    w = weights.copy()
    for i in range(len(subset)):
        step_returns = subset.iloc[i].values
        weighted_return = float(np.sum(step_returns * w))
        portfolio_returns.append(weighted_return)
        w = w * np.exp(step_returns) / (1 + weighted_return)
    return portfolio_returns


def get_statistics(portfolio, weights, oos_log_returns):
    """
    Compute all performance metrics for one portfolio on an OOS window.

    :param portfolio: list of ticker strings.
    :param weights: weight allocations.
    :param oos_log_returns: OOS log returns DataFrame (dates x tickers).
    :return: dict keyed by metric name.
    """
    portfolio_returns = run_portfolio(portfolio, weights, oos_log_returns)
    max_dd = maximum_drawdown(portfolio_returns)
    dd = downside_deviation(portfolio_returns)
    r = np.mean(portfolio_returns) * TRADING_DAYS_PER_YEAR
    std = np.std(portfolio_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = r / std if std != 0 else 0.0
    return dict(zip(METRIC_NAMES, [
        r, std, sharpe, dd, max_dd,
        calmar_ratio(r, max_dd),
        sortino_ratio(r, dd),
    ]))


def fitness(portfolio_returns):
    """
    Calculates the portfolio Sharpe Ratio.

    :portfolio_returns: The input portfolio returns. List of floats.
    :return: The fitness of the portfolio.
    """
    portfolio_std = np.std(portfolio_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if portfolio_std == 0:
        return 0.0
    return (np.mean(portfolio_returns) * TRADING_DAYS_PER_YEAR) / portfolio_std


# ─── GA Worker Helpers ────────────────────────────────────────────────────────

# Process-local state for multiprocessing workers.  Stored in a dict rather
# than bare module globals for explicitness and easier cleanup in tests.
_worker_state = {}


def _init_worker(training_prices, use_forecast):
    """Pool initializer — stores training data in each worker process."""
    _worker_state['prices'] = training_prices
    _worker_state['forecast'] = use_forecast


def create_portfolio(num_children, *, prices=None, use_forecast=None):
    """Create a cardinality-constrained portfolio using PygadOptimiser.

    When called inside a worker pool, reads from _worker_state (set by
    _init_worker).  For direct calls or testing, pass *prices* and
    *use_forecast* explicitly.

    :param num_children: number of GA children.
    :param prices: training prices DataFrame (optional, for direct use).
    :param use_forecast: whether to use forecasts (optional, for direct use).
    :return: list of selected ticker strings.
    """
    p = prices if prices is not None else _worker_state['prices']
    f = use_forecast if use_forecast is not None else _worker_state['forecast']
    opt = PygadOptimiser(
        num_children=num_children,
        target_return=None,
        use_forecasts=f,
    )
    result = opt.optimise(p)
    return result.selected_tickers


# ─── Statistical Testing ─────────────────────────────────────────────────────


def difference_of_means_hypothesis_test(sample_1, sample_2):
    """
    Calculates the t statistic for the difference of means.

    Second sample mean minus the first. (i.e. if positive,
    the second is greater than the first.)

    :sample_1: The first sample. List of floats.
    :sample_2: The second sample. List of floats.
    :return: The t statistic.
    """
    if not sample_1 or not sample_2:
        raise ValueError("Both samples must be non-empty")
    denominator = np.sqrt(
        np.var(sample_1) / len(sample_1) + np.var(sample_2) / len(sample_2)
    )
    if denominator == 0:
        raise ValueError(
            "t-statistic is undefined: both samples have zero variance"
        )
    return (np.mean(sample_2) - np.mean(sample_1)) / denominator


def paired_t_test(sharpes_a, sharpes_b):
    """
    Paired t-test across windows (same window, different methods).

    Controls for market-regime effects by pairing observations from
    the same OOS period.

    :param sharpes_a: dict {window_label: mean_sharpe} for method A.
    :param sharpes_b: dict {window_label: mean_sharpe} for method B.
    :return: (t_statistic, p_value) tuple.
    """
    common = sorted(set(sharpes_a) & set(sharpes_b))
    if len(common) < 2:
        raise ValueError(
            f"Need at least 2 common windows for paired test, got {len(common)}"
        )
    a = [sharpes_a[w] for w in common]
    b = [sharpes_b[w] for w in common]
    # ttest_rel computes first - second; swap so positive t = b > a
    return ttest_rel(b, a)


def friedman_test(all_results, categories):
    """
    Non-parametric Friedman test for comparing K methods across W windows.

    :param all_results: list of WindowResult objects.
    :param categories: list of category names to compare.
    :return: (chi2_statistic, p_value) tuple.
    """
    # Build matrix: one column per method, one row per window
    # Value = mean Sharpe of that method in that window
    columns = {}
    for cat in categories:
        values = []
        for wr in all_results:
            if cat in wr.method_results:
                values.append(wr.method_results[cat].mean_sharpe)
        columns[cat] = values

    # All methods must have the same number of windows
    n_windows = len(all_results)
    valid_cats = [c for c in categories if len(columns.get(c, [])) == n_windows]
    if len(valid_cats) < 3:
        raise ValueError(
            f"Friedman test requires >= 3 methods present in all windows, "
            f"got {len(valid_cats)}"
        )
    arrays = [columns[c] for c in valid_cats]
    return friedmanchisquare(*arrays)


def aggregate_cross_window(all_results):
    """
    Build a summary table of mean Sharpe per method per window.

    :param all_results: list of WindowResult objects.
    :return: DataFrame with methods as rows, windows + mean + std as columns.
    """
    data = {}
    all_categories = set()
    for wr in all_results:
        for cat in wr.method_results:
            all_categories.add(cat)

    for cat in sorted(all_categories):
        row = {}
        values = []
        for wr in all_results:
            if cat in wr.method_results:
                val = wr.method_results[cat].mean_sharpe
                row[wr.window.label] = val
                values.append(val)
            else:
                row[wr.window.label] = np.nan
        row['mean'] = np.nanmean(values) if values else np.nan
        row['std'] = np.nanstd(values) if values else np.nan
        data[cat] = row

    return pd.DataFrame(data).T


# ─── Per-Window Evaluation ────────────────────────────────────────────────────


def slice_window_data(window, full_prices):
    """Slice train/test data from full prices for a single backtest window.

    Prepends the last training price to the test period so the first OOS
    log return is log(test_price[0] / train_price[-1]) rather than 0.

    :param window: WindowSpec with train/test boundaries.
    :param full_prices: complete price DataFrame.
    :return: (train_prices, oos_log_returns) tuple.
    """
    train_prices = full_prices.loc[window.train_start:window.train_end]
    test_prices = full_prices.loc[window.test_start:window.test_end]
    boundary_price = train_prices.iloc[[-1]]
    test_with_boundary = pd.concat([boundary_price, test_prices])
    oos_log_returns = calculate_log_returns(test_with_boundary).iloc[1:]
    return train_prices, oos_log_returns


def create_random_portfolios(tickers, num_portfolios,
                             min_securities=GA_MIN_SECURITIES,
                             max_securities=GA_MAX_SECURITIES):
    """Create random portfolios by sampling binary selection vectors.

    :param tickers: list or Index of available ticker names.
    :param num_portfolios: number of portfolios to create.
    :param min_securities: minimum number of selected assets.
    :param max_securities: maximum number of selected assets.
    :return: list of lists of ticker strings.
    """
    n = len(tickers)
    p = max_securities / n
    portfolios = []
    for _ in range(num_portfolios):
        individual = np.random.binomial(1, p, n)
        while np.count_nonzero(individual) < min_securities:
            individual = np.random.binomial(1, p, n)
        portfolios.append(list(np.array(tickers)[individual.astype(bool)]))
    return portfolios


def evaluate_portfolios(portfolios, weights_list, oos_log_returns,
                        train_log_returns=None, category=''):
    """Evaluate a set of portfolios on OOS data, returning MethodResults.

    :param portfolios: list of portfolios (each a list of ticker strings).
    :param weights_list: list of weight arrays (same order as portfolios).
    :param oos_log_returns: OOS log returns DataFrame.
    :param train_log_returns: training log returns for IS Sharpe (optional).
    :param category: category name for the method.
    :return: MethodResults dataclass.
    """
    prs = []
    for p, w in zip(portfolios, weights_list):
        metrics = get_statistics(p, w, oos_log_returns)
        is_sr = None
        if train_log_returns is not None:
            try:
                is_stats = get_statistics(p, w, train_log_returns)
                is_sr = is_stats['sharpe_ratio']
            except (KeyError, ValueError, ArithmeticError):
                logger.warning("IS Sharpe failed for %s", p, exc_info=True)
        prs.append(PortfolioResult(
            portfolio=p, weights=w, metrics=metrics, is_sharpe=is_sr,
        ))
    return MethodResults(category=category, portfolios=prs)


def _create_ga_portfolios(train_prices, num_portfolios, num_children,
                          use_forecast):
    """Create portfolios via GA in a multiprocessing pool."""
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
            min_securities=GA_MIN_SECURITIES,
            max_securities=GA_MAX_SECURITIES,
        )
        if solution is not None:
            portfolios.append(list(train_prices.columns[solution == 1]))
        else:
            portfolios.extend(create_random_portfolios(
                train_prices.columns, 1))
    logger.info("  MC done in %.1fs", time.time() - start)
    return portfolios


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
            degradation = (
                (mean_is - mean_oos) / mean_is * 100
                if mean_is > 0 else float('nan')
            )
            logger.info(
                "    %-25s  IS_sharpe=%.4f  OOS_sharpe=%.4f  degradation=%.0f%%",
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

    OVERFITTING AWARENESS: The GA optimises on training data, producing
    in-sample (IS) Sharpe ratios that are biased upward due to selection
    bias. Typical IS -> OOS degradation is 30-50%.
    See CLAUDE.md "Sharpe Ratio Overfitting" section.
    """
    window_start = time.time()
    result = WindowResult(window=window)

    # ── Slice data ────────────────────────────────────────────────────────
    train_prices, oos_log_returns = slice_window_data(window, full_prices)
    train_log_returns = calculate_log_returns(train_prices)
    logger.info(
        "  Window %s: train=%d rows, test=%d rows, %d tickers",
        window.label, len(train_prices), len(oos_log_returns),
        train_prices.shape[1],
    )

    # ── Prepare optimiser instance ────────────────────────────────────────
    optimiser = PygadOptimiser(
        num_children=num_children,
        target_return=None,
        use_forecasts=False,
        conn=conn,
    )
    optimiser._prepare_inputs(train_prices)

    # ── Create portfolios ─────────────────────────────────────────────────
    ga_portfolios = _create_ga_portfolios(
        train_prices, num_portfolios, num_children, use_forecast=False)
    random_portfolios = create_random_portfolios(
        train_prices.columns, num_portfolios)
    mc_portfolios = _create_mc_portfolios(
        train_prices, num_portfolios, mc_trials)

    forecast_portfolios = []
    if use_forecast:
        forecast_portfolios = _create_ga_portfolios(
            train_prices, num_portfolios, num_children, use_forecast=True)

    # ── Compute weights and evaluate ──────────────────────────────────────
    logger.info("  Optimising weights and running OOS evaluation...")
    categories = [
        ('cc_optimised',      ga_portfolios,     lambda p: optimal_weights(p, optimiser, use_copulae=False)),
        ('cc_copulae',        ga_portfolios,     lambda p: optimal_weights(p, optimiser, use_copulae=True)),
        ('cc_random_weights', ga_portfolios,     get_random_weights),
        ('mc_optimised',      mc_portfolios,     lambda p: optimal_weights(p, optimiser)),
        ('mc_random_weights', mc_portfolios,     get_random_weights),
        ('random_optimised',  random_portfolios, lambda p: optimal_weights(p, optimiser)),
        ('random_random',     random_portfolios, get_random_weights),
    ]
    for cat_name, portfolios, weight_fn in tqdm(categories, desc="  Evaluating", leave=False):
        weights_list = [weight_fn(p) for p in portfolios]
        result.method_results[cat_name] = evaluate_portfolios(
            portfolios, weights_list, oos_log_returns,
            train_log_returns, cat_name)

    if use_forecast and forecast_portfolios:
        forecast_optimiser = PygadOptimiser(
            target_return=None, use_forecasts=True, conn=conn,
        )
        forecast_optimiser._prepare_inputs(train_prices)
        weights_list = [optimal_weights(p, forecast_optimiser)
                        for p in forecast_portfolios]
        result.method_results['cc_forecast'] = evaluate_portfolios(
            forecast_portfolios, weights_list, oos_log_returns,
            train_log_returns, 'cc_forecast')

    result.elapsed_seconds = time.time() - window_start
    _log_window_summary(window, result)
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    from src import db

    bt_start = time.time()

    # ── Load prices from DB (CSV fallback) ────────────────────────────────
    from src.portfolio_utils import load_prices
    conn = db.get_connection()
    data = load_prices(exchange='US', csv_fallback='data/NZ_ETF_Prices.csv',
                       conn=conn)
    logger.info("Loaded price data: %d rows x %d columns", *data.shape)

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
    if len(core_categories) >= 3 and len(all_results) >= 2:
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
    from src.logging_config import setup_logging
    setup_logging()
    main()
