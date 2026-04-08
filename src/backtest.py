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

from src.portfolio_utils import (
    calculate_log_returns,
    calculate_expected_returns,
    calculate_covariance_matrix,
    optimise_weights,
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
    BACKTEST_MAX_WEIGHT_FLOOR,
    TRADING_DAYS_PER_YEAR,
    GA_MIN_SECURITIES,
    GA_MAX_SECURITIES,
    GA_MIN_WEIGHT,
    GA_MAX_WEIGHT,
    GA_NUM_GENERATIONS,
    NZ_ETF_PRICES_CSV,
)

logger = logging.getLogger(__name__)

NUM_JOBS = cpu_count()

METRIC_NAMES = [
    'annualised_return', 'annualised_volatility', 'sharpe_ratio',
    'downside_deviation', 'max_drawdown', 'calmar_ratio', 'sortino_ratio',
]

# Module-level state set before spawning worker pools.
_backtest_data = None
_use_forecast = False
# Log returns (transposed: tickers×dates) and expected returns for weight optimisation.
_backtest_log_returns = None
_backtest_expected_returns = None


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


def optimal_weights(portfolio, use_copulae=False):
    """
    Finds the optimal weights (allocations) for the
    input portfolio.

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
        from src.portfolio_utils import estimate_corr_using_copulas
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
    Creates a cardinality-constrained portfolio with the
    training data.

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
    """Slice full prices into train/test sets and compute OOS log returns.

    :param window: WindowSpec defining train/test boundaries.
    :param full_prices: complete price DataFrame.
    :return: (train_prices, oos_log_returns) tuple.
    """
    train_prices = full_prices.loc[window.train_start:window.train_end]
    test_prices = full_prices.loc[window.test_start:window.test_end]
    assert train_prices.index.max() < test_prices.index.min(), (
        f"Window {window.label}: train data ends at {train_prices.index.max()} "
        f"but test data starts at {test_prices.index.min()}. "
        f"This would leak test-period data into training."
    )
    boundary_price = train_prices.iloc[[-1]]
    test_with_boundary = pd.concat([boundary_price, test_prices])
    oos_log_returns = calculate_log_returns(test_with_boundary).iloc[1:]
    logger.info(
        "  Window %s: train=%d rows, test=%d rows, %d tickers",
        window.label, len(train_prices), len(test_prices),
        train_prices.shape[1],
    )
    return train_prices, oos_log_returns


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
    """Create portfolios via GA using a worker pool.

    :param train_prices: training price DataFrame.
    :param num_portfolios: number of portfolios to create.
    :param num_children: GA population size.
    :param use_forecast: whether to use forecast-based GA.
    :return: list of portfolios (each a list of ticker strings).
    """
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
    """Create portfolios via Monte Carlo search.

    :param train_prices: training price DataFrame.
    :param num_portfolios: number of portfolios to create.
    :param mc_trials: Monte Carlo trials per portfolio.
    :return: list of portfolios (each a list of ticker strings).
    """
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
    """Compute weights for all (portfolio, mode) pairs in parallel.

    :param categories: list of (cat_name, portfolios, mode) tuples.
    :param log_returns_T: transposed training log returns (tickers x dates).
    :param expected_returns: Series of annualised expected returns.
    :return: dict mapping cat_name -> list of weight arrays.
    """
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


def _evaluate_oos(portfolios, weights_list, oos_log_returns,
                  train_log_returns):
    """Evaluate portfolios out-of-sample and compute IS Sharpe diagnostic.

    :param portfolios: list of portfolios (each a list of ticker strings).
    :param weights_list: list of weight arrays (same order as portfolios).
    :param oos_log_returns: OOS log returns DataFrame.
    :param train_log_returns: training log returns for IS Sharpe computation.
    :return: list of PortfolioResult objects.
    """
    results = []
    for p, w in zip(portfolios, weights_list):
        metrics = get_statistics(p, w, oos_log_returns)
        try:
            is_stats = get_statistics(p, w, train_log_returns)
            is_sr = is_stats['sharpe_ratio']
        except Exception:
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
    and OOS evaluation. See CLAUDE.md "Sharpe Ratio Overfitting" for
    details on IS vs OOS Sharpe degradation.

    :param window: WindowSpec defining train/test boundaries.
    :param full_prices: complete price DataFrame (will be sliced).
    :param conn: sqlite3 connection (for forecast loading).
    :param num_portfolios: portfolios per method.
    :param num_children: GA population size.
    :param mc_trials: Monte Carlo trials per portfolio.
    :param use_forecast: whether to also run forecast-based GA.
    :return: WindowResult with all method results.
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
    from src import db
    from src.portfolio_utils import load_training_data

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
