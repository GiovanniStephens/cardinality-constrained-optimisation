"""Portfolio simulation: weight generation, OOS evaluation, and metrics."""

import logging
from typing import List

import numpy as np

from src.config import (
    BACKTEST_MAX_WEIGHT_FLOOR,
    TRADING_DAYS_PER_YEAR,
    GA_MIN_SECURITIES,
    GA_MAX_SECURITIES,
)
from src.returns import calculate_log_returns, calculate_expected_returns
from src.covariance import calculate_covariance_matrix
from src.metrics import (
    maximum_drawdown,
    downside_deviation,
    sortino_ratio,
    calmar_ratio,
)

from .types import MethodResults, PortfolioResult

logger = logging.getLogger(__name__)

METRIC_NAMES = [
    'annualised_return', 'annualised_volatility', 'sharpe_ratio',
    'downside_deviation', 'max_drawdown', 'calmar_ratio', 'sortino_ratio',
]

# Module-level state set before spawning worker pools.
_backtest_data = None
# Log returns (transposed: tickers x dates) and expected returns for weight optimisation.
_backtest_log_returns = None
_backtest_expected_returns = None


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


def _resolve_subset_and_er(portfolio, expected_returns_override=None):
    """Slice global state for a portfolio and resolve expected returns.

    Shared prelude for the weight helpers.

    :param portfolio: list of ticker strings.
    :param expected_returns_override: optional 1-D array aligned with
        ``portfolio`` order. When provided, replaces the global ER lookup.
    :return: tuple (subset, er, max_weight) where subset is the
        per-portfolio log-returns DataFrame (dates × tickers), er is the
        annualised expected-returns array, and max_weight is the per-asset
        upper bound used for SLSQP.
    """
    if len(portfolio) < 2:
        raise ValueError("Portfolio must contain at least 2 assets.")
    missing = set(portfolio) - set(_backtest_log_returns.index)
    if missing:
        raise KeyError(f"Tickers not found in data: {missing}")
    subset = _backtest_log_returns.loc[portfolio, :].transpose()
    if expected_returns_override is not None:
        er = np.asarray(expected_returns_override, dtype=float)
        if er.shape != (len(portfolio),):
            raise ValueError(
                f"expected_returns_override shape {er.shape} does not match "
                f"portfolio length {len(portfolio)}")
    else:
        er = _backtest_expected_returns.loc[subset.columns].values
    max_weight = max(1 / (len(portfolio) - 1), BACKTEST_MAX_WEIGHT_FLOOR)
    return subset, er, max_weight


def _resolve_cov_matrix(subset, *, use_copulae=False, forecast_variances=None):
    """Build the covariance matrix used by SLSQP weight optimisation.

    Three modes, mutually exclusive in caller intent:

    * ``forecast_variances=None``, ``use_copulae=False`` → Ledoit-Wolf
      shrunk sample covariance (current default).
    * ``forecast_variances=Series``, ``use_copulae=False`` → CCC
      ``D × R × D`` where R is shrunk sample correlation and D uses the
      provided (already-annualised) variances.
    * ``use_copulae=True``, ``forecast_variances`` optional → Copula-CCC.
      D uses ``forecast_variances`` if provided, else historical std.

    :param subset: log-returns DataFrame (dates × tickers).
    :param use_copulae: route via the copula correlation estimator.
    :param forecast_variances: optional Series of annualised variances
        indexed by ticker. Must contain every column in ``subset``.
    :return: numpy array covariance matrix.
    """
    if use_copulae and forecast_variances is None:
        # Backward-compatible copulae mode used by cc_copulae: copula
        # correlation but historical std for the diagonal. Kept distinct
        # from the CCC path so existing OOS Sharpe values are unchanged.
        from src.covariance import estimate_corr_using_copulas
        corr = estimate_corr_using_copulas(subset)
        D = np.diag(subset.std().values * np.sqrt(TRADING_DAYS_PER_YEAR))
        return np.matmul(np.matmul(D, corr), D)
    cov = calculate_covariance_matrix(
        subset, forecast_variances=forecast_variances, use_copulae=use_copulae,
    )
    return cov.values if hasattr(cov, 'values') else cov


def _max_sharpe_weights(portfolio, *, use_copulae=False,
                        forecast_variances=None,
                        expected_returns_override=None):
    """SLSQP max-Sharpe weights with pluggable ER and covariance sources.

    Reads the worker globals ``_backtest_log_returns`` and (unless
    overridden) ``_backtest_expected_returns``.
    """
    subset, er, max_weight = _resolve_subset_and_er(
        portfolio, expected_returns_override=expected_returns_override)
    cov = _resolve_cov_matrix(
        subset, use_copulae=use_copulae,
        forecast_variances=forecast_variances)

    from src.weights import optimise_weights
    result = optimise_weights(
        expected_returns=er, cov_matrix=cov,
        max_weight=max_weight,
        initial_weights=get_random_weights(portfolio),
    )
    if not result.success:
        logger.warning("Weight optimization did not converge: %s", result.message)
    return result['x']


def _min_variance_weights(portfolio, *, use_copulae=False,
                          forecast_variances=None):
    """SLSQP min-variance weights (objective = wᵀΣw, ER ignored)."""
    subset, _, max_weight = _resolve_subset_and_er(portfolio)
    cov = _resolve_cov_matrix(
        subset, use_copulae=use_copulae,
        forecast_variances=forecast_variances)

    from src.weights import optimise_weights
    result = optimise_weights(
        expected_returns=np.zeros(len(portfolio)), cov_matrix=cov,
        max_weight=max_weight,
        initial_weights=get_random_weights(portfolio),
        minimize_variance=True,
    )
    if not result.success:
        logger.warning("Min-variance optimization did not converge: %s",
                       result.message)
    return result['x']


def _equal_weights(portfolio):
    """1/N weighting for the portfolio."""
    if len(portfolio) < 1:
        raise ValueError("Cannot generate equal weights for an empty portfolio.")
    n = len(portfolio)
    return np.ones(n) / n


def _inverse_vol_weights(portfolio):
    """Naive risk parity: w_i ∝ 1/σ_i (per-ticker historical std).

    No covariance estimation — just per-ticker volatility from training-window
    log returns. Often surprisingly competitive vs full SLSQP per
    DeMiguel et al. (2009).
    """
    if len(portfolio) < 1:
        raise ValueError("Cannot generate inverse-vol weights for an empty portfolio.")
    subset, _, _ = _resolve_subset_and_er(portfolio)
    stds = subset.std().values
    # Guard against zero-variance series (would otherwise divide by zero)
    stds = np.where(stds > 0, stds, np.nan)
    inv = 1.0 / stds
    if not np.isfinite(inv).any():
        return _equal_weights(portfolio)
    inv = np.where(np.isfinite(inv), inv, 0.0)
    total = inv.sum()
    return inv / total if total > 0 else _equal_weights(portfolio)


def _risk_parity_weights(portfolio, *, use_copulae=False,
                         forecast_variances=None):
    """Equal Risk Contribution (ERC) weights.

    Each asset contributes equally to portfolio variance. Uses the existing
    SLSQP risk-budget objective in :mod:`src.weights`. Bridgewater-style
    All-Weather paradigm.
    """
    subset, _, max_weight = _resolve_subset_and_er(portfolio)
    cov = _resolve_cov_matrix(
        subset, use_copulae=use_copulae,
        forecast_variances=forecast_variances)
    from src.weights import optimise_weights
    result = optimise_weights(
        expected_returns=np.zeros(len(portfolio)), cov_matrix=cov,
        max_weight=max_weight,
        initial_weights=get_random_weights(portfolio),
        risk_parity=True,
    )
    if not result.success:
        logger.warning("Risk-parity optimization did not converge: %s",
                       result.message)
    return result['x']


def _max_diversification_weights(portfolio, *, use_copulae=False,
                                  forecast_variances=None):
    """Choueifaty & Coignard (2008) most-diversified portfolio.

    Maximises the diversification ratio DR = (w'σ) / sqrt(w'Σw), where σ is
    the per-asset volatility vector. Equivalent to maximising the ratio of
    weighted volatilities to portfolio volatility.

    Implemented as SLSQP minimising −DR² (squared to avoid negative-sqrt
    issues during line search). Constraints: long-only, weights sum to 1,
    per-asset cap = ``max_weight``.
    """
    from scipy.optimize import minimize

    subset, _, max_weight = _resolve_subset_and_er(portfolio)
    cov = _resolve_cov_matrix(
        subset, use_copulae=use_copulae,
        forecast_variances=forecast_variances)
    sigma = np.sqrt(np.diag(cov))

    def neg_dr_squared(w):
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return 0.0
        weighted_vol = float(w @ sigma)
        return -(weighted_vol * weighted_vol) / port_var

    n = len(portfolio)
    x0 = np.asarray(get_random_weights(portfolio), dtype=float)
    bounds = [(0.0, max_weight)] * n
    constraints = ({'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)},)
    result = minimize(neg_dr_squared, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 200, 'ftol': 1e-9})
    if not result.success:
        logger.warning("Max-diversification optimization did not converge: %s",
                       result.message)
        return _inverse_vol_weights(portfolio)
    return result.x


def optimal_weights(portfolio, use_copulae=False):
    """Backwards-compatible max-Sharpe wrapper.

    Kept for the public ``src.backtest`` API and existing tests. Delegates
    to :func:`_max_sharpe_weights`.
    """
    return _max_sharpe_weights(portfolio, use_copulae=use_copulae)


_BENCHMARK_DEFINITIONS = {
    'bench_spy': (['SPY'], np.array([1.0])),
    'bench_6040': (['SPY', 'AGG'], np.array([0.6, 0.4])),
}


def benchmark_portfolio(category, train_log_returns, oos_log_returns):
    """Return ``(tickers, weights)`` for a fixed benchmark.

    Returns ``(None, None)`` if any required ticker is missing from either
    the training or OOS DataFrame so the caller can skip the benchmark for
    that window. Bypasses :func:`_max_sharpe_weights` entirely — the
    ``max_weight`` floor ``1/(n-1)`` would divide by zero for ``n=1``.

    :param category: one of ``'bench_spy'``, ``'bench_6040'``.
    :param train_log_returns: DataFrame of training-window log returns
        (dates × tickers). Used to assert ticker availability.
    :param oos_log_returns: DataFrame of OOS log returns
        (dates × tickers). Used to assert ticker availability.
    :return: tuple of (list of tickers, np.ndarray of weights), or
        ``(None, None)`` if the benchmark is unavailable for the window.
    """
    spec = _BENCHMARK_DEFINITIONS.get(category)
    if spec is None:
        raise ValueError(f"Unknown benchmark category: {category!r}")
    tickers, weights = spec
    train_cols = set(train_log_returns.columns)
    oos_cols = set(oos_log_returns.columns)
    if not all(t in train_cols and t in oos_cols for t in tickers):
        return None, None
    return list(tickers), weights.copy()


def run_portfolio(portfolio, weights, oos_log_returns):
    """
    Buy-and-hold simulation over the OOS period with natural weight drift.

    :param portfolio: list of ticker strings.
    :param weights: initial weight allocations (same order as portfolio).
    :param oos_log_returns: DataFrame with dates as rows, tickers as columns.
                            Only the OOS period -- no offset needed.
    :return: list of daily portfolio log returns.
    """
    subset = oos_log_returns[portfolio]
    portfolio_returns = []
    w = weights.copy()
    for i in range(len(subset)):
        step_returns = subset.iloc[i].values
        # Portfolio growth factor: each asset's value scales by exp(r_log),
        # so the day's true portfolio log return is log(sum w_i * exp(r_i)).
        # (Pre-July-2026 this divided by 1 + sum(w*r_log) — a log/simple mix
        # that de-normalised the weights into ~6-10% phantom leverage over a
        # 5y window and biased every stored backtest Sharpe slightly upward.)
        growth = float(np.sum(w * np.exp(step_returns)))
        portfolio_returns.append(float(np.log(growth)))
        w = w * np.exp(step_returns) / growth
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
    # Annualise the downside deviation so it lives on the same scale as r/std
    # (pre-July-2026 the daily value was stored and fed to Sortino against an
    # annualised return — a ~sqrt(252) = 16x scale error in stored Sortinos).
    dd = downside_deviation(portfolio_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    r = np.mean(portfolio_returns) * TRADING_DAYS_PER_YEAR
    std = np.std(portfolio_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = r / std if std != 0 else 0.0
    return dict(zip(METRIC_NAMES, [
        r, std, sharpe, dd, max_dd,
        calmar_ratio(r, max_dd),
        sortino_ratio(r, dd),
    ]))


def get_statistics_with_sleeve(portfolio, weights, oos_log_returns,
                               sleeve_returns, alpha):
    """Like :func:`get_statistics`, but blends a managed-futures sleeve return
    series into the book at the portfolio level::

        combined = (1 - alpha) * book + alpha * sleeve

    ``sleeve_returns`` is the full-history sleeve Series; it is reindexed onto
    ``oos_log_returns.index`` and any missing date is treated as flat (0.0).
    ``alpha = 0`` reproduces :func:`get_statistics` exactly (the additivity
    contract). The existing single-portfolio path is left untouched.

    :param portfolio: list of ticker strings.
    :param weights: weight allocations (same order as portfolio).
    :param oos_log_returns: OOS log returns DataFrame (dates x tickers).
    :param sleeve_returns: daily sleeve log-return Series (datetime index).
    :param alpha: fraction of the book allocated to the sleeve (0..1).
    :return: dict keyed by metric name (same shape as get_statistics).
    """
    book = run_portfolio(portfolio, weights, oos_log_returns)
    sleeve = sleeve_returns.reindex(oos_log_returns.index).fillna(0.0).to_numpy()
    combined = ((1.0 - alpha) * np.asarray(book, dtype=float)
                + alpha * sleeve).tolist()
    max_dd = maximum_drawdown(combined)
    # Annualised, matching get_statistics (see the note there).
    dd = downside_deviation(combined) * np.sqrt(TRADING_DAYS_PER_YEAR)
    r = np.mean(combined) * TRADING_DAYS_PER_YEAR
    std = np.std(combined) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = r / std if std != 0 else 0.0
    return dict(zip(METRIC_NAMES, [
        r, std, sharpe, dd, max_dd,
        calmar_ratio(r, max_dd),
        sortino_ratio(r, dd),
    ]))


def evaluate_portfolios_with_sleeve(portfolios, weights_list, oos_log_returns,
                                    train_log_returns, category,
                                    sleeve_returns, alpha):
    """:func:`evaluate_portfolios` variant that blends the sleeve into both the
    OOS metrics and the in-sample Sharpe, so DSR / IS-OOS degradation / PBO stay
    consistent with what the sleeved arm actually traded.

    Reuses the base method's already-computed ``portfolios`` and ``weights_list``
    (only the evaluation differs), so registering a sleeve arm costs no extra
    weight optimisation.

    :param sleeve_returns: full-history sleeve log-return Series; reindexed
        internally onto both the OOS and the training index.
    :param alpha: fraction of the book allocated to the sleeve (0..1).
    :return: MethodResults object.
    """
    prs = []
    for p, w in zip(portfolios, weights_list):
        metrics = get_statistics_with_sleeve(
            p, w, oos_log_returns, sleeve_returns, alpha)
        try:
            is_stats = get_statistics_with_sleeve(
                p, w, train_log_returns, sleeve_returns, alpha)
            is_sr = is_stats['sharpe_ratio']
        except Exception:
            is_sr = None
        prs.append(PortfolioResult(
            portfolio=p, weights=w, metrics=metrics, is_sharpe=is_sr,
        ))
    return MethodResults(category=category, portfolios=prs)


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


def create_random_portfolios(columns, num_portfolios, min_securities=None,
                             max_securities=None):
    """
    Create a list of random portfolios by selecting random subsets of tickers.

    :param columns: available ticker names (Index or list).
    :param num_portfolios: number of portfolios to generate.
    :param min_securities: minimum cardinality (default: GA_MIN_SECURITIES).
    :param max_securities: maximum cardinality (default: GA_MAX_SECURITIES).
    :return: list of lists of ticker strings.
    """
    if min_securities is None:
        min_securities = GA_MIN_SECURITIES
    if max_securities is None:
        max_securities = GA_MAX_SECURITIES
    num_tickers = len(columns)
    ticker_names = list(columns)
    portfolios = []
    for _ in range(num_portfolios):
        portfolios.append(list(_random_selection(
            num_tickers, min_securities, max_securities, ticker_names)))
    return portfolios


def evaluate_portfolios(portfolios, weights_list, oos_log_returns,
                        train_log_returns, category):
    """
    Evaluate a set of portfolios and return a MethodResults object.

    :param portfolios: list of portfolios (each a list of ticker strings).
    :param weights_list: list of weight arrays (same length as portfolios).
    :param oos_log_returns: OOS log returns DataFrame (dates x tickers).
    :param train_log_returns: training-period log returns for IS Sharpe.
    :param category: category name for the MethodResults.
    :return: MethodResults object.
    """
    prs = []
    for p, w in zip(portfolios, weights_list):
        metrics = get_statistics(p, w, oos_log_returns)
        try:
            is_stats = get_statistics(p, w, train_log_returns)
            is_sr = is_stats['sharpe_ratio']
        except Exception:
            is_sr = None
        prs.append(PortfolioResult(
            portfolio=p, weights=w, metrics=metrics, is_sharpe=is_sr,
        ))
    return MethodResults(category=category, portfolios=prs)


# ---- GA Worker Helpers -------------------------------------------------------


def _random_selection(num_tickers, min_k, max_k, ticker_names):
    """Generate a random portfolio by selecting min_k..max_k tickers."""
    k = np.random.randint(min_k, max_k + 1)
    chosen = np.random.choice(num_tickers, k, replace=False)
    return [ticker_names[i] for i in chosen]


def _init_worker(training_data):
    """Pool initializer -- sets module globals in each worker process."""
    global _backtest_data
    _backtest_data = training_data


def _init_weight_worker(log_returns_T, expected_returns):
    """Pool initializer for weight computation workers."""
    global _backtest_log_returns, _backtest_expected_returns
    _backtest_log_returns = log_returns_T
    _backtest_expected_returns = expected_returns


def _compute_weights_for_portfolio(args):
    """Top-level function for Pool.map -- computes weights for a single portfolio.

    Must be top-level (not a lambda/closure) for macOS spawn-based multiprocessing.

    Accepts either the legacy 2-tuple ``(portfolio, mode)`` or the new
    3-tuple ``(portfolio, mode, kwargs)`` where ``kwargs`` carries small
    per-task overrides — typically the portfolio-sliced forecast variance
    Series or expected-return array. Two-tuple form remains supported so
    older test fixtures don't break.
    """
    if len(args) == 2:
        portfolio, mode = args
        kwargs = {}
    else:
        portfolio, mode, kwargs = args

    if mode == 'random':
        return get_random_weights(portfolio)
    if mode == 'optimal':
        return _max_sharpe_weights(portfolio)
    if mode == 'copulae':
        return _max_sharpe_weights(portfolio, use_copulae=True)
    if mode == 'optimal_ccc':
        return _max_sharpe_weights(
            portfolio, forecast_variances=kwargs['var'])
    if mode == 'min_variance':
        return _min_variance_weights(portfolio)
    if mode == 'equal':
        return _equal_weights(portfolio)
    if mode == 'inverse_vol':
        return _inverse_vol_weights(portfolio)
    if mode == 'risk_parity':
        return _risk_parity_weights(portfolio)
    if mode == 'max_diversification':
        return _max_diversification_weights(portfolio)
    if mode == 'optimal_arima_er':
        return _max_sharpe_weights(
            portfolio, expected_returns_override=kwargs['er'])
    if mode == 'optimal_garch':
        return _max_sharpe_weights(
            portfolio, forecast_variances=kwargs['var'])
    if mode == 'optimal_garch_copula':
        return _max_sharpe_weights(
            portfolio, use_copulae=True, forecast_variances=kwargs['var'])
    if mode == 'optimal_arima_garch':
        return _max_sharpe_weights(
            portfolio,
            expected_returns_override=kwargs['er'],
            forecast_variances=kwargs['var'])
    if mode == 'optimal_arima_garch_copula':
        return _max_sharpe_weights(
            portfolio, use_copulae=True,
            expected_returns_override=kwargs['er'],
            forecast_variances=kwargs['var'])
    raise ValueError(f"Unknown weight-computation mode: {mode!r}")


def create_portfolio(num_children):
    """
    Creates a cardinality-constrained portfolio using the C++ island GA.

    Subprocesses ``cpp/optimisation`` with ISLAND_GA params for a much
    stronger search than the previous pygad-based implementation.

    Each backtest worker runs one C++ call with ``num-islands=1`` because
    the outer ``mp.Pool`` is already parallelising 20 portfolios across
    cores; setting num-islands>1 would oversubscribe the machine.

    :num_children: GA population size — passed through as ``--pop-size``.
        BACKTEST_NUM_CHILDREN should be set in config to match the strength
        of the standalone island_ga (ISLAND_GA_POPULATION_SIZE).
    :return: A list of tickers.
    """
    import json
    import os
    import subprocess
    import tempfile

    from src.binary_io import write_binary_data
    from src.config import (
        CPP_BINARY_PATH,
        ISLAND_GA_NUM_GENERATIONS,
        ISLAND_GA_NUM_ELITES,
        ISLAND_GA_MIGRATION_INTERVAL,
        ISLAND_GA_MIGRATION_RATE,
        ISLAND_GA_MUTATION_RATE_INITIAL,
        ISLAND_GA_MUTATION_RATE_FINAL,
        ISLAND_GA_MIN_SECURITIES,
        ISLAND_GA_MAX_SECURITIES,
        RISK_FREE_RATE,
    )

    log_returns = calculate_log_returns(_backtest_data)
    fd, bin_path = tempfile.mkstemp(suffix='.bin', prefix='backtest_ga_')
    os.close(fd)
    try:
        write_binary_data(log_returns, bin_path)
        cmd = [
            CPP_BINARY_PATH, '--binary', '--data', bin_path,
            '--mode', 'ga',
            '--pop-size', str(num_children),
            '--generations', str(ISLAND_GA_NUM_GENERATIONS),
            '--num-islands', '1',
            '--num-elites', str(ISLAND_GA_NUM_ELITES),
            '--migration-interval', str(ISLAND_GA_MIGRATION_INTERVAL),
            '--migration-rate', str(ISLAND_GA_MIGRATION_RATE),
            '--mutation-initial', str(ISLAND_GA_MUTATION_RATE_INITIAL),
            '--mutation-final', str(ISLAND_GA_MUTATION_RATE_FINAL),
            '--min-etfs', str(ISLAND_GA_MIN_SECURITIES),
            '--max-etfs', str(ISLAND_GA_MAX_SECURITIES),
            '--risk-free-rate', str(RISK_FREE_RATE),
            '--top-k', '1',
            '--seed', '-1',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f'cpp/optimisation failed (rc={result.returncode}): '
                f'{result.stderr[-500:]}')
        out = json.loads(result.stdout)
        if not out.get('selected_tickers'):
            raise RuntimeError(
                f'cpp/optimisation returned no selection: {result.stdout[:500]}')
        return out['selected_tickers']
    finally:
        try:
            os.unlink(bin_path)
        except OSError:
            pass
