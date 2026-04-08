"""Shared portfolio utility functions used across all optimisation methods."""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from src.config import (
    TRADING_DAYS_PER_YEAR,
    COV_SHRINKAGE_ENABLED,
    COV_MIN_OBS_RATIO,
    COV_MIN_OBS_RATIO_ERROR,
)

_cov_logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


# ─── Common interface ────────────────────────────────────────────────────────


@dataclass
class OptimisationResult:
    """Standard output from any optimiser.

    WARNING: sharpe_ratio is an in-sample value computed on the training data.
    It is biased upward by selection bias — typical IS-to-OOS degradation is
    30-50%. See CLAUDE.md "Sharpe Ratio Overfitting" for details.
    """
    selected_tickers: List[str]
    weights: np.ndarray
    sharpe_ratio: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate result integrity (skip for empty-result sentinels)."""
        if len(self.weights) == 0:
            return
        if len(self.weights) != len(self.selected_tickers):
            raise ValueError(
                f"weights length ({len(self.weights)}) != "
                f"selected_tickers length ({len(self.selected_tickers)})"
            )
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contain NaN or inf values")
        if np.any(self.weights < -1e-10):
            raise ValueError("weights contain negative values")
        if not np.isfinite(self.sharpe_ratio):
            raise ValueError(f"sharpe_ratio is not finite: {self.sharpe_ratio}")


def load_prices(exchange='US', csv_fallback=None, conn=None,
                last_n_days=None, min_coverage=None, ffill_limit=None):
    """Load price data from DB with CSV fallback, applying standard filters.

    :param exchange: exchange code for DB query (default 'US').
    :param csv_fallback: CSV path to use if DB is empty.
    :param conn: sqlite3 connection. If None, opens and closes one automatically.
    :param last_n_days: if set, keep only the most recent N calendar days.
    :param min_coverage: minimum non-null fraction to keep a column (default from config).
    :param ffill_limit: max consecutive NaN to forward-fill (default from config).
    :return: cleaned DataFrame with dates as index, tickers as columns.
    """
    from src.config import DATA_MIN_COVERAGE, DATA_FFILL_LIMIT
    if min_coverage is None:
        min_coverage = DATA_MIN_COVERAGE
    if ffill_limit is None:
        ffill_limit = DATA_FFILL_LIMIT

    own_conn = False
    if conn is None:
        from src import db
        conn = db.get_connection()
        own_conn = True

    try:
        from src import db as _db
        data = _db.load_prices(conn, exchange=exchange)
    finally:
        if own_conn:
            conn.close()

    if data.empty and csv_fallback:
        logger.info("No data in DB, falling back to CSV: %s", csv_fallback)
        return load_prices_csv(csv_fallback, min_coverage=min_coverage,
                               last_n_days=last_n_days)
    if data.empty:
        return data

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    if last_n_days is not None:
        cutoff = data.index[-1] - pd.Timedelta(days=last_n_days)
        data = data[data.index >= cutoff]
    data = data.dropna(axis=1, thresh=int(min_coverage * len(data)))
    data = data.ffill(limit=ffill_limit)
    return data


def load_training_data(exchange='US', csv_fallback=None, lookback_days=None,
                       min_coverage=None):
    """Load and filter price data for training, with DB-first-CSV-fallback.

    Convenience wrapper for entry-point scripts that consolidates the common
    pattern of loading from DB, falling back to CSV, and applying standard
    filters (lookback, coverage, forward-fill).

    :param exchange: exchange code for DB query (default 'US').
    :param csv_fallback: CSV path to use if DB is empty.
    :param lookback_days: restrict to last N calendar days (default from config).
    :param min_coverage: minimum non-null fraction to keep a column (default from config).
    :return: cleaned DataFrame.
    :raises ValueError: if the resulting DataFrame is empty.
    """
    from src.config import DATA_LOOKBACK_DAYS, DATA_MIN_COVERAGE
    if lookback_days is None:
        lookback_days = DATA_LOOKBACK_DAYS
    if min_coverage is None:
        min_coverage = DATA_MIN_COVERAGE

    data = load_prices(exchange=exchange, csv_fallback=csv_fallback,
                       last_n_days=lookback_days, min_coverage=min_coverage)
    if data.empty:
        raise ValueError("No price data available from DB or CSV fallback.")
    logger.info("Loaded price data: %d rows x %d columns", *data.shape)
    return data


def load_prices_csv(filename, min_coverage=0.95, last_n_days=None):
    """Load price data from CSV with coverage filtering and forward-fill.

    :param filename: path to CSV file (index_col=0 assumed).
    :param min_coverage: minimum fraction of non-null rows to keep a column.
    :param last_n_days: if set, keep only the most recent N calendar days.
    :return: cleaned DataFrame with dates as index, tickers as columns.
    """
    prices_df = pd.read_csv(filename, index_col=0)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()
    if last_n_days is not None:
        cutoff = prices_df.index[-1] - pd.Timedelta(days=last_n_days)
        prices_df = prices_df[prices_df.index >= cutoff]
    thresh = int(min_coverage * len(prices_df))
    prices_df = prices_df.loc[:, prices_df.notna().sum() >= thresh]
    prices_df = prices_df.ffill(limit=5)
    return prices_df


def load_data(filename, min_coverage=0.10, last_n_days=None):
    """Load price data with permissive coverage filtering.

    Convenience wrapper around load_prices_csv with a lower default
    min_coverage (0.10 vs 0.95) — used by entry points and scripts.
    Raises ValueError if the resulting DataFrame is empty.
    """
    prices_df = load_prices_csv(filename, min_coverage=min_coverage,
                                last_n_days=last_n_days)
    if prices_df.empty:
        raise ValueError(f"Loaded CSV '{filename}' is empty.")
    return prices_df


def calculate_log_returns(prices):
    """Calculate log returns, replacing NaN and inf with 0.

    :param prices: DataFrame of prices.
    :return: DataFrame of log returns.
    :raises ValueError: if the index is not monotonically increasing.
    """
    if not prices.index.is_monotonic_increasing:
        raise ValueError(
            "prices index must be sorted in ascending order. "
            "Unsorted data causes .shift(1) to compute returns against wrong date pairs."
        )
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.fillna(0)
    log_returns = log_returns.replace([np.inf, -np.inf], 0)
    return log_returns


def check_observation_ratio(T, N, context=""):
    """Guard against ill-conditioned covariance estimation.

    :param T: number of time-series observations.
    :param N: number of assets (columns).
    :param context: descriptive label for log messages.
    :raises ValueError: if T/N < COV_MIN_OBS_RATIO_ERROR.
    """
    if N <= 0:
        return
    ratio = T / N
    if ratio < COV_MIN_OBS_RATIO_ERROR:
        raise ValueError(
            f"T/N ratio ({T}/{N}={ratio:.1f}) is below {COV_MIN_OBS_RATIO_ERROR}. "
            f"Covariance matrix will be singular. {context}"
        )
    if ratio < COV_MIN_OBS_RATIO:
        _cov_logger.warning(
            "T/N ratio (%d/%d=%.1f) is below %d — covariance estimate may be "
            "noisy. %s", T, N, ratio, COV_MIN_OBS_RATIO, context,
        )


def _ledoit_wolf_covariance(log_returns):
    """Ledoit-Wolf shrinkage covariance estimate.

    :param log_returns: DataFrame of log returns.
    :return: (cov_matrix as DataFrame, shrinkage_coefficient).
    """
    from sklearn.covariance import ledoit_wolf
    cov_array, shrinkage = ledoit_wolf(log_returns.values)
    return pd.DataFrame(cov_array, index=log_returns.columns,
                        columns=log_returns.columns), shrinkage


def shrink_correlation_matrix(corr_matrix, log_returns):
    """Shrink a correlation matrix toward identity using Ledoit-Wolf alpha.

    For CCC paths: estimates alpha from ledoit_wolf(), returns (1-a)*R + a*I.

    :param corr_matrix: numpy array correlation matrix.
    :param log_returns: DataFrame of log returns (used to estimate shrinkage intensity).
    :return: shrunk correlation matrix as numpy array.
    """
    from sklearn.covariance import ledoit_wolf
    _, alpha = ledoit_wolf(log_returns.values)
    N = corr_matrix.shape[0]
    return (1 - alpha) * corr_matrix + alpha * np.eye(N)


def estimate_corr_using_copulas(data: pd.DataFrame,
                                diagnostics: bool = False) -> np.ndarray:
    """Estimate the correlation matrix using the copula method.

    Fits AR(1)-GARCH(1,1) with skew-t innovations to each series, then
    fits a Student-t copula to the standardised residuals and extracts
    the correlation matrix (``cop.sigma``).

    Falls back to the (optionally shrunk) sample correlation matrix if
    copula fitting fails.

    :param data: DataFrame of log returns.
    :param diagnostics: if True, log GARCH residual adequacy tests and
        copula model comparison (t-copula vs Gaussian).
    :return: numpy array correlation matrix.
    """
    from copulae import GaussianCopula, TCopula
    from muarch import MUArch
    from statsmodels.stats.diagnostic import acorr_ljungbox

    try:
        # scale=10 multiplies returns before fitting for numerical stability
        # (daily returns are ~0.001), then divides back internally.
        models = MUArch(data.shape[1], mean='AR', lags=1, dist='skewt', scale=10)
        models.fit(data)
        residuals = models.residuals()

        if diagnostics:
            for i, col in enumerate(data.columns):
                sq_resid = residuals[:, i] ** 2
                lb_result = acorr_ljungbox(sq_resid, lags=[10], return_df=True)
                p_value = lb_result['lb_pvalue'].values[0]
                if p_value < 0.05:
                    _cov_logger.warning(
                        "GARCH residuals for %s show remaining autocorrelation "
                        "(Ljung-Box p=%.4f < 0.05). Model may be inadequate.",
                        col, p_value)
                else:
                    _cov_logger.info(
                        "GARCH residuals for %s pass Ljung-Box test (p=%.4f).",
                        col, p_value)

        cop = TCopula(dim=data.shape[1])
        cop.fit(residuals)

        if diagnostics:
            gauss_cop = GaussianCopula(dim=data.shape[1])
            gauss_cop.fit(residuals)
            _cov_logger.info(
                "Copula comparison — t-copula log-lik: %.2f, "
                "Gaussian copula log-lik: %.2f",
                cop.log_lik(residuals), gauss_cop.log_lik(residuals))

        return cop.sigma
    except Exception as e:
        _cov_logger.warning(
            "Copula estimation failed: %s; falling back to sample correlation.", e)
        _cov_logger.debug("Copula traceback:", exc_info=True)
        corr = data.corr().values if isinstance(data, pd.DataFrame) else np.corrcoef(data, rowvar=False)
        if COV_SHRINKAGE_ENABLED:
            corr = shrink_correlation_matrix(corr, data)
        return corr


def calculate_covariance_matrix(log_returns, annualise=True, shrinkage=None, *,
                                 forecast_variances=None, use_copulae=False):
    """Covariance matrix of log returns.

    Supports three modes depending on the arguments:

    1. **Sample covariance** (default): Ledoit-Wolf shrinkage or raw sample
       covariance, controlled by *shrinkage*.
    2. **CCC model** (``forecast_variances`` provided): Bollerslev (1990)
       Cov = D × R × D, where D uses GARCH-forecast volatilities and R is
       the (optionally shrunk) historical correlation matrix.
    3. **Copula-CCC** (``use_copulae=True``): same as CCC but R is estimated
       via a Student-t copula on AR(1)-GARCH standardised residuals.

    When ``forecast_variances`` is provided, columns missing from it cause a
    logged warning and automatic fallback to mode 1.

    :param log_returns: DataFrame of log returns.
    :param annualise: multiply by 252 trading days (default True).
    :param shrinkage: True/False to force shrinkage on/off, or None to use
        the COV_SHRINKAGE_ENABLED config default.
    :param forecast_variances: optional Series of GARCH-forecast annualised
        variances, indexed by ticker.  Enables CCC mode.
    :param use_copulae: if True, estimate R via copula instead of sample
        correlation (requires copulae/muarch packages).
    :return: covariance matrix.  DataFrame in mode 1, numpy array in
        modes 2/3.
    """
    T, N = log_returns.shape
    if N >= 2:
        check_observation_ratio(T, N)

    # ── CCC / Copula-CCC path ────────────────────────────────────────────
    if forecast_variances is not None or use_copulae:
        # Validate forecast variances if provided
        if forecast_variances is not None:
            missing = set(log_returns.columns) - set(forecast_variances.index)
            if missing:
                _cov_logger.warning(
                    "Columns missing from forecast variances: %s. "
                    "Falling back to historical covariance.", missing)
                return calculate_covariance_matrix(
                    log_returns, annualise=annualise, shrinkage=shrinkage)

        # Correlation matrix R
        if use_copulae:
            corr = estimate_corr_using_copulas(log_returns)
        else:
            corr = log_returns.corr().values
            if COV_SHRINKAGE_ENABLED if shrinkage is None else shrinkage:
                corr = shrink_correlation_matrix(corr, log_returns)

        # Diagonal volatility matrix D
        if forecast_variances is not None:
            var_values = forecast_variances.loc[log_returns.columns].values.flatten()
            if np.any(var_values < 0):
                _cov_logger.warning(
                    "Negative forecast variances found; clipping to 0.")
                var_values = np.clip(var_values, 0, None)
            diag = np.sqrt(var_values)
        else:
            diag = log_returns.std().values * np.sqrt(TRADING_DAYS_PER_YEAR)

        D = np.diag(diag)
        cov = D @ corr @ D
        if annualise and forecast_variances is None:
            # forecast variances are already annualised; only annualise when
            # using historical std (which is daily scale).
            cov = cov * TRADING_DAYS_PER_YEAR
        return cov

    # ── Standard sample-covariance path ──────────────────────────────────
    use_shrinkage = COV_SHRINKAGE_ENABLED if shrinkage is None else shrinkage
    if use_shrinkage and N >= 2:
        cov, _ = _ledoit_wolf_covariance(log_returns)
    else:
        cov = log_returns.cov()

    if annualise:
        cov = cov * TRADING_DAYS_PER_YEAR
    return cov


def calculate_expected_returns(log_returns, annualise=True):
    """Mean log returns per asset.

    :param log_returns: DataFrame of log returns.
    :param annualise: multiply by 252 trading days (default True).
    :return: Series of expected returns.
    """
    er = log_returns.mean()
    if annualise:
        er = er * TRADING_DAYS_PER_YEAR
    return er


def calculate_variances(log_returns, annualise=True):
    """Variance of log returns per asset.

    :param log_returns: DataFrame of log returns.
    :param annualise: multiply by 252 trading days (default True).
    :return: Series of variances.
    """
    var = log_returns.var()
    if annualise:
        var = var * TRADING_DAYS_PER_YEAR
    return var


def prepare_portfolio_inputs(prices):
    """Compute standard portfolio inputs from a price DataFrame.

    :param prices: DataFrame with dates as index, tickers as columns.
    :return: (log_returns, expected_returns, cov_matrix) as numpy arrays.
    """
    log_returns = calculate_log_returns(prices)
    expected_returns = calculate_expected_returns(log_returns).values
    cov_matrix = calculate_covariance_matrix(log_returns).values
    return log_returns, expected_returns, cov_matrix


def sharpe_ratio(weights, expected_returns, cov_matrix):
    """Portfolio Sharpe ratio (positive).

    :param weights: array of portfolio weights.
    :param expected_returns: array/Series of expected returns.
    :param cov_matrix: covariance matrix (array or DataFrame).
    :return: Sharpe ratio as a float.
    """
    if len(weights) != len(expected_returns) or len(weights) != cov_matrix.shape[0]:
        raise ValueError(
            "weights, expected_returns, and cov_matrix dimensions must match"
        )
    p_return = np.sum(weights * expected_returns)
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if p_volatility == 0:
        return 0.0
    return p_return / p_volatility


def sharpe_loss(weights, expected_returns, cov_matrix):
    """Negated Sharpe ratio for use as a minimisation objective (e.g. SLSQP).

    :param weights: array of portfolio weights.
    :param expected_returns: array/Series of expected returns.
    :param cov_matrix: covariance matrix (array or DataFrame).
    :return: negative Sharpe ratio as a float.
    """
    return -sharpe_ratio(weights, expected_returns, cov_matrix)


# Backward-compatible alias
negative_sharpe_ratio = sharpe_loss


def equal_weight_fitness(selection_mask, expected_returns, cov_matrix,
                         min_count, max_count, min_return=None):
    """Equal-weight Sharpe ratio for a binary selection vector with cardinality constraints.

    :param selection_mask: binary array (1 = selected).
    :param expected_returns: array of annualised expected returns per asset.
    :param cov_matrix: annualised covariance matrix (array).
    :param min_count: minimum number of selected assets.
    :param max_count: maximum number of selected assets.
    :param min_return: optional minimum portfolio return threshold.
    :return: Sharpe ratio (float), or -1e4 if constraints violated.
    """
    selected = selection_mask == 1
    n = np.sum(selected)
    if n < min_count or n > max_count:
        return -1e4
    if not np.any(selected):
        return 0.0
    filtered_returns = expected_returns[selected]
    filtered_cov = cov_matrix[np.ix_(selected, selected)]
    weights = np.ones(n) / n
    port_return = np.dot(weights, filtered_returns)
    if min_return is not None and port_return < min_return:
        return -1e4
    port_variance = np.dot(weights, np.dot(filtered_cov, weights))
    return port_return / np.sqrt(port_variance) if port_variance > 0 else 0.0


# ─── Performance Metrics ──────────────────────────────────────────────────────


def maximum_drawdown(portfolio_returns):
    """
    Calculates the out-of-sample maximum drawdown
    from the simulation.

    :portfolio_returns: The input portfolio returns. List of floats.
    :return: The maximum drawdown, which is the percentage drawdown
             from the highest peak to the lowest low.
    """
    if not portfolio_returns:
        raise ValueError("portfolio_returns cannot be empty")
    cummax = []
    cum_return = []
    drawdowns = []

    cummax.append(np.exp(portfolio_returns[0]))
    cum_return.append(np.exp(portfolio_returns[0]))
    drawdowns.append(0)
    for i in range(1, len(portfolio_returns)):
        cummax.append(max(np.exp(portfolio_returns[i]) * cum_return[i-1], cum_return[i-1]))
        cum_return.append(np.exp(portfolio_returns[i]) * cum_return[i-1])
        drawdowns.append(cum_return[i] / cummax[i] - 1)
    return min(drawdowns)


def downside_deviation(portfolio_returns, mar=0):
    """
    Calculates the downside deviation of the portfolio
    returns.

    :portfolio_returns: The input portfolio returns. List of floats.
    :mar: threshold below which one would calculate the deviation.
    :return: downside deviation.
    """
    if not portfolio_returns:
        return 0.0
    squared_dev = 0
    for i in portfolio_returns:
        if i < mar:
            squared_dev += (i - mar)**2
    return (squared_dev / len(portfolio_returns))**0.5


def sortino_ratio(r, downside_deviation, MAR=0):
    """
    Calculates the Sortino ratio given the inputs.

    :r: float for the portfolio returns (annualised)
    :downside_deviation: the standard deviation of the
                         returns below MAR.
    :MAR: The threshold under which the deviation is calculated.
    :return: float for the Sortino Ratio.
    """
    if downside_deviation == 0:
        return 0.0
    return (r - MAR) / downside_deviation


def calmar_ratio(r, downside_drawdown):
    """
    Calculates that the portfolio Calmar ratio would be.

    :r: float for the portfolio returns (annualised)
    :downside_deviation: The maximum drawdown over a period in % terms.
    :return: a float for the Calmar ratio
    """
    if downside_drawdown == 0:
        return 0.0
    return r / abs(downside_drawdown)


# ─── Overfitting Detection ──────────────────────────────────────────────────

from scipy.stats import norm

# In-sample Sharpe thresholds for annual equity portfolios.
# See CLAUDE.md "Sharpe Ratio Overfitting" for academic backing.
SHARPE_WARN_THRESHOLD = 2.0
SHARPE_CRITICAL_THRESHOLD = 3.0

_overfit_logger = logging.getLogger(__name__)


def sharpe_ratio_variance(sr, n, skewness=0.0, excess_kurtosis=0.0):
    """Variance of the Sharpe ratio estimator (Lo 2002, Bailey & López de Prado 2014).

    Accounts for non-normality of returns via skewness and excess kurtosis.
    For normal returns (skewness=0, excess_kurtosis=0), simplifies to
    (1 + sr^2/2) / n.

    :param sr: observed Sharpe ratio.
    :param n: number of return observations.
    :param skewness: sample skewness of returns (default 0).
    :param excess_kurtosis: sample excess kurtosis of returns (default 0).
    :return: variance of the SR estimator.
    """
    return (1 - skewness * sr + (excess_kurtosis / 4) * sr ** 2) / n


def deflated_sharpe_ratio(observed_sr, n, num_trials, skewness=0.0,
                          excess_kurtosis=0.0, sr_benchmark=0.0):
    """Probability that observed Sharpe is genuine after multiple testing correction.

    Implements the Deflated Sharpe Ratio of Bailey & López de Prado (2014).
    Returns P(SR > E[max SR under null]) — values near 1.0 indicate the
    observed Sharpe likely reflects genuine skill, values near 0 indicate
    it is likely due to overfitting.

    :param observed_sr: the best observed Sharpe ratio.
    :param n: number of return observations used to compute the SR.
    :param num_trials: number of independent strategy variations tested
        (for GA: ~population_size * generations).
    :param skewness: sample skewness of returns.
    :param excess_kurtosis: sample excess kurtosis of returns.
    :param sr_benchmark: Sharpe ratio of the null hypothesis (default 0).
    :return: DSR probability in [0, 1].
    """
    sr_var = sharpe_ratio_variance(observed_sr, n, skewness, excess_kurtosis)
    sr_std = np.sqrt(max(sr_var, 1e-10))

    # Expected maximum SR under the null (Euler-Mascheroni approximation)
    euler_gamma = 0.5772156649
    if num_trials <= 1:
        expected_max_sr = sr_benchmark
    else:
        z = norm.ppf(1 - 1 / num_trials)
        expected_max_sr = sr_std * (
            (1 - euler_gamma) * z
            + euler_gamma * norm.ppf(1 - 1 / (num_trials * np.e))
        ) + sr_benchmark

    # DSR = P(observed SR > expected max SR under null)
    return float(norm.cdf((observed_sr - expected_max_sr) / sr_std))


def warn_if_sharpe_suspicious(sr, context, logger=None):
    """Log warnings if a Sharpe ratio is suspiciously high (likely overfit).

    :param sr: observed Sharpe ratio.
    :param context: descriptive label for log messages (e.g. "GA in-sample").
    :param logger: logger instance (defaults to module logger).
    """
    log = logger or _overfit_logger
    if sr > SHARPE_CRITICAL_THRESHOLD:
        log.warning(
            "%s: Sharpe=%.2f exceeds %.1f — almost certainly overfit on annual "
            "data (Harvey et al. 2016). Apply heavy OOS discount. "
            "See CLAUDE.md 'Sharpe Ratio Overfitting'.",
            context, sr, SHARPE_CRITICAL_THRESHOLD,
        )
    elif sr > SHARPE_WARN_THRESHOLD:
        log.warning(
            "%s: Sharpe=%.2f exceeds %.1f — likely inflated by in-sample "
            "optimisation bias. Expect 30-50%% OOS degradation. "
            "See CLAUDE.md 'Sharpe Ratio Overfitting'.",
            context, sr, SHARPE_WARN_THRESHOLD,
        )


# ─── Binary data format for C++ optimiser ────────────────────────────────────


def write_binary_data(log_returns, path):
    """Write log returns matrix in binary format for the C++ optimiser.

    Format: uint32 num_rows, uint32 num_cols, then num_cols null-terminated
    ticker strings, then num_rows * num_cols float64 values in row-major order.

    :param log_returns: DataFrame of log returns (index=dates, columns=tickers).
    :param path: output file path.
    """
    import struct

    tickers = list(log_returns.columns)
    mat = log_returns.values.astype(np.float64)
    num_rows, num_cols = mat.shape

    with open(path, 'wb') as f:
        f.write(struct.pack('<II', num_rows, num_cols))
        for ticker in tickers:
            f.write(ticker.encode('utf-8') + b'\x00')
        f.write(mat.tobytes(order='C'))


def read_binary_data(path):
    """Read binary data file written by write_binary_data.

    :param path: input file path.
    :return: (log_returns DataFrame, tickers list).
    """
    import struct

    with open(path, 'rb') as f:
        num_rows, num_cols = struct.unpack('<II', f.read(8))
        tickers = []
        for _ in range(num_cols):
            chars = []
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                chars.append(c)
            tickers.append(b''.join(chars).decode('utf-8'))
        data = np.frombuffer(f.read(num_rows * num_cols * 8), dtype=np.float64)
        mat = data.reshape(num_rows, num_cols)

    return pd.DataFrame(mat, columns=tickers), tickers


# ─── Portfolio Variance ──────────────────────────────────────────────────────


def calculate_portfolio_variance(weights, cov_matrix):
    """Portfolio variance: w^T @ Cov @ w.

    Accepts numpy arrays or pandas DataFrames for *cov_matrix*.

    :param weights: array of portfolio weights.
    :param cov_matrix: covariance matrix (array or DataFrame).
    :return: portfolio variance as a float.
    """
    cov = cov_matrix.values if hasattr(cov_matrix, 'values') else np.asarray(cov_matrix)
    w = np.asarray(weights).flatten()
    return float(np.dot(w, np.dot(cov, w)))


# ─── Risk-Parity Helpers ────────────────────────────────────────────────────


def _risk_parity_portfolio_var(w, V):
    """Portfolio variance via np.matrix (internal, used by risk-parity)."""
    w = np.matrix(w)
    return (w * V * w.T)[0, 0]


def calculate_risk_contribution(w, V):
    """Asset contribution to total risk for risk-parity objective.

    :param w: weight vector.
    :param V: covariance matrix (np.matrix).
    :return: column vector of risk contributions.
    """
    w = np.matrix(w)
    portfolio_var = _risk_parity_portfolio_var(w, V)
    if portfolio_var <= 0:
        return np.zeros_like(w.T)
    sigma = np.sqrt(portfolio_var)
    MRC = V * w.T
    RC = np.multiply(MRC, w.T) / sigma
    return RC


def risk_budget_objective(x, pars):
    """Risk-parity objective: minimise deviation from equal risk contribution.

    :param x: weight vector.
    :param pars: [covariance_matrix, risk_target_proportions].
    :return: sum of squared deviations from target risk contributions.
    """
    V = pars[0]     # covariance table
    x_t = pars[1]   # risk target in percent of portfolio risk
    sig_p = np.sqrt(_risk_parity_portfolio_var(x, V))
    risk_target = np.asmatrix(np.multiply(sig_p, x_t))
    asset_RC = calculate_risk_contribution(x, V)
    J = sum(np.square(asset_RC - risk_target.T))[0, 0]
    return J


# ─── Weight Optimisation ─────────────────────────────────────────────────────


def optimise_weights(selection_vector=None, data=None, min_weight=0.0,
                     max_weight=1.0, min_return=None, *,
                     expected_returns=None, cov_matrix=None,
                     target_return=None, target_risk=None,
                     initial_weights=None, risk_parity=False):
    """SLSQP weight optimisation for a portfolio.

    Can be called in two modes:

    1. **Selection-vector mode** (original API): pass ``selection_vector``
       and ``data`` (prices DataFrame).  Log returns, expected returns, and
       covariance are computed internally.

    2. **Pre-computed mode**: pass ``expected_returns`` and ``cov_matrix``
       directly (as numpy arrays).  Useful when the caller has already
       computed these (e.g. backtest weight workers, GA fitness closures).

    :param selection_vector: binary array (1 = selected, 0 = not).
        Required in mode 1, ignored in mode 2.
    :param data: DataFrame of prices (index=dates, columns=tickers).
        Required in mode 1, ignored in mode 2.
    :param min_weight: lower bound per position weight.
    :param max_weight: upper bound per position weight.
    :param min_return: if set, adds an inequality constraint for minimum
        annualised portfolio return.
    :param expected_returns: pre-computed expected returns (array).
        If provided, ``cov_matrix`` must also be provided.
    :param cov_matrix: pre-computed covariance matrix (array).
        If provided, ``expected_returns`` must also be provided.
    :param target_return: if set, adds an equality constraint for the
        annualised portfolio return.
    :param target_risk: if set, adds an equality constraint for the
        annualised portfolio standard deviation.
    :param initial_weights: starting point for the optimiser.  Defaults to
        equal weights.
    :param risk_parity: if True, minimise risk-budget deviation instead of
        negative Sharpe ratio.
    :return: scipy.optimize.OptimizeResult with optimised weights in .x
    """
    from scipy.optimize import minimize

    # ── Resolve inputs ────────────────────────────────────────────────────
    if expected_returns is not None and cov_matrix is not None:
        # Pre-computed mode
        er = np.asarray(expected_returns)
        cov = np.asarray(cov_matrix)
        n = len(er)
    elif selection_vector is not None and data is not None:
        # Selection-vector mode (original API)
        selected = data.columns[selection_vector == 1]
        log_returns = calculate_log_returns(data[selected])
        er = calculate_expected_returns(log_returns).values
        cov = calculate_covariance_matrix(log_returns).values
        n = len(selected)
    else:
        raise ValueError(
            "Either (selection_vector, data) or "
            "(expected_returns, cov_matrix) must be provided."
        )

    if initial_weights is not None:
        x0 = np.asarray(initial_weights, dtype=float)
    else:
        x0 = np.ones(n) / n

    # ── Constraints ───────────────────────────────────────────────────────
    bounds = [(min_weight, max_weight) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    if min_return is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, _er=er: np.dot(_er, x) - min_return,
        })
    if target_return is not None and target_risk is None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x, _er=er: np.sum(_er * x) - target_return,
        })
    if target_risk is not None and target_return is None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x, _cov=cov: target_risk -
            np.sqrt(np.dot(x.T, np.dot(_cov, x))),
        })

    # ── Objective ─────────────────────────────────────────────────────────
    if risk_parity:
        risk_proportion = [1 / n] * n
        return minimize(risk_budget_objective, x0,
                        args=([np.matrix(cov), risk_proportion]),
                        method='SLSQP', bounds=bounds,
                        constraints=constraints)

    def objective(x):
        return sharpe_loss(x, er, cov)

    return minimize(objective, x0=x0, method='SLSQP',
                    bounds=bounds, constraints=constraints)


# ─── DB Persistence ──────────────────────────────────────────────────────────


def save_optimisation_result(conn, selected_tickers, weights, prices,
                             script_name, params=None, exchange='US',
                             elapsed_seconds=None):
    """Compute portfolio metrics and persist an optimisation run to DB.

    Consolidates the common pattern: compute log returns for selected
    tickers → expected returns and covariance → portfolio return/vol →
    save to ``optimisation_runs`` and ``portfolio_holdings``.

    :param conn: sqlite3 connection.
    :param selected_tickers: list of ticker symbols in the portfolio.
    :param weights: array of portfolio weights (same order as tickers).
    :param prices: full price DataFrame (will be sliced to selected_tickers).
    :param script_name: identifier for the optimisation script.
    :param params: extra parameter dict merged into the DB params column.
    :param exchange: exchange code (default 'US').
    :param elapsed_seconds: optional wall-clock time for the run.
    :return: run_id (int) from the database.
    """
    from src import db

    log_returns = calculate_log_returns(prices[selected_tickers])
    er = calculate_expected_returns(log_returns)
    cov = calculate_covariance_matrix(log_returns)
    w = np.asarray(weights)
    portfolio_return = float(np.dot(w, er))
    portfolio_vol = float(
        np.sqrt(calculate_portfolio_variance(w, cov)))
    sr = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0

    all_params = {'script': script_name}
    if params:
        all_params.update(params)

    results = {
        'best_sharpe': sr,
        'portfolio_return': portfolio_return,
        'portfolio_volatility': portfolio_vol,
        'num_selected': len(selected_tickers),
    }
    if elapsed_seconds is not None:
        results['elapsed_seconds'] = elapsed_seconds

    run_id = db.save_optimisation_run(
        conn, params=all_params, results=results,
        holdings=list(zip(selected_tickers, weights)),
        exchange=exchange,
    )
    return run_id
