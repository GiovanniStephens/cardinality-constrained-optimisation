"""Covariance estimation: sample, Ledoit-Wolf shrinkage, CCC, and copula-CCC."""

import hashlib
import logging

import numpy as np
import pandas as pd

from src.config import (
    TRADING_DAYS_PER_YEAR,
    COV_SHRINKAGE_ENABLED,
    COV_MIN_OBS_RATIO,
    COV_MIN_OBS_RATIO_ERROR,
    COPULA_GARCH_SCALE,
    COPULA_DIAGNOSTIC_LAGS,
    COPULA_TYPE,
    STATISTICAL_SIGNIFICANCE_LEVEL,
)

_cov_logger = logging.getLogger(__name__)


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


# Per-process cache of standardised AR(1)-GARCH(1,1) residuals, keyed by
# (ticker_name, hash of the series values). Survives across calls within
# a single Python process; multiprocessing workers each hold their own.
_garch_residuals_cache: dict[str, np.ndarray] = {}


def _garch_cache_key(name: str, values: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(str(name).encode('utf-8'))
    h.update(b':')
    h.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    return h.hexdigest()


def clear_garch_cache() -> None:
    """Clear the per-process GARCH residuals cache."""
    _garch_residuals_cache.clear()


def _fit_garch_residuals_cached(values: np.ndarray, name: str,
                                 scale: float = COPULA_GARCH_SCALE) -> np.ndarray:
    """Fit AR(1)-GARCH(1,1) skew-t to a single series and return standardised
    residuals. Caches by (name, series bytes) so repeated tickers within a
    process pay no fit cost.

    Mirrors muarch's per-series behaviour (mean='AR', lags=1, dist='skewt',
    scale=COPULA_GARCH_SCALE) but goes through arch directly to enable the
    cache.
    """
    key = _garch_cache_key(name, values)
    cached = _garch_residuals_cache.get(key)
    if cached is not None:
        return cached
    from arch import arch_model
    res = arch_model(
        values * scale, mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='skewt',
        rescale=False,
    ).fit(disp='off', show_warning=False)
    std_resid = np.asarray(res.std_resid, dtype=np.float64)
    # Drop any leading NaN from the AR(1) initial-condition slot.
    std_resid = std_resid[~np.isnan(std_resid)]
    _garch_residuals_cache[key] = std_resid
    return std_resid


def estimate_corr_using_copulas(data: pd.DataFrame,
                                diagnostics: bool = False,
                                copula_type: str | None = None,
                                strict: bool = False) -> np.ndarray:
    """Estimate the correlation matrix using the copula method.

    Fits AR(1)-GARCH(1,1) with skew-t innovations to each series, then
    fits a copula (Gaussian or Student-t, see ``copula_type``) to the
    standardised residuals and extracts the correlation matrix
    (``cop.sigma``).

    Per-ticker GARCH fits are cached in :data:`_garch_residuals_cache`,
    keyed by (ticker name, series bytes). Repeat fits for the same
    ticker+series within a process are O(dict-lookup).

    Falls back to the (optionally shrunk) sample correlation matrix if
    fitting fails.

    :param data: DataFrame of log returns.
    :param diagnostics: if True, log GARCH residual adequacy tests and
        copula model comparison (t-copula vs Gaussian).
    :param copula_type: ``'gaussian'`` (closed-form, O(N²)) or ``'t'``
        (iterative, super-cubic). Defaults to :data:`config.COPULA_TYPE`.
    :param strict: if True, re-raise on fitting failure instead of degrading
        to sample correlation. The production rebalance passes True so a
        broken copulae/scipy ABI stops the run rather than silently shipping
        a sample-correlation book labelled cc_copulae; long backtest/CPCV
        runs keep the default (crashing mid-run is worse there).
    :return: numpy array correlation matrix.
    """
    if copula_type is None:
        copula_type = COPULA_TYPE
    if copula_type not in ('gaussian', 't'):
        raise ValueError(
            f"copula_type must be 'gaussian' or 't', got {copula_type!r}")

    try:
        # Imported inside the try so a broken copulae/scipy ABI (e.g. scipy
        # dropping a C symbol copulae's extension links against) degrades to the
        # sample-correlation fallback below instead of crashing the caller.
        from copulae import GaussianCopula, TCopula
        from statsmodels.stats.diagnostic import acorr_ljungbox
        residuals_cols = []
        for col in data.columns:
            values = np.asarray(data[col].values, dtype=np.float64)
            residuals_cols.append(_fit_garch_residuals_cached(values, str(col)))
        # Align lengths in case different fits trimmed differently.
        min_len = min(len(r) for r in residuals_cols)
        residuals = np.column_stack([r[-min_len:] for r in residuals_cols])

        if diagnostics:
            for i, col in enumerate(data.columns):
                sq_resid = residuals[:, i] ** 2
                lb_result = acorr_ljungbox(sq_resid, lags=[COPULA_DIAGNOSTIC_LAGS], return_df=True)
                p_value = lb_result['lb_pvalue'].values[0]
                if p_value < STATISTICAL_SIGNIFICANCE_LEVEL:
                    _cov_logger.warning(
                        "GARCH residuals for %s show remaining autocorrelation "
                        "(Ljung-Box p=%.4f < %.2f). Model may be inadequate.",
                        col, p_value, STATISTICAL_SIGNIFICANCE_LEVEL)
                else:
                    _cov_logger.info(
                        "GARCH residuals for %s pass Ljung-Box test (p=%.4f).",
                        col, p_value)

        cop_cls = GaussianCopula if copula_type == 'gaussian' else TCopula
        cop = cop_cls(dim=data.shape[1])
        cop.fit(residuals)

        if diagnostics:
            other_cls = TCopula if copula_type == 'gaussian' else GaussianCopula
            other = other_cls(dim=data.shape[1])
            other.fit(residuals)
            _cov_logger.info(
                "Copula comparison — %s log-lik: %.2f, %s log-lik: %.2f",
                cop_cls.__name__, cop.log_lik(residuals),
                other_cls.__name__, other.log_lik(residuals))

        return cop.sigma
    except (ValueError, RuntimeError, TypeError, ImportError, np.linalg.LinAlgError) as e:
        if strict:
            _cov_logger.error(
                "Copula estimation failed (%s: %s) and strict=True — "
                "refusing to degrade to sample correlation.",
                type(e).__name__, e)
            raise
        _cov_logger.warning(
            "Copula estimation failed (%s: %s); falling back to sample "
            "correlation.", type(e).__name__, e)
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
        correlation (requires the copulae package).
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
