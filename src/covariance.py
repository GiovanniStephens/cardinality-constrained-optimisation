"""Covariance estimation: sample, Ledoit-Wolf shrinkage, CCC, and copula-CCC."""

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
        models = MUArch(data.shape[1], mean='AR', lags=1, dist='skewt', scale=COPULA_GARCH_SCALE)
        models.fit(data)
        residuals = models.residuals()

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
    except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as e:
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
