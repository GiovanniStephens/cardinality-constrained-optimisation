class PortfolioError(Exception):
    """Base exception for portfolio optimisation errors."""


class DownloadError(PortfolioError):
    """Error during data download (network, API, rate limit)."""


class ValidationError(PortfolioError):
    """Error during data or ticker validation."""


class OptimisationError(PortfolioError):
    """Error during portfolio optimisation."""


class DatabaseError(PortfolioError):
    """Error during database operations."""


class ForecastError(PortfolioError):
    """Error during ARIMA/GARCH forecasting."""
