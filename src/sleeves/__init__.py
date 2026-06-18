"""Portfolio sleeves: separately-managed return streams blended into the book.

Currently houses the synthetic trend-following (TSMOM) managed-futures sleeve
used by the research backtest. See ``trend`` (signal engine) and ``overlay``
(full-history cache + per-window slicing).
"""

from src.sleeves.trend import compute_tsmom_returns

__all__ = ["compute_tsmom_returns"]
