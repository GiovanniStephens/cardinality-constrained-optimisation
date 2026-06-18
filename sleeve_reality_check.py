"""Reality-check: synthetic TSMOM sleeve vs real managed-futures ETFs.

NON-BLOCKING sanity check (not the primary measurement). Confirms the
parameter-free synthetic trend sleeve behaves like real CTAs over the window
where real funds exist. DBMF (2019) and KMLM (2020) are excluded from the
backtest by the 1260-day history filter, so we load them directly here.

A good result (meaningfully positive correlation + crisis-quarter sign
agreement) licenses trusting the synthetic sleeve's CPCV verdict as a real read
on managed-futures diversification. A poor result means a null/positive backtest
result is uninformative — the synthetic proxy isn't a faithful CTA stand-in.

Run:  python sleeve_reality_check.py
"""

import logging

import numpy as np
import pandas as pd

from src import db
from src.returns import calculate_log_returns
from src.sleeves.overlay import get_cached_sleeve_series

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REAL_CTA_ETFS = ["DBMF", "KMLM"]
TRADING_DAYS = 252
# Quarters where real CTAs are expected to print strongly positive (crisis
# alpha): the 2022 equity+bond selloff and (more mixed) the 2020 COVID quarter.
CRISIS_QUARTERS = ["2022Q2", "2022Q3", "2020Q1"]


def _load_real_returns(conn, symbols):
    prices = db.load_prices(
        conn, exchange="US", tickers=symbols,
        exclude_flagged=False, min_coverage=None, ffill_limit=5)
    if prices.empty:
        return pd.DataFrame()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return calculate_log_returns(prices)


def _summarise(name, synth, real):
    """Compare one real CTA return series against the synthetic sleeve."""
    common = synth.index.intersection(real.index)
    if len(common) < 60:
        logger.info("  %-5s  only %d common days — skipping", name, len(common))
        return
    s = synth.reindex(common)
    r = real.reindex(common)

    corr = float(np.corrcoef(s.values, r.values)[0, 1])
    synth_vol = float(s.std() * np.sqrt(TRADING_DAYS))
    real_vol = float(r.std() * np.sqrt(TRADING_DAYS))

    # Quarterly log returns + sign agreement.
    sq = s.resample("QE").sum()
    rq = r.resample("QE").sum()
    sign_agree = float((np.sign(sq) == np.sign(rq)).mean())

    logger.info("  %-5s  common=%d days (%s → %s)", name, len(common),
                common.min().date(), common.max().date())
    logger.info("        daily corr      = %+.3f   %s", corr,
                "PASS" if corr >= 0.3 else "WEAK/CHECK")
    logger.info("        synth vol=%.1f%%  real vol=%.1f%%  (target 10%%)",
                synth_vol * 100, real_vol * 100)
    logger.info("        quarterly sign agreement = %.0f%%", sign_agree * 100)

    # Crisis-quarter behaviour.
    for q in CRISIS_QUARTERS:
        if q in sq.index.to_period("Q").astype(str):
            mask = sq.index.to_period("Q").astype(str) == q
            sv = float(sq[mask].iloc[0])
            rv = float(rq[mask].iloc[0])
            agree = "agree" if np.sign(sv) == np.sign(rv) else "DISAGREE"
            logger.info("        %s  synth=%+.1f%%  real=%+.1f%%  [%s]",
                        q, sv * 100, rv * 100, agree)


def main():
    conn = db.get_connection()
    try:
        logger.info("Building synthetic TSMOM sleeve (full history)...")
        synth = get_cached_sleeve_series(conn)
        logger.info("Synthetic sleeve: %d days, %s → %s, ann.vol=%.1f%%",
                    len(synth), synth.index.min().date(),
                    synth.index.max().date(),
                    float(synth.std() * np.sqrt(TRADING_DAYS)) * 100)

        real = _load_real_returns(conn, REAL_CTA_ETFS)
        if real.empty:
            logger.warning("No real CTA ETF prices found in the DB (%s). "
                           "Backfill DBMF/KMLM to run the cross-check.",
                           REAL_CTA_ETFS)
            return

        logger.info("\nSynthetic vs real CTAs:")
        for sym in REAL_CTA_ETFS:
            if sym in real.columns:
                _summarise(sym, synth, real[sym])
            else:
                logger.info("  %-5s  not in DB — skipping", sym)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
