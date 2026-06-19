"""Synthetic trend-following (TSMOM) sleeve.

Canonical time-series momentum (Moskowitz, Ooi & Pedersen 2012) on a fixed
basket of long-history liquid ETFs, used as a managed-futures proxy that spans
the full backtest period. The spec is FROZEN and parameter-free: every constant
comes from ``src.config`` and is never tuned to results, so the sleeve adds no
overfitting surface. Only the book-level blend weight ``alpha`` is swept
(in the backtest runner), never anything in here.

CAUSALITY (no look-ahead). The return realized on day *t* is
``position_t * r_t``, where ``r_t = log(p_t / p_{t-1})`` is the return *earned*
over the period ending at *t*, and ``position_t`` is decided from prices
strictly before *t* — the signal and inverse-vol sizing are ``.shift(1)``-ed and
held constant between monthly rebalances. Perturbing the price on day *t*
therefore changes the returns on days *t* and *t+1* (they contain ``p_t``) but
NEVER any sleeve return on a day before *t*, and never a position decided before
*t*. This is what ``tests/test_sleeves.py::test_sleeve_is_causal`` verifies.

Because the series is causal per-date, computing it once over full history and
then *slicing* a window out of it is leakage-free for that window — see
``src.sleeves.overlay``.
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    TSMOM_BASKET,
    TSMOM_BASKET_MULTI,
    TSMOM_USE_MULTI_MARKET,
    TSMOM_LOOKBACK_DAYS,
    TSMOM_VOL_LOOKBACK,
    TSMOM_TARGET_VOL_INSTR,
    TSMOM_TARGET_VOL_BOOK,
    TSMOM_REBALANCE_DAYS,
    TSMOM_ALLOW_SHORT,
    TRADING_DAYS_PER_YEAR,
)
from src.returns import calculate_log_returns

logger = logging.getLogger(__name__)

# Numerical safeguard (NOT a tuned knob): floor the annualised vol estimate to
# avoid divide-by-zero / runaway leverage when a rolling window is ~flat. Real
# ETF vols never approach this; only synthetic constant-price tests can.
_VOL_FLOOR = 1e-4


def _resolve_slot(prices, slot, candidates, min_obs):
    """First fallback ticker present in ``prices`` with >= ``min_obs`` non-NaN
    observations (enough to form a signal), or ``None`` if the slot is unusable.
    """
    for tk in candidates:
        if tk in prices.columns and prices[tk].notna().sum() >= min_obs:
            return tk
    logger.warning("TSMOM basket: no usable ticker for slot %r (tried %s)",
                   slot, candidates)
    return None


def _resolve_basket(prices, basket, min_obs):
    """Pick one ticker per slot. Supports two basket shapes:

    * **flat** ``{slot: [fallback_chain]}`` → ``{slot: ticker}`` (the original
      5-ETF basket; legs are equal-weighted).
    * **nested** ``{cluster: {slot: [fallback_chain]}}`` → ``{cluster: {slot:
      ticker}}`` (the multi-market basket; legs are cluster-balanced — see
      ``_aggregate``).

    Slots with no usable ticker are dropped; nested clusters left empty are
    dropped too. Resolved once per run so the basket never drifts between
    windows. The shape is detected from whether the basket's values are dicts.
    """
    nested = all(isinstance(v, dict) for v in basket.values())
    if not nested:
        resolved = {}
        for slot, candidates in basket.items():
            tk = _resolve_slot(prices, slot, candidates, min_obs)
            if tk is not None:
                resolved[slot] = tk
        logger.info("TSMOM basket resolved (flat): %s", resolved)
        return resolved

    resolved = {}
    for cluster, slots in basket.items():
        cl = {}
        for slot, candidates in slots.items():
            tk = _resolve_slot(prices, slot, candidates, min_obs)
            if tk is not None:
                cl[slot] = tk
        if cl:
            resolved[cluster] = cl
    logger.info("TSMOM basket resolved (nested, %d clusters): %s",
                len(resolved), resolved)
    return resolved


def _hold_monthly(daily, rebalance_days):
    """Hold an already-causal daily decision series constant between monthly
    rebalances: keep the value sampled every ``rebalance_days`` rows, NaN the
    rest, then forward-fill. The sampled values are themselves ``.shift(1)``-ed,
    so this never introduces look-ahead — it only reduces turnover.
    """
    keep = pd.Series(False, index=daily.index)
    keep.iloc[::rebalance_days] = True
    return daily.where(keep).ffill()


def _instrument_position(r, lookback, vol_lookback, target_vol_instr,
                         allow_short, rebalance_days):
    """Causal, monthly-rebalanced signed inverse-vol position for one
    instrument's daily log-return series ``r``.

    Signal = sign of the trailing ``lookback``-day cumulative return, using only
    data up to the day BEFORE the position is held. Size = ``target_vol_instr``
    / trailing annualised vol (also lagged). Both held constant between monthly
    rebalances. Returns a target-weight Series (NaN during warmup / flat vol).
    """
    trailing = r.rolling(lookback).sum().shift(1)
    signal = np.sign(trailing)
    if not allow_short:
        signal = signal.clip(lower=0.0)
    vol = (r.rolling(vol_lookback).std().shift(1)
           * np.sqrt(TRADING_DAYS_PER_YEAR))
    vol = vol.where(vol > _VOL_FLOOR)          # ~flat → NaN → no position
    size = signal * (target_vol_instr / vol)
    return _hold_monthly(size, rebalance_days)


def _resolved_tickers(resolved):
    """Flat list of every resolved ticker, for either basket shape:
    flat ``{slot: ticker}`` or nested ``{cluster: {slot: ticker}}``.
    """
    nested = all(isinstance(v, dict) for v in resolved.values())
    if nested:
        return [tk for cl in resolved.values() for tk in cl.values()]
    return list(resolved.values())


def _aggregate(resolved, log_returns, lookback, vol_lookback, target_vol_instr,
               allow_short, rebalance_days):
    """Combine per-leg P&L (``position * return``) into one daily book series.

    * **flat** ``{slot: ticker}`` → equal-weight across legs (the original
      5-ETF behaviour, byte-for-byte: a single ``mean(axis=1)`` over legs).
    * **nested** ``{cluster: {slot: ticker}}`` → two-level equal weight: mean
      legs *within* a cluster, then mean *across* clusters. Each cluster gets
      ``1/N_clusters`` of book risk regardless of how many legs it holds, so a
      numerous-but-correlated cluster (equities) can't dominate. Reduces to the
      flat result exactly when every cluster has one leg.

    The aggregate is a linear combination of the causal ``position * return``
    legs, so causality is inherited from ``_instrument_position`` — no new
    dependence on future prices is introduced here.
    """
    def _leg_pnl(tk):
        r = log_returns[tk]
        pos = _instrument_position(r, lookback, vol_lookback, target_vol_instr,
                                   allow_short, rebalance_days)
        return pos * r

    nested = all(isinstance(v, dict) for v in resolved.values())
    if not nested:
        legs = [_leg_pnl(tk) for tk in resolved.values()]
        return pd.concat(legs, axis=1).mean(axis=1)

    cluster_pnl = []
    for slots in resolved.values():
        legs = pd.concat([_leg_pnl(tk) for tk in slots.values()], axis=1)
        cluster_pnl.append(legs.mean(axis=1))          # equal-weight within
    return pd.concat(cluster_pnl, axis=1).mean(axis=1)  # equal-weight across


def compute_tsmom_returns(prices, basket=None, lookback=None, vol_lookback=None,
                          target_vol_instr=None, target_vol_book=None,
                          rebalance_days=None, allow_short=None):
    """Daily TSMOM sleeve log-return series over the full history of ``prices``.

    :param prices: wide DataFrame (dates x tickers) containing the basket
        candidate columns; index must be sorted ascending.
    :param basket: flat ``{slot: [ticker, fallback, ...]}`` or nested
        ``{cluster: {slot: [fallbacks]}}``. Default follows the
        ``config.TSMOM_USE_MULTI_MARKET`` flag (``TSMOM_BASKET_MULTI`` when on,
        flat ``TSMOM_BASKET`` when off). All other params default to their
        frozen ``config.TSMOM_*`` values — overridable only for tests.
    :return: ``pd.Series`` of daily sleeve log returns, indexed like ``prices``,
        with the warmup period (insufficient history) filled with 0.0.
    """
    if basket is None:
        basket = TSMOM_BASKET_MULTI if TSMOM_USE_MULTI_MARKET else TSMOM_BASKET
    lookback = TSMOM_LOOKBACK_DAYS if lookback is None else lookback
    vol_lookback = TSMOM_VOL_LOOKBACK if vol_lookback is None else vol_lookback
    target_vol_instr = (TSMOM_TARGET_VOL_INSTR if target_vol_instr is None
                        else target_vol_instr)
    target_vol_book = (TSMOM_TARGET_VOL_BOOK if target_vol_book is None
                       else target_vol_book)
    rebalance_days = (TSMOM_REBALANCE_DAYS if rebalance_days is None
                      else rebalance_days)
    allow_short = TSMOM_ALLOW_SHORT if allow_short is None else allow_short

    resolved = _resolve_basket(prices, basket, min_obs=lookback + vol_lookback)
    if not resolved:
        raise ValueError(
            "TSMOM basket resolved to zero instruments; cannot build the "
            "sleeve. Check the basket tickers exist in the price data.")

    log_returns = calculate_log_returns(prices[_resolved_tickers(resolved)])

    # Per-leg daily P&L (position * return), aggregated to one book series —
    # equal-weight legs (flat basket) or cluster-balanced (nested basket).
    combined = _aggregate(resolved, log_returns, lookback, vol_lookback,
                          target_vol_instr, allow_short, rebalance_days)

    # Book-level vol target: scale the aggregate to the book vol target using a
    # trailing realized-vol estimate, lagged one day and held monthly (causal).
    book_vol = (combined.rolling(vol_lookback).std().shift(1)
                * np.sqrt(TRADING_DAYS_PER_YEAR))
    book_vol = book_vol.where(book_vol > _VOL_FLOOR)
    book_scale = _hold_monthly(target_vol_book / book_vol, rebalance_days)
    sleeve = book_scale * combined

    return sleeve.replace([np.inf, -np.inf], np.nan).fillna(0.0)
