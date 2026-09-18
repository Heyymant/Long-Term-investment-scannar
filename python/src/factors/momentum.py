"""Momentum factor.

    M = 0.35 Z(Mom12) + 0.35 Z(Mom6) + 0.30 Z(Momentum/Volatility)

Two details matter for India specifically:

  * the most recent month is skipped (12-1, 6-1) to avoid short-term reversal
  * the vol-adjusted leg damps momentum's crash risk - plain momentum in India
    suffered roughly a -70% drawdown in 2008 with a 65-month recovery
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import MomentumWeights
from .base import combine_scores, zscore

TRADING_DAYS_PER_MONTH = 21


def momentum_return(
    prices: pd.DataFrame, lookback_months: int = 12, skip_months: int = 1
) -> pd.Series:
    """Cumulative return from t-lookback to t-skip, as of the last row."""
    lb = lookback_months * TRADING_DAYS_PER_MONTH
    sk = skip_months * TRADING_DAYS_PER_MONTH
    if len(prices) < lb + 1:
        return pd.Series(np.nan, index=prices.columns)

    end = prices.iloc[-(sk + 1)] if sk > 0 else prices.iloc[-1]
    start = prices.iloc[-(lb + 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (end / start) - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def realized_volatility(
    prices: pd.DataFrame, lookback_days: int = 252, annualize: bool = True
) -> pd.Series:
    """Trailing volatility of daily returns."""
    rets = prices.pct_change().iloc[-lookback_days:]
    if len(rets) < 20:
        return pd.Series(np.nan, index=prices.columns)
    vol = rets.std(ddof=0)
    return vol * np.sqrt(252) if annualize else vol


def vol_adjusted_momentum(
    prices: pd.DataFrame, lookback_months: int = 12, skip_months: int = 1,
    vol_days: int = 126,
) -> pd.Series:
    """Momentum per unit of risk - the crash-damped variant."""
    mom = momentum_return(prices, lookback_months, skip_months)
    vol = realized_volatility(prices, vol_days)
    return (mom / vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def path_smoothness(prices: pd.DataFrame, lookback_days: int = 252) -> pd.Series:
    """Fraction of up-days ("frog in the pan": smooth trends persist better)."""
    rets = prices.pct_change().iloc[-lookback_days:]
    if len(rets) < 20:
        return pd.Series(np.nan, index=prices.columns)
    return (rets > 0).sum() / rets.notna().sum().replace(0, np.nan)


def compute_momentum_score(
    prices: pd.DataFrame,
    weights: MomentumWeights | None = None,
    winsor_pct: float = 0.01,
    universe: list[str] | None = None,
) -> pd.Series:
    """Composite momentum score at the last date of `prices`."""
    w = weights or MomentumWeights()
    px = prices[universe] if universe else prices
    px = px.dropna(axis=1, how="all")
    if px.empty:
        return pd.Series(dtype=float)

    components = {
        "mom_12_1": zscore(momentum_return(px, 12, 1), winsor_pct),
        "mom_6_1": zscore(momentum_return(px, 6, 1), winsor_pct),
        "vol_adjusted": zscore(vol_adjusted_momentum(px, 12, 1), winsor_pct),
    }
    return combine_scores(
        components,
        {"mom_12_1": w.mom_12_1, "mom_6_1": w.mom_6_1, "vol_adjusted": w.vol_adjusted},
    )


def absolute_momentum(prices: pd.DataFrame, lookback_months: int = 12) -> pd.Series:
    """Raw trailing return - the time-series (not cross-sectional) signal."""
    return momentum_return(prices, lookback_months, skip_months=0)
