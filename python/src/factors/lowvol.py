"""Low-volatility factor:  LV = -Z(sigma_12m)

The volatility effect is well documented in India (Nifty 500 evidence shows
low-vol beating high-vol in both absolute and risk-adjusted terms, and it is
not explained by size, value or momentum). We score the *negative* of
volatility so that higher is better, consistent with every other factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import zscore


def trailing_volatility(
    prices: pd.DataFrame, lookback_days: int = 252, min_obs: int | None = None
) -> pd.Series:
    """Annualized volatility of daily returns over the lookback."""
    rets = prices.pct_change().iloc[-lookback_days:]
    min_obs = min_obs or max(20, lookback_days // 4)
    vol = rets.std(ddof=0)
    return vol.where(rets.notna().sum() >= min_obs) * np.sqrt(252)


def downside_volatility(prices: pd.DataFrame, lookback_days: int = 252) -> pd.Series:
    """Volatility of negative returns only."""
    rets = prices.pct_change().iloc[-lookback_days:]
    downside = rets.where(rets < 0)
    return downside.std(ddof=0) * np.sqrt(252)


def beta_to_market(
    prices: pd.DataFrame, market: pd.Series, lookback_days: int = 252
) -> pd.Series:
    """OLS beta against a market series."""
    rets = prices.pct_change().iloc[-lookback_days:]
    mkt = market.pct_change().reindex(rets.index)
    var = mkt.var(ddof=0)
    if var is None or var == 0 or np.isnan(var):
        return pd.Series(np.nan, index=prices.columns)
    return rets.apply(lambda col: col.cov(mkt) / var)


def compute_low_vol_score(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    winsor_pct: float = 0.01,
    universe: list[str] | None = None,
    use_beta: bool = False,
    market: pd.Series | None = None,
) -> pd.Series:
    """Low-volatility score at the last date (higher = calmer)."""
    px = prices[universe] if universe else prices
    px = px.dropna(axis=1, how="all")
    if px.empty:
        return pd.Series(dtype=float)

    if use_beta and market is not None:
        risk = beta_to_market(px, market, lookback_days)
    else:
        risk = trailing_volatility(px, lookback_days)

    return zscore(-risk, winsor_pct)
