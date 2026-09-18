"""Volatility scaling (Barroso & Santa-Clara style).

Momentum's crash risk is largely predictable from its *own* recent
volatility - scaling exposure inversely to realized vol historically improved
momentum's Sharpe and cut the tail. Kept optional and capped at 1.0 because
this is a long-only, unlevered book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import VolScalingConfig


def realized_portfolio_vol(
    returns: pd.Series, lookback_days: int = 126, annualize: bool = True
) -> pd.Series:
    """Trailing realized volatility of a return stream."""
    vol = returns.rolling(lookback_days, min_periods=max(20, lookback_days // 4)).std()
    return vol * np.sqrt(252) if annualize else vol


def vol_scale_factor(
    returns: pd.Series, config: VolScalingConfig
) -> pd.Series:
    """Exposure multiplier that targets a constant volatility."""
    if not config.enabled or returns.empty:
        return pd.Series(1.0, index=returns.index)

    vol = realized_portfolio_vol(returns, config.lookback_days)
    scale = config.target_vol / vol.replace(0, np.nan)
    scale = scale.clip(upper=config.max_leverage).fillna(config.max_leverage)
    # Lag by one day - today's vol is not knowable before today's close.
    return scale.shift(1).fillna(config.max_leverage)


def apply_vol_scaling(
    weights: pd.Series, scale: float, max_gross: float = 1.0
) -> pd.Series:
    """Scale a weight vector, respecting the gross-exposure cap."""
    if weights.empty:
        return weights
    scaled = weights * float(np.clip(scale, 0.0, max_gross))
    total = scaled.sum()
    if total > max_gross:
        scaled = scaled * (max_gross / total)
    return scaled


def ex_ante_volatility(
    weights: pd.Series, returns: pd.DataFrame, lookback_days: int = 252,
    shrinkage: float = 0.1,
) -> float:
    """Forecast portfolio volatility from a shrunk covariance matrix.

    Ledoit-Wolf style shrinkage toward a diagonal target - sample covariance
    is badly conditioned when names outnumber observations.
    """
    if weights.empty or returns.empty:
        return np.nan
    cols = [c for c in weights.index if c in returns.columns]
    if not cols:
        return np.nan

    hist = returns[cols].iloc[-lookback_days:].dropna(how="all")
    if len(hist) < 20:
        return np.nan

    w = weights.reindex(cols).fillna(0.0).values
    cov = hist.cov().values
    if shrinkage > 0:
        target = np.diag(np.diag(cov))
        cov = (1 - shrinkage) * cov + shrinkage * target

    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(252))
