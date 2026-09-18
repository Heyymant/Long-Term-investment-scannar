"""Cross-sectional residual reversion - Sleeve B's primary formulation.

Rather than betting on a handful of cointegrated pairs, we strip market and
sector effects out of every stock's recent return and buy the names whose
*residual* is most negative - statistically oversold relative to peers.

Why this is preferred over long-only pairs here:
  * diversified across dozens of names instead of a few fragile relationships
  * no dependence on a single pair's cointegration surviving
  * the residual is already market/sector neutral by construction, which
    partly offsets the fact that we cannot short

The binding constraint is cost: this signal decays in days, so turnover is
high and the cost model decides whether anything survives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


@dataclass
class ResidualSignal:
    date: pd.Timestamp
    residual_z: pd.Series        # negative = oversold
    selected: list[str]
    weights: pd.Series
    diagnostics: dict

    def to_frame(self) -> pd.DataFrame:
        df = self.residual_z.to_frame("residual_z")
        df["selected"] = df.index.isin(self.selected)
        df["weight"] = self.weights.reindex(df.index).fillna(0.0)
        df["date"] = self.date
        return df.reset_index().rename(columns={"index": "isin"})


def compute_residuals(
    returns: pd.DataFrame,
    sectors: pd.Series | None = None,
    market_returns: pd.Series | None = None,
    lookback_days: int = 60,
) -> pd.Series:
    """Cumulative return over the lookback, net of market and sector effects.

    We regress each stock's cumulative return on a market term and sector
    dummies; the residual is the idiosyncratic move that is a candidate to
    revert.
    """
    window = returns.iloc[-lookback_days:]
    if len(window) < 10:
        return pd.Series(dtype=float)

    cum = (1 + window).prod() - 1.0
    cum = cum.dropna()
    if len(cum) < 5:
        return pd.Series(dtype=float)

    resid = cum.copy()

    # 1) Remove the market component (equal-weight proxy if none supplied).
    if market_returns is not None and not market_returns.empty:
        mkt = float((1 + market_returns.iloc[-lookback_days:]).prod() - 1.0)
    else:
        mkt = float(cum.mean())

    betas = _estimate_betas(window, market_returns)
    if betas is not None:
        resid = cum - betas.reindex(cum.index).fillna(1.0) * mkt
    else:
        resid = cum - mkt

    # 2) Remove sector means.
    if sectors is not None and not sectors.empty:
        sec = sectors.reindex(resid.index).fillna("UNKNOWN")
        resid = resid - resid.groupby(sec).transform("mean")

    return resid


def _estimate_betas(
    window: pd.DataFrame, market_returns: pd.Series | None
) -> pd.Series | None:
    if market_returns is None or market_returns.empty:
        return None
    mkt = market_returns.reindex(window.index).dropna()
    if len(mkt) < 10:
        return None
    var = mkt.var()
    if var == 0:
        return None
    aligned = window.reindex(mkt.index)
    return aligned.apply(lambda col: col.cov(mkt) / var)


def generate_residual_signal(
    prices: pd.DataFrame,
    dt: pd.Timestamp,
    universe: list[str],
    sectors: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    lookback_days: int = 60,
    n_positions: int = 20,
    entry_z: float = -1.5,
    max_weight: float = 0.05,
) -> ResidualSignal:
    """Buy the most oversold residuals as of `dt` (long-only)."""
    dt = pd.Timestamp(dt)
    px = prices.loc[prices.index <= dt]
    cols = [c for c in universe if c in px.columns]
    if len(cols) < 10 or len(px) < lookback_days + 5:
        return ResidualSignal(dt, pd.Series(dtype=float), [], pd.Series(dtype=float),
                              {"reason": "insufficient data"})

    returns = px[cols].pct_change()
    market = benchmark.loc[benchmark.index <= dt].pct_change() if benchmark is not None else None

    resid = compute_residuals(returns, sectors, market, lookback_days)
    if resid.empty:
        return ResidualSignal(dt, pd.Series(dtype=float), [], pd.Series(dtype=float),
                              {"reason": "no residuals"})

    std = resid.std(ddof=0)
    if std == 0 or np.isnan(std):
        return ResidualSignal(dt, pd.Series(dtype=float), [], pd.Series(dtype=float),
                              {"reason": "zero dispersion"})
    z = (resid - resid.mean()) / std

    # Long-only: take only the oversold tail.
    oversold = z[z <= entry_z].sort_values()
    selected = list(oversold.index[:n_positions])

    if not selected:
        return ResidualSignal(dt, z, [], pd.Series(dtype=float),
                              {"n_candidates": 0, "min_z": float(z.min())})

    # Weight by conviction (depth of the dislocation), capped.
    conviction = (-z[selected]).clip(lower=0.0)
    weights = conviction / conviction.sum() if conviction.sum() > 0 else pd.Series(
        1.0 / len(selected), index=selected
    )
    weights = weights.clip(upper=max_weight)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    return ResidualSignal(
        date=dt, residual_z=z, selected=selected, weights=weights,
        diagnostics={
            "n_candidates": int((z <= entry_z).sum()),
            "n_selected": len(selected),
            "min_z": float(z.min()),
            "mean_selected_z": float(z[selected].mean()),
            "lookback_days": lookback_days,
        },
    )


def residual_reversion_signals(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    sectors: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    lookback_days: int = 60,
    n_positions: int = 20,
    entry_z: float = -1.5,
    max_weight: float = 0.05,
) -> dict[pd.Timestamp, pd.Series]:
    """Target weights for every rebalance date (feeds the shared engine)."""
    from ..data.universe import eligible_on

    out: dict[pd.Timestamp, pd.Series] = {}
    for dt in rebalance_dates:
        universe = eligible_on(eligibility, dt)
        if not universe:
            continue
        sig = generate_residual_signal(
            prices, dt, universe, sectors, benchmark,
            lookback_days, n_positions, entry_z, max_weight,
        )
        if not sig.weights.empty:
            out[dt] = sig.weights
    return out


def reversion_dates(index: pd.DatetimeIndex, horizon_days: int, warmup: int) -> pd.DatetimeIndex:
    """Rebalance every `horizon_days` - reversion needs frequent refreshes."""
    usable = index[warmup:] if len(index) > warmup else index[-1:]
    return pd.DatetimeIndex(usable[::max(1, horizon_days)])
