"""Portfolio-level risk management across sleeves.

Individual sleeves each have their own caps, but risk is a property of the
*combined* book: two sleeves can independently look prudent and still stack
into the same names, sectors or market beta. This module enforces the
whole-portfolio limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import get_logger
from ..schema import RiskConfig

log = get_logger(__name__)


@dataclass
class RiskState:
    """Risk assessment at a point in time."""

    date: pd.Timestamp
    gross_exposure: float
    cash_weight: float
    drawdown: float
    exposure_multiplier: float = 1.0      # from drawdown de-risking
    vol_multiplier: float = 1.0           # from vol targeting
    forecast_vol: float = np.nan
    breaches: list[str] = field(default_factory=list)
    sector_weights: dict[str, float] = field(default_factory=dict)
    concentration: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "date": str(self.date.date()) if self.date is not None else None,
            "gross_exposure": round(self.gross_exposure, 4),
            "cash_weight": round(self.cash_weight, 4),
            "drawdown": round(self.drawdown, 4),
            "exposure_multiplier": round(self.exposure_multiplier, 4),
            "vol_multiplier": round(self.vol_multiplier, 4),
            "forecast_vol": None if np.isnan(self.forecast_vol) else round(self.forecast_vol, 4),
            "breaches": self.breaches,
            "sector_weights": {k: round(v, 4) for k, v in self.sector_weights.items()},
            "concentration": {k: round(v, 4) for k, v in self.concentration.items()},
        }


# --------------------------------------------------------------------------- #
# Drawdown de-risking
# --------------------------------------------------------------------------- #
def drawdown_exposure_multiplier(
    current_drawdown: float, config: RiskConfig, currently_derisked: bool = False
) -> float:
    """Staged exposure cuts as drawdown deepens (a strategy circuit breaker).

    Re-entry requires recovering past the threshold plus a buffer, so we do
    not flip-flop right at the boundary.
    """
    if not config.drawdown_derisk_enabled:
        return 1.0

    multiplier = 1.0
    for threshold, exposure in zip(config.drawdown_thresholds, config.drawdown_exposures):
        trigger = threshold + (config.drawdown_recovery_buffer if currently_derisked else 0.0)
        if current_drawdown <= trigger:
            multiplier = exposure
    return multiplier


def running_drawdown(equity: pd.Series, lookback_days: int | None = None) -> pd.Series:
    """Drawdown against a rolling peak.

    `lookback_days=None` gives the classic all-time-high drawdown, which is the
    right *reporting* measure. For de-risking decisions a rolling peak is
    essential: measuring against an all-time high means a de-risked portfolio
    can never recover enough to re-enter, and the circuit breaker latches
    permanently.
    """
    if equity.empty:
        return equity
    if lookback_days is None:
        peak = equity.cummax()
    else:
        peak = equity.rolling(lookback_days, min_periods=1).max()
    return equity / peak - 1.0


# --------------------------------------------------------------------------- #
# Volatility targeting
# --------------------------------------------------------------------------- #
def vol_target_multiplier(
    returns: pd.Series, config: RiskConfig
) -> float:
    """Multiplier that scales realized vol toward the target."""
    if not config.vol_target_enabled or returns.empty:
        return 1.0
    window = returns.iloc[-config.vol_lookback_days:]
    if len(window) < 20:
        return 1.0
    realized = float(window.std() * np.sqrt(252))
    if realized <= 0:
        return 1.0
    return float(np.clip(config.target_vol / realized, 0.0, config.max_gross_exposure))


# --------------------------------------------------------------------------- #
# Liquidity
# --------------------------------------------------------------------------- #
def days_to_liquidate(
    weights: pd.Series, portfolio_value: float, adv: pd.Series,
    participation: float = 0.10,
) -> pd.Series:
    """Trading days needed to exit each position at `participation` of ADV."""
    if weights.empty or adv.empty:
        return pd.Series(dtype=float)
    position_value = weights * portfolio_value
    capacity = adv.reindex(weights.index) * participation
    return (position_value / capacity.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def apply_liquidity_cap(
    weights: pd.Series, portfolio_value: float, adv: pd.Series,
    max_pct_of_adv: float = 0.10,
) -> pd.Series:
    """Trim positions that are too large relative to daily traded value."""
    if weights.empty or adv.empty or portfolio_value <= 0:
        return weights
    max_value = adv.reindex(weights.index).fillna(0.0) * max_pct_of_adv
    max_weight = (max_value / portfolio_value).replace([np.inf, -np.inf], np.nan)
    capped = pd.concat([weights, max_weight], axis=1).min(axis=1)
    capped = capped.fillna(weights).clip(lower=0.0)

    trimmed = float((weights - capped).sum())
    if trimmed > 1e-6:
        log.debug("Liquidity caps trimmed %.2f%% of gross exposure", trimmed * 100)
    return capped


# --------------------------------------------------------------------------- #
# Cross-sleeve aggregation
# --------------------------------------------------------------------------- #
def merge_sleeve_weights(
    sleeve_weights: dict[str, pd.Series], allocations: dict[str, float]
) -> pd.Series:
    """Combine per-sleeve weights into one book.

    A name held by several sleeves gets a single combined position (a top-up),
    not duplicate entries.
    """
    combined: dict[str, float] = {}
    for sleeve, weights in sleeve_weights.items():
        alloc = float(allocations.get(sleeve, 0.0))
        if alloc <= 0 or weights is None or weights.empty:
            continue
        for isin, w in weights.items():
            combined[isin] = combined.get(isin, 0.0) + float(w) * alloc
    if not combined:
        return pd.Series(dtype=float)
    return pd.Series(combined).sort_values(ascending=False)


def sleeve_correlation(sleeve_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation between sleeve return streams.

    If the "diversifying" sleeves are 0.9 correlated with the core, they are
    not diversifying - they are leverage.
    """
    valid = {k: v for k, v in sleeve_returns.items() if v is not None and not v.empty}
    if len(valid) < 2:
        return pd.DataFrame()
    return pd.DataFrame(valid).corr()


def apply_risk_limits(
    weights: pd.Series,
    config: RiskConfig,
    sectors: pd.Series | None = None,
    portfolio_value: float = 0.0,
    adv: pd.Series | None = None,
    equity_curve: pd.Series | None = None,
    returns: pd.Series | None = None,
    currently_derisked: bool = False,
    date: pd.Timestamp | None = None,
) -> tuple[pd.Series, RiskState]:
    """Apply every portfolio-level limit and report what bound."""
    from ..portfolio.construction import apply_weight_caps

    breaches: list[str] = []
    w = weights.clip(lower=0.0).copy() if not weights.empty else pd.Series(dtype=float)

    # 1) Combined per-name / per-sector caps.
    if not w.empty:
        before = w.copy()
        w = apply_weight_caps(
            w, config.max_weight_per_stock_total, sectors, config.max_weight_per_sector_total
        )
        # apply_weight_caps renormalizes to 1.0; restore the original gross.
        gross_before = before.sum()
        if gross_before > 0:
            w = w * min(gross_before, config.max_gross_exposure)
        if (before > config.max_weight_per_stock_total + 1e-9).any():
            breaches.append("max_weight_per_stock_total")

    # 2) Liquidity caps.
    if not w.empty and adv is not None and portfolio_value > 0:
        before_sum = w.sum()
        w = apply_liquidity_cap(w, portfolio_value, adv, config.max_pct_of_adv)
        if w.sum() < before_sum - 1e-9:
            breaches.append("liquidity")

    # 3) Drawdown de-risking, measured against a rolling peak.
    dd = 0.0
    dd_mult = 1.0
    if equity_curve is not None and not equity_curve.empty:
        dd = float(running_drawdown(equity_curve, config.drawdown_lookback_days).iloc[-1])
        dd_mult = drawdown_exposure_multiplier(dd, config, currently_derisked)
        if dd_mult < 1.0:
            breaches.append(f"drawdown_derisk({dd:.1%})")

    # 4) Volatility targeting.
    vol_mult = vol_target_multiplier(returns, config) if returns is not None else 1.0

    if not w.empty:
        w = w * dd_mult * vol_mult

    # 5) Gross exposure and cash floor.
    max_gross = min(config.max_gross_exposure, 1.0 - config.min_cash)
    gross = float(w.sum()) if not w.empty else 0.0
    if gross > max_gross and gross > 0:
        w = w * (max_gross / gross)
        breaches.append("max_gross_exposure")

    gross = float(w.sum()) if not w.empty else 0.0
    sec_w = {}
    if sectors is not None and not w.empty:
        sec_w = w.groupby(sectors.reindex(w.index).fillna("UNKNOWN")).sum().to_dict()

    state = RiskState(
        date=pd.Timestamp(date) if date is not None else pd.Timestamp("now"),
        gross_exposure=gross,
        cash_weight=max(0.0, 1.0 - gross),
        drawdown=dd,
        exposure_multiplier=dd_mult,
        vol_multiplier=vol_mult,
        breaches=breaches,
        sector_weights={str(k): float(v) for k, v in sec_w.items()},
        concentration={
            "max_position": float(w.max()) if not w.empty else 0.0,
            "top5": float(w.nlargest(5).sum()) if not w.empty else 0.0,
            "effective_n": float(1.0 / (w**2).sum()) if gross > 0 else 0.0,
        },
    )
    return w, state
