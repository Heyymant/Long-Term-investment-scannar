"""Composite factor score.

    Score = 0.35 Q + 0.30 M + 0.20 V + 0.15 LV

with a Value x Quality gate and optional sector-neutralization. This is the
single ranking that Sleeve A selects from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import get_logger
from ..schema import FactorConfig
from .base import combine_scores
from .lowvol import compute_low_vol_score
from .momentum import compute_momentum_score
from .quality import compute_quality_score
from .sector_neutral import sector_neutralize
from .value import apply_value_quality_gate, compute_value_score

log = get_logger(__name__)


@dataclass
class CompositeScores:
    """Composite plus the underlying legs (kept for attribution/diagnostics)."""

    composite: pd.Series
    quality: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    value: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    momentum: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    low_vol: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "composite": self.composite,
            "quality": self.quality,
            "value": self.value,
            "momentum": self.momentum,
            "low_vol": self.low_vol,
        })

    def components(self) -> dict[str, pd.Series]:
        return {
            "quality": self.quality, "value": self.value,
            "momentum": self.momentum, "low_vol": self.low_vol,
        }


def compute_composite(
    prices: pd.DataFrame,
    dt: pd.Timestamp,
    config: FactorConfig,
    universe: list[str],
    fundamentals: pd.DataFrame | None = None,
    sectors: pd.Series | None = None,
    sector_neutralize_scores: bool = True,
    dynamic_weights: dict[str, float] | None = None,
) -> CompositeScores:
    """Composite score on `dt` using only data available at that time.

    `prices` must already be truncated to <= dt by the caller; we assert it
    here because a single leaked row silently invents alpha.
    """
    px = prices.loc[prices.index <= pd.Timestamp(dt)]
    if px.empty or not universe:
        return CompositeScores(composite=pd.Series(dtype=float))

    available = [c for c in universe if c in px.columns]
    if not available:
        return CompositeScores(composite=pd.Series(dtype=float))

    w = config.winsorize_pct

    momentum = compute_momentum_score(px, config.momentum, w, available)
    low_vol = compute_low_vol_score(px, config.low_vol_lookback_days, w, available)

    quality = pd.Series(dtype=float)
    value = pd.Series(dtype=float)
    if fundamentals is not None and not fundamentals.empty:
        quality = compute_quality_score(fundamentals, dt, config.quality, w, available)
        value = compute_value_score(fundamentals, dt, config.value, w, available)
        if config.value_quality_gate and not value.empty and not quality.empty:
            value = apply_value_quality_gate(value, quality, config.value_quality_gate_strength)

    weights = dynamic_weights or config.composite_weights()
    components = {
        "quality": quality, "momentum": momentum,
        "value": value, "low_vol": low_vol,
    }
    # Drop legs with no data or zero weight so renormalization stays honest.
    active = {k: v for k, v in components.items()
              if v is not None and not v.empty and weights.get(k, 0.0) != 0.0}
    if not active:
        log.warning("No factor components available on %s", pd.Timestamp(dt).date())
        return CompositeScores(composite=pd.Series(dtype=float))

    composite = combine_scores(active, weights)

    if sector_neutralize_scores and sectors is not None and not sectors.empty:
        composite = sector_neutralize(composite, sectors)

    return CompositeScores(
        composite=composite.dropna(),
        quality=quality, value=value, momentum=momentum, low_vol=low_vol,
    )


def build_score_panel(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    config: FactorConfig,
    eligibility: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    sectors: pd.Series | None = None,
    sector_neutralize_scores: bool = True,
) -> dict[str, pd.DataFrame]:
    """Composite and per-leg score panels across all rebalance dates."""
    from ..data.universe import eligible_on

    panels: dict[str, dict] = {k: {} for k in ("composite", "quality", "value", "momentum", "low_vol")}

    for dt in rebalance_dates:
        universe = eligible_on(eligibility, dt)
        if not universe:
            continue
        cs = compute_composite(
            prices, dt, config, universe, fundamentals, sectors, sector_neutralize_scores
        )
        if cs.composite.empty:
            continue
        panels["composite"][dt] = cs.composite
        panels["quality"][dt] = cs.quality
        panels["value"][dt] = cs.value
        panels["momentum"][dt] = cs.momentum
        panels["low_vol"][dt] = cs.low_vol

    return {
        name: (pd.DataFrame(rows).T.sort_index() if rows else pd.DataFrame())
        for name, rows in panels.items()
    }
