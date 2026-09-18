"""Sleeve A portfolio construction.

Pipeline: composite score -> rank-buffered selection -> weighting ->
per-stock and per-sector caps -> regime overlay -> target weights.

Long-only throughout, and weights never sum above the gross-exposure cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import get_logger
from ..schema import SleeveAConfig, WeightScheme
from .buffer import apply_rank_buffer

log = get_logger(__name__)


@dataclass
class Portfolio:
    """A set of target weights plus how they were arrived at."""

    weights: pd.Series                      # index=isin, sums to <= gross exposure
    date: pd.Timestamp
    exposure: float = 1.0                   # regime-driven gross exposure
    cash_weight: float = 0.0
    selected: list[str] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def n_positions(self) -> int:
        return int((self.weights > 0).sum())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"isin": self.weights.index, "weight": self.weights.values})


# --------------------------------------------------------------------------- #
# Weighting schemes
# --------------------------------------------------------------------------- #
def equal_weights(names: list[str]) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(names), index=names)


def inverse_vol_weights(
    names: list[str], prices: pd.DataFrame, lookback_days: int = 252,
    floor_vol: float = 0.05,
) -> pd.Series:
    """Weight inversely to trailing volatility (risk parity, ignoring correlation)."""
    if not names:
        return pd.Series(dtype=float)
    cols = [n for n in names if n in prices.columns]
    if not cols:
        return equal_weights(names)

    rets = prices[cols].pct_change().iloc[-lookback_days:]
    vol = rets.std(ddof=0) * np.sqrt(252)
    vol = vol.replace(0, np.nan).fillna(vol.median()).clip(lower=floor_vol)
    if vol.isna().all():
        return equal_weights(names)

    inv = 1.0 / vol
    w = inv / inv.sum()
    return w.reindex(names).fillna(0.0)


def score_tilted_weights(
    names: list[str], scores: pd.Series, power: float = 1.0
) -> pd.Series:
    """Weight proportional to positive score rank - a conviction tilt."""
    if not names:
        return pd.Series(dtype=float)
    s = scores.reindex(names)
    # Shift into positive territory so the weakest selected name still gets weight.
    adj = (s - s.min() + 0.1) ** power
    total = adj.sum()
    if total <= 0 or np.isnan(total):
        return equal_weights(names)
    return adj / total


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #
def effective_caps(
    n_names: int,
    max_per_stock: float,
    sectors: pd.Series | None = None,
    names: pd.Index | None = None,
    max_per_sector: float | None = None,
) -> tuple[float, float | None]:
    """Relax caps that make full investment impossible.

    With 16 holdings and a 5% cap the most you can ever be invested is 80%;
    silently holding 20% cash would look like the strategy underperforming
    when it is really a constraint conflict. So when a cap is infeasible we
    widen it to the minimum feasible level and say so.
    """
    stock_cap = max_per_stock
    if n_names > 0 and n_names * max_per_stock < 1.0:
        stock_cap = 1.0 / n_names
        log.info(
            "Per-stock cap %.1f%% infeasible for %d holdings (max %.0f%% invested); "
            "relaxed to %.1f%%", max_per_stock * 100, n_names,
            n_names * max_per_stock * 100, stock_cap * 100,
        )

    sector_cap = max_per_sector
    if max_per_sector is not None and sectors is not None and names is not None and len(names):
        n_sectors = sectors.reindex(names).fillna("UNKNOWN").nunique()
        if n_sectors > 0 and n_sectors * max_per_sector < 1.0:
            sector_cap = 1.0 / n_sectors
            log.info(
                "Per-sector cap %.1f%% infeasible across %d sectors; relaxed to %.1f%%",
                max_per_sector * 100, n_sectors, sector_cap * 100,
            )
    return stock_cap, sector_cap


def apply_weight_caps(
    weights: pd.Series,
    max_per_stock: float,
    sectors: pd.Series | None = None,
    max_per_sector: float | None = None,
    max_iterations: int = 50,
    auto_relax: bool = True,
) -> pd.Series:
    """Enforce per-stock and per-sector caps, redistributing the excess.

    Iterative because capping one name pushes weight onto others, which can
    breach a different cap. Converges quickly in practice.
    """
    if weights.empty:
        return weights

    w = weights.clip(lower=0.0).copy()
    total = w.sum()
    if total <= 0:
        return w
    w = w / total

    if auto_relax:
        max_per_stock, max_per_sector = effective_caps(
            len(w), max_per_stock, sectors, w.index, max_per_sector
        )

    for _ in range(max_iterations):
        changed = False

        over = w > max_per_stock + 1e-12
        if over.any():
            excess = (w[over] - max_per_stock).sum()
            w[over] = max_per_stock
            room = ~over & (w > 0)
            if room.any() and excess > 0:
                headroom = (max_per_stock - w[room]).clip(lower=0)
                if headroom.sum() > 0:
                    w[room] += excess * headroom / headroom.sum()
                    changed = True
            else:
                break

        if sectors is not None and max_per_sector is not None:
            sec = sectors.reindex(w.index).fillna("UNKNOWN")
            sec_tot = w.groupby(sec).sum()
            breached = sec_tot[sec_tot > max_per_sector + 1e-12]
            for sector, tot in breached.items():
                members = w.index[sec == sector]
                w[members] *= max_per_sector / tot
                changed = True
            if len(breached):
                others = w.index[~sec.isin(breached.index)]
                deficit = 1.0 - w.sum()
                if deficit > 1e-12 and len(others):
                    headroom = (max_per_stock - w[others]).clip(lower=0)
                    if headroom.sum() > 0:
                        w[others] += deficit * headroom / headroom.sum()

        if not changed:
            break

    total = w.sum()
    if total > 1.0:
        w = w / total
    return w


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def construct_portfolio(
    scores: pd.Series,
    prices: pd.DataFrame,
    config: SleeveAConfig,
    dt: pd.Timestamp,
    current_holdings: set[str] | None = None,
    sectors: pd.Series | None = None,
    exposure: float = 1.0,
) -> Portfolio:
    """Build target weights for one rebalance date."""
    dt = pd.Timestamp(dt)
    valid = scores.dropna()
    if valid.empty:
        return Portfolio(weights=pd.Series(dtype=float), date=dt, exposure=exposure,
                         cash_weight=1.0)

    selected = apply_rank_buffer(
        valid, current_holdings, config.top_n, config.rank_buffer_multiple
    )
    if not selected:
        return Portfolio(weights=pd.Series(dtype=float), date=dt, exposure=exposure,
                         cash_weight=1.0)

    px = prices.loc[prices.index <= dt]
    if config.weight_scheme == WeightScheme.EQUAL:
        raw = equal_weights(selected)
    elif config.weight_scheme == WeightScheme.SCORE_TILTED:
        raw = score_tilted_weights(selected, valid)
    else:
        raw = inverse_vol_weights(selected, px, config.factors.low_vol_lookback_days)

    capped = apply_weight_caps(
        raw, config.max_weight_per_stock, sectors, config.max_weight_per_sector
    )

    final = capped * float(np.clip(exposure, 0.0, 1.0))
    final = final[final > 1e-6]

    return Portfolio(
        weights=final,
        date=dt,
        exposure=float(exposure),
        cash_weight=float(max(0.0, 1.0 - final.sum())),
        selected=selected,
        diagnostics={
            "n_selected": len(selected),
            "n_scored": len(valid),
            "max_weight": float(final.max()) if not final.empty else 0.0,
            "effective_n": float(1.0 / (final**2).sum()) if final.sum() > 0 else 0.0,
        },
    )


def rebalance_dates(
    index: pd.DatetimeIndex, frequency: str, warmup_days: int = 252
) -> pd.DatetimeIndex:
    """Rebalance dates (last trading day of each period) after a warm-up."""
    if len(index) == 0:
        return pd.DatetimeIndex([])

    usable = index[warmup_days:] if len(index) > warmup_days else index[-1:]
    if len(usable) == 0:
        return pd.DatetimeIndex([])

    rule = {
        "monthly": "ME", "quarterly": "QE",
        "semiannual": "2QE", "annual": "YE",
    }.get(str(frequency).lower(), "QE")

    s = pd.Series(usable, index=usable)
    picked = s.resample(rule).last().dropna()
    return pd.DatetimeIndex(picked.values)
