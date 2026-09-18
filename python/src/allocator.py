"""Merge the three sleeves into one book.

The important behaviour: a stock held by more than one sleeve becomes a single
larger position (a top-up), never two separate lines. Combined per-name and
per-sector limits are then enforced on the merged book, because risk is a
property of the whole portfolio, not of each sleeve in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import get_logger
from .risk.portfolio_risk import RiskState, apply_risk_limits, merge_sleeve_weights
from .schema import StrategyConfig

log = get_logger(__name__)


@dataclass
class AllocatedPortfolio:
    date: pd.Timestamp
    weights: pd.Series
    sleeve_weights: dict[str, pd.Series] = field(default_factory=dict)
    sleeve_contributions: dict[str, float] = field(default_factory=dict)
    risk_state: RiskState | None = None
    overlaps: list[str] = field(default_factory=list)

    @property
    def gross_exposure(self) -> float:
        return float(self.weights.sum()) if not self.weights.empty else 0.0

    @property
    def cash_weight(self) -> float:
        return max(0.0, 1.0 - self.gross_exposure)


def allocate(
    config: StrategyConfig,
    date: pd.Timestamp,
    sleeve_a_weights: pd.Series | None = None,
    sleeve_b_weights: pd.Series | None = None,
    sleeve_c_weights: pd.Series | None = None,
    sectors: pd.Series | None = None,
    portfolio_value: float = 0.0,
    adv: pd.Series | None = None,
    equity_curve: pd.Series | None = None,
    returns: pd.Series | None = None,
) -> AllocatedPortfolio:
    """Combine sleeve weights and apply portfolio-level limits."""
    sleeves = {
        "A": sleeve_a_weights if sleeve_a_weights is not None else pd.Series(dtype=float),
        "B": sleeve_b_weights if sleeve_b_weights is not None else pd.Series(dtype=float),
        "C": sleeve_c_weights if sleeve_c_weights is not None else pd.Series(dtype=float),
    }
    allocations = {
        "A": config.allocation.sleeve_a if config.sleeve_a.enabled else 0.0,
        "B": config.allocation.sleeve_b if config.sleeve_b.enabled else 0.0,
        "C": config.allocation.sleeve_c if config.sleeve_c.enabled else 0.0,
    }

    combined = merge_sleeve_weights(sleeves, allocations)
    overlaps = _find_overlaps(sleeves, allocations)
    if overlaps:
        log.debug("%d names held by multiple sleeves (topped up, not duplicated)", len(overlaps))

    final, state = apply_risk_limits(
        combined, config.risk, sectors,
        portfolio_value=portfolio_value, adv=adv,
        equity_curve=equity_curve, returns=returns, date=date,
    )

    contributions = {
        name: float((w * allocations.get(name, 0.0)).sum())
        for name, w in sleeves.items() if w is not None and not w.empty
    }

    return AllocatedPortfolio(
        date=pd.Timestamp(date),
        weights=final,
        sleeve_weights={k: v for k, v in sleeves.items() if not v.empty},
        sleeve_contributions=contributions,
        risk_state=state,
        overlaps=overlaps,
    )


def _find_overlaps(
    sleeves: dict[str, pd.Series], allocations: dict[str, float]
) -> list[str]:
    counts: dict[str, int] = {}
    for name, w in sleeves.items():
        if allocations.get(name, 0.0) <= 0 or w is None or w.empty:
            continue
        for isin in w.index:
            counts[isin] = counts.get(isin, 0) + 1
    return [k for k, v in counts.items() if v > 1]


def merge_signal_dates(
    *signal_dicts: dict[pd.Timestamp, pd.Series]
) -> list[pd.Timestamp]:
    """Union of all sleeve rebalance dates, sorted."""
    dates: set[pd.Timestamp] = set()
    for d in signal_dicts:
        if d:
            dates.update(d.keys())
    return sorted(dates)


def combined_signals(
    config: StrategyConfig,
    sleeve_a: dict[pd.Timestamp, pd.Series] | None = None,
    sleeve_b: dict[pd.Timestamp, pd.Series] | None = None,
    sleeve_c: dict[pd.Timestamp, pd.Series] | None = None,
    sectors: pd.Series | None = None,
) -> dict[pd.Timestamp, pd.Series]:
    """Pre-merge sleeve signals into a single weight series per date.

    Sleeves rebalance on different cadences (quarterly factors, weekly events),
    so on any given date we carry forward each sleeve's most recent target.
    """
    a, b, c = sleeve_a or {}, sleeve_b or {}, sleeve_c or {}
    dates = merge_signal_dates(a, b, c)
    out: dict[pd.Timestamp, pd.Series] = {}

    last = {"A": pd.Series(dtype=float), "B": pd.Series(dtype=float), "C": pd.Series(dtype=float)}
    for dt in dates:
        if dt in a:
            last["A"] = a[dt]
        if dt in b:
            last["B"] = b[dt]
        if dt in c:
            last["C"] = c[dt]

        allocated = allocate(
            config, dt,
            sleeve_a_weights=last["A"], sleeve_b_weights=last["B"], sleeve_c_weights=last["C"],
            sectors=sectors,
        )
        if not allocated.weights.empty:
            out[dt] = allocated.weights
    return out
