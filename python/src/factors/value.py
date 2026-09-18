"""Value factor.

    V = 0.25 Z(FCF Yield) + 0.25 Z(Earnings Yield)
      + 0.20 Z(-EV/EBITDA) + 0.15 Z(-P/B) + 0.15 Z(-P/E)

Value in India is strong but cyclical, and the research compilation is blunt
about the trap: *cheap is not the same as undervalued*. A stock is usually
cheap for a reason, so value is used in combination with quality rather than
on its own (see `value_quality_gate`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.fundamentals.base import as_of
from ..schema import ValueWeights
from .base import combine_scores, zscore

# Metric -> (weight attribute, higher_is_better)
VALUE_SPEC = {
    "fcf_yield": ("fcf_yield", True),
    "earnings_yield": ("earnings_yield", True),
    "ev_ebitda": ("neg_ev_ebitda", False),
    "pb": ("neg_pb", False),
    "pe": ("neg_pe", False),
}


def compute_value_score(
    fundamentals: pd.DataFrame,
    dt: pd.Timestamp,
    weights: ValueWeights | None = None,
    winsor_pct: float = 0.01,
    universe: list[str] | None = None,
) -> pd.Series:
    """Value score as known on `dt` (higher = cheaper)."""
    w = weights or ValueWeights()
    snap = as_of(fundamentals, dt, metrics=list(VALUE_SPEC))
    if snap.empty:
        return pd.Series(dtype=float)

    if universe:
        snap = snap.reindex([i for i in universe if i in snap.index])
    if snap.empty:
        return pd.Series(dtype=float)

    components: dict[str, pd.Series] = {}
    weight_map: dict[str, float] = {}
    for metric, (attr, higher_better) in VALUE_SPEC.items():
        if metric not in snap.columns:
            continue
        series = pd.to_numeric(snap[metric], errors="coerce")
        # Negative multiples are meaningless as "cheap" - drop them.
        if metric in ("pe", "pb", "ev_ebitda"):
            series = series.where(series > 0)
        if not higher_better:
            series = -series
        components[attr] = zscore(series, winsor_pct)
        weight_map[attr] = getattr(w, attr, 0.0)

    if not components:
        return pd.Series(dtype=float)
    return combine_scores(components, weight_map, min_components=max(1, len(components) // 2))


def apply_value_quality_gate(
    value_score: pd.Series, quality_score: pd.Series, strength: float = 0.5
) -> pd.Series:
    """Penalize cheap-but-junk names.

    Cheap stocks with poor quality are discounted; cheap stocks with decent
    quality keep their score. `strength` scales how harsh the penalty is
    (0 = off, 1 = full).
    """
    if value_score.empty or quality_score.empty or strength <= 0:
        return value_score

    q = quality_score.reindex(value_score.index)
    # Only penalize below-average quality, and only for names that look cheap.
    junk = (-q).clip(lower=0.0).fillna(0.0)
    cheap = value_score.clip(lower=0.0).fillna(0.0)
    penalty = strength * junk * (cheap / (cheap.abs().max() or 1.0))
    return value_score - penalty
