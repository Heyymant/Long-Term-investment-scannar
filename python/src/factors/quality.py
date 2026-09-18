"""Quality factor.

    Q = 0.25 Z(ROIC) + 0.20 Z(GrossProfitability) + 0.15 Z(CFO/PAT)
      + 0.15 Z(FCF/Assets) + 0.15 Z(-Debt/Assets) + 0.10 Z(Payout)

Quality is the strongest and most robust Indian factor in the literature
(IIMA: QMJ earns ~0.92%/month four-factor alpha, with low turnover and shallow
drawdowns, and it works long-only). The profitability and payout legs do most
of the work, consistent with the tunnelling hypothesis - in a market with
uneven governance enforcement, cash actually returned to shareholders is
informative.
"""

from __future__ import annotations

import pandas as pd

from ..data.fundamentals.base import as_of
from ..schema import QualityWeights
from .base import combine_scores, zscore

# Metric -> (weight attribute, higher_is_better)
QUALITY_SPEC = {
    "roic": ("roic", True),
    "gross_profitability": ("gross_profitability", True),
    "cfo_to_pat": ("cfo_to_pat", True),
    "fcf_to_assets": ("fcf_to_assets", True),
    "debt_to_assets": ("neg_debt_to_assets", False),   # less debt = better
    "payout": ("payout", True),
}


def compute_quality_score(
    fundamentals: pd.DataFrame,
    dt: pd.Timestamp,
    weights: QualityWeights | None = None,
    winsor_pct: float = 0.01,
    universe: list[str] | None = None,
) -> pd.Series:
    """Quality score as known on `dt` (point-in-time via announce_date)."""
    w = weights or QualityWeights()
    snap = as_of(fundamentals, dt, metrics=list(QUALITY_SPEC))
    if snap.empty:
        return pd.Series(dtype=float)

    if universe:
        snap = snap.reindex([i for i in universe if i in snap.index])
    if snap.empty:
        return pd.Series(dtype=float)

    components: dict[str, pd.Series] = {}
    weight_map: dict[str, float] = {}
    for metric, (attr, higher_better) in QUALITY_SPEC.items():
        if metric not in snap.columns:
            continue
        series = pd.to_numeric(snap[metric], errors="coerce")
        if not higher_better:
            series = -series
        components[attr] = zscore(series, winsor_pct)
        weight_map[attr] = getattr(w, attr, 0.0)

    if not components:
        return pd.Series(dtype=float)
    # Require at least half the legs so a single stray metric can't define quality.
    return combine_scores(components, weight_map, min_components=max(1, len(components) // 2))


def quality_panel(
    fundamentals: pd.DataFrame,
    dates: pd.DatetimeIndex,
    universe: list[str],
    weights: QualityWeights | None = None,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """Quality scores over time (used for QMJ construction and IC analysis)."""
    rows = {}
    for dt in dates:
        s = compute_quality_score(fundamentals, dt, weights, winsor_pct, universe)
        if not s.empty:
            rows[dt] = s
    if not rows:
        return pd.DataFrame(index=dates, columns=universe, dtype=float)
    return pd.DataFrame(rows).T.reindex(index=dates, columns=universe)
