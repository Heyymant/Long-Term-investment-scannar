"""Sector-neutralization and factor-concentration guards.

Without neutralization a "quality" portfolio in India quietly becomes a bet on
IT and FMCG, and a "value" portfolio becomes a bet on PSU banks and metals.
Neutralizing within sectors makes the bets stock-specific, which is what we
actually want to be paid for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_logger
from .base import standardize_within

log = get_logger(__name__)


def sector_neutralize(
    scores: pd.Series, sectors: pd.Series, method: str = "demean", min_size: int = 3
) -> pd.Series:
    """Remove sector effects from a cross-sectional score.

    method:
      demean       - subtract the sector mean (keeps cross-sector dispersion scale)
      standardize  - full within-sector z-score
      rank         - within-sector percentile rank, recentred
    """
    if scores.empty or sectors.empty:
        return scores

    sec = sectors.reindex(scores.index).fillna("UNKNOWN")
    counts = sec.value_counts()
    # Tiny sectors can't be neutralized meaningfully; pool them.
    small = set(counts[counts < min_size].index)
    if small:
        sec = sec.where(~sec.isin(small), "OTHER")

    if method == "standardize":
        return standardize_within(scores, sec)
    if method == "rank":
        pct = scores.groupby(sec).rank(pct=True)
        return (pct - 0.5) * 2.0
    if method == "demean":
        return scores - scores.groupby(sec).transform("mean")
    raise ValueError(f"unknown neutralization method '{method}'")


def neutralize_to_factor(scores: pd.Series, exposure: pd.Series) -> pd.Series:
    """Regress out a continuous exposure (e.g. size or beta); keep residuals."""
    df = pd.concat([scores.rename("y"), exposure.rename("x")], axis=1).dropna()
    if len(df) < 5:
        return scores
    x, y = df["x"].values, df["y"].values
    var = x.var()
    if var == 0:
        return scores
    beta = np.cov(x, y, ddof=0)[0, 1] / var
    alpha = y.mean() - beta * x.mean()
    resid = pd.Series(y - (alpha + beta * x), index=df.index)
    return resid.reindex(scores.index)


def check_factor_concentration(
    weights: pd.Series,
    factor_scores: dict[str, pd.Series],
    max_abs_exposure: float = 1.5,
) -> dict[str, float]:
    """Weighted factor exposures of a portfolio, with a warning if extreme.

    Catches the case where a "diversified multi-factor" portfolio has silently
    become a pure momentum bet.
    """
    out: dict[str, float] = {}
    if weights.empty:
        return out
    w = weights / weights.sum() if weights.sum() else weights
    for name, scores in factor_scores.items():
        s = scores.reindex(w.index)
        exposure = float((w * s).sum(skipna=True))
        out[name] = exposure
        if abs(exposure) > max_abs_exposure:
            log.warning(
                "Factor concentration: portfolio exposure to '%s' is %.2f (limit %.2f)",
                name, exposure, max_abs_exposure,
            )
    return out


def sector_weights(weights: pd.Series, sectors: pd.Series) -> pd.Series:
    """Aggregate portfolio weight by sector."""
    if weights.empty:
        return pd.Series(dtype=float)
    sec = sectors.reindex(weights.index).fillna("UNKNOWN")
    return weights.groupby(sec).sum().sort_values(ascending=False)
