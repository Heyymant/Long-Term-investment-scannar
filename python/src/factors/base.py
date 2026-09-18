"""Cross-sectional scoring primitives shared by every factor.

All factors follow the same recipe: take raw metrics, winsorize the tails,
convert to cross-sectional z-scores within the eligible universe, then combine
with weights. Doing this consistently is what makes factor weights comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    """Clip both tails at the given percentile.

    Indian fundamentals in particular contain wild outliers (a near-zero
    denominator gives a P/E of 5000); without clipping, one bad print can
    dominate an entire z-score distribution.
    """
    if s.dropna().empty or pct <= 0:
        return s
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series, winsor_pct: float = 0.01, robust: bool = False) -> pd.Series:
    """Cross-sectional z-score.

    `robust=True` uses median/MAD, which is safer for heavy-tailed metrics.
    """
    x = winsorize(s.astype(float), winsor_pct)
    valid = x.dropna()
    if len(valid) < 3:
        return pd.Series(np.nan, index=s.index)

    if robust:
        center = valid.median()
        scale = (valid - center).abs().median() * 1.4826
    else:
        center = valid.mean()
        scale = valid.std(ddof=0)

    if scale is None or scale == 0 or np.isnan(scale):
        return pd.Series(0.0, index=s.index).where(x.notna())
    return (x - center) / scale


def zscore_frame(df: pd.DataFrame, winsor_pct: float = 0.01, robust: bool = False) -> pd.DataFrame:
    """Row-wise (per date) cross-sectional z-scores."""
    return df.apply(lambda row: zscore(row, winsor_pct, robust), axis=1)


def rank_normalize(s: pd.Series) -> pd.Series:
    """Map to approximately N(0,1) via ranks - fully outlier-immune."""
    valid = s.dropna()
    if len(valid) < 3:
        return pd.Series(np.nan, index=s.index)
    from scipy.stats import norm

    ranks = valid.rank(method="average") / (len(valid) + 1)
    return pd.Series(norm.ppf(ranks), index=valid.index).reindex(s.index)


def combine_scores(
    components: dict[str, pd.Series], weights: dict[str, float], min_components: int = 1
) -> pd.Series:
    """Weighted blend that renormalizes when components are missing.

    A stock lacking one input is not discarded outright; its remaining
    components are reweighted, provided at least `min_components` are present.
    """
    if not components:
        return pd.Series(dtype=float)

    index = None
    for s in components.values():
        index = s.index if index is None else index.union(s.index)

    num = pd.Series(0.0, index=index)
    den = pd.Series(0.0, index=index)
    count = pd.Series(0, index=index)

    for name, series in components.items():
        w = float(weights.get(name, 0.0))
        if w == 0:
            continue
        aligned = series.reindex(index)
        present = aligned.notna()
        num[present] += aligned[present] * w
        den[present] += abs(w)
        count[present] += 1

    out = num / den.replace(0, np.nan)
    return out.where(count >= min_components)


def neutralize(scores: pd.Series, groups: pd.Series) -> pd.Series:
    """Demean scores within each group (e.g. sector)."""
    aligned = groups.reindex(scores.index).fillna("UNKNOWN")
    return scores - scores.groupby(aligned).transform("mean")


def standardize_within(scores: pd.Series, groups: pd.Series) -> pd.Series:
    """Full within-group standardization (demean and rescale)."""
    aligned = groups.reindex(scores.index).fillna("UNKNOWN")
    g = scores.groupby(aligned)
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    out = (scores - mean) / std
    # Singleton groups have no dispersion; treat them as neutral.
    return out.fillna(0.0).where(scores.notna())
