"""Cointegration screening with false-discovery control.

Testing every pair in a 500-stock universe means ~125,000 hypothesis tests.
At a 5% significance level you would expect ~6,000 "significant" pairs purely
by chance. Benjamini-Hochberg FDR control is therefore not optional - without
it, a pairs backtest is mostly trading noise it mistook for structure.

We also require an economic rationale (same sector) rather than screening the
entire cross-product, which cuts the multiple-testing burden at the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


@dataclass
class PairCandidate:
    y: str
    x: str
    sector: str
    eg_pvalue: float
    adf_stat: float
    hedge_ratio: float
    half_life: float
    correlation: float
    johansen_trace: float = np.nan
    johansen_significant: bool = False
    passes_fdr: bool = False

    def to_dict(self) -> dict:
        return {
            "y": self.y, "x": self.x, "sector": self.sector,
            "eg_pvalue": self.eg_pvalue, "adf_stat": self.adf_stat,
            "hedge_ratio": self.hedge_ratio, "half_life": self.half_life,
            "correlation": self.correlation,
            "johansen_trace": self.johansen_trace,
            "johansen_significant": self.johansen_significant,
            "passes_fdr": self.passes_fdr,
        }


def engle_granger_test(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Engle-Granger two-step: regress, then ADF-test the residuals."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 60:
        return np.nan, np.nan
    try:
        from statsmodels.tsa.stattools import coint

        stat, pvalue, _ = coint(df["y"], df["x"], trend="c", autolag="AIC")
        return float(pvalue), float(stat)
    except Exception as exc:  # noqa: BLE001
        log.debug("Engle-Granger failed: %s", exc)
        return np.nan, np.nan


def adf_test(series: pd.Series) -> tuple[float, float]:
    """Augmented Dickey-Fuller on a single series."""
    s = series.dropna()
    if len(s) < 30:
        return np.nan, np.nan
    try:
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(s, autolag="AIC")
        return float(result[1]), float(result[0])
    except Exception as exc:  # noqa: BLE001
        log.debug("ADF failed: %s", exc)
        return np.nan, np.nan


def johansen_test(
    y: pd.Series, x: pd.Series, det_order: int = 0, k_ar_diff: int = 1
) -> tuple[float, bool]:
    """Johansen trace test - a second opinion on cointegration rank."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 60:
        return np.nan, False
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        res = coint_johansen(df.values, det_order, k_ar_diff)
        trace = float(res.lr1[0])
        crit_95 = float(res.cvt[0, 1])
        return trace, bool(trace > crit_95)
    except Exception as exc:  # noqa: BLE001
        log.debug("Johansen failed: %s", exc)
        return np.nan, False


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH step-up procedure; returns a boolean mask of discoveries.

    Controls the expected proportion of false positives among the pairs we
    accept, which is exactly the quantity that matters when screening
    thousands of candidates.
    """
    p = np.asarray(pvalues, dtype=float)
    valid = ~np.isnan(p)
    out = np.zeros(len(p), dtype=bool)
    if valid.sum() == 0:
        return out

    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)

    threshold_rank = 0
    for rank, i in enumerate(order, start=1):
        if p[i] <= alpha * rank / m:
            threshold_rank = rank
    if threshold_rank > 0:
        out[order[:threshold_rank]] = True
    return out


def screen_pairs(
    prices: pd.DataFrame,
    sectors: pd.Series,
    formation_window: int = 252,
    max_pvalue: float = 0.05,
    fdr_alpha: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 60.0,
    min_correlation: float = 0.5,
    within_sector_only: bool = True,
    use_johansen: bool = True,
    max_pairs_tested: int = 5000,
) -> list[PairCandidate]:
    """Find cointegrated pairs, then apply FDR control to the survivors."""
    from .spread import build_spread

    window = prices.iloc[-formation_window:] if len(prices) > formation_window else prices
    window = window.dropna(axis=1, thresh=int(len(window) * 0.9))
    if window.shape[1] < 2:
        return []

    # Candidate generation: same sector keeps the economic story intact.
    if within_sector_only and not sectors.empty:
        groups: dict[str, list[str]] = {}
        for isin in window.columns:
            groups.setdefault(str(sectors.get(isin, "UNKNOWN")), []).append(isin)
        pairs = [
            (a, b, sector)
            for sector, members in groups.items()
            if len(members) >= 2
            for a, b in combinations(sorted(members), 2)
        ]
    else:
        pairs = [(a, b, "ALL") for a, b in combinations(sorted(window.columns), 2)]

    if len(pairs) > max_pairs_tested:
        log.warning(
            "%d candidate pairs exceeds the cap of %d; truncating. Every extra test "
            "inflates the multiple-testing burden.", len(pairs), max_pairs_tested,
        )
        pairs = pairs[:max_pairs_tested]

    log.info("Screening %d candidate pairs", len(pairs))
    candidates: list[PairCandidate] = []

    for a, b, sector in pairs:
        ya, xb = window[a], window[b]
        corr = float(ya.corr(xb))
        # Correlation is a cheap pre-filter - it is NOT cointegration, but
        # uncorrelated series are rarely worth the expensive test.
        if np.isnan(corr) or abs(corr) < min_correlation:
            continue

        pval, stat = engle_granger_test(ya, xb)
        if np.isnan(pval) or pval > max_pvalue:
            continue

        model = build_spread(ya, xb, "ols")
        if not np.isfinite(model.half_life):
            continue
        if not (min_half_life <= model.half_life <= max_half_life):
            continue

        trace, sig = johansen_test(ya, xb) if use_johansen else (np.nan, True)
        candidates.append(PairCandidate(
            y=a, x=b, sector=sector, eg_pvalue=pval, adf_stat=stat,
            hedge_ratio=model.hedge_ratio, half_life=model.half_life,
            correlation=corr, johansen_trace=trace, johansen_significant=sig,
        ))

    if not candidates:
        log.info("No pairs survived the cointegration screen")
        return []

    # FDR across everything that reached the testing stage.
    pvals = np.array([c.eg_pvalue for c in candidates])
    mask = benjamini_hochberg(pvals, fdr_alpha)
    for c, keep in zip(candidates, mask):
        c.passes_fdr = bool(keep)

    survivors = [c for c in candidates if c.passes_fdr and (c.johansen_significant or not use_johansen)]
    log.info(
        "Cointegration screen: %d tested -> %d significant -> %d after BH-FDR(%.2f)",
        len(pairs), len(candidates), len(survivors), fdr_alpha,
    )
    return sorted(survivors, key=lambda c: c.eg_pvalue)


def pairs_to_frame(pairs: list[PairCandidate]) -> pd.DataFrame:
    return pd.DataFrame([p.to_dict() for p in pairs]) if pairs else pd.DataFrame()
