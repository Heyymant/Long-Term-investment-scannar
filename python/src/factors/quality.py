"""Quality factor.

    Q = 0.25 Z(ROIC) + 0.20 Z(GrossProfitability) + 0.15 Z(CFO/PAT)
      + 0.15 Z(FCF/Assets) + 0.15 Z(-Debt/Assets) + 0.10 Z(Payout)

Quality is the strongest and most robust Indian factor in the literature
(IIMA: QMJ earns ~0.92%/month four-factor alpha, with low turnover and shallow
drawdowns, and it works long-only). The profitability and payout legs do most
of the work, consistent with the tunnelling hypothesis - in a market with
uneven governance enforcement, cash actually returned to shareholders is
informative.

NSE quarterly XBRL typically fills the P&L (revenue, PAT, PBT, EPS) and not
the balance sheet, so when the classic six legs are empty we fall back to
trailing net/pretax margin plus an EBIT/equity ROIC proxy - still point-in-time
via announce_date. A single stray metric is never enough to define quality.
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
    "net_margin": ("net_margin", True),
    "pretax_margin": ("pretax_margin", True),
}

# Honest P&L stand-in when balance-sheet quality legs are missing.
PNL_QUALITY_WEIGHTS = QualityWeights(
    roic=0.35,
    gross_profitability=0.15,
    cfo_to_pat=0.0,
    fcf_to_assets=0.0,
    neg_debt_to_assets=0.0,
    payout=0.0,
    net_margin=0.35,
    pretax_margin=0.15,
)

# At least two independent legs must print; one ratio is not "quality".
_MIN_QUALITY_LEGS = 2


def compute_quality_score(
    fundamentals: pd.DataFrame,
    dt: pd.Timestamp,
    weights: QualityWeights | None = None,
    winsor_pct: float = 0.01,
    universe: list[str] | None = None,
    min_components: int | None = None,
) -> pd.Series:
    """Quality score as known on `dt` (point-in-time via announce_date)."""
    w = weights or QualityWeights()
    need = min_components if min_components is not None else _MIN_QUALITY_LEGS
    snap = as_of(fundamentals, dt, metrics=list(QUALITY_SPEC))
    if snap.empty:
        return pd.Series(dtype=float)

    if universe:
        snap = snap.reindex([i for i in universe if i in snap.index])
    if snap.empty:
        return pd.Series(dtype=float)

    components, weight_map = _quality_components(snap, w, winsor_pct)
    if float(getattr(w, "margin_growth", 0.0) or 0.0) > 0:
        growth = compute_growth_score(fundamentals, dt, universe, winsor_pct)
        if not growth.empty:
            components["margin_growth"] = growth
            weight_map["margin_growth"] = float(w.margin_growth)
    if len(components) < need:
        classic = (
            w.roic + w.gross_profitability + w.cfo_to_pat
            + w.fcf_to_assets + w.neg_debt_to_assets
        )
        if classic > 0:
            fb_c, fb_w = _quality_components(snap, PNL_QUALITY_WEIGHTS, winsor_pct)
            for key, series in fb_c.items():
                if key not in components:
                    components[key] = series
                    weight_map[key] = fb_w[key]
    if not components:
        return pd.Series(dtype=float)
    return combine_scores(components, weight_map, min_components=need)


def _quality_components(
    snap: pd.DataFrame, w: QualityWeights, winsor_pct: float
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    components: dict[str, pd.Series] = {}
    weight_map: dict[str, float] = {}
    for metric, (attr, higher_better) in QUALITY_SPEC.items():
        weight = float(getattr(w, attr, 0.0) or 0.0)
        if weight == 0 or metric not in snap.columns:
            continue
        series = pd.to_numeric(snap[metric], errors="coerce")
        if series.notna().sum() < 3:
            continue
        if not higher_better:
            series = -series
        components[attr] = zscore(series, winsor_pct)
        weight_map[attr] = weight
    return components, weight_map


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


PROFIT_METRICS = ["net_margin", "pretax_margin", "roic", "gross_profitability"]


def compute_growth_score(
    fundamentals: pd.DataFrame,
    dt: pd.Timestamp,
    universe: list[str] | None = None,
    winsor_pct: float = 0.01,
) -> pd.Series:
    """IIMA QMJ growth: change in profitability vs one year earlier.

    The paper uses a five-year window. NSE XBRL here only covers a few years,
    so this is trailing one-year change in the P&L profitability legs — still
    dated by announce_date, never invented.

    Prior is the last print from a year before *that name's latest print*,
    not a calendar year before `dt`. Otherwise a stale XBRL cache (latest
    filing in 2024, screener date 2026) would score every name as zero growth.
    """
    if fundamentals.empty or "announce_date" not in fundamentals.columns:
        return pd.Series(dtype=float)
    ts = pd.Timestamp(dt)
    vis = fundamentals.loc[pd.to_datetime(fundamentals["announce_date"]) <= ts].copy()
    if vis.empty:
        return pd.Series(dtype=float)
    vis["announce_date"] = pd.to_datetime(vis["announce_date"])
    vis["isin"] = vis["isin"].astype(str)
    if universe:
        vis = vis[vis["isin"].isin({str(i) for i in universe})]
    if vis.empty:
        return pd.Series(dtype=float)
    latest = vis.sort_values("announce_date").groupby("isin").tail(1).set_index("isin")
    cut = latest["announce_date"] - pd.DateOffset(years=1)
    vis = vis.merge(cut.rename("cut"), left_on="isin", right_index=True, how="inner")
    vis = vis[vis["announce_date"] <= vis["cut"]]
    if vis.empty:
        return pd.Series(dtype=float)
    prior = vis.sort_values("announce_date").groupby("isin").tail(1).set_index("isin")
    prior = prior.reindex(latest.index)
    same = pd.to_datetime(latest["announce_date"]) == pd.to_datetime(prior["announce_date"])
    components: dict[str, pd.Series] = {}
    for metric in PROFIT_METRICS:
        if metric not in latest.columns or metric not in prior.columns:
            continue
        delta = pd.to_numeric(latest[metric], errors="coerce") - pd.to_numeric(prior[metric], errors="coerce")
        delta = delta.where(~same.fillna(True))
        if delta.notna().sum() < 3:
            continue
        components[metric] = zscore(delta, winsor_pct)
    if not components:
        return pd.Series(dtype=float)
    return combine_scores(components, {k: 1.0 for k in components}, min_components=1)


def compute_iima_dimensions(
    fundamentals: pd.DataFrame,
    dt: pd.Timestamp,
    universe: list[str] | None = None,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """Cross-section of the four IIMA QMJ dimensions on `dt`.

    Profitability and payout are the legs Jacob, Pradeep & Varma found drove
    Indian QMJ alpha (tunnelling). Growth and safety are scored when the
    underlying prints exist; empty columns stay empty.
    """
    prof_w = QualityWeights(
        roic=0.35, gross_profitability=0.15, cfo_to_pat=0.0, fcf_to_assets=0.0,
        neg_debt_to_assets=0.0, payout=0.0, net_margin=0.35, pretax_margin=0.15,
    )
    pay_w = QualityWeights(
        roic=0.0, gross_profitability=0.0, cfo_to_pat=0.0, fcf_to_assets=0.0,
        neg_debt_to_assets=0.0, payout=1.0, net_margin=0.0, pretax_margin=0.0,
    )
    safe_w = QualityWeights(
        roic=0.0, gross_profitability=0.0, cfo_to_pat=0.0, fcf_to_assets=0.0,
        neg_debt_to_assets=1.0, payout=0.0, net_margin=0.0, pretax_margin=0.0,
    )
    profitability = compute_quality_score(fundamentals, dt, prof_w, winsor_pct, universe, min_components=1)
    payout = compute_quality_score(fundamentals, dt, pay_w, winsor_pct, universe, min_components=1)
    safety = compute_quality_score(fundamentals, dt, safe_w, winsor_pct, universe, min_components=1)
    growth = compute_growth_score(fundamentals, dt, universe, winsor_pct)
    frame = pd.DataFrame({
        "profitability": profitability,
        "growth": growth,
        "safety": safety,
        "payout_z": payout,
    })
    return frame.dropna(how="all")
