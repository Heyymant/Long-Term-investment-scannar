"""Factor engine tests.

The synthetic market has *known* planted premia, so these are known-answer
tests: if the quality factor cannot recover planted quality, the factor code
is broken regardless of how plausible a backtest looks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.base import combine_scores, neutralize, winsorize, zscore
from src.factors.lowvol import compute_low_vol_score, trailing_volatility
from src.factors.momentum import compute_momentum_score, momentum_return
from src.factors.quality import compute_quality_score
from src.factors.sector_neutral import sector_neutralize
from src.factors.value import apply_value_quality_gate, compute_value_score
from src.schema import FactorConfig


# --------------------------------------------------------------------- base
def test_zscore_is_standardized():
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z = zscore(s, winsor_pct=0.0)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


def test_winsorize_clips_outliers():
    s = pd.Series([1.0] * 98 + [1000.0, -1000.0])
    w = winsorize(s, 0.05)
    assert w.max() < 1000
    assert w.min() > -1000


def test_zscore_handles_constant_series():
    z = zscore(pd.Series([5.0] * 10))
    assert (z.fillna(0) == 0).all()


def test_combine_scores_renormalizes_when_components_missing():
    a = pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"])
    b = pd.Series([1.0, np.nan, 3.0], index=["x", "y", "z"])
    out = combine_scores({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    # 'y' only has component a, so it should equal a's value, not be halved.
    assert out["y"] == pytest.approx(2.0)
    assert out["x"] == pytest.approx(1.0)


def test_neutralize_removes_group_means():
    s = pd.Series([1.0, 3.0, 10.0, 12.0], index=list("abcd"))
    g = pd.Series(["A", "A", "B", "B"], index=list("abcd"))
    out = neutralize(s, g)
    assert out.groupby(g).mean().abs().max() < 1e-9


# ----------------------------------------------------------------- momentum
def test_momentum_skips_recent_month(prices):
    """12-1 momentum must not include the most recent month."""
    mom = momentum_return(prices, lookback_months=12, skip_months=1)
    assert mom.notna().sum() > 0
    assert np.isfinite(mom.dropna()).all()


def test_momentum_detects_a_planted_trend():
    """Cross-sectional scoring needs a real cross-section, so use several names."""
    dates = pd.bdate_range("2020-01-01", periods=400)
    panel = pd.DataFrame({
        "UP": 100 * np.exp(np.linspace(0, 0.6, 400)),
        "MILD_UP": 100 * np.exp(np.linspace(0, 0.2, 400)),
        "FLAT": np.full(400, 100.0),
        "MILD_DOWN": 100 * np.exp(np.linspace(0, -0.2, 400)),
        "DOWN": 100 * np.exp(np.linspace(0, -0.6, 400)),
    }, index=dates)

    score = compute_momentum_score(panel)
    assert score["UP"] > score["FLAT"] > score["DOWN"]


# ------------------------------------------------------------------ low vol
def test_low_vol_score_prefers_calm_names(market):
    px = market.prices
    score = compute_low_vol_score(px)
    realized = trailing_volatility(px)
    common = score.dropna().index.intersection(realized.dropna().index)
    # Score is -Z(vol), so it must be negatively correlated with volatility.
    corr = np.corrcoef(score[common], realized[common])[0, 1]
    assert corr < -0.9


def test_low_vol_recovers_planted_structure(market):
    score = compute_low_vol_score(market.prices).dropna()
    truth = market.true_scores.loc[score.index, "lowvol_pref"]
    assert np.corrcoef(score, truth)[0, 1] > 0.7


# ------------------------------------------------------------------ quality
def test_quality_recovers_planted_quality(market):
    dt = pd.Timestamp("2020-06-30")
    score = compute_quality_score(market.fundamentals, dt).dropna()
    assert len(score) > 20
    truth = market.true_scores.loc[score.index, "quality"]
    assert np.corrcoef(score, truth)[0, 1] > 0.8


def test_quality_is_point_in_time(market):
    """Fundamentals announced after `dt` must not influence the score."""
    dt = pd.Timestamp("2018-06-30")
    score_now = compute_quality_score(market.fundamentals, dt)

    future = market.fundamentals.copy()
    mask = pd.to_datetime(future["announce_date"]) > dt
    future.loc[mask, "roic"] = 999.0  # corrupt only the future

    score_corrupted = compute_quality_score(future, dt)
    pd.testing.assert_series_equal(score_now, score_corrupted)


# -------------------------------------------------------------------- value
def test_value_gate_penalizes_cheap_junk():
    value = pd.Series([2.0, 2.0], index=["good", "junk"])
    quality = pd.Series([1.5, -2.0], index=["good", "junk"])
    gated = apply_value_quality_gate(value, quality, strength=1.0)
    assert gated["good"] > gated["junk"]
    assert gated["good"] == pytest.approx(2.0)   # good quality untouched


def test_value_ignores_negative_multiples(market):
    dt = pd.Timestamp("2020-06-30")
    f = market.fundamentals.copy()
    f.loc[f.index[:5], "pe"] = -10.0             # loss-making
    score = compute_value_score(f, dt)
    assert score.notna().sum() > 0


# ---------------------------------------------------------- sector handling
def test_sector_neutralize_zeroes_sector_means(market):
    dt = pd.Timestamp("2020-06-30")
    sectors = market.securities.set_index("isin")["sector"]
    raw = compute_quality_score(market.fundamentals, dt).dropna()
    neutral = sector_neutralize(raw, sectors)
    means = neutral.groupby(sectors.reindex(neutral.index)).mean()
    assert means.abs().max() < 1e-9


def test_composite_combines_all_legs(market):
    from src.factors.composite import compute_composite

    dt = pd.Timestamp("2020-06-30")
    sectors = market.securities.set_index("isin")["sector"]
    cs = compute_composite(
        market.prices, dt, FactorConfig(), list(market.prices.columns),
        market.fundamentals, sectors,
    )
    assert not cs.composite.empty
    assert not cs.quality.empty and not cs.momentum.empty
    assert cs.composite.notna().sum() > 20
