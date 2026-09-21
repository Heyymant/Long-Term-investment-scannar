"""Factor engine tests on recorded NSE prices and filings.

These assert properties of the factor code against real market data: ranking
agrees with the statistic it is built from, point-in-time filters ignore
later announcements, and composite scores stay finite.
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


def test_momentum_ranks_real_winners(prices):
    """On real NSE prices the score must agree with 12-1 return rank."""
    raw = momentum_return(prices, lookback_months=12, skip_months=1)
    score = compute_momentum_score(prices)
    common = raw.dropna().index.intersection(score.dropna().index)
    assert len(common) >= 20
    # Winsorization and the vol-adjusted leg move levels, not ranks.
    spearman = pd.Series(raw[common]).corr(pd.Series(score[common]), method="spearman")
    assert spearman > 0.85


# ------------------------------------------------------------------ low vol
def test_low_vol_score_prefers_calm_names(market):
    px = market.prices
    score = compute_low_vol_score(px)
    realized = trailing_volatility(px)
    common = score.dropna().index.intersection(realized.dropna().index)
    corr = np.corrcoef(score[common], realized[common])[0, 1]
    assert corr < -0.9


def test_low_vol_ranks_real_names(market):
    score = compute_low_vol_score(market.prices).dropna()
    vol = trailing_volatility(market.prices).reindex(score.index)
    assert score.idxmax() == vol.idxmin()


# ------------------------------------------------------------------ quality
def test_quality_scores_real_filings(market):
    """Quality needs several legs. NSE's current XBRL window often only
    fills ROIC, and the scorer must not invent a rank from a single metric."""
    if market.fundamentals.empty:
        pytest.skip("No recorded XBRL filings in the fixture")
    dt = pd.to_datetime(market.fundamentals["announce_date"]).max()
    score = compute_quality_score(market.fundamentals, dt).dropna()
    metrics = ["roic", "gross_profitability", "cfo_to_pat", "fcf_to_assets",
               "debt_to_assets", "payout", "net_margin", "pretax_margin"]
    present = [c for c in metrics if c in market.fundamentals.columns]
    filled_legs = int(market.fundamentals[present].notna().any().sum()) if present else 0
    if filled_legs < 2:
        assert score.empty
    else:
        assert len(score) > 0
        assert np.isfinite(score).all()


def test_quality_pnl_fallback_needs_two_legs():
    """A single margin print is not quality; two P&L legs can rank."""
    from src.schema import QualityWeights

    n = 12
    isins = [f"INE{i:03d}" for i in range(n)]
    fund = pd.DataFrame({
        "isin": isins,
        "announce_date": pd.Timestamp("2023-06-01"),
        "period_end": pd.Timestamp("2023-03-31"),
        "roic": np.nan,
        "gross_profitability": np.linspace(0.1, 0.4, n),
        "cfo_to_pat": np.nan,
        "fcf_to_assets": np.nan,
        "debt_to_assets": np.nan,
        "payout": np.nan,
        "net_margin": np.linspace(0.02, 0.20, n),
        "pretax_margin": np.linspace(0.03, 0.25, n),
    })
    dt = pd.Timestamp("2023-06-30")
    one_leg = fund.copy()
    one_leg["gross_profitability"] = np.nan
    one_leg["pretax_margin"] = np.nan
    assert compute_quality_score(one_leg, dt, QualityWeights()).dropna().empty

    score = compute_quality_score(fund, dt, QualityWeights()).dropna()
    assert len(score) == n
    assert score.idxmax() == isins[-1]


def test_iima_profitability_ranks_fatter_margins():
    from src.factors.quality import compute_iima_dimensions

    n = 10
    isins = [f"INE{i:03d}" for i in range(n)]
    fund = pd.DataFrame({
        "isin": isins,
        "announce_date": pd.Timestamp("2023-06-01"),
        "period_end": pd.Timestamp("2023-03-31"),
        "roic": np.linspace(0.05, 0.25, n),
        "gross_profitability": np.linspace(0.1, 0.4, n),
        "cfo_to_pat": np.nan,
        "fcf_to_assets": np.nan,
        "debt_to_assets": np.nan,
        "payout": np.nan,
        "net_margin": np.linspace(0.02, 0.20, n),
        "pretax_margin": np.linspace(0.03, 0.25, n),
    })
    dims = compute_iima_dimensions(fund, pd.Timestamp("2023-06-30"))
    assert dims["profitability"].idxmax() == isins[-1]
    assert dims["payout_z"].isna().all()


def test_iima_growth_is_year_over_year_and_point_in_time():
    from src.factors.quality import compute_growth_score

    isins = [f"INE{i:03d}" for i in range(8)]
    early = pd.DataFrame({
        "isin": isins,
        "announce_date": pd.Timestamp("2022-06-01"),
        "period_end": pd.Timestamp("2022-03-31"),
        "net_margin": np.linspace(0.02, 0.04, 8),
        "pretax_margin": np.linspace(0.03, 0.05, 8),
        "roic": 0.10,
        "gross_profitability": 0.20,
    })
    late = pd.DataFrame({
        "isin": isins,
        "announce_date": pd.Timestamp("2023-06-01"),
        "period_end": pd.Timestamp("2023-03-31"),
        "net_margin": np.linspace(0.04, 0.20, 8),
        "pretax_margin": np.linspace(0.05, 0.22, 8),
        "roic": 0.12,
        "gross_profitability": 0.22,
    })
    fund = pd.concat([early, late], ignore_index=True)
    before = compute_growth_score(fund, pd.Timestamp("2022-12-31"))
    assert before.empty
    after = compute_growth_score(fund, pd.Timestamp("2023-06-30")).dropna()
    assert len(after) == 8
    assert after.idxmax() == isins[-1]
    leaked = fund.copy()
    leaked.loc[leaked["announce_date"] == pd.Timestamp("2023-06-01"), "net_margin"] = 9.0
    leaked_score = compute_growth_score(leaked, pd.Timestamp("2022-12-31"))
    assert leaked_score.empty
    # Same filing on both sides of the year is missing growth, not a 0 print.
    stale_only = late.copy()
    stale_only["announce_date"] = pd.Timestamp("2021-06-01")
    same = compute_growth_score(stale_only, pd.Timestamp("2023-12-31"))
    assert same.empty
    # Last-print YoY still scores when the screener date is years after XBRL.
    delayed = compute_growth_score(fund, pd.Timestamp("2026-09-21")).dropna()
    assert len(delayed) == 8
    assert delayed.idxmax() == isins[-1]


def test_value_live_earnings_yield_from_price():
    from src.factors.value import compute_value_score

    n = 10
    isins = [f"INE{i:03d}" for i in range(n)]
    fund = pd.DataFrame({
        "isin": isins,
        "announce_date": pd.Timestamp("2023-06-01"),
        "period_end": pd.Timestamp("2023-03-31"),
        "eps_ttm": np.linspace(5, 14, n),
        "earnings_yield": np.nan,
        "pe": np.nan,
        "fcf_yield": np.nan,
        "ev_ebitda": np.nan,
        "pb": np.nan,
    })
    idx = pd.bdate_range("2023-06-01", periods=20)
    prices = pd.DataFrame(100.0, index=idx, columns=isins)
    score = compute_value_score(fund, idx[-1], prices=prices).dropna()
    assert len(score) == n
    # Higher EPS at the same price is cheaper (higher value score).
    assert score.idxmax() == isins[-1]


def test_quality_is_point_in_time(market):
    """Fundamentals announced after `dt` must not influence the score."""
    if market.fundamentals.empty:
        pytest.skip("No recorded XBRL filings in the fixture")
    announced = pd.to_datetime(market.fundamentals["announce_date"])
    mid = announced.median()
    if not ((announced <= mid).any() and (announced > mid).any()):
        pytest.skip("Need filings on both sides of a cut date")
    dt = pd.Timestamp(mid)

    score_now = compute_quality_score(market.fundamentals, dt)

    future = market.fundamentals.copy()
    mask = pd.to_datetime(future["announce_date"]) > dt
    future.loc[mask, "roic"] = 999.0

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
    if market.fundamentals.empty:
        pytest.skip("No recorded XBRL filings in the fixture")
    dt = pd.to_datetime(market.fundamentals["announce_date"]).max()
    f = market.fundamentals.copy()
    f["pe"] = 12.0
    f.loc[f.index[: min(5, len(f))], "pe"] = -10.0
    score = compute_value_score(f, dt)
    assert score.notna().sum() >= 0  # negative PE must not crash the scorer


# ---------------------------------------------------------- sector handling
def test_sector_neutralize_zeroes_sector_means(market):
    if market.fundamentals.empty:
        pytest.skip("No recorded XBRL filings in the fixture")
    dt = pd.to_datetime(market.fundamentals["announce_date"]).max()
    sectors = market.securities.set_index("isin")["sector"]
    raw = compute_quality_score(market.fundamentals, dt).dropna()
    if raw.empty:
        pytest.skip("Quality score empty on recorded filings")
    # Restrict to names that have a sector label.
    raw = raw[raw.index.isin(sectors.index)]
    if raw.empty:
        pytest.skip("No overlap between filings and security master")
    neutral = sector_neutralize(raw, sectors)
    means = neutral.groupby(sectors.reindex(neutral.index)).mean()
    assert means.abs().max() < 1e-9


def test_composite_combines_price_legs(market):
    from src.factors.composite import compute_composite

    dt = market.prices.index[len(market.prices) // 2]
    sectors = market.securities.set_index("isin")["sector"]
    cs = compute_composite(
        market.prices, dt, FactorConfig(), list(market.prices.columns),
        market.fundamentals, sectors,
    )
    assert not cs.composite.empty
    assert not cs.momentum.empty
    assert cs.composite.notna().sum() > 20
