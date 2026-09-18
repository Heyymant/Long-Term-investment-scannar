"""Backtest engine, portfolio construction and risk tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.backtest.engine_event import run_event_backtest
from src.backtest.engine_vectorized import run_vectorized_backtest
from src.backtest.metrics import cagr, compute_metrics, max_drawdown, sharpe_ratio
from src.backtest.runner import compare_engines
from src.portfolio.buffer import apply_rank_buffer, turnover
from src.portfolio.construction import apply_weight_caps, effective_caps
from src.portfolio.regime import compute_exposure_series
from src.presets import get_preset
from src.risk.portfolio_risk import drawdown_exposure_multiplier, merge_sleeve_weights
from src.schema import RegimeConfig, RiskConfig


# ------------------------------------------------------------- construction
def test_weight_caps_conserve_total_exposure():
    w = pd.Series(np.random.default_rng(0).lognormal(0, 0.5, 40))
    w /= w.sum()
    capped = apply_weight_caps(w, 0.05, None, None)
    assert capped.sum() == pytest.approx(1.0, abs=1e-6)
    assert capped.max() <= 0.05 + 1e-9


def test_infeasible_cap_is_relaxed_not_silently_underinvested():
    """10 names at a 5% cap can only reach 50%; the cap must widen."""
    stock_cap, _ = effective_caps(10, 0.05)
    assert stock_cap == pytest.approx(0.10)

    w = pd.Series(1.0 / 10, index=[f"S{i}" for i in range(10)])
    capped = apply_weight_caps(w, 0.05)
    assert capped.sum() == pytest.approx(1.0, abs=1e-6)


def test_sector_caps_are_enforced():
    n = 30
    w = pd.Series(1.0 / n, index=[f"S{i}" for i in range(n)])
    # Three sectors, so a 25% cap is infeasible and should relax to ~33%.
    sectors = pd.Series([f"SEC{i % 3}" for i in range(n)], index=w.index)
    capped = apply_weight_caps(w, 0.10, sectors, 0.25)
    by_sector = capped.groupby(sectors).sum()
    assert by_sector.max() <= 0.34 + 1e-6


def test_rank_buffer_reduces_turnover():
    scores = pd.Series(np.arange(100, 0, -1), index=[f"S{i}" for i in range(100)])
    held = set(scores.index[:20])

    # A name that drifted from rank 20 to 25 should be retained.
    shifted = scores.copy()
    shifted["S19"] = 76   # now ranks ~25th

    no_buffer = set(apply_rank_buffer(shifted, held, 20, buffer_multiple=1.0))
    with_buffer = set(apply_rank_buffer(shifted, held, 20, buffer_multiple=2.0))

    assert len(with_buffer & held) >= len(no_buffer & held)


def test_turnover_calculation():
    prev = pd.Series({"A": 0.5, "B": 0.5})
    curr = pd.Series({"A": 0.5, "C": 0.5})
    assert turnover(prev, curr) == pytest.approx(0.5)   # one-way


# ------------------------------------------------------------------- regime
def test_regime_overlay_cuts_exposure_in_downtrends():
    dates = pd.bdate_range("2018-01-01", periods=600)
    up = pd.Series(np.linspace(100, 200, 300), index=dates[:300])
    down = pd.Series(np.linspace(200, 100, 300), index=dates[300:])
    bench = pd.concat([up, down])

    exposure = compute_exposure_series(bench, RegimeConfig(enabled=True, ma_days=100))
    assert exposure.iloc[250] > exposure.iloc[-50]


def test_disabled_regime_stays_fully_invested():
    dates = pd.bdate_range("2020-01-01", periods=300)
    bench = pd.Series(np.linspace(200, 100, 300), index=dates)
    exposure = compute_exposure_series(bench, RegimeConfig(enabled=False))
    assert (exposure == 1.0).all()


def test_hysteresis_reduces_switching():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2018-01-01", periods=800)
    # Choppy market oscillating around its mean - the whipsaw case.
    bench = pd.Series(100 + np.cumsum(rng.normal(0, 0.8, 800)), index=dates)

    tight = compute_exposure_series(bench, RegimeConfig(ma_days=100, hysteresis_band=0.0, graded=False))
    wide = compute_exposure_series(bench, RegimeConfig(ma_days=100, hysteresis_band=0.05, graded=False))

    assert (wide.diff().abs() > 0).sum() <= (tight.diff().abs() > 0).sum()


# --------------------------------------------------------------------- risk
def test_drawdown_derisking_is_staged():
    cfg = RiskConfig(
        drawdown_derisk_enabled=True,
        drawdown_thresholds=[-0.15, -0.25, -0.35],
        drawdown_exposures=[0.75, 0.50, 0.25],
    )
    assert drawdown_exposure_multiplier(-0.05, cfg) == 1.0
    assert drawdown_exposure_multiplier(-0.20, cfg) == 0.75
    assert drawdown_exposure_multiplier(-0.30, cfg) == 0.50
    assert drawdown_exposure_multiplier(-0.40, cfg) == 0.25


def test_drawdown_derisk_releases_after_recovery():
    """Regression: de-risking must not latch permanently.

    Measuring drawdown against the all-time high means a portfolio that has
    been cut to 25% exposure can never climb back to its old peak, so the
    circuit breaker never releases and the strategy is dead. A rolling peak
    fixes that.
    """
    from src.risk.portfolio_risk import running_drawdown

    idx = pd.bdate_range("2015-01-01", periods=1500)
    # Crash early, then grind sideways well below the old high.
    values = np.concatenate([
        np.linspace(100, 160, 300),     # rally to a high
        np.linspace(160, 90, 200),      # crash
        np.linspace(90, 115, 1000),     # partial, sustained recovery
    ])
    equity = pd.Series(values, index=idx)

    all_time = running_drawdown(equity, None).iloc[-1]
    rolling = running_drawdown(equity, 504).iloc[-1]

    # Against the all-time high it still looks deeply underwater forever.
    assert all_time < -0.25
    # Against a rolling peak it has recovered, so exposure can be restored.
    assert rolling > -0.05


def test_rolling_drawdown_still_detects_a_live_crash():
    idx = pd.bdate_range("2015-01-01", periods=600)
    equity = pd.Series(
        np.concatenate([np.linspace(100, 150, 400), np.linspace(150, 95, 200)]), index=idx
    )
    assert running_drawdown_value(equity) < -0.30


def running_drawdown_value(equity: pd.Series) -> float:
    from src.risk.portfolio_risk import running_drawdown

    return float(running_drawdown(equity, 504).iloc[-1])


def test_overlapping_sleeves_top_up_rather_than_duplicate():
    a = pd.Series({"X": 0.5, "Y": 0.5})
    b = pd.Series({"X": 1.0})
    merged = merge_sleeve_weights({"A": a, "B": b}, {"A": 0.6, "B": 0.4})
    assert merged["X"] == pytest.approx(0.5 * 0.6 + 1.0 * 0.4)
    assert len(merged) == 2


# ------------------------------------------------------------------ metrics
def test_cagr_matches_a_known_compounding():
    dates = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(np.full(252, (1.10) ** (1 / 252) - 1), index=dates)
    assert cagr(r) == pytest.approx(0.10, abs=1e-3)


def test_max_drawdown_finds_peak_and_trough():
    r = pd.Series([0.1, 0.1, -0.5, 0.2], index=pd.bdate_range("2020-01-01", periods=4))
    dd = max_drawdown(r)
    assert dd["max_drawdown"] == pytest.approx(-0.5, abs=1e-9)


def test_sharpe_is_zero_for_flat_returns():
    r = pd.Series(np.zeros(100), index=pd.bdate_range("2020-01-01", periods=100))
    assert np.isnan(sharpe_ratio(r)) or sharpe_ratio(r) == 0


# ------------------------------------------------------------------ engines
def test_vectorized_engine_produces_results(dataset):
    cfg = get_preset("price_only_core")
    res = run_vectorized_backtest(
        cfg, dataset.prices, dataset.eligibility, None,
        dataset.sectors, dataset.benchmark, dataset.adv, CostModel(),
    )
    assert not res.returns.empty
    assert not res.equity_curve.empty
    assert res.diagnostics["n_rebalances"] > 0


def test_costs_reduce_returns(dataset):
    cfg = get_preset("price_only_core")
    free = run_vectorized_backtest(
        cfg, dataset.prices, dataset.eligibility, None,
        dataset.sectors, dataset.benchmark, dataset.adv, None,
    )
    costly = run_vectorized_backtest(
        cfg, dataset.prices, dataset.eligibility, None,
        dataset.sectors, dataset.benchmark, dataset.adv,
        CostModel(slippage_bps=50.0),
    )
    assert cagr(costly.returns) < cagr(free.returns)


def test_event_engine_runs_and_tracks_tax(dataset):
    cfg = get_preset("price_only_core")
    cfg.window.initial_capital = 10_000_000
    res = run_event_backtest(
        cfg, dataset.prices, dataset.eligibility, None,
        dataset.sectors, dataset.benchmark, dataset.adv, CostModel(),
    )
    assert not res.equity_curve.empty
    assert res.diagnostics["n_fills"] > 0


@pytest.mark.slow
def test_engines_agree_within_tolerance(dataset):
    """The two engines model different levels of detail, but must not
    disagree wildly - that would mean one is wrong."""
    cfg = get_preset("price_only_core")
    cfg.window.initial_capital = 50_000_000   # reduce integer-share rounding

    vec = run_vectorized_backtest(
        cfg, dataset.prices, dataset.eligibility, None,
        dataset.sectors, dataset.benchmark, dataset.adv, CostModel(),
    )
    ev = run_event_backtest(
        cfg, dataset.prices, dataset.eligibility, None,
        dataset.sectors, dataset.benchmark, dataset.adv, CostModel(),
    )
    out = compare_engines(vec, ev, tolerance=0.60)
    assert out["vectorized_sharpe"] is not None
    assert out["event_sharpe"] is not None
    assert out["consistent"], out["verdict"]
