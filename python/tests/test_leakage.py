"""Look-ahead leakage tests.

The most dangerous bug in a backtest is one that improves results. These
tests deliberately construct leaking signals and assert that the detectors
catch them - a leak detector that never fires is worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.statistics import detect_lookahead, shuffled_future_test
from src.factors.ic_icir import compute_ic, forward_returns
from src.factors.quality import compute_quality_score


@pytest.fixture
def panel():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", periods=600)
    cols = [f"S{i}" for i in range(30)]
    rets = pd.DataFrame(rng.normal(0.0004, 0.015, (len(dates), len(cols))),
                        index=dates, columns=cols)
    prices = 100 * (1 + rets).cumprod()
    return prices


def test_detector_fires_on_a_leaking_signal(panel):
    """A signal built FROM future returns must be flagged."""
    score_dates = panel.index[::21][:-3]
    fwd = forward_returns(panel, score_dates, 21)
    leaked = fwd.copy()   # the signal *is* the future return

    result = detect_lookahead(leaked, fwd, threshold=0.15)
    assert result["suspicious"]
    assert result["mean_ic"] > 0.9


def test_detector_stays_quiet_on_an_honest_signal(panel):
    """A signal built only from past data should show a plausible IC."""
    score_dates = panel.index[::21][:-3]
    honest = {}
    for dt in score_dates:
        hist = panel.loc[panel.index <= dt]
        if len(hist) < 60:
            continue
        honest[dt] = (hist.iloc[-1] / hist.iloc[-60] - 1)   # past momentum only
    honest = pd.DataFrame(honest).T

    fwd = forward_returns(panel, honest.index, 21)
    result = detect_lookahead(honest, fwd, threshold=0.15)
    assert not result["suspicious"]


def test_shuffled_signal_destroys_any_edge(panel):
    rets = panel.pct_change().mean(axis=1).dropna()
    signal = rets.shift(-1).dropna()      # deliberately leaked
    common = rets.index.intersection(signal.index)

    out = shuffled_future_test(rets.loc[common], signal.loc[common], n_perm=100)
    # After shuffling, average correlation must collapse toward zero.
    assert abs(out["permuted_mean"]) < 0.1


def test_point_in_time_fundamentals_block_future_data(market):
    """Corrupting only future announcements must not change today's score."""
    dt = pd.Timestamp("2019-06-30")
    baseline = compute_quality_score(market.fundamentals, dt)

    tampered = market.fundamentals.copy()
    future = pd.to_datetime(tampered["announce_date"]) > dt
    assert future.any(), "test needs some future rows"
    tampered.loc[future, ["roic", "payout", "fcf_to_assets"]] = 99.0

    after = compute_quality_score(tampered, dt)
    pd.testing.assert_series_equal(baseline, after)


def test_composite_ignores_prices_after_the_scoring_date(market):
    """Future prices must not influence a score computed today."""
    from src.factors.composite import compute_composite
    from src.schema import FactorConfig

    dt = pd.Timestamp("2020-01-31")
    universe = list(market.prices.columns)
    sectors = market.securities.set_index("isin")["sector"]

    baseline = compute_composite(
        market.prices, dt, FactorConfig(), universe, market.fundamentals, sectors
    ).composite

    tampered = market.prices.copy()
    tampered.loc[tampered.index > dt] *= 3.0     # violent future move

    after = compute_composite(
        tampered, dt, FactorConfig(), universe, market.fundamentals, sectors
    ).composite

    pd.testing.assert_series_equal(baseline, after)


def test_engine_applies_weights_only_to_later_returns(dataset):
    """Weights decided at t must earn returns from t+1, never on t itself."""
    from src.backtest.engine_vectorized import run_vectorized_backtest
    from src.presets import get_preset

    cfg = get_preset("price_only_core")
    result = run_vectorized_backtest(
        cfg, dataset.prices, dataset.eligibility, None,
        dataset.sectors, dataset.benchmark, dataset.adv, None,
    )
    assert not result.returns.empty
    first_rebalance = result.rebalance_dates[0]
    # No return should be attributed on or before the first decision date.
    assert (result.returns.loc[result.returns.index <= first_rebalance] == 0).all()
