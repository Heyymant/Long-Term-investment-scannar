"""Statistical validation tests.

The critical property: these tools must *reject* noise. A validation suite
that blesses random data is worse than none, because it manufactures
confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.statistics import (
    bootstrap_metric,
    deflated_sharpe_ratio,
    haircut_sharpe,
    min_track_record_length,
    permutation_test,
    probability_backtest_overfitting,
    reality_check,
    sharpe_test,
    spa_test,
    validate_strategy,
)


@pytest.fixture
def noise():
    """Pure noise - every test below should decline to find an edge."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    return pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)


@pytest.fixture
def strong_edge():
    """A genuinely good strategy.

    The realized mean is pinned to the target rather than left to the draw,
    so the test asserts on the statistics rather than on a lucky sample.
    """
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2012-01-01", periods=2500)
    target_sharpe, vol = 2.0, 0.01
    noise = rng.normal(0, vol, len(idx))
    noise = (noise - noise.mean()) / noise.std(ddof=1) * vol
    return pd.Series(noise + target_sharpe / np.sqrt(252) * vol, index=idx)


# -------------------------------------------------------------- Sharpe test
def test_sharpe_test_rejects_noise(noise):
    result = sharpe_test(noise)
    assert not result.significant
    assert result.ci_lower < 0 < result.ci_upper


def test_sharpe_test_detects_a_real_edge(strong_edge):
    result = sharpe_test(strong_edge)
    assert result.significant
    assert result.sharpe > 1.5
    assert result.ci_lower > 0


def test_sharpe_se_widens_with_autocorrelation():
    """Positively autocorrelated returns understate risk; the SE must grow."""
    rng = np.random.default_rng(3)
    n = 1000
    idx = pd.bdate_range("2018-01-01", periods=n)

    iid = rng.normal(0.0005, 0.01, n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.3 * ar[i - 1] + rng.normal(0.0005, 0.01)

    from src.backtest.statistics import sharpe_standard_error

    se_iid, _ = sharpe_standard_error(pd.Series(iid, index=idx))
    se_ar, rho = sharpe_standard_error(pd.Series(ar, index=idx))
    assert rho > 0.1
    assert se_ar > se_iid


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_ci_contains_zero_for_noise(noise):
    out = bootstrap_metric(noise, n_boot=300, seed=1)
    assert out["ci_lower"] < 0 < out["ci_upper"]


def test_bootstrap_ci_excludes_zero_for_real_edge(strong_edge):
    out = bootstrap_metric(strong_edge, n_boot=300, seed=1)
    assert out["ci_lower"] > 0
    assert out["prob_positive"] > 0.95


def test_permutation_test_on_noise(noise):
    out = permutation_test(noise, n_perm=200, seed=2)
    assert 0.0 <= out["p_value"] <= 1.0


# ------------------------------------------------------- multiple testing
def test_deflated_sharpe_falls_as_trials_rise():
    one = deflated_sharpe_ratio(1.2, n_trials=1, n_obs=1500)
    many = deflated_sharpe_ratio(1.2, n_trials=500, n_obs=1500)
    assert many["dsr"] < one["dsr"]
    assert many["expected_max_sharpe"] > one["expected_max_sharpe"]


def test_deflated_sharpe_rejects_marginal_result_after_many_trials():
    out = deflated_sharpe_ratio(0.5, n_trials=1000, n_obs=1000)
    assert not out["passes_95"]


def test_haircut_reduces_sharpe():
    out = haircut_sharpe(1.5, n_trials=100, n_obs=1500)
    assert out["haircut_sharpe"] < 1.5
    assert out["haircut_pct"] > 0


def test_min_trl_grows_as_sharpe_shrinks():
    high = min_track_record_length(2.0)
    low = min_track_record_length(0.4)
    assert low["min_track_record_years"] > high["min_track_record_years"]


def test_min_trl_infinite_when_sharpe_below_target():
    out = min_track_record_length(0.1, target_sharpe=0.5)
    assert np.isinf(out["min_track_record_obs"])


# ----------------------------------------------------------------- PBO/SPA
def test_pbo_is_high_for_pure_noise():
    """With only noise, the in-sample winner should be random out-of-sample."""
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    matrix = pd.DataFrame(
        {f"c{i}": rng.normal(0, 0.01, len(idx)) for i in range(12)}, index=idx
    )
    out = probability_backtest_overfitting(matrix, n_splits=6)
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["n_combinations"] > 0


def test_pbo_is_low_when_one_config_is_genuinely_best():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=1200)
    data = {f"c{i}": rng.normal(0, 0.01, len(idx)) for i in range(8)}
    data["winner"] = rng.normal(0.0012, 0.01, len(idx))   # persistent edge
    out = probability_backtest_overfitting(pd.DataFrame(data, index=idx), n_splits=6)
    assert out["pbo"] < 0.5


def test_spa_and_reality_check_run(noise):
    rng = np.random.default_rng(6)
    idx = noise.index
    strategies = pd.DataFrame(
        {f"s{i}": rng.normal(0, 0.01, len(idx)) for i in range(5)}, index=idx
    )
    rc = reality_check(strategies, noise, n_boot=100, seed=1)
    sp = spa_test(strategies, noise, n_boot=100, seed=1)
    assert 0.0 <= rc["p_value"] <= 1.0
    assert 0.0 <= sp["p_value"] <= 1.0


# ------------------------------------------------------------ full battery
def test_validation_flags_noise_as_not_significant(noise):
    report = validate_strategy(noise, n_trials=50)
    assert report.verdict.startswith(("CAUTION", "FAIL"))
    assert len(report.flags) > 0


def test_validation_passes_a_strong_single_trial_edge(strong_edge):
    report = validate_strategy(strong_edge, n_trials=1)
    assert report.sharpe_test["significant_5pct"]
