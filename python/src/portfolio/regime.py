"""Trend / absolute-momentum regime overlay.

This is the main drawdown lever: Indian momentum lost ~70% in 2008 and took 65
months to recover, and a trend gate is the cheapest way to avoid the worst of
that.

A naive "below the 200-DMA => go to cash" rule whipsaws in choppy markets, so
we support:
  * hysteresis bands - exit below (1-band)*MA, re-enter only above (1+band)*MA
  * graded exposure  - scale smoothly with distance from the MA instead of 0/1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import RegimeConfig


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(20, window // 4)).mean()


def compute_exposure_series(
    benchmark: pd.Series, config: RegimeConfig
) -> pd.Series:
    """Target gross equity exposure over time, in [min_exposure, max_exposure]."""
    if not config.enabled or benchmark.empty:
        return pd.Series(config.max_exposure, index=benchmark.index)

    ma = moving_average(benchmark, config.ma_days)
    ratio = benchmark / ma

    if config.graded:
        exposure = _graded_exposure(ratio, config)
    else:
        exposure = _hysteresis_exposure(ratio, config)

    exposure = exposure.clip(config.min_exposure, config.max_exposure)
    # Before the MA warms up, stay invested rather than sitting in cash.
    return exposure.where(ma.notna(), config.max_exposure)


def _hysteresis_exposure(ratio: pd.Series, config: RegimeConfig) -> pd.Series:
    """Binary in/out with separate entry and exit thresholds."""
    lower = 1.0 - config.hysteresis_band
    upper = 1.0 + config.hysteresis_band

    out = np.full(len(ratio), np.nan)
    state = config.max_exposure  # start invested
    values = ratio.values
    for i, r in enumerate(values):
        if np.isnan(r):
            out[i] = state
            continue
        if state > config.min_exposure and r < lower:
            state = config.min_exposure      # risk-off
        elif state <= config.min_exposure and r > upper:
            state = config.max_exposure      # risk-on
        out[i] = state
    return pd.Series(out, index=ratio.index)


def _graded_exposure(ratio: pd.Series, config: RegimeConfig) -> pd.Series:
    """Smoothly scale exposure across the hysteresis band.

    Fully invested at/above (1+band)*MA, fully defensive at/below (1-band)*MA,
    linear in between. Avoids all-or-nothing switching.
    """
    band = max(config.hysteresis_band, 1e-6)
    lower, upper = 1.0 - band, 1.0 + band
    scaled = (ratio - lower) / (upper - lower)
    scaled = scaled.clip(0.0, 1.0)
    return config.min_exposure + scaled * (config.max_exposure - config.min_exposure)


def exposure_on(exposure: pd.Series, dt: pd.Timestamp, default: float = 1.0) -> float:
    """Exposure as known on `dt` (last value at or before the date)."""
    if exposure.empty:
        return default
    prior = exposure.loc[exposure.index <= pd.Timestamp(dt)]
    return default if prior.empty else float(prior.iloc[-1])


def regime_label(
    benchmark: pd.Series, ma_days: int = 200, vol_window: int = 63
) -> pd.Series:
    """Classify bull / bear / sideways - used for per-regime reporting."""
    ma = moving_average(benchmark, ma_days)
    rets = benchmark.pct_change()
    vol = rets.rolling(vol_window, min_periods=20).std() * np.sqrt(252)
    trend = benchmark / ma - 1.0
    median_vol = vol.median()

    labels = pd.Series("sideways", index=benchmark.index, dtype=object)
    labels[trend > 0.05] = "bull"
    labels[trend < -0.05] = "bear"
    # High volatility below trend is the genuinely dangerous state.
    labels[(trend < -0.02) & (vol > median_vol * 1.3)] = "bear"
    return labels.where(ma.notna(), "unknown")


def overlay_diagnostics(
    benchmark: pd.Series, exposure: pd.Series
) -> dict[str, float]:
    """Did the overlay actually earn its keep?

    Compares buy-and-hold against the overlay applied with a one-day lag
    (you can only act on yesterday's close).
    """
    rets = benchmark.pct_change().fillna(0.0)
    applied = exposure.shift(1).reindex(rets.index).fillna(1.0)
    overlay_rets = rets * applied

    def _stats(r: pd.Series) -> tuple[float, float, float]:
        ann = float((1 + r).prod() ** (252 / max(len(r), 1)) - 1)
        vol = float(r.std() * np.sqrt(252))
        curve = (1 + r).cumprod()
        dd = float((curve / curve.cummax() - 1).min())
        return ann, vol, dd

    bh_ret, bh_vol, bh_dd = _stats(rets)
    ov_ret, ov_vol, ov_dd = _stats(overlay_rets)
    switches = int((applied.diff().abs() > 1e-9).sum())

    return {
        "buy_hold_cagr": bh_ret, "overlay_cagr": ov_ret,
        "buy_hold_vol": bh_vol, "overlay_vol": ov_vol,
        "buy_hold_max_dd": bh_dd, "overlay_max_dd": ov_dd,
        "buy_hold_sharpe": bh_ret / bh_vol if bh_vol else np.nan,
        "overlay_sharpe": ov_ret / ov_vol if ov_vol else np.nan,
        "n_switches": switches,
        "avg_exposure": float(applied.mean()),
    }
