"""Spread construction, hedge ratios and Ornstein-Uhlenbeck dynamics.

For a pair (y, x) we estimate a hedge ratio b and form the spread
s_t = y_t - b*x_t. If s is stationary it mean-reverts, and the OU model gives
us the speed of that reversion (the half-life), which tells us how long a
trade should take and therefore whether it can survive costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SpreadModel:
    hedge_ratio: float
    intercept: float
    spread: pd.Series
    half_life: float
    method: str = "ols"
    hedge_ratio_series: pd.Series | None = None   # Kalman only

    @property
    def is_tradeable(self) -> bool:
        return bool(np.isfinite(self.half_life) and 0 < self.half_life < 252)


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Static hedge ratio via ordinary least squares."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 20:
        return np.nan, np.nan
    xv, yv = df["x"].values, df["y"].values
    var = xv.var()
    if var == 0:
        return np.nan, np.nan
    beta = float(np.cov(xv, yv, ddof=0)[0, 1] / var)
    alpha = float(yv.mean() - beta * xv.mean())
    return beta, alpha


def total_least_squares_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Orthogonal regression - treats both legs as noisy.

    OLS assumes x is measured without error, which is untrue for two traded
    prices, and biases the hedge ratio toward zero.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 20:
        return np.nan, np.nan
    data = df[["x", "y"]].values
    centered = data - data.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    vx, vy = vt[-1]
    if vy == 0:
        return np.nan, np.nan
    beta = float(-vx / vy)
    alpha = float(data[:, 1].mean() - beta * data[:, 0].mean())
    return beta, alpha


def kalman_hedge_ratio(
    y: pd.Series, x: pd.Series,
    delta: float = 1e-4, observation_var: float = 1e-3,
) -> tuple[pd.Series, pd.Series]:
    """Time-varying hedge ratio via a hand-rolled Kalman filter.

    State = [beta, alpha], random-walk transition. `delta` controls how fast
    the hedge ratio may drift: too high and it chases noise, too low and it
    lags structural breaks. Not automatically better than static OLS - which
    is why the config lets you choose.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(df)
    if n < 20:
        empty = pd.Series(dtype=float)
        return empty, empty

    trans_cov = delta / (1 - delta) * np.eye(2)
    state = np.zeros(2)
    state_cov = np.ones((2, 2))

    betas = np.zeros(n)
    spreads = np.zeros(n)

    for i, (_, row) in enumerate(df.iterrows()):
        obs = np.array([row["x"], 1.0])

        # Predict
        state_cov = state_cov + trans_cov

        # Update
        pred = float(obs @ state)
        resid = float(row["y"]) - pred
        innov_var = float(obs @ state_cov @ obs) + observation_var
        if innov_var <= 0:
            innov_var = observation_var
        gain = (state_cov @ obs) / innov_var
        state = state + gain * resid
        state_cov = state_cov - np.outer(gain, obs) @ state_cov

        betas[i] = state[0]
        spreads[i] = resid

    return pd.Series(betas, index=df.index), pd.Series(spreads, index=df.index)


def ou_half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion from the OU/AR(1) discretization.

        ds_t = theta*(mu - s_t)*dt + sigma*dW  ->  half_life = ln(2)/theta

    A half-life of 2 days means a spread typically closes half its gap in two
    days; 200 days means capital is tied up far too long to be worth trading.
    """
    s = spread.dropna()
    if len(s) < 20:
        return np.nan

    lagged = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    idx = lagged.index.intersection(delta.index)
    if len(idx) < 20:
        return np.nan

    xv = lagged.loc[idx].values
    yv = delta.loc[idx].values
    var = xv.var()
    if var == 0:
        return np.nan

    theta = float(np.cov(xv, yv, ddof=0)[0, 1] / var)
    if theta >= 0:          # not mean-reverting (explosive or random walk)
        return np.inf
    return float(-np.log(2) / theta)


def build_spread(
    y: pd.Series, x: pd.Series, method: str = "ols",
    kalman_delta: float = 1e-4,
) -> SpreadModel:
    """Estimate the hedge ratio and return the spread plus its half-life."""
    if method == "kalman":
        betas, spread = kalman_hedge_ratio(y, x, kalman_delta)
        if spread.empty:
            return SpreadModel(np.nan, np.nan, pd.Series(dtype=float), np.nan, "kalman")
        return SpreadModel(
            hedge_ratio=float(betas.iloc[-1]), intercept=np.nan, spread=spread,
            half_life=ou_half_life(spread), method="kalman", hedge_ratio_series=betas,
        )

    beta, alpha = (
        total_least_squares_hedge_ratio(y, x) if method == "tls" else ols_hedge_ratio(y, x)
    )
    if np.isnan(beta):
        return SpreadModel(np.nan, np.nan, pd.Series(dtype=float), np.nan, method)

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    spread = df["y"] - beta * df["x"] - (alpha if np.isfinite(alpha) else 0.0)
    return SpreadModel(beta, alpha, spread, ou_half_life(spread), method)


def rolling_zscore(spread: pd.Series, window: int, shift: int = 1) -> pd.Series:
    """Rolling z-score of the spread.

    `shift=1` is mandatory for honesty: the mean and standard deviation used
    on day t must come from data up to t-1, otherwise today's observation
    helps define its own z-score.
    """
    mean = spread.rolling(window, min_periods=max(5, window // 3)).mean().shift(shift)
    std = spread.rolling(window, min_periods=max(5, window // 3)).std().shift(shift)
    return (spread - mean) / std.replace(0, np.nan)


def zscore_window_from_half_life(
    half_life: float, multiple: float = 4.0, lo: int = 5, hi: int = 252
) -> int:
    """Pick a z-score window scaled to the spread's own reversion speed."""
    if not np.isfinite(half_life) or half_life <= 0:
        return 60
    return int(np.clip(round(half_life * multiple), lo, hi))
