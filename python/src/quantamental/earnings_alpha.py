"""Optional quantamental layer: predict earnings change, confirm with price.

The research compilation frames this as going beyond pure factor investing:

  Layer 1  statistical alpha  (Quality + Value + Momentum + LowVol)
  Layer 2  earnings alpha     (predict dEPS from operating drivers)
  Layer 3  fundamental quality
  Layer 4  price confirmation (only act when momentum > 0)

This is deliberately last in the build order and off by default. It adds
parameters and a fitted model, which is exactly where overfitting creeps in,
so it must earn its place through walk-forward evidence rather than a
flattering in-sample fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

DRIVER_COLUMNS = [
    "revenue_growth", "revenue_acceleration", "margin_change",
    "operating_leverage", "working_capital_change", "asset_turnover_change",
]


@dataclass
class EarningsAlphaModel:
    """Ridge-style linear model predicting next-quarter EPS growth."""

    coefficients: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    intercept: float = 0.0
    r_squared: float = np.nan
    n_train: int = 0
    features: list[str] = field(default_factory=list)
    fitted: bool = False

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.fitted or X.empty:
            return pd.Series(dtype=float)
        cols = [c for c in self.features if c in X.columns]
        if not cols:
            return pd.Series(dtype=float)
        vals = X[cols].fillna(0.0)
        return vals.mul(self.coefficients.reindex(cols).fillna(0.0), axis=1).sum(axis=1) + self.intercept


def build_drivers(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Derive the operating drivers from a fundamentals history."""
    if fundamentals.empty:
        return pd.DataFrame()

    df = fundamentals.sort_values(["isin", "period_end"]).copy()
    g = df.groupby("isin")

    if "revenue" in df.columns:
        df["revenue_growth"] = g["revenue"].pct_change(4)
        df["revenue_acceleration"] = g["revenue_growth"].diff()
    if {"gross_profitability"}.issubset(df.columns):
        df["margin_change"] = g["gross_profitability"].diff()
    if {"revenue", "net_income"}.issubset(df.columns):
        rev_g = g["revenue"].pct_change()
        ni_g = g["net_income"].pct_change()
        # Operating leverage: profit responsiveness to sales.
        df["operating_leverage"] = (ni_g / rev_g.replace(0, np.nan)).clip(-10, 10)
    if "total_assets" in df.columns and "revenue" in df.columns:
        turnover = df["revenue"] / df["total_assets"].replace(0, np.nan)
        df["asset_turnover_change"] = turnover.groupby(df["isin"]).diff()
    if "cfo_to_pat" in df.columns:
        # Falling cash conversion is a classic warning sign.
        df["working_capital_change"] = -g["cfo_to_pat"].diff()

    for c in DRIVER_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    return df.replace([np.inf, -np.inf], np.nan)


def fit_earnings_model(
    drivers: pd.DataFrame, target_col: str = "eps_growth_next",
    ridge_alpha: float = 1.0, min_samples: int = 200,
) -> EarningsAlphaModel:
    """Fit a regularized linear model on the operating drivers.

    Ridge (not OLS) because the drivers are collinear; regularization keeps
    coefficients stable instead of exploding on correlated inputs.
    """
    if drivers.empty or target_col not in drivers.columns:
        return EarningsAlphaModel()

    cols = [c for c in DRIVER_COLUMNS if c in drivers.columns]
    data = drivers[[*cols, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < min_samples:
        log.info("Only %d samples for the earnings model (need %d) - skipping",
                 len(data), min_samples)
        return EarningsAlphaModel()

    X = data[cols].values
    y = data[target_col].values

    # Standardize so the ridge penalty is applied evenly.
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    n_features = Xs.shape[1]
    A = Xs.T @ Xs + ridge_alpha * np.eye(n_features)
    try:
        beta = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    except np.linalg.LinAlgError:
        return EarningsAlphaModel()

    pred = Xs @ beta + y.mean()
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())

    return EarningsAlphaModel(
        coefficients=pd.Series(beta / sigma, index=cols),
        intercept=float(y.mean() - (beta / sigma * mu).sum()),
        r_squared=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        n_train=len(data),
        features=cols,
        fitted=True,
    )


def prepare_training_data(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach the forward EPS-growth target to the driver table."""
    drivers = build_drivers(fundamentals)
    if drivers.empty or "eps" not in drivers.columns:
        return pd.DataFrame()
    g = drivers.groupby("isin")
    drivers["eps_next"] = g["eps"].shift(-1)
    drivers["eps_growth_next"] = (
        (drivers["eps_next"] - drivers["eps"]) / drivers["eps"].abs().replace(0, np.nan)
    ).clip(-3, 3)
    return drivers


def score_earnings_alpha(
    model: EarningsAlphaModel,
    drivers: pd.DataFrame,
    dt: pd.Timestamp,
    universe: list[str],
    momentum_scores: pd.Series | None = None,
    require_positive_momentum: bool = True,
) -> pd.Series:
    """Predicted EPS growth, gated by price confirmation (Layer 4).

    The gate matters: a fundamentally attractive name the market is still
    selling is a thesis, not a signal. Requiring positive momentum avoids
    catching knives.
    """
    if not model.fitted or drivers.empty:
        return pd.Series(dtype=float)

    visible = drivers[pd.to_datetime(drivers["announce_date"]) <= pd.Timestamp(dt)]
    if visible.empty:
        return pd.Series(dtype=float)

    latest = visible.sort_values("announce_date").groupby("isin").tail(1).set_index("isin")
    latest = latest.reindex([i for i in universe if i in latest.index])
    if latest.empty:
        return pd.Series(dtype=float)

    pred = model.predict(latest)
    if pred.empty:
        return pred

    from ..factors.base import zscore

    scores = zscore(pred)
    if require_positive_momentum and momentum_scores is not None and not momentum_scores.empty:
        mom = momentum_scores.reindex(scores.index)
        scores = scores.where(mom > 0)

    return scores.dropna()
