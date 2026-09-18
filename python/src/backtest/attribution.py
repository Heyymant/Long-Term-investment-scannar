"""True-alpha attribution.

    Rp - Rf = alpha + bM(MKT-Rf) + bS*SMB + bV*HML + bMOM*WML + bQ*QMJ + e

Beating the Nifty by 5% is not 5% of alpha. If the portfolio is tilted to
momentum and small caps, most of that gap is *factor exposure* you could have
bought cheaply. Alpha is the regression intercept - what is left after paying
for the factor bets.

Standard errors are Newey-West (HAC) because portfolio returns are
autocorrelated and heteroskedastic; OLS standard errors would overstate
significance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


@dataclass
class AttributionResult:
    alpha: float                    # per period (monthly if monthly data)
    alpha_annualized: float
    alpha_tstat: float
    alpha_pvalue: float
    betas: dict[str, float] = field(default_factory=dict)
    beta_tstats: dict[str, float] = field(default_factory=dict)
    beta_pvalues: dict[str, float] = field(default_factory=dict)
    r_squared: float = np.nan
    adj_r_squared: float = np.nan
    n_obs: int = 0
    frequency: str = "monthly"
    factors_used: list[str] = field(default_factory=list)
    hac_lags: int = 0
    note: str = ""

    def significant_at(self, level: float = 0.05) -> bool:
        if self.alpha_pvalue is None or np.isnan(self.alpha_pvalue):
            return False
        return bool(self.alpha_pvalue < level)

    @property
    def is_significant(self) -> bool:
        return self.significant_at(0.05)

    def to_dict(self) -> dict:
        return {
            "alpha_per_period": _r(self.alpha, 6),
            "alpha_annualized": _r(self.alpha_annualized, 5),
            "alpha_tstat": _r(self.alpha_tstat, 3),
            "alpha_pvalue": _r(self.alpha_pvalue, 5),
            "alpha_significant_5pct": self.is_significant,
            "betas": {k: _r(v, 4) for k, v in self.betas.items()},
            "beta_tstats": {k: _r(v, 3) for k, v in self.beta_tstats.items()},
            "beta_pvalues": {k: _r(v, 5) for k, v in self.beta_pvalues.items()},
            "r_squared": _r(self.r_squared, 4),
            "adj_r_squared": _r(self.adj_r_squared, 4),
            "n_obs": self.n_obs,
            "frequency": self.frequency,
            "factors_used": self.factors_used,
            "hac_lags": self.hac_lags,
            "note": self.note,
        }


def _r(v: float, nd: int) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), nd)


def run_attribution(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    frequency: str = "monthly",
    risk_free: pd.Series | None = None,
    hac_lags: int | None = None,
) -> AttributionResult:
    """Regress excess portfolio returns on the IIMA factor set.

    `factors` should contain MKT (already excess of Rf), SMB, HML, WML and
    optionally QMJ, plus RF if no separate risk-free series is given.
    """
    periods_per_year = 12 if frequency.startswith("month") else 252

    if factors is None or factors.empty:
        return _market_only_fallback(portfolio_returns, periods_per_year, frequency)

    df = pd.DataFrame({"port": portfolio_returns}).join(factors, how="inner").dropna()
    if len(df) < 12:
        return AttributionResult(
            alpha=np.nan, alpha_annualized=np.nan, alpha_tstat=np.nan, alpha_pvalue=np.nan,
            n_obs=len(df), frequency=frequency,
            note="Not enough overlapping observations for attribution (need >= 12)",
        )

    rf = df["RF"] if "RF" in df.columns else (
        risk_free.reindex(df.index).fillna(0.0) if risk_free is not None
        else pd.Series(0.0, index=df.index)
    )
    y = (df["port"] - rf).values

    factor_names = [c for c in ("MKT", "SMB", "HML", "WML", "QMJ") if c in df.columns]
    if not factor_names:
        return _market_only_fallback(portfolio_returns, periods_per_year, frequency)

    X = df[factor_names].values
    X = np.column_stack([np.ones(len(X)), X])

    try:
        import statsmodels.api as sm

        if hac_lags is None:
            hac_lags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
        hac_lags = max(0, min(hac_lags, len(y) - 2))

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        params, tvals, pvals = model.params, model.tvalues, model.pvalues
        r2, adj_r2 = float(model.rsquared), float(model.rsquared_adj)
    except ImportError:
        params, tvals, pvals, r2, adj_r2 = _ols_hac(y, X, hac_lags or 0)
        hac_lags = hac_lags or 0

    alpha = float(params[0])
    return AttributionResult(
        alpha=alpha,
        alpha_annualized=float((1 + alpha) ** periods_per_year - 1),
        alpha_tstat=float(tvals[0]),
        alpha_pvalue=float(pvals[0]),
        betas={n: float(params[i + 1]) for i, n in enumerate(factor_names)},
        beta_tstats={n: float(tvals[i + 1]) for i, n in enumerate(factor_names)},
        beta_pvalues={n: float(pvals[i + 1]) for i, n in enumerate(factor_names)},
        r_squared=r2, adj_r_squared=adj_r2,
        n_obs=len(y), frequency=frequency,
        factors_used=factor_names, hac_lags=int(hac_lags),
    )


def _market_only_fallback(
    portfolio_returns: pd.Series, periods_per_year: int, frequency: str
) -> AttributionResult:
    """Without the factor library we can still report CAPM-style alpha."""
    return AttributionResult(
        alpha=np.nan, alpha_annualized=np.nan, alpha_tstat=np.nan, alpha_pvalue=np.nan,
        n_obs=len(portfolio_returns), frequency=frequency,
        note=("IIMA factor library not available - true-alpha attribution skipped. "
              "Download the factor CSV to data/ to enable it."),
    )


def capm_alpha_beta(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series,
    risk_free: pd.Series | None = None, periods_per_year: int = 252,
) -> dict[str, float]:
    """Single-factor alpha/beta against a benchmark (reported alongside)."""
    df = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
    if len(df) < 10:
        return {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan, "n_obs": len(df)}

    rf = risk_free.reindex(df.index).fillna(0.0) if risk_free is not None else 0.0
    y, x = (df["p"] - rf).values, (df["b"] - rf).values

    var = x.var()
    if var == 0:
        return {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan, "n_obs": len(df)}

    beta = float(np.cov(x, y, ddof=0)[0, 1] / var)
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan

    return {
        "alpha_per_period": alpha,
        "alpha_annualized": float((1 + alpha) ** periods_per_year - 1),
        "beta": beta,
        "r_squared": r2,
        "n_obs": len(df),
    }


def _ols_hac(
    y: np.ndarray, X: np.ndarray, max_lags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Minimal OLS + Newey-West fallback when statsmodels is unavailable."""
    from scipy import stats as sps

    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for lag in range(1, max_lags + 1):
        w = 1.0 - lag / (max_lags + 1)
        Xl, rl = X[lag:], resid[lag:]
        Xr, rr = X[:-lag], resid[:-lag]
        G = (Xl * rl[:, None]).T @ (Xr * rr[:, None])
        S += w * (G + G.T)

    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / se
    pvals = 2 * (1 - sps.t.cdf(np.abs(tvals), df=max(n - k, 1)))

    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / max(n - k, 1) if not np.isnan(r2) else np.nan
    return beta, tvals, pvals, r2, adj
