"""Performance metrics catalog.

Return, risk, risk-adjusted, trading and rolling statistics, plus per-year and
per-regime breakdowns. Deliberately verbose: a single headline CAGR hides the
drawdown and turnover that decide whether a strategy is actually livable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if returns.empty:
        return np.nan
    total = float((1 + returns).prod())
    years = len(returns) / periods_per_year
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else np.nan


def downside_deviation(
    returns: pd.Series, mar: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    downside = returns[returns < mar]
    if len(downside) < 2:
        return np.nan
    return float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free: pd.Series | float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    excess = _excess(returns, risk_free, periods_per_year)
    sd = excess.std(ddof=1)
    if sd is None or sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free: pd.Series | float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    excess = _excess(returns, risk_free, periods_per_year)
    dd = downside_deviation(excess, 0.0, periods_per_year)
    if dd is None or dd == 0 or np.isnan(dd):
        return np.nan
    return float(excess.mean() * periods_per_year / dd)


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)["max_drawdown"]
    c = cagr(returns, periods_per_year)
    if mdd is None or mdd == 0 or np.isnan(mdd) or np.isnan(c):
        return np.nan
    return float(c / abs(mdd))


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    gains = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns <= threshold]).sum()
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def information_ratio(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    df = pd.concat([returns, benchmark], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = active.std(ddof=1)
    if te == 0 or np.isnan(te):
        return np.nan
    return float(active.mean() / te * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> dict[str, Any]:
    """Max drawdown with its duration and recovery time."""
    if returns.empty:
        return {"max_drawdown": np.nan, "peak_date": None, "trough_date": None,
                "recovery_date": None, "drawdown_days": np.nan, "recovery_days": np.nan}

    curve = (1 + returns).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    trough = dd.idxmin()
    mdd = float(dd.min())

    before = curve.loc[:trough]
    peak_date = before.idxmax() if len(before) else None

    after = curve.loc[trough:]
    peak_value = curve.loc[peak_date] if peak_date is not None else np.nan
    recovered = after[after >= peak_value]
    recovery_date = recovered.index[0] if len(recovered) else None

    return {
        "max_drawdown": mdd,
        "peak_date": peak_date,
        "trough_date": trough,
        "recovery_date": recovery_date,
        "drawdown_days": (trough - peak_date).days if peak_date is not None else np.nan,
        "recovery_days": (recovery_date - trough).days if recovery_date is not None else np.nan,
    }


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = (1 + returns).cumprod()
    return curve / curve.cummax() - 1.0


def ulcer_index(returns: pd.Series) -> float:
    """RMS drawdown - penalizes deep AND long drawdowns."""
    dd = drawdown_series(returns)
    return float(np.sqrt((dd**2).mean())) if len(dd) else np.nan


def value_at_risk(returns: pd.Series, level: float = 0.05, method: str = "historical") -> float:
    if len(returns) < 20:
        return np.nan
    if method == "historical":
        return float(np.percentile(returns, level * 100))
    if method == "cornish_fisher":
        from scipy import stats as sps

        z = sps.norm.ppf(level)
        s, k = float(returns.skew()), float(returns.kurtosis())
        z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24
                - (2 * z**3 - 5 * z) * s**2 / 36)
        return float(returns.mean() + z_cf * returns.std())
    raise ValueError(f"unknown VaR method '{method}'")


def conditional_var(returns: pd.Series, level: float = 0.05) -> float:
    """Expected shortfall - average loss in the worst `level` tail."""
    if len(returns) < 20:
        return np.nan
    var = value_at_risk(returns, level)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else np.nan


def _excess(
    returns: pd.Series, risk_free: pd.Series | float, periods_per_year: int
) -> pd.Series:
    if isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(returns.index).fillna(0.0)
        rf_period = (1 + rf) ** (1 / periods_per_year) - 1
    else:
        rf_period = (1 + float(risk_free)) ** (1 / periods_per_year) - 1
    return returns - rf_period


# --------------------------------------------------------------------------- #
# Breakdowns
# --------------------------------------------------------------------------- #
def yearly_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return (1 + returns).resample("YE").prod() - 1


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Classic year x month grid."""
    if returns.empty:
        return pd.DataFrame()
    monthly = (1 + returns).resample("ME").prod() - 1
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    table = df.pivot_table(index="year", columns="month", values="ret")
    table.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in table.columns]
    table["Year"] = yearly_returns(returns).values[: len(table)]
    return table


def rolling_metrics(
    returns: pd.Series, window: int = 252, benchmark: pd.Series | None = None
) -> pd.DataFrame:
    """Rolling Sharpe / vol / beta."""
    if len(returns) < window:
        return pd.DataFrame()

    out = pd.DataFrame(index=returns.index)
    out["rolling_vol"] = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    out["rolling_sharpe"] = (
        returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    )
    out["rolling_return"] = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1

    if benchmark is not None:
        b = benchmark.reindex(returns.index)
        cov = returns.rolling(window).cov(b)
        var = b.rolling(window).var()
        out["rolling_beta"] = cov / var.replace(0, np.nan)

    return out.dropna(how="all")


def regime_breakdown(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Performance split by market regime (bull/bear/sideways)."""
    df = pd.concat([returns.rename("ret"), regimes.rename("regime")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for regime, grp in df.groupby("regime"):
        r = grp["ret"]
        rows.append({
            "regime": regime,
            "n_days": len(r),
            "pct_of_time": len(r) / len(df),
            "total_return": float((1 + r).prod() - 1),
            "annualized": cagr(r),
            "volatility": annualized_volatility(r),
            "sharpe": sharpe_ratio(r),
            "max_drawdown": max_drawdown(r)["max_drawdown"],
            "hit_rate": float((r > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("n_days", ascending=False)


# --------------------------------------------------------------------------- #
# Full catalog
# --------------------------------------------------------------------------- #
@dataclass
class PerformanceMetrics:
    # Return
    total_return: float = np.nan
    cagr: float = np.nan
    best_year: float = np.nan
    worst_year: float = np.nan
    # Risk
    volatility: float = np.nan
    downside_deviation: float = np.nan
    max_drawdown: float = np.nan
    drawdown_days: float = np.nan
    recovery_days: float = np.nan
    ulcer_index: float = np.nan
    var_95: float = np.nan
    cvar_95: float = np.nan
    skew: float = np.nan
    kurtosis: float = np.nan
    # Risk-adjusted
    sharpe: float = np.nan
    sortino: float = np.nan
    calmar: float = np.nan
    omega: float = np.nan
    information_ratio: float = np.nan
    # vs benchmark
    benchmark_cagr: float = np.nan
    excess_return: float = np.nan
    beta: float = np.nan
    tracking_error: float = np.nan
    # Trading
    turnover: float = np.nan
    avg_holding_days: float = np.nan
    hit_rate: float = np.nan
    n_periods: int = 0
    avg_exposure: float = np.nan
    # Tax
    total_tax: float = np.nan
    after_tax_cagr: float = np.nan
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free: pd.Series | float = 0.0,
    turnover: float | None = None,
    after_tax_returns: pd.Series | None = None,
    total_tax: float | None = None,
    exposure: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceMetrics:
    """Compute the full catalog for one return stream."""
    r = returns.dropna()
    if r.empty:
        return PerformanceMetrics()

    dd = max_drawdown(r)
    yearly = yearly_returns(r)

    m = PerformanceMetrics(
        total_return=float((1 + r).prod() - 1),
        cagr=cagr(r, periods_per_year),
        best_year=float(yearly.max()) if len(yearly) else np.nan,
        worst_year=float(yearly.min()) if len(yearly) else np.nan,
        volatility=annualized_volatility(r, periods_per_year),
        downside_deviation=downside_deviation(r, 0.0, periods_per_year),
        max_drawdown=dd["max_drawdown"],
        drawdown_days=dd["drawdown_days"],
        recovery_days=dd["recovery_days"],
        ulcer_index=ulcer_index(r),
        var_95=value_at_risk(r, 0.05),
        cvar_95=conditional_var(r, 0.05),
        skew=float(r.skew()),
        kurtosis=float(r.kurtosis()),
        sharpe=sharpe_ratio(r, risk_free, periods_per_year),
        sortino=sortino_ratio(r, risk_free, periods_per_year),
        calmar=calmar_ratio(r, periods_per_year),
        omega=omega_ratio(r),
        hit_rate=float((r > 0).mean()),
        n_periods=len(r),
    )

    if benchmark_returns is not None:
        b = benchmark_returns.reindex(r.index).dropna()
        if len(b) > 10:
            common = r.index.intersection(b.index)
            rr, bb = r.loc[common], b.loc[common]
            m.benchmark_cagr = cagr(bb, periods_per_year)
            m.excess_return = m.cagr - m.benchmark_cagr if not np.isnan(m.cagr) else np.nan
            m.information_ratio = information_ratio(rr, bb, periods_per_year)
            m.tracking_error = float((rr - bb).std(ddof=1) * np.sqrt(periods_per_year))
            var = bb.var()
            m.beta = float(np.cov(rr, bb, ddof=0)[0, 1] / var) if var > 0 else np.nan

    if turnover is not None:
        m.turnover = float(turnover)
        if turnover > 0:
            m.avg_holding_days = float(periods_per_year / turnover)

    if exposure is not None and not exposure.empty:
        m.avg_exposure = float(exposure.mean())

    if total_tax is not None:
        m.total_tax = float(total_tax)
    if after_tax_returns is not None and not after_tax_returns.empty:
        m.after_tax_cagr = cagr(after_tax_returns.dropna(), periods_per_year)

    return m
