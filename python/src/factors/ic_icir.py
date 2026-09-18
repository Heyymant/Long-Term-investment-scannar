"""Information Coefficient diagnostics.

    IC_{f,t}  = Corr(score_{i,t}, forward_return_{i,t+1})
    ICIR_f    = mean(IC) / std(IC)

This answers the only question that matters about a signal: does it actually
predict forward returns? ICIR is the signal's own "Sharpe ratio", and it can
optionally drive dynamic factor weighting (w_f proportional to ICIR_f) -
though the Indian evidence says factor *timing* is unreliable, so that stays
off by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..config import get_logger

log = get_logger(__name__)


@dataclass
class ICResult:
    factor: str
    mean_ic: float
    std_ic: float
    icir: float
    t_stat: float
    p_value: float
    hit_rate: float          # fraction of periods with IC > 0
    n_periods: int
    ic_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "factor": self.factor,
            "mean_ic": round(self.mean_ic, 5),
            "std_ic": round(self.std_ic, 5),
            "icir": round(self.icir, 4),
            "t_stat": round(self.t_stat, 3),
            "p_value": round(self.p_value, 5),
            "hit_rate": round(self.hit_rate, 4),
            "n_periods": self.n_periods,
        }


def forward_returns(
    prices: pd.DataFrame, dates: pd.DatetimeIndex, horizon_days: int
) -> pd.DataFrame:
    """Return over the next `horizon_days`, observed at each date in `dates`."""
    out = {}
    idx = prices.index
    for dt in dates:
        pos = idx.searchsorted(dt)
        fwd = pos + horizon_days
        if pos >= len(idx) or fwd >= len(idx):
            continue
        out[dt] = (prices.iloc[fwd] / prices.iloc[pos]) - 1.0
    return pd.DataFrame(out).T if out else pd.DataFrame()


def compute_ic(
    scores: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
    min_names: int = 10,
) -> pd.Series:
    """Per-period cross-sectional correlation between score and forward return.

    Spearman (rank IC) by default - robust to outliers, which matters a lot
    with Indian small caps.
    """
    ics = {}
    common_dates = scores.index.intersection(fwd_returns.index)
    for dt in common_dates:
        s = scores.loc[dt]
        r = fwd_returns.loc[dt]
        df = pd.concat([s.rename("s"), r.rename("r")], axis=1).dropna()
        if len(df) < min_names:
            continue
        if method == "spearman":
            c, _ = stats.spearmanr(df["s"], df["r"])
        else:
            c, _ = stats.pearsonr(df["s"], df["r"])
        if not np.isnan(c):
            ics[dt] = c
    return pd.Series(ics).sort_index()


def summarize_ic(ic_series: pd.Series, factor: str = "factor") -> ICResult:
    """Mean/std/ICIR plus a HAC-corrected t-stat on the IC series."""
    ic = ic_series.dropna()
    if len(ic) < 3:
        return ICResult(factor, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, len(ic), ic)

    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    icir = mean / std if std > 0 else np.nan
    t_stat = _hac_tstat(ic.values)
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(ic) - 1))) if not np.isnan(t_stat) else np.nan

    return ICResult(
        factor=factor, mean_ic=mean, std_ic=std, icir=float(icir),
        t_stat=float(t_stat), p_value=p_value,
        hit_rate=float((ic > 0).mean()), n_periods=len(ic), ic_series=ic,
    )


def _hac_tstat(x: np.ndarray, max_lags: int | None = None) -> float:
    """Newey-West t-statistic for the mean of an autocorrelated series."""
    n = len(x)
    if n < 3:
        return np.nan
    mean = x.mean()
    resid = x - mean
    if max_lags is None:
        max_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    max_lags = max(0, min(max_lags, n - 2))

    gamma0 = float((resid @ resid) / n)
    var = gamma0
    for lag in range(1, max_lags + 1):
        cov = float((resid[lag:] @ resid[:-lag]) / n)
        weight = 1.0 - lag / (max_lags + 1)
        var += 2 * weight * cov
    if var <= 0:
        return np.nan
    se = np.sqrt(var / n)
    return float(mean / se) if se > 0 else np.nan


def analyze_factors(
    factor_panels: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    horizon_days: int = 21,
    method: str = "spearman",
) -> pd.DataFrame:
    """IC/ICIR table for every factor."""
    results = []
    for name, panel in factor_panels.items():
        if panel is None or panel.empty:
            continue
        fwd = forward_returns(prices, panel.index, horizon_days)
        if fwd.empty:
            continue
        ic = compute_ic(panel, fwd, method=method)
        results.append(summarize_ic(ic, name).to_dict())
    return pd.DataFrame(results)


def icir_weights(
    ic_results: dict[str, ICResult],
    floor: float = 0.0,
    default_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Dynamic factor weights proportional to ICIR.

    Negative-ICIR factors are floored to zero rather than shorted. If nothing
    has positive ICIR we fall back to the static weights - a signal that
    dynamic weighting should not be trusted in this window.
    """
    raw = {k: max(floor, v.icir) for k, v in ic_results.items() if not np.isnan(v.icir)}
    total = sum(raw.values())
    if total <= 0:
        log.warning("No factor has positive ICIR; falling back to static weights")
        return default_weights or {}
    return {k: v / total for k, v in raw.items()}
