"""Standardized Unexpected Earnings (SUE) and earnings-day reaction.

    UE_t  = EPS_t - EPS_{t-4}                      (seasonal random walk)
    SUE_t = UE_t / std(UE over the last 8 quarters)

The seasonal differencing handles the fact that Indian quarters are strongly
seasonal, and scaling by the firm's own surprise volatility makes SUE
comparable across companies. Crucially this needs only reported EPS history -
no analyst-estimate feed, which would be expensive and patchy for Indian mid
and small caps.

Evidence: on Nifty 500 (2002-2017) positive- minus negative-surprise portfolios
earned roughly 6% over ~64 post-announcement days. But NSE large-caps show
little drift, so the liquidity/size filters in the config matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


@dataclass
class SUEObservation:
    isin: str
    announce_date: pd.Timestamp
    period_end: pd.Timestamp
    eps: float
    eps_year_ago: float
    unexpected_earnings: float
    sue: float
    n_quarters: int

    def to_dict(self) -> dict:
        return {
            "isin": self.isin, "announce_date": self.announce_date,
            "period_end": self.period_end, "eps": self.eps,
            "eps_year_ago": self.eps_year_ago,
            "unexpected_earnings": self.unexpected_earnings,
            "sue": self.sue, "n_quarters": self.n_quarters,
        }


def compute_sue_series(
    earnings_history: pd.DataFrame, lookback_quarters: int = 8, min_quarters: int = 6
) -> pd.DataFrame:
    """SUE for every reported quarter of one company."""
    if earnings_history.empty or len(earnings_history) < 5:
        return pd.DataFrame()

    df = earnings_history.sort_values("period_end").reset_index(drop=True).copy()
    df["eps_year_ago"] = df["eps"].shift(4)
    df["unexpected_earnings"] = df["eps"] - df["eps_year_ago"]

    # Standard deviation of past surprises, excluding the current one so the
    # scaling uses only prior information.
    df["ue_std"] = (
        df["unexpected_earnings"]
        .shift(1)
        .rolling(lookback_quarters, min_periods=min_quarters - 4)
        .std()
    )
    df["n_quarters"] = np.arange(1, len(df) + 1)
    df["sue"] = df["unexpected_earnings"] / df["ue_std"].replace(0, np.nan)
    df.loc[df["n_quarters"] < min_quarters, "sue"] = np.nan

    return df.replace([np.inf, -np.inf], np.nan)


def compute_sue_panel(
    earnings: pd.DataFrame, lookback_quarters: int = 8, min_quarters: int = 6
) -> pd.DataFrame:
    """SUE for every company and quarter (long format)."""
    if earnings.empty:
        return pd.DataFrame()

    frames = []
    for isin, grp in earnings.groupby("isin"):
        s = compute_sue_series(grp, lookback_quarters, min_quarters)
        if s.empty:
            continue
        s["isin"] = isin
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    cols = ["isin", "announce_date", "period_end", "fiscal_quarter", "eps",
            "eps_year_ago", "unexpected_earnings", "sue", "n_quarters"]
    return out[[c for c in cols if c in out.columns]].dropna(subset=["sue"])


def earnings_day_reaction(
    prices: pd.DataFrame,
    benchmark: pd.Series | None,
    isin: str,
    announce_date: pd.Timestamp,
    window: int = 1,
) -> dict[str, float]:
    """Abnormal return and volume around the announcement.

    A positive surprise that the market ignored is a weaker signal than one it
    immediately rewarded, so this is used as a confirmation gate.
    """
    if isin not in prices.columns:
        return {"abnormal_return": np.nan, "raw_return": np.nan}

    idx = prices.index
    pos = idx.searchsorted(pd.Timestamp(announce_date))
    if pos <= 0 or pos + window >= len(idx):
        return {"abnormal_return": np.nan, "raw_return": np.nan}

    start, end = idx[pos - 1], idx[min(pos + window, len(idx) - 1)]
    p0, p1 = prices[isin].get(start), prices[isin].get(end)
    if p0 is None or p1 is None or np.isnan(p0) or np.isnan(p1) or p0 <= 0:
        return {"abnormal_return": np.nan, "raw_return": np.nan}

    raw = float(p1 / p0 - 1.0)
    abnormal = raw
    if benchmark is not None and not benchmark.empty:
        b0, b1 = benchmark.get(start), benchmark.get(end)
        if b0 and b1 and not np.isnan(b0) and not np.isnan(b1) and b0 > 0:
            abnormal = raw - float(b1 / b0 - 1.0)

    return {"raw_return": raw, "abnormal_return": abnormal}


def rank_sue_cross_section(
    sue_panel: pd.DataFrame,
    dt: pd.Timestamp,
    universe: list[str],
    lookback_days: int = 90,
) -> pd.Series:
    """Most recent SUE per stock among companies that reported recently.

    Only announcements already published (and within the recency window) are
    considered, so nothing unpublished leaks in.
    """
    if sue_panel.empty:
        return pd.Series(dtype=float)

    ts = pd.Timestamp(dt)
    d = pd.to_datetime(sue_panel["announce_date"])
    recent = sue_panel[(d <= ts) & (d >= ts - pd.Timedelta(days=lookback_days))]
    if recent.empty:
        return pd.Series(dtype=float)

    recent = recent[recent["isin"].isin(universe)]
    if recent.empty:
        return pd.Series(dtype=float)

    latest = recent.sort_values("announce_date").groupby("isin").tail(1)
    return latest.set_index("isin")["sue"].sort_values(ascending=False)
