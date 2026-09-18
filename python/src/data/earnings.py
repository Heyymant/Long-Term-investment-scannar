"""Quarterly results calendar and EPS history (Sleeve C input).

Two things decide whether a PEAD study is honest:

1. **The announcement timestamp.** A result published after the close on
   day t is only tradeable from t+1. Treating it as available on t manufactures
   a day of free drift.
2. **Split-adjusted, consistently-consolidated EPS.** A 1:5 split makes EPS
   fall 80% and would register as a catastrophic earnings miss. Comparing
   consolidated to standalone figures across quarters does the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig, get_logger

log = get_logger(__name__)

EARNINGS_COLUMNS = [
    "isin", "announce_date", "announce_time", "period_end",
    "fiscal_quarter", "eps", "revenue", "net_income", "basis",
]


@dataclass
class EarningsCalendar:
    """Results history keyed by ISIN and announcement date."""

    data: pd.DataFrame

    @classmethod
    def empty(cls) -> EarningsCalendar:
        return cls(pd.DataFrame(columns=EARNINGS_COLUMNS))

    @classmethod
    def load(cls, data_dir: str | Path) -> EarningsCalendar:
        p = Path(data_dir) / "earnings.parquet"
        if not p.exists():
            return cls.empty()
        return cls(pd.read_parquet(p))

    def save(self, data_dir: str | Path) -> None:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self.data.to_parquet(Path(data_dir) / "earnings.parquet", index=False)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, split_adjust: pd.DataFrame | None = None) -> EarningsCalendar:
        out = df.copy()
        for col in EARNINGS_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        out["announce_date"] = pd.to_datetime(out["announce_date"], errors="coerce")
        out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
        out = out.dropna(subset=["isin", "announce_date", "eps"])
        out = out.sort_values(["isin", "period_end"]).drop_duplicates(
            subset=["isin", "period_end"], keep="last"
        )
        if split_adjust is not None and not split_adjust.empty:
            out = _apply_split_adjustment(out, split_adjust)
        return cls(out[EARNINGS_COLUMNS].reset_index(drop=True))

    # -- access ----------------------------------------------------------------
    def visible_on(self, dt: pd.Timestamp) -> pd.DataFrame:
        """Results already published as of `dt`."""
        if self.data.empty:
            return self.data
        return self.data[pd.to_datetime(self.data["announce_date"]) <= pd.Timestamp(dt)]

    def announced_between(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if self.data.empty:
            return self.data
        d = pd.to_datetime(self.data["announce_date"])
        return self.data[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]

    def upcoming(self, dt: pd.Timestamp, days: int = 30) -> pd.DataFrame:
        """Results expected in the near future (for the dashboard Events view)."""
        if self.data.empty:
            return self.data
        d = pd.to_datetime(self.data["announce_date"])
        ts = pd.Timestamp(dt)
        return self.data[(d > ts) & (d <= ts + pd.Timedelta(days=days))]

    def history_for(self, isin: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        df = self.visible_on(as_of) if as_of is not None else self.data
        return df[df["isin"] == isin].sort_values("period_end")

    def effective_entry_date(
        self, announce_date: pd.Timestamp, announce_time: str | None, lag_days: int = 1
    ) -> pd.Timestamp:
        """First tradeable date after an announcement.

        Post-market results can only be traded the following session; we apply
        the configured lag on top of that.
        """
        base = pd.Timestamp(announce_date)
        if isinstance(announce_time, str) and announce_time.lower() in ("amc", "after", "post"):
            base = base + pd.Timedelta(days=1)
        return base + pd.Timedelta(days=lag_days)


def _apply_split_adjustment(earnings: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Scale historical EPS for splits/bonuses so the series is comparable."""
    from .corporate_actions import ActionType, _share_multiplier

    out = earnings.copy()
    splits = actions[actions["action_type"].isin([ActionType.SPLIT.value, ActionType.BONUS.value])]
    for _, a in splits.iterrows():
        isin, ex = a["isin"], pd.Timestamp(a["ex_date"])
        mult = _share_multiplier(a)
        if not mult or mult <= 0:
            continue
        mask = (out["isin"] == isin) & (out["period_end"] < ex)
        out.loc[mask, "eps"] = out.loc[mask, "eps"] / mult
    return out


def build_earnings_from_fundamentals(fundamentals: pd.DataFrame) -> EarningsCalendar:
    """Derive an earnings calendar when the provider ships EPS in fundamentals."""
    if fundamentals.empty:
        return EarningsCalendar.empty()

    df = fundamentals.copy()
    if "eps" not in df.columns:
        if {"net_income", "shares_outstanding"}.issubset(df.columns):
            df["eps"] = df["net_income"] / df["shares_outstanding"].replace(0, np.nan)
        else:
            log.warning("Fundamentals contain no EPS; Sleeve C cannot compute SUE")
            return EarningsCalendar.empty()

    keep = [c for c in ("isin", "announce_date", "period_end", "fiscal_quarter", "eps") if c in df.columns]
    return EarningsCalendar.from_frame(df[keep])


def fetch_nse_results_calendar(cfg: AppConfig, days_ahead: int = 30) -> pd.DataFrame:
    """Placeholder for the NSE board-meeting/results calendar.

    NSE publishes forthcoming results dates, but the endpoint requires
    browser-like headers and changes periodically, so this is intentionally a
    stub: supply a CSV at data/earnings_calendar.csv or use a paid provider.
    """
    path = cfg.paths.data_dir / "earnings_calendar.csv"
    if path.exists():
        return pd.read_csv(path)
    log.info(
        "No earnings calendar at %s. Provide a CSV (isin, announce_date) or use "
        "a paid provider to enable forward-looking event views.", path,
    )
    return pd.DataFrame(columns=["isin", "announce_date", "announce_time"])
