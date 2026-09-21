"""FundamentalsProvider interface + point-in-time alignment.

Quality and Value need per-stock fundamentals, which Kite does not supply.
Providers are pluggable so the prototype can run on free data while the real
backtest uses EODHD or CMIE Prowess.

The non-negotiable rule: a fundamental value may only be used on or after its
**announcement date**. Using period-end dates instead is the single most
common source of fake alpha in factor backtests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import get_logger

log = get_logger(__name__)

# Canonical schema every provider must emit.
FUNDAMENTAL_COLUMNS = [
    "isin",
    "announce_date",     # point-in-time key
    "period_end",
    "fiscal_quarter",
    # Quality inputs
    "roic",
    "gross_profitability",
    "cfo_to_pat",
    "fcf_to_assets",
    "debt_to_assets",
    "payout",
    # P&L profitability (available from quarterly Ind-AS filings when
    # balance-sheet ROIC/GP legs are missing).
    "net_margin",
    "pretax_margin",
    # Value inputs
    "earnings_yield",
    "fcf_yield",
    "ev_ebitda",
    "pb",
    "pe",
    # Reported line items. EPS is carried here (not just in the earnings
    # table) because Sleeve C's SUE is computed from reported EPS history.
    "eps",
    "eps_ttm",
    "revenue",
    "net_income",
    # Context
    "total_assets",
    "market_cap",
]

QUALITY_METRICS = [
    "roic", "gross_profitability", "cfo_to_pat", "fcf_to_assets",
    "debt_to_assets", "payout", "net_margin", "pretax_margin",
]
VALUE_METRICS = ["earnings_yield", "fcf_yield", "ev_ebitda", "pb", "pe"]


class FundamentalsProvider(ABC):
    """Base class for all fundamentals sources."""

    name: str = "base"
    #: True if the source supplies real announcement dates. When False we fall
    #: back to period_end + reporting_lag_days, which is an approximation.
    has_announce_dates: bool = False

    def __init__(self, reporting_lag_days: int = 45):
        self.reporting_lag_days = reporting_lag_days

    @abstractmethod
    def fetch(self, isins: list[str], start: str, end: str) -> pd.DataFrame:
        """Return rows conforming to FUNDAMENTAL_COLUMNS."""

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce to the canonical schema and enforce point-in-time dating."""
        if df.empty:
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

        out = df.copy()
        for col in FUNDAMENTAL_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")

        if self.has_announce_dates and out["announce_date"].notna().any():
            out["announce_date"] = pd.to_datetime(out["announce_date"], errors="coerce")
            missing = out["announce_date"].isna()
            if missing.any():
                out.loc[missing, "announce_date"] = (
                    out.loc[missing, "period_end"] + pd.Timedelta(days=self.reporting_lag_days)
                )
        else:
            log.warning(
                "%s has no announcement dates; approximating as period_end + %d days. "
                "This is a look-ahead approximation.", self.name, self.reporting_lag_days,
            )
            out["announce_date"] = out["period_end"] + pd.Timedelta(days=self.reporting_lag_days)

        out = out.dropna(subset=["isin", "announce_date"])
        out = out.sort_values(["isin", "announce_date"])
        out = out.drop_duplicates(subset=["isin", "period_end"], keep="last")
        return out[FUNDAMENTAL_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Point-in-time access
# --------------------------------------------------------------------------- #
def as_of(
    fundamentals: pd.DataFrame, dt: pd.Timestamp, metrics: list[str] | None = None
) -> pd.DataFrame:
    """Latest published fundamentals per ISIN, as known on `dt`.

    Filters on announce_date <= dt, so nothing unpublished leaks in.
    """
    if fundamentals.empty:
        return pd.DataFrame()
    ts = pd.Timestamp(dt)
    visible = fundamentals[pd.to_datetime(fundamentals["announce_date"]) <= ts]
    if visible.empty:
        return pd.DataFrame()
    latest = visible.sort_values("announce_date").groupby("isin").tail(1).set_index("isin")
    cols = metrics or [c for c in FUNDAMENTAL_COLUMNS if c not in ("isin",)]
    return latest[[c for c in cols if c in latest.columns]]


def build_metric_panel(
    fundamentals: pd.DataFrame, metric: str, dates: pd.DatetimeIndex, isins: list[str]
) -> pd.DataFrame:
    """Forward-filled (date x isin) panel of one metric, point-in-time safe."""
    if fundamentals.empty or metric not in fundamentals.columns:
        return pd.DataFrame(index=dates, columns=isins, dtype=float)

    f = fundamentals[["isin", "announce_date", metric]].copy()
    f["announce_date"] = pd.to_datetime(f["announce_date"])
    wide = (
        f.pivot_table(index="announce_date", columns="isin", values=metric, aggfunc="last")
        .sort_index()
    )
    # Union the announce dates with the target calendar, ffill, then restrict.
    combined = wide.reindex(wide.index.union(dates)).ffill().reindex(dates)
    return combined.reindex(columns=isins)


def staleness_days(
    fundamentals: pd.DataFrame, dates: pd.DatetimeIndex, isins: list[str]
) -> pd.DataFrame:
    """Age (days) of the most recent published report - a data-quality guard."""
    if fundamentals.empty:
        return pd.DataFrame(index=dates, columns=isins, dtype=float)
    f = fundamentals[["isin", "announce_date"]].copy()
    f["announce_date"] = pd.to_datetime(f["announce_date"])
    f["_v"] = f["announce_date"]
    wide = f.pivot_table(index="announce_date", columns="isin", values="_v", aggfunc="last")
    wide = wide.reindex(wide.index.union(dates)).ffill().reindex(dates).reindex(columns=isins)
    age = pd.DataFrame(
        {c: (dates - pd.to_datetime(wide[c])).days for c in wide.columns}, index=dates
    )
    return age


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class FundamentalsCache:
    def __init__(self, data_dir: str | Path):
        self.path = Path(data_dir) / "fundamentals.parquet"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
        return pd.read_parquet(self.path)

    def write(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.path, index=False)

    def upsert(self, df: pd.DataFrame) -> pd.DataFrame:
        merged = pd.concat([self.read(), df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["isin", "period_end"], keep="last")
        merged = merged.sort_values(["isin", "announce_date"]).reset_index(drop=True)
        self.write(merged)
        return merged
