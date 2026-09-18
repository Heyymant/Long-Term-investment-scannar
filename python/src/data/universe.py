"""Point-in-time investable universe.

Survivorship bias is the classic way to make a backtest look brilliant: if you
only ever test today's NSE 500, you have quietly excluded every company that
blew up. This module reconstructs *who was actually in the index on each date*
and then applies tradeability filters using only information available then.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig, get_logger

log = get_logger(__name__)

MEMBERSHIP_COLUMNS = ["isin", "index_name", "start_date", "end_date"]


@dataclass
class UniverseFilters:
    min_adv_inr: float = 50_000_000.0
    min_price_inr: float = 20.0
    min_listing_days: int = 365
    min_financial_quarters: int = 8
    max_circuit_frequency: float = 0.20
    exclude_flagged_accounting: bool = True

    @classmethod
    def from_config(cls, cfg: AppConfig) -> UniverseFilters:
        f = cfg.get("universe.filters", {}) or {}
        return cls(
            min_adv_inr=float(f.get("min_adv_inr", 50_000_000.0)),
            min_price_inr=float(f.get("min_price_inr", 20.0)),
            min_listing_days=int(f.get("min_listing_days", 365)),
            min_financial_quarters=int(f.get("min_financial_quarters", 8)),
            max_circuit_frequency=float(f.get("exclude_circuit_hits_pct", 0.20)),
            exclude_flagged_accounting=bool(f.get("exclude_flagged_accounting", True)),
        )


class IndexMembership:
    """Point-in-time index constituents as [start_date, end_date) intervals."""

    def __init__(self, membership: pd.DataFrame):
        df = membership.copy()
        for c in ("start_date", "end_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        self.df = df

    @classmethod
    def empty(cls) -> IndexMembership:
        return cls(pd.DataFrame(columns=MEMBERSHIP_COLUMNS))

    @classmethod
    def load(cls, data_dir: str | Path) -> IndexMembership:
        p = Path(data_dir) / "index_membership.parquet"
        if p.exists():
            return cls(pd.read_parquet(p))
        csv = Path(data_dir) / "index_membership.csv"
        if csv.exists():
            return cls(pd.read_csv(csv))
        return cls.empty()

    def save(self, data_dir: str | Path) -> None:
        self.df.to_parquet(Path(data_dir) / "index_membership.parquet", index=False)

    @classmethod
    def from_static(
        cls, isins: list[str], index_name: str, start: str | pd.Timestamp = "1990-01-01"
    ) -> IndexMembership:
        """Fallback when no change history exists.

        This reintroduces survivorship bias - callers should surface the caveat.
        """
        log.warning(
            "Building static membership for %s from %d current constituents; "
            "results will carry survivorship bias.", index_name, len(isins),
        )
        return cls(pd.DataFrame({
            "isin": isins, "index_name": index_name,
            "start_date": pd.Timestamp(start), "end_date": pd.NaT,
        }))

    def members_on(self, dt: pd.Timestamp, index_name: str | None = None) -> set[str]:
        df = self.df
        if df.empty:
            return set()
        if index_name:
            df = df[df["index_name"] == index_name]
        ts = pd.Timestamp(dt)
        mask = (df["start_date"] <= ts) & (df["end_date"].isna() | (df["end_date"] > ts))
        return set(df.loc[mask, "isin"])

    def add_change(
        self, isin: str, index_name: str,
        added: str | pd.Timestamp, removed: str | pd.Timestamp | None = None,
    ) -> None:
        row = {
            "isin": isin, "index_name": index_name,
            "start_date": pd.Timestamp(added),
            "end_date": pd.Timestamp(removed) if removed else pd.NaT,
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)


# --------------------------------------------------------------------------- #
# Tradeability metrics
# --------------------------------------------------------------------------- #
def average_daily_value(
    prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 63
) -> pd.DataFrame:
    """Rolling average daily traded value in INR (price * shares)."""
    common = prices.columns.intersection(volumes.columns)
    value = prices[common] * volumes[common]
    return value.rolling(window, min_periods=max(5, window // 3)).mean()


def circuit_frequency(
    prices: pd.DataFrame, window: int = 252, band: float = 0.19
) -> pd.DataFrame:
    """Fraction of recent days with |return| at/above a circuit-like band.

    Names that repeatedly lock limit-up/limit-down are not realistically
    tradeable at model prices.
    """
    rets = prices.pct_change()
    hit = (rets.abs() >= band).astype(float)
    return hit.rolling(window, min_periods=max(20, window // 5)).mean()


def listing_age_days(prices: pd.DataFrame) -> pd.DataFrame:
    """Trading days of history available per name at each date."""
    return prices.notna().cumsum()


# --------------------------------------------------------------------------- #
# Universe construction
# --------------------------------------------------------------------------- #
class UniverseBuilder:
    """Produces a boolean (date x isin) eligibility mask."""

    def __init__(
        self,
        filters: UniverseFilters,
        membership: IndexMembership | None = None,
        index_name: str = "NSE500",
        custom_list: list[str] | None = None,
    ):
        self.filters = filters
        self.membership = membership or IndexMembership.empty()
        self.index_name = index_name
        self.custom_list = set(custom_list) if custom_list else None

    def build(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
        fundamentals_coverage: pd.DataFrame | None = None,
        accounting_flags: set[str] | None = None,
    ) -> pd.DataFrame:
        """Eligibility mask aligned to `prices` (index=date, cols=isin)."""
        eligible = prices.notna()

        # 1) Index membership (point-in-time) or an explicit tradeable list.
        if self.custom_list is not None:
            keep = [c for c in prices.columns if c in self.custom_list]
            mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
            mask[keep] = True
            eligible &= mask
        elif not self.membership.df.empty:
            eligible &= self._membership_mask(prices)

        # 2) Price floor - penny stocks are not investable at scale.
        eligible &= prices >= self.filters.min_price_inr

        # 3) Liquidity floor.
        if volumes is not None and not volumes.empty:
            adv = average_daily_value(prices, volumes)
            adv = adv.reindex(columns=prices.columns)
            eligible &= adv >= self.filters.min_adv_inr

        # 4) Minimum listing history.
        if self.filters.min_listing_days > 0:
            eligible &= listing_age_days(prices) >= self.filters.min_listing_days

        # 5) Circuit-hit frequency.
        if self.filters.max_circuit_frequency < 1.0:
            cf = circuit_frequency(prices)
            eligible &= (cf <= self.filters.max_circuit_frequency) | cf.isna()

        # 6) Sufficient reported financial history.
        if (
            fundamentals_coverage is not None
            and not fundamentals_coverage.empty
            and self.filters.min_financial_quarters > 0
        ):
            cov = fundamentals_coverage.reindex(
                index=prices.index, columns=prices.columns
            ).ffill().fillna(0)
            passes = cov >= self.filters.min_financial_quarters

            # Guard against a data gap masquerading as an uninvestable market.
            # If this single filter removes almost everything that is otherwise
            # eligible, the cause is missing fundamentals, not 2,000 unsuitable
            # companies - and an empty universe produces a backtest that looks
            # like a strategy failure instead of a data failure.
            before = int(eligible.iloc[-1].sum())
            after = int((eligible & passes).iloc[-1].sum())
            if before > 0 and after < before * 0.05:
                log.error(
                    "Fundamentals filter (>= %d quarters) would drop %d of %d eligible names. "
                    "This indicates missing fundamentals data, not an uninvestable universe, "
                    "so the filter is being SKIPPED. Fetch more history "
                    "(scripts/fetch_nse.py --earnings) or set "
                    "universe.min_financial_quarters = 0 for price-only strategies.",
                    self.filters.min_financial_quarters, before - after, before,
                )
            else:
                eligible &= passes

        # 7) Accounting-quality exclusions.
        if self.filters.exclude_flagged_accounting and accounting_flags:
            for isin in accounting_flags:
                if isin in eligible.columns:
                    eligible[isin] = False

        n = int(eligible.iloc[-1].sum()) if len(eligible) else 0
        log.info("Universe built: %d eligible names on %s", n, prices.index[-1].date() if len(prices) else "n/a")
        return eligible

    def _membership_mask(self, prices: pd.DataFrame) -> pd.DataFrame:
        mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        df = self.membership.df
        df = df[df["index_name"] == self.index_name] if "index_name" in df.columns else df
        for _, row in df.iterrows():
            isin = row["isin"]
            if isin not in mask.columns:
                continue
            start = pd.Timestamp(row["start_date"])
            end = pd.Timestamp(row["end_date"]) if pd.notna(row["end_date"]) else None
            sel = mask.index >= start
            if end is not None:
                sel &= mask.index < end
            mask.loc[sel, isin] = True
        return mask


def eligible_on(mask: pd.DataFrame, dt: pd.Timestamp) -> list[str]:
    """Names eligible on (or on the last date at/before) `dt`."""
    if mask.empty:
        return []
    ts = pd.Timestamp(dt)
    if ts in mask.index:
        row = mask.loc[ts]
    else:
        prior = mask.index[mask.index <= ts]
        if len(prior) == 0:
            return []
        row = mask.loc[prior[-1]]
    return list(row[row].index)


def fundamentals_coverage_matrix(
    fundamentals: pd.DataFrame, dates: pd.DatetimeIndex, isins: list[str]
) -> pd.DataFrame:
    """Count of quarters reported as-of each date (point-in-time).

    Uses announce_date, so a quarter only counts once it has been published.
    """
    if fundamentals.empty:
        return pd.DataFrame(index=dates, columns=isins, dtype=float)
    f = fundamentals.copy()
    f["announce_date"] = pd.to_datetime(f["announce_date"])
    counts = (
        f.sort_values("announce_date")
        .groupby("isin")
        .apply(lambda g: pd.Series(np.arange(1, len(g) + 1), index=g["announce_date"]), include_groups=False)
    )
    out = pd.DataFrame(index=dates, columns=isins, dtype=float)
    for isin in isins:
        if isin not in counts.index.get_level_values(0):
            continue
        s = counts.loc[isin]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[isin] = s.reindex(dates, method="ffill")
    return out.fillna(0.0)
