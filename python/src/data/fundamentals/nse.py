"""NSE fundamentals provider - point-in-time by construction.

This is the only free source in the framework that carries a real
**announcement timestamp**. Every other free option forces us to guess the
publication date as `period_end + lag`, which quietly leaks future information
into factor scores. Here the timestamp comes from the exchange itself.

Two consistency traps this handles explicitly:

1. **Consolidated vs standalone.** Companies file both, often minutes apart. A
   series that flips between the two is not a time series of anything. We pick
   one basis (consolidated by default) and stick to it per company.
2. **Restatements and re-filings.** The same quarter can be filed more than
   once. We keep the *first* disseminated version for point-in-time work,
   because that is what the market actually saw.

Balance-sheet items are sparse in quarterly Ind-AS filings, so ratios needing
assets or equity are computed only where the data exists; the composite scorer
already renormalizes around missing legs.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ...config import get_logger
from ..nse_client import NSEClient, NSEAPIUnavailable
from .base import FundamentalsProvider

log = get_logger(__name__)


class NSEFundamentalsProvider(FundamentalsProvider):
    """Quarterly fundamentals from NSE filings, keyed on dissemination time."""

    name = "nse"
    has_announce_dates = True   # the whole point of this provider

    def __init__(
        self,
        client: NSEClient | None = None,
        reporting_lag_days: int = 45,
        basis: str = "consolidated",      # consolidated | standalone
        fetch_xbrl: bool = True,
        max_xbrl: int = 2000,
        cache_dir: str | None = None,
        xbrl_isins: list[str] | None = None,
    ):
        super().__init__(reporting_lag_days)
        self.client = client or NSEClient(cache_dir=cache_dir, allow_api=True)
        self.basis = basis.lower()
        self.fetch_xbrl = fetch_xbrl
        self.max_xbrl = max_xbrl
        self.xbrl_isins = list(xbrl_isins) if xbrl_isins else None

    # -- fetch ----------------------------------------------------------------
    def fetch(
        self, isins: list[str], start: str, end: str, chunk_months: int = 3,
    ) -> pd.DataFrame:
        """Pull filings over the window and map them to the canonical schema.

        NSE's results endpoint is date-windowed rather than symbol-keyed, so we
        walk the calendar in chunks and filter to our universe afterwards.
        """
        if not self.client.api_available:
            raise NSEAPIUnavailable(
                "NSE fundamentals need the JSON API (allow_api=True + curl_cffi installed)."
            )

        filings = self._fetch_filings(start, end, chunk_months)
        if filings.empty:
            log.warning("NSE returned no filings for %s..%s", start, end)
            return self.normalize(pd.DataFrame())

        filings = self._select_basis(filings)
        if isins:
            wanted = set(isins)
            filings = filings[filings["isin"].isin(wanted)]
            log.info("Filtered to %d filings inside the universe", len(filings))
        if filings.empty:
            return self.normalize(pd.DataFrame())

        if self.xbrl_isins:
            prio = set(self.xbrl_isins)
            filings = filings.copy()
            filings["_prio"] = filings["isin"].isin(prio).astype(int)
            filings = filings.sort_values(
                ["_prio", "period_end"], ascending=[False, True]
            ).drop(columns=["_prio"])

        records = self._build_records(filings)
        raw = enrich_pnl_metrics(pd.DataFrame(records))
        return self.normalize(raw)

    def _fetch_filings(self, start: str, end: str, chunk_months: int) -> pd.DataFrame:
        """Walk the window in chunks; the endpoint caps how much it returns."""
        frames = []
        cursor = pd.Timestamp(start)
        stop = pd.Timestamp(end)

        while cursor <= stop:
            chunk_end = min(cursor + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1), stop)
            try:
                df = self.client.financial_results(cursor, chunk_end, period="Quarterly")
                if not df.empty:
                    frames.append(df)
                    log.info("NSE results %s..%s: %d filings",
                             cursor.date(), chunk_end.date(), len(df))
            except Exception as exc:  # noqa: BLE001 - one bad window shouldn't abort
                log.warning("results fetch failed for %s..%s: %s",
                            cursor.date(), chunk_end.date(), exc)
            cursor = chunk_end + pd.Timedelta(days=1)

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return out.dropna(subset=["isin", "announce_date"])

    def _select_basis(self, filings: pd.DataFrame) -> pd.DataFrame:
        """Keep one reporting basis per company-quarter, and the first filing.

        Mixing consolidated and standalone across quarters produces artificial
        jumps that momentum and SUE would read as real news.
        """
        df = filings.copy()
        if "consolidated" not in df.columns:
            return df

        flag = df["consolidated"].astype(str).str.lower()
        df["_is_consolidated"] = flag.str.startswith("consolidated")
        # Prefer the requested basis, then fall back if a company only files one.
        df["_preferred"] = (
            df["_is_consolidated"] if self.basis == "consolidated" else ~df["_is_consolidated"]
        )
        df = df.sort_values(
            ["isin", "period_end", "_preferred", "announce_datetime"],
            ascending=[True, True, False, True],
        )
        # First row per company-quarter = preferred basis, earliest dissemination.
        out = df.groupby(["isin", "period_end"], as_index=False).first()

        dropped = len(df) - len(out)
        if dropped:
            log.info("Dropped %d duplicate filings (basis/restatements)", dropped)
        return out

    def _build_records(self, filings: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        xbrl_budget = self.max_xbrl if self.fetch_xbrl else 0

        for i, row in enumerate(filings.itertuples(index=False), 1):
            facts: dict[str, float] = {}
            xbrl_url = getattr(row, "xbrl", None) or getattr(row, "xbrl_url", None)
            isin = getattr(row, "isin", None)
            if (
                xbrl_budget > 0
                and isinstance(xbrl_url, str)
                and xbrl_url.startswith("http")
                and (self.xbrl_isins is None or isin in self.xbrl_isins)
            ):
                facts = self.client.download_xbrl(xbrl_url)
                xbrl_budget -= 1
                if i % 50 == 0:
                    log.info("XBRL: %d/%d filings parsed", i, len(filings))

            records.append(self._to_canonical(row, facts))
        return records

    def _to_canonical(self, row: Any, facts: dict[str, float]) -> dict[str, Any]:
        """Map one filing + its XBRL facts onto FUNDAMENTAL_COLUMNS."""
        period_end = getattr(row, "period_end", None)
        revenue = facts.get("revenue")
        net_income = facts.get("net_income")
        pbt = facts.get("profit_before_tax")
        eps = facts.get("eps_basic")
        equity_capital = facts.get("equity_capital")

        return {
            "isin": getattr(row, "isin", None),
            "symbol": getattr(row, "symbol", None),
            "announce_date": getattr(row, "announce_date", None),
            "announce_datetime": getattr(row, "announce_datetime", None),
            "period_end": period_end,
            "fiscal_quarter": (
                f"{period_end.year}Q{period_end.quarter}" if pd.notna(period_end) else None
            ),
            "basis": getattr(row, "consolidated", None),
            # Quality inputs. Balance-sheet legs stay empty; P&L proxies are
            # filled by enrich_pnl_metrics once the full company series exists.
            "roic": np.nan,
            "gross_profitability": np.nan,
            "cfo_to_pat": np.nan,
            "fcf_to_assets": np.nan,
            "debt_to_assets": np.nan,
            "payout": np.nan,
            "net_margin": _div(net_income, revenue),
            "pretax_margin": _div(pbt, revenue),
            # Valuation needs a price, joined in later.
            "earnings_yield": np.nan,
            "fcf_yield": np.nan,
            "ev_ebitda": np.nan,
            "pb": np.nan,
            "pe": np.nan,
            # Raw line items, kept for SUE, TTM rolls, and the quantamental layer.
            "revenue": revenue,
            "net_income": net_income,
            "profit_before_tax": pbt,
            "eps": eps,
            "eps_ttm": np.nan,
            "eps_diluted": facts.get("eps_diluted"),
            "total_expenses": facts.get("total_expenses"),
            "finance_cost": facts.get("finance_cost"),
            "depreciation": facts.get("depreciation"),
            "tax_expense": facts.get("tax_expense"),
            "equity_capital": equity_capital,
            "total_assets": np.nan,
            "market_cap": np.nan,
        }


def _div(a: float | None, b: float | None) -> float:
    if a is None or b is None or b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return a / b


def enrich_pnl_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Fill Quality legs that a quarterly P&L can actually support.

    NSE Ind-AS quarterlies rarely carry invested capital or total assets, so
    textbook ROIC / gross-profitability / FCF-to-assets stay empty. Trailing
    net margin, pretax margin and an EBIT/equity proxy are honest substitutes
    from the same filing - still dated by announce_date, never invented.
    """
    if df.empty:
        return df

    out = df.sort_values(["isin", "period_end"]).copy()
    for col in ("revenue", "net_income", "profit_before_tax", "finance_cost",
                "eps", "equity_capital"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    def _ttm(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(np.nan, index=out.index)
        return out.groupby("isin")[col].transform(
            lambda s: s.rolling(4, min_periods=2).sum()
        )

    ttm_rev = _ttm("revenue")
    ttm_ni = _ttm("net_income")
    ttm_pbt = _ttm("profit_before_tax")
    ttm_int = _ttm("finance_cost")
    out["eps_ttm"] = _ttm("eps")

    ebit = ttm_pbt.add(ttm_int, fill_value=0)
    ebit = ebit.where(ttm_pbt.notna() | ttm_int.notna())

    out["net_margin"] = ttm_ni / ttm_rev.replace(0, np.nan)
    out["pretax_margin"] = ttm_pbt / ttm_rev.replace(0, np.nan)
    out["gross_profitability"] = ebit / ttm_rev.replace(0, np.nan)
    equity = out["equity_capital"] if "equity_capital" in out.columns else np.nan
    out["roic"] = ebit / pd.to_numeric(equity, errors="coerce").replace(0, np.nan)
    return out


# --------------------------------------------------------------------------- #
# Price-dependent enrichment
# --------------------------------------------------------------------------- #
def attach_valuation_ratios(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    shares_outstanding: pd.Series | None = None,
) -> pd.DataFrame:
    """Add earnings yield, P/E and market cap using the price on the filing date.

    Valuation is a price-over-fundamental ratio, so it must be recomputed as
    prices move. Anchoring it to the announcement date keeps the snapshot
    point-in-time; the composite scorer refreshes ratios at each rebalance.

    EPS is trailing-twelve-month (four quarters) rather than a single quarter,
    since one quarter annualized is far too noisy for a value signal.
    """
    if fundamentals.empty or prices.empty:
        return fundamentals

    df = fundamentals.sort_values(["isin", "period_end"]).copy()
    if "eps_ttm" not in df.columns or df["eps_ttm"].isna().all():
        df["eps_ttm"] = (
            df.groupby("isin")["eps"]
            .rolling(4, min_periods=2).sum()
            .reset_index(level=0, drop=True)
        )

    price_on_announce = []
    for row in df.itertuples(index=False):
        price_on_announce.append(_price_asof(prices, row.isin, row.announce_date))
    df["price_at_announce"] = price_on_announce

    df["pe"] = np.where(
        (df["eps_ttm"] > 0) & df["price_at_announce"].notna(),
        df["price_at_announce"] / df["eps_ttm"],
        np.nan,
    )
    df["earnings_yield"] = np.where(df["pe"] > 0, 1.0 / df["pe"], np.nan)

    if shares_outstanding is not None and not shares_outstanding.empty:
        shares = df["isin"].map(shares_outstanding)
        df["market_cap"] = df["price_at_announce"] * shares
        df["fcf_yield"] = np.nan

    return df


def _price_asof(prices: pd.DataFrame, isin: str, when: Any) -> float:
    """Last traded price at or before a date."""
    if isin not in prices.columns or pd.isna(when):
        return np.nan
    series = prices[isin].dropna()
    prior = series.loc[series.index <= pd.Timestamp(when)]
    return float(prior.iloc[-1]) if len(prior) else np.nan


def build_eps_history(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Extract the EPS series Sleeve C needs, with real announcement dates."""
    if fundamentals.empty or "eps" not in fundamentals.columns:
        return pd.DataFrame()

    cols = [c for c in ("isin", "symbol", "announce_date", "announce_datetime",
                        "period_end", "fiscal_quarter", "eps", "revenue", "net_income")
            if c in fundamentals.columns]
    out = fundamentals[cols].dropna(subset=["eps"]).copy()
    out["basis"] = fundamentals.get("basis")
    return out.sort_values(["isin", "period_end"]).reset_index(drop=True)
