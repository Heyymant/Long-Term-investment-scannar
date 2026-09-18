"""NSE fundamentals provider - free, point-in-time, exchange-sourced.

Combines two NSE feeds:
  * the financial-results index, which carries ``broadCastDate`` (the moment
    the market learned the numbers) and a link to each XBRL filing
  * the XBRL filings themselves, which carry revenue, PAT and EPS

This is the only free provider in the framework with **genuine announcement
timestamps**, which is what makes it usable for a serious backtest rather than
just a prototype.

Honest limitations:
  * Quarterly filings give P&L detail; balance-sheet items (total assets,
    borrowings) appear only in half-yearly/annual filings, so the balance-sheet
    driven Quality legs are sparser than the profitability ones.
  * The results API returns a rolling recent window, so deep history has to be
    accumulated over time (the cache is append-only for exactly this reason).
  * Valuation ratios need market cap, which is joined from price data rather
    than taken from the filing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config import get_logger
from ..nse import NSEClient
from ..xbrl import facts_to_frame, parse_xbrl
from .base import FundamentalsProvider

log = get_logger(__name__)


class NSEProvider(FundamentalsProvider):
    """Fundamentals assembled from NSE results filings + XBRL."""

    name = "nse"
    has_announce_dates = True      # broadCastDate is a real announcement time

    def __init__(
        self,
        client: NSEClient | None = None,
        cache_dir: str | Path | None = None,
        reporting_lag_days: int = 45,
        max_filings: int | None = None,
        prefer_consolidated: bool = True,
    ):
        super().__init__(reporting_lag_days)
        self.client = client or NSEClient(cache_dir=cache_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_filings = max_filings
        self.prefer_consolidated = prefer_consolidated

    # -- fetch -----------------------------------------------------------------
    def fetch(
        self,
        isins: list[str],
        start: str,
        end: str,
        prices: pd.DataFrame | None = None,
        shares_outstanding: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fetch results filings and parse their XBRL into our schema."""
        index = self.client.fetch_financial_results("Quarterly")
        if index.empty:
            log.warning("NSE returned no financial results; fundamentals unavailable")
            return self.normalize(pd.DataFrame())

        index = self._filter_filings(index, isins, start, end)
        if index.empty:
            return self.normalize(pd.DataFrame())

        cached = self._load_xbrl_cache()
        parsed: list[dict[str, Any]] = []
        fetched = 0

        for _, row in index.iterrows():
            key = _filing_key(row)
            if key in cached:
                parsed.append(cached[key])
                continue
            if self.max_filings is not None and fetched >= self.max_filings:
                continue

            url = row.get("xbrl_url")
            content = self.client.fetch_xbrl(url) if isinstance(url, str) else None
            fetched += 1
            if not content:
                continue

            facts = parse_xbrl(content, row.get("period_start"), row.get("period_end"))
            if facts is None:
                continue

            record = facts.to_dict()
            record.update({
                "isin": row.get("isin") or record.get("isin"),
                "symbol": row.get("symbol") or record.get("symbol"),
                "announce_date": row.get("announce_date"),
                "period_end": row.get("period_end") or record.get("period_end"),
                "consolidated": row.get("consolidated"),
                "_key": key,
            })
            parsed.append(record)
            cached[key] = record

            if fetched % 25 == 0:
                log.info("NSE XBRL: %d filings parsed", fetched)

        self._save_xbrl_cache(cached)

        if not parsed:
            log.warning("No XBRL filings could be parsed")
            return self.normalize(pd.DataFrame())

        raw = pd.DataFrame(parsed)
        raw = self._pick_reporting_basis(raw)
        derived = self._derive_metrics(raw, prices, shares_outstanding)
        log.info("NSE fundamentals: %d company-quarters", len(derived))
        return self.normalize(derived)

    # -- helpers ---------------------------------------------------------------
    def _filter_filings(
        self, index: pd.DataFrame, isins: list[str], start: str, end: str
    ) -> pd.DataFrame:
        df = index.copy()
        if isins:
            df = df[df["isin"].isin(set(isins))]
        pe = pd.to_datetime(df["period_end"], errors="coerce")
        df = df[(pe >= pd.Timestamp(start)) & (pe <= pd.Timestamp(end))]
        return df.dropna(subset=["xbrl_url"]) if "xbrl_url" in df.columns else df

    def _pick_reporting_basis(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per (isin, period_end).

        Companies file both standalone and consolidated results. Mixing the two
        across quarters would create fake earnings jumps, so we pick one basis
        consistently - consolidated by default, since it reflects the whole group.
        """
        if df.empty or "period_end" not in df.columns:
            return df

        out = df.copy()
        basis = out.get("consolidated", pd.Series(index=out.index, dtype=str)).astype(str)
        out["_is_consolidated"] = basis.str.contains("consolidat", case=False, na=False)
        out["_rank"] = (~out["_is_consolidated"]).astype(int)
        if not self.prefer_consolidated:
            out["_rank"] = 1 - out["_rank"]

        out = (
            out.sort_values(["isin", "period_end", "_rank"])
            .groupby(["isin", "period_end"], as_index=False)
            .first()
        )
        return out.drop(columns=[c for c in ("_rank",) if c in out.columns])

    def _derive_metrics(
        self,
        raw: pd.DataFrame,
        prices: pd.DataFrame | None,
        shares_outstanding: pd.Series | None,
    ) -> pd.DataFrame:
        """Map XBRL line items onto the canonical Quality/Value inputs."""
        df = raw.sort_values(["isin", "period_end"]).copy()
        g = df.groupby("isin")

        revenue = pd.to_numeric(df.get("revenue"), errors="coerce")
        net_income = pd.to_numeric(df.get("net_income"), errors="coerce")
        pbt = pd.to_numeric(df.get("profit_before_tax"), errors="coerce")
        finance_cost = pd.to_numeric(df.get("finance_cost"), errors="coerce")
        eps = pd.to_numeric(df.get("eps_basic"), errors="coerce")
        equity_capital = pd.to_numeric(df.get("equity_capital"), errors="coerce")

        out = pd.DataFrame({
            "isin": df["isin"],
            "symbol": df.get("symbol"),
            "announce_date": pd.to_datetime(df.get("announce_date"), errors="coerce"),
            "period_end": pd.to_datetime(df.get("period_end"), errors="coerce"),
            "revenue": revenue,
            "net_income": net_income,
            "eps": eps,
        })
        out["fiscal_quarter"] = out["period_end"].dt.to_period("Q").astype(str)

        # Trailing-twelve-month aggregates smooth Indian seasonality and are
        # the right basis for margin/profitability ratios.
        ttm_revenue = g["revenue"].transform(lambda s: s.rolling(4, min_periods=2).sum())
        ttm_income = g["net_income"].transform(lambda s: s.rolling(4, min_periods=2).sum())
        out["ttm_revenue"] = ttm_revenue
        out["ttm_net_income"] = ttm_income

        # EBIT proxy: PBT + finance cost (interest added back).
        ebit = pbt.fillna(0) + finance_cost.fillna(0)
        out["net_margin"] = _safe_div(ttm_income, ttm_revenue)

        # Quality legs available from quarterly P&L filings.
        out["gross_profitability"] = _safe_div(ebit, ttm_revenue)
        out["roic"] = _safe_div(ebit * 4, equity_capital)      # crude: equity capital only
        out["cfo_to_pat"] = np.nan        # cash flow is not in quarterly filings
        out["fcf_to_assets"] = np.nan
        out["debt_to_assets"] = np.nan
        out["payout"] = np.nan
        out["total_assets"] = np.nan

        # Valuation needs a price; join it if the caller supplied one.
        out["market_cap"] = np.nan
        out["earnings_yield"] = np.nan
        out["fcf_yield"] = np.nan
        out["ev_ebitda"] = np.nan
        out["pb"] = np.nan
        out["pe"] = np.nan

        if prices is not None and not prices.empty:
            out = _attach_valuation(out, prices, shares_outstanding)

        return out

    # -- XBRL cache ------------------------------------------------------------
    def _cache_file(self) -> Path | None:
        return self.cache_dir / "nse_xbrl_cache.parquet" if self.cache_dir else None

    def _load_xbrl_cache(self) -> dict[str, dict[str, Any]]:
        path = self._cache_file()
        if not path or not path.exists():
            return {}
        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError):
            return {}
        if "_key" not in df.columns:
            return {}
        return {row["_key"]: row.to_dict() for _, row in df.iterrows()}

    def _save_xbrl_cache(self, cache: dict[str, dict[str, Any]]) -> None:
        path = self._cache_file()
        if not path or not cache:
            return
        try:
            pd.DataFrame(list(cache.values())).to_parquet(path, index=False)
        except (OSError, ValueError) as exc:
            log.debug("Could not persist XBRL cache: %s", exc)


def _filing_key(row: pd.Series) -> str:
    pe = pd.to_datetime(row.get("period_end"), errors="coerce")
    basis = str(row.get("consolidated", ""))[:4]
    return f"{row.get('isin')}|{pe:%Y%m%d}|{basis}" if pd.notna(pe) else f"{row.get('isin')}|na|{basis}"


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _attach_valuation(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    shares_outstanding: pd.Series | None,
) -> pd.DataFrame:
    """Add price-dependent ratios, valued at the announcement date.

    Using the price on the announcement date (not today's) keeps the valuation
    point-in-time consistent with the rest of the pipeline.
    """
    out = fundamentals.copy()
    px_on_announce = []

    for _, row in out.iterrows():
        isin, dt = row["isin"], row["announce_date"]
        if isin not in prices.columns or pd.isna(dt):
            px_on_announce.append(np.nan)
            continue
        series = prices[isin].loc[prices.index <= dt].dropna()
        px_on_announce.append(float(series.iloc[-1]) if len(series) else np.nan)

    out["price_at_announce"] = px_on_announce

    if shares_outstanding is not None and not shares_outstanding.empty:
        shares = out["isin"].map(shares_outstanding)
        out["market_cap"] = out["price_at_announce"] * shares
        out["earnings_yield"] = _safe_div(out["ttm_net_income"], out["market_cap"])
        out["pe"] = _safe_div(out["market_cap"], out["ttm_net_income"])
    else:
        # Without a share count we can still use per-share earnings yield:
        # (TTM EPS / price) is equivalent and needs no share data.
        ttm_eps = (
            out.sort_values(["isin", "period_end"])
            .groupby("isin")["eps"]
            .transform(lambda s: s.rolling(4, min_periods=2).sum())
        )
        out["ttm_eps"] = ttm_eps
        out["earnings_yield"] = _safe_div(ttm_eps, out["price_at_announce"])
        out["pe"] = _safe_div(out["price_at_announce"], ttm_eps)

    return out


def build_earnings_from_nse(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Earnings calendar (for Sleeve C) from parsed NSE filings."""
    if fundamentals.empty or "eps" not in fundamentals.columns:
        return pd.DataFrame()

    cols = ["isin", "symbol", "announce_date", "period_end", "fiscal_quarter", "eps"]
    out = fundamentals[[c for c in cols if c in fundamentals.columns]].copy()
    if "revenue" in fundamentals.columns:
        out["revenue"] = fundamentals["revenue"]
    if "net_income" in fundamentals.columns:
        out["net_income"] = fundamentals["net_income"]
    out["basis"] = "nse_xbrl"
    return out.dropna(subset=["isin", "announce_date", "eps"])
