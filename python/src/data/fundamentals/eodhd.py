"""EODHD fundamentals adapter (recommended default paid source).

Provides NSE/BSE historical financial statements via REST. Good coverage and
an actual API (no fragile scraping). Statement dates are used as the
point-in-time key when EODHD supplies a filing date; otherwise we fall back to
period_end + reporting lag.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from ...config import env, get_logger
from .base import FundamentalsProvider

log = get_logger(__name__)

BASE_URL = "https://eodhd.com/api/fundamentals"


class EODHDProvider(FundamentalsProvider):
    name = "eodhd"
    has_announce_dates = True   # 'filing_date' when present

    def __init__(self, api_key: str | None = None, reporting_lag_days: int = 45,
                 exchange: str = "NSE", timeout: int = 30, rate_limit_per_sec: float = 4.0):
        super().__init__(reporting_lag_days)
        self.api_key = api_key or env("EODHD_API_KEY")
        self.exchange = exchange
        self.timeout = timeout
        self._min_gap = 1.0 / rate_limit_per_sec if rate_limit_per_sec > 0 else 0.0
        self._last = 0.0

    def _throttle(self) -> None:
        if self._min_gap <= 0:
            return
        delta = time.monotonic() - self._last
        if delta < self._min_gap:
            time.sleep(self._min_gap - delta)
        self._last = time.monotonic()

    def fetch(self, isins: list[str], start: str, end: str,
              symbols: dict[str, str] | None = None) -> pd.DataFrame:
        """Fetch fundamentals for each ISIN.

        EODHD is keyed by TICKER.EXCHANGE, so `symbols` should map isin ->
        NSE symbol (from the security master).
        """
        if not self.api_key:
            raise RuntimeError("EODHD_API_KEY not set (see .env.example)")

        rows: list[dict[str, Any]] = []
        for i, isin in enumerate(isins, 1):
            sym = (symbols or {}).get(isin)
            if not sym:
                log.debug("No symbol for %s; skipping", isin)
                continue
            try:
                payload = self._get(f"{sym}.{self.exchange}")
            except Exception as exc:  # noqa: BLE001
                log.warning("EODHD fetch failed for %s: %s", sym, exc)
                continue
            rows.extend(self._parse(payload, isin))
            if i % 50 == 0:
                log.info("EODHD: %d/%d instruments", i, len(isins))

        df = pd.DataFrame(rows)
        if df.empty:
            return self.normalize(df)
        df = df[(df["period_end"] >= pd.Timestamp(start)) & (df["period_end"] <= pd.Timestamp(end))]
        return self.normalize(df)

    def _get(self, ticker: str) -> dict[str, Any]:
        self._throttle()
        resp = requests.get(
            f"{BASE_URL}/{ticker}",
            params={"api_token": self.api_key, "fmt": "json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _parse(self, payload: dict[str, Any], isin: str) -> list[dict[str, Any]]:
        """Map EODHD's nested statements into our canonical row schema."""
        fin = payload.get("Financials", {}) or {}
        bs = (fin.get("Balance_Sheet", {}) or {}).get("quarterly", {}) or {}
        inc = (fin.get("Income_Statement", {}) or {}).get("quarterly", {}) or {}
        cf = (fin.get("Cash_Flow", {}) or {}).get("quarterly", {}) or {}
        highlights = payload.get("Highlights", {}) or {}
        valuation = payload.get("Valuation", {}) or {}
        shares = (payload.get("SharesStats", {}) or {}).get("SharesOutstanding")

        out: list[dict[str, Any]] = []
        for period, b in bs.items():
            i_stmt = inc.get(period, {}) or {}
            c_stmt = cf.get(period, {}) or {}

            total_assets = _f(b.get("totalAssets"))
            total_debt = _f(b.get("shortLongTermDebtTotal")) or _f(b.get("netDebt"))
            equity = _f(b.get("totalStockholderEquity"))
            revenue = _f(i_stmt.get("totalRevenue"))
            gross = _f(i_stmt.get("grossProfit"))
            net_income = _f(i_stmt.get("netIncome"))
            ebit = _f(i_stmt.get("ebit"))
            cfo = _f(c_stmt.get("totalCashFromOperatingActivities"))
            capex = _f(c_stmt.get("capitalExpenditures"))
            dividends = abs(_f(c_stmt.get("dividendsPaid")) or 0.0)

            fcf = (cfo - abs(capex)) if (cfo is not None and capex is not None) else None
            invested = (total_debt or 0.0) + (equity or 0.0)

            out.append({
                "isin": isin,
                "announce_date": _d(b.get("filing_date")),
                "period_end": _d(period),
                "fiscal_quarter": _quarter_label(period),
                "roic": _safe_div(ebit, invested),
                "gross_profitability": _safe_div(gross, total_assets),
                "cfo_to_pat": _safe_div(cfo, net_income),
                "fcf_to_assets": _safe_div(fcf, total_assets),
                "debt_to_assets": _safe_div(total_debt, total_assets),
                "payout": _safe_div(dividends, net_income),
                # Valuation ratios: EODHD publishes current snapshots only, so
                # these are approximations for historical periods.
                "earnings_yield": _inv(_f(highlights.get("PERatio"))),
                "fcf_yield": np.nan,
                "ev_ebitda": _f(valuation.get("EnterpriseValueEbitda")),
                "pb": _f(highlights.get("PriceBookMRQ")),
                "pe": _f(highlights.get("PERatio")),
                "total_assets": total_assets,
                "market_cap": _f(highlights.get("MarketCapitalization")),
            })
        return out


def _f(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _d(v: Any) -> pd.Timestamp | None:
    if not v:
        return None
    try:
        return pd.Timestamp(v)
    except (TypeError, ValueError):
        return None


def _safe_div(a: float | None, b: float | None) -> float:
    if a is None or b is None or b == 0:
        return np.nan
    return a / b


def _inv(v: float | None) -> float:
    if v is None or v == 0:
        return np.nan
    return 1.0 / v


def _quarter_label(period: str) -> str:
    ts = _d(period)
    return f"{ts.year}Q{ts.quarter}" if ts is not None else ""
