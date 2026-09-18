"""Free / prototype fundamentals adapters.

Two sources, both with real limitations that are surfaced rather than hidden:

  * CSVProvider    - you export from screener.in (or anywhere) into a CSV.
  * YFinanceProvider - yfinance `.NS` tickers; patchy for Indian names and
    offers only a few years of quarterly statements with no filing dates.

Neither is point-in-time. Both approximate the announcement date as
period_end + reporting lag, which is fine for prototyping and NOT fine for the
final backtest. Use EODHD or Prowess for that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config import get_logger
from .base import FundamentalsProvider

log = get_logger(__name__)


class CSVProvider(FundamentalsProvider):
    """Read fundamentals from a local CSV you maintain yourself."""

    name = "csv"
    has_announce_dates = False

    def __init__(self, path: str | Path, reporting_lag_days: int = 45):
        super().__init__(reporting_lag_days)
        self.path = Path(path)

    def fetch(self, isins: list[str], start: str, end: str) -> pd.DataFrame:
        if not self.path.exists():
            log.warning("Fundamentals CSV not found at %s", self.path)
            return self.normalize(pd.DataFrame())
        df = pd.read_csv(self.path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "announce_date" in df.columns and df["announce_date"].notna().any():
            self.has_announce_dates = True
        if isins:
            df = df[df["isin"].isin(isins)]
        return self.normalize(df)


class YFinanceProvider(FundamentalsProvider):
    """yfinance adapter - prototype only.

    Caveats worth repeating: limited history (typically ~4-8 quarters),
    inconsistent coverage for Indian mid/small caps, and no filing dates.
    """

    name = "yfinance"
    has_announce_dates = False

    def __init__(self, reporting_lag_days: int = 45):
        super().__init__(reporting_lag_days)

    def fetch(self, isins: list[str], start: str, end: str,
              symbols: dict[str, str] | None = None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            log.error("yfinance not installed. `pip install yfinance` or pick another provider.")
            return self.normalize(pd.DataFrame())

        rows: list[dict[str, Any]] = []
        for i, isin in enumerate(isins, 1):
            sym = (symbols or {}).get(isin)
            if not sym:
                continue
            try:
                rows.extend(self._one(yf.Ticker(f"{sym}.NS"), isin))
            except Exception as exc:  # noqa: BLE001
                log.debug("yfinance failed for %s: %s", sym, exc)
            if i % 25 == 0:
                log.info("yfinance: %d/%d", i, len(isins))

        df = pd.DataFrame(rows)
        if df.empty:
            return self.normalize(df)
        df = df[(df["period_end"] >= pd.Timestamp(start)) & (df["period_end"] <= pd.Timestamp(end))]
        return self.normalize(df)

    def _one(self, tk: Any, isin: str) -> list[dict[str, Any]]:
        bs = _safe_frame(tk, "quarterly_balance_sheet")
        inc = _safe_frame(tk, "quarterly_financials")
        cf = _safe_frame(tk, "quarterly_cashflow")
        info = getattr(tk, "fast_info", {}) or {}
        mcap = _get(info, "market_cap")

        out = []
        for period in list(bs.columns) if not bs.empty else []:
            assets = _row(bs, "Total Assets", period)
            debt = _row(bs, "Total Debt", period)
            equity = _row(bs, "Stockholders Equity", period)
            revenue = _row(inc, "Total Revenue", period)
            gross = _row(inc, "Gross Profit", period)
            ni = _row(inc, "Net Income", period)
            ebit = _row(inc, "EBIT", period) or _row(inc, "Operating Income", period)
            cfo = _row(cf, "Operating Cash Flow", period)
            capex = _row(cf, "Capital Expenditure", period)
            divs = abs(_row(cf, "Cash Dividends Paid", period) or 0.0)

            fcf = (cfo - abs(capex)) if (cfo is not None and capex is not None) else None
            invested = (debt or 0.0) + (equity or 0.0)
            ts = pd.Timestamp(period)

            out.append({
                "isin": isin,
                "announce_date": None,
                "period_end": ts,
                "fiscal_quarter": f"{ts.year}Q{ts.quarter}",
                "roic": _div(ebit, invested),
                "gross_profitability": _div(gross, assets),
                "cfo_to_pat": _div(cfo, ni),
                "fcf_to_assets": _div(fcf, assets),
                "debt_to_assets": _div(debt, assets),
                "payout": _div(divs, ni),
                "earnings_yield": _div(ni, mcap),
                "fcf_yield": _div(fcf, mcap),
                "ev_ebitda": np.nan,
                "pb": _div(mcap, equity),
                "pe": _div(mcap, ni),
                "total_assets": assets,
                "market_cap": mcap,
            })
        return out


class SyntheticProvider(FundamentalsProvider):
    """Serves the synthetic dataset - lets the full pipeline run offline."""

    name = "synthetic"
    has_announce_dates = True

    def __init__(self, fundamentals: pd.DataFrame, reporting_lag_days: int = 45):
        super().__init__(reporting_lag_days)
        self._data = fundamentals

    def fetch(self, isins: list[str], start: str, end: str) -> pd.DataFrame:
        df = self._data.copy()
        if isins:
            df = df[df["isin"].isin(isins)]
        if not df.empty:
            pe = pd.to_datetime(df["period_end"])
            df = df[(pe >= pd.Timestamp(start)) & (pe <= pd.Timestamp(end))]
        return self.normalize(df)


def _safe_frame(tk: Any, attr: str) -> pd.DataFrame:
    try:
        df = getattr(tk, attr)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _row(df: pd.DataFrame, label: str, col: Any) -> float | None:
    if df.empty or label not in df.index or col not in df.columns:
        return None
    try:
        v = df.loc[label, col]
        return None if pd.isna(v) else float(v)
    except (KeyError, TypeError, ValueError):
        return None


def _get(info: Any, key: str) -> float | None:
    try:
        v = info[key] if not hasattr(info, key) else getattr(info, key)
        return float(v) if v else None
    except Exception:  # noqa: BLE001
        return None


def _div(a: float | None, b: float | None) -> float:
    if a is None or b is None or b == 0:
        return np.nan
    return a / b
