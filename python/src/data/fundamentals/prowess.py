"""CMIE Prowess adapter (academic gold standard).

Prowess is the source behind the IIMA Indian factor research: point-in-time,
survivorship-adjusted, with real result-announcement dates. It has no public
REST API - access is via institutional subscription and data is exported to
CSV/Excel. This adapter therefore reads local exports and maps CMIE's column
names onto our canonical schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...config import get_logger
from .base import FundamentalsProvider

log = get_logger(__name__)

# CMIE column name -> canonical name. Prowess exports vary by query template,
# so this map is intentionally easy to extend.
COLUMN_MAP = {
    "company_name": "name",
    "isin_code": "isin",
    "isin": "isin",
    "result_announcement_date": "announce_date",
    "date_of_board_meeting": "announce_date",
    "period_end_date": "period_end",
    "quarter_ending": "period_end",
    "total_assets": "total_assets",
    "market_capitalisation": "market_cap",
    "pat": "net_income",
    "profit_after_tax": "net_income",
    "net_sales": "revenue",
    "gross_profit": "gross_profit",
    "pbit": "ebit",
    "cash_flow_from_operating_activities": "cfo",
    "capital_expenditure": "capex",
    "total_borrowings": "total_debt",
    "net_worth": "equity",
    "dividend_paid": "dividends",
    "price_to_book": "pb",
    "price_to_earnings": "pe",
    "ev_by_ebitda": "ev_ebitda",
}


class ProwessProvider(FundamentalsProvider):
    name = "prowess"
    has_announce_dates = True

    def __init__(self, export_dir: str | Path, reporting_lag_days: int = 45):
        super().__init__(reporting_lag_days)
        self.export_dir = Path(export_dir)

    def fetch(self, isins: list[str], start: str, end: str) -> pd.DataFrame:
        files = sorted([*self.export_dir.glob("*.csv"), *self.export_dir.glob("*.xlsx")])
        if not files:
            log.warning("No Prowess exports found in %s", self.export_dir)
            return self.normalize(pd.DataFrame())

        frames = []
        for f in files:
            try:
                df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f)
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not read %s: %s", f.name, exc)

        if not frames:
            return self.normalize(pd.DataFrame())

        raw = pd.concat(frames, ignore_index=True)
        raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
        raw = raw.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in raw.columns})

        if "isin" not in raw.columns:
            log.error("Prowess export lacks an ISIN column; cannot join to the security master")
            return self.normalize(pd.DataFrame())

        df = self._derive(raw)
        if isins:
            df = df[df["isin"].isin(isins)]
        df = df[
            (pd.to_datetime(df["period_end"]) >= pd.Timestamp(start))
            & (pd.to_datetime(df["period_end"]) <= pd.Timestamp(end))
        ]
        return self.normalize(df)

    def _derive(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Build canonical ratios from raw CMIE line items."""
        g = raw.get
        assets = _num(g("total_assets"))
        debt = _num(g("total_debt"))
        equity = _num(g("equity"))
        ebit = _num(g("ebit"))
        net_income = _num(g("net_income"))
        gross = _num(g("gross_profit"))
        cfo = _num(g("cfo"))
        capex = _num(g("capex"))
        divs = _num(g("dividends"))
        mcap = _num(g("market_cap"))

        fcf = cfo - capex.abs() if cfo is not None and capex is not None else None
        invested = debt.fillna(0) + equity.fillna(0) if debt is not None and equity is not None else None

        out = pd.DataFrame({
            "isin": raw["isin"],
            "announce_date": pd.to_datetime(raw.get("announce_date"), errors="coerce"),
            "period_end": pd.to_datetime(raw.get("period_end"), errors="coerce"),
        })
        out["fiscal_quarter"] = out["period_end"].dt.to_period("Q").astype(str)
        out["roic"] = _div(ebit, invested)
        out["gross_profitability"] = _div(gross, assets)
        out["cfo_to_pat"] = _div(cfo, net_income)
        out["fcf_to_assets"] = _div(fcf, assets)
        out["debt_to_assets"] = _div(debt, assets)
        out["payout"] = _div(divs, net_income)
        out["earnings_yield"] = _div(net_income, mcap)
        out["fcf_yield"] = _div(fcf, mcap)
        out["ev_ebitda"] = _num(raw.get("ev_ebitda"))
        out["pb"] = _num(raw.get("pb"))
        out["pe"] = _num(raw.get("pe"))
        out["total_assets"] = assets
        out["market_cap"] = mcap
        return out


def _num(s) -> pd.Series | None:
    if s is None:
        return None
    return pd.to_numeric(s, errors="coerce")


def _div(a, b):
    if a is None or b is None:
        return np.nan
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
