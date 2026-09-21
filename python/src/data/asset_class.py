"""Classify Indian instruments as equity, ETF, or mutual fund.

NSE equity shares use ISIN prefix INE. Mutual-fund products — including
exchange-traded funds — use INF. Kite's MF book is the non-exchange (folio)
channel; listed ETFs trade on NSE/BSE like shares.

This split matters for the research rules: Quality/Value/QMJ are stock
fundamentals. ETFs are priced as portfolios, so they get momentum, low-vol
and dual-momentum (Antonacci / Moskowitz) instead of a value×quality gate.
"""

from __future__ import annotations

ETF_TOKENS = (
    "ETF", "BEES", "IETF", "NIFTYBEES", "GOLDBEES", "SILVERBE", "BANKBEES",
    "LIQUIDBE", "JUNIORBE", "SETFNIF", "SETFNN50", "MON100", "MOSNIFTY",
    "ICICINIFTY", "HDFCNIFTY", "KOTAKNIFTY", "UTINIFTETF", "NIF100BEES",
    "ITBEES", "PHARMABEES", "AUTOBEES", "PSUBNKBEES", "INFRABEES",
)


def classify(isin: str | None, symbol: str | None = None, exchange: str | None = None) -> str:
    """Return ``equity``, ``etf`` or ``mf``."""
    isin_u = (isin or "").strip().upper()
    sym = (symbol or "").strip().upper()
    ex = (exchange or "").strip().upper()

    if ex in {"MF", "MFSS"}:
        return "mf"

    etf_name = any(tok in sym for tok in ETF_TOKENS)
    fund_isin = isin_u.startswith("INF")

    if fund_isin or etf_name:
        if ex in {"NSE", "BSE", "NFO", ""}:
            return "etf"
        return "mf"

    return "equity"


def is_etf(isin: str | None, symbol: str | None = None, exchange: str | None = None) -> bool:
    return classify(isin, symbol, exchange) == "etf"


def is_mf(isin: str | None, symbol: str | None = None, exchange: str | None = None) -> bool:
    return classify(isin, symbol, exchange) == "mf"
