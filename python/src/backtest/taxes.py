"""Indian capital gains tax with lot-level FIFO accounting.

Tax is a first-order drag for an actively rebalanced Indian portfolio, and it
is *path dependent* - which specific lots you sell determines what you owe. So
we track individual lots rather than applying an average cost basis.

Rules modelled (all rates configurable, since they change):
  * STCG - holding <= 12 months
  * LTCG - holding > 12 months, with an annual exemption
  * FIFO matching, as required for Indian equities
  * dividends taxed at slab

Tax-aware rebalancing then uses this to defer sales that are close to
crossing into long-term treatment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import AppConfig


@dataclass
class Lot:
    """One purchase lot."""

    isin: str
    quantity: float
    cost_per_share: float
    purchase_date: pd.Timestamp

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.cost_per_share

    def holding_days(self, on: pd.Timestamp) -> int:
        return int((pd.Timestamp(on) - self.purchase_date).days)


@dataclass
class RealizedGain:
    isin: str
    quantity: float
    proceeds: float
    cost_basis: float
    purchase_date: pd.Timestamp
    sale_date: pd.Timestamp
    is_long_term: bool

    @property
    def gain(self) -> float:
        return self.proceeds - self.cost_basis


@dataclass
class TaxConfig:
    enabled: bool = True
    stcg_rate: float = 0.20
    ltcg_rate: float = 0.125
    ltcg_exemption_per_year: float = 125_000.0
    ltcg_holding_days: int = 365
    dividend_rate: float = 0.30
    tax_aware_rebalance: bool = True

    @classmethod
    def from_config(cls, cfg: AppConfig) -> TaxConfig:
        t = cfg.taxes or {}
        return cls(
            enabled=bool(t.get("enabled", True)),
            stcg_rate=float(t.get("stcg_rate", 0.20)),
            ltcg_rate=float(t.get("ltcg_rate", 0.125)),
            ltcg_exemption_per_year=float(t.get("ltcg_exemption_per_year", 125_000.0)),
            ltcg_holding_days=int(t.get("ltcg_holding_days", 365)),
            dividend_rate=float(t.get("dividend_rate", 0.30)),
            tax_aware_rebalance=bool(t.get("tax_aware_rebalance", True)),
        )


class TaxLotBook:
    """FIFO lot ledger + running tax liability."""

    def __init__(self, config: TaxConfig):
        self.config = config
        self.lots: dict[str, list[Lot]] = {}
        self.realized: list[RealizedGain] = []
        self.dividend_income: float = 0.0

    # -- positions -------------------------------------------------------------
    def buy(self, isin: str, quantity: float, price: float, date: pd.Timestamp) -> None:
        if quantity <= 0:
            return
        self.lots.setdefault(isin, []).append(
            Lot(isin, quantity, price, pd.Timestamp(date))
        )

    def sell(
        self, isin: str, quantity: float, price: float, date: pd.Timestamp
    ) -> list[RealizedGain]:
        """Sell FIFO; returns the realized gains created."""
        if quantity <= 0 or isin not in self.lots:
            return []

        remaining = quantity
        date = pd.Timestamp(date)
        gains: list[RealizedGain] = []
        lots = self.lots[isin]

        while remaining > 1e-9 and lots:
            lot = lots[0]
            take = min(remaining, lot.quantity)
            holding = lot.holding_days(date)

            gains.append(RealizedGain(
                isin=isin, quantity=take, proceeds=take * price,
                cost_basis=take * lot.cost_per_share,
                purchase_date=lot.purchase_date, sale_date=date,
                is_long_term=holding > self.config.ltcg_holding_days,
            ))

            lot.quantity -= take
            remaining -= take
            if lot.quantity <= 1e-9:
                lots.pop(0)

        if not lots:
            self.lots.pop(isin, None)
        self.realized.extend(gains)
        return gains

    def quantity(self, isin: str) -> float:
        return sum(l.quantity for l in self.lots.get(isin, []))

    def cost_basis(self, isin: str) -> float:
        return sum(l.cost_basis for l in self.lots.get(isin, []))

    def positions(self) -> dict[str, float]:
        return {k: self.quantity(k) for k in self.lots}

    def add_dividend(self, amount: float) -> None:
        self.dividend_income += amount

    # -- tax -------------------------------------------------------------------
    def tax_for_year(self, year: int) -> dict[str, float]:
        """Tax owed for one financial year (Apr-Mar)."""
        if not self.config.enabled:
            return {"stcg_tax": 0.0, "ltcg_tax": 0.0, "dividend_tax": 0.0, "total": 0.0}

        in_year = [g for g in self.realized if _fy(g.sale_date) == year]
        stcg = sum(g.gain for g in in_year if not g.is_long_term)
        ltcg = sum(g.gain for g in in_year if g.is_long_term)

        stcg_tax = max(0.0, stcg) * self.config.stcg_rate
        taxable_ltcg = max(0.0, ltcg - self.config.ltcg_exemption_per_year)
        ltcg_tax = taxable_ltcg * self.config.ltcg_rate

        return {
            "stcg_gain": stcg, "ltcg_gain": ltcg,
            "stcg_tax": stcg_tax, "ltcg_tax": ltcg_tax,
            "dividend_tax": 0.0,
            "total": stcg_tax + ltcg_tax,
        }

    def total_tax(self) -> dict[str, float]:
        years = {_fy(g.sale_date) for g in self.realized}
        out = {"stcg_tax": 0.0, "ltcg_tax": 0.0, "dividend_tax": 0.0, "total": 0.0}
        for y in years:
            t = self.tax_for_year(y)
            for k in out:
                out[k] += t.get(k, 0.0)
        div_tax = self.dividend_income * self.config.dividend_rate
        out["dividend_tax"] += div_tax
        out["total"] += div_tax
        return out

    def realized_frame(self) -> pd.DataFrame:
        if not self.realized:
            return pd.DataFrame(columns=[
                "isin", "quantity", "proceeds", "cost_basis", "gain",
                "purchase_date", "sale_date", "is_long_term", "holding_days",
            ])
        return pd.DataFrame([{
            "isin": g.isin, "quantity": g.quantity, "proceeds": g.proceeds,
            "cost_basis": g.cost_basis, "gain": g.gain,
            "purchase_date": g.purchase_date, "sale_date": g.sale_date,
            "is_long_term": g.is_long_term,
            "holding_days": (g.sale_date - g.purchase_date).days,
        } for g in self.realized])


def _fy(date: pd.Timestamp) -> int:
    """Indian financial year (April-March), labelled by its starting year."""
    d = pd.Timestamp(date)
    return d.year if d.month >= 4 else d.year - 1


# --------------------------------------------------------------------------- #
# Tax-aware rebalancing
# --------------------------------------------------------------------------- #
def tax_aware_sell_priority(
    book: TaxLotBook, candidates: dict[str, float], prices: dict[str, float],
    date: pd.Timestamp, defer_window_days: int = 45,
) -> dict[str, float]:
    """Reduce (not forbid) sales that are about to become long-term.

    A position 20 days short of the LTCG threshold is usually worth holding:
    deferring converts a 20% STCG bill into a 12.5% LTCG bill. We scale the
    intended sale down rather than blocking it, so the portfolio still tracks
    its target.
    """
    adjusted: dict[str, float] = {}
    date = pd.Timestamp(date)

    for isin, sell_value in candidates.items():
        if sell_value <= 0 or isin not in book.lots:
            adjusted[isin] = sell_value
            continue

        price = prices.get(isin, 0.0)
        if price <= 0:
            adjusted[isin] = sell_value
            continue

        near_lt_qty = 0.0
        for lot in book.lots[isin]:
            days = lot.holding_days(date)
            crossing_soon = (
                book.config.ltcg_holding_days - defer_window_days
                < days <= book.config.ltcg_holding_days
            )
            # Only worth deferring if the lot is actually in profit.
            if crossing_soon and price > lot.cost_per_share:
                near_lt_qty += lot.quantity

        near_lt_value = near_lt_qty * price
        if near_lt_value <= 0:
            adjusted[isin] = sell_value
        else:
            adjusted[isin] = max(0.0, sell_value - near_lt_value)

    return adjusted


def harvest_losses(
    book: TaxLotBook, prices: dict[str, float], date: pd.Timestamp,
    min_loss_pct: float = 0.10,
) -> dict[str, float]:
    """Positions sitting on material unrealized losses worth harvesting."""
    out: dict[str, float] = {}
    for isin, lots in book.lots.items():
        price = prices.get(isin, 0.0)
        if price <= 0:
            continue
        loss = sum((price - l.cost_per_share) * l.quantity for l in lots
                   if price < l.cost_per_share * (1 - min_loss_pct))
        if loss < 0:
            out[isin] = abs(loss)
    return out


def after_tax_equity_curve(
    pre_tax_equity: pd.Series, book: TaxLotBook
) -> pd.Series:
    """Subtract each financial year's tax bill at year end (March 31)."""
    if not book.config.enabled or pre_tax_equity.empty:
        return pre_tax_equity.copy()

    after = pre_tax_equity.copy()
    cumulative = 0.0
    for year in sorted({_fy(g.sale_date) for g in book.realized}):
        tax = book.tax_for_year(year)["total"]
        if tax <= 0:
            continue
        cumulative += tax
        pay_date = pd.Timestamp(f"{year + 1}-03-31")
        after.loc[after.index >= pay_date] -= cumulative
    return after
