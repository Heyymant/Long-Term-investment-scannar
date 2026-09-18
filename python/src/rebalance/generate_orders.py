"""Rebalance checklist generator - the thing you actually act on.

Takes target weights, diffs them against your live Zerodha holdings, and
writes a `rebalance_orders.csv` you execute manually.

This module never places orders. It only reads holdings and writes a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

ORDER_COLUMNS = [
    "symbol", "isin", "action", "current_qty", "target_qty", "delta_qty",
    "price", "current_weight", "target_weight", "delta_value",
    "est_cost", "pct_of_adv", "notes",
]


@dataclass
class RebalancePlan:
    orders: pd.DataFrame
    summary: dict
    as_of: pd.Timestamp

    @property
    def n_buys(self) -> int:
        return int((self.orders["action"] == "BUY").sum()) if not self.orders.empty else 0

    @property
    def n_sells(self) -> int:
        return int((self.orders["action"] == "SELL").sum()) if not self.orders.empty else 0

    def actionable(self) -> pd.DataFrame:
        if self.orders.empty:
            return self.orders
        return self.orders[self.orders["action"].isin(["BUY", "SELL"])]


def generate_rebalance(
    target_weights: pd.Series,
    current_holdings: pd.DataFrame,
    prices: pd.Series,
    portfolio_value: float | None = None,
    securities: pd.DataFrame | None = None,
    adv: pd.Series | None = None,
    cost_model=None,
    min_order_value: float = 5000.0,
    tradeable_universe: set[str] | None = None,
    as_of: pd.Timestamp | None = None,
) -> RebalancePlan:
    """Build the manual execution checklist.

    `current_holdings` is Kite's holdings frame (needs isin/tradingsymbol and
    quantity). Orders smaller than `min_order_value` are suppressed - churning
    tiny amounts just donates money to costs.
    """
    as_of = pd.Timestamp(as_of or date.today())

    current_qty = _current_quantities(current_holdings)
    symbol_map = _symbol_map(securities, current_holdings)

    # Current portfolio value from live holdings unless told otherwise.
    if portfolio_value is None:
        holdings_value = sum(
            qty * float(prices.get(isin, np.nan))
            for isin, qty in current_qty.items()
            if isin in prices.index and not np.isnan(prices.get(isin, np.nan))
        )
        cash = _cash_from_holdings(current_holdings)
        portfolio_value = holdings_value + cash

    if portfolio_value <= 0:
        log.warning("Portfolio value is zero or unknown; cannot size orders")
        return RebalancePlan(pd.DataFrame(columns=ORDER_COLUMNS), {"error": "zero portfolio value"}, as_of)

    if tradeable_universe is not None:
        dropped = [i for i in target_weights.index if i not in tradeable_universe]
        if dropped:
            log.info("Dropping %d target names outside the tradeable universe", len(dropped))
        target_weights = target_weights[target_weights.index.isin(tradeable_universe)]

    rows = []
    all_isins = set(target_weights.index) | set(current_qty)

    for isin in sorted(all_isins):
        price = float(prices.get(isin, np.nan))
        if np.isnan(price) or price <= 0:
            rows.append(_row(isin, symbol_map, 0, 0, np.nan, 0.0, 0.0,
                             note="no price available - review manually"))
            continue

        cur_q = float(current_qty.get(isin, 0.0))
        tgt_w = float(target_weights.get(isin, 0.0))
        cur_w = cur_q * price / portfolio_value
        tgt_q = np.floor(tgt_w * portfolio_value / price)
        delta_q = tgt_q - cur_q
        delta_value = delta_q * price

        if abs(delta_value) < min_order_value:
            action, delta_q, delta_value = "HOLD", 0.0, 0.0
            tgt_q = cur_q
        else:
            action = "BUY" if delta_q > 0 else "SELL"

        est_cost = 0.0
        if cost_model is not None and action != "HOLD":
            name_adv = float(adv[isin]) if (adv is not None and isin in adv.index) else None
            est_cost = cost_model.trade_cost(
                abs(delta_value), action == "BUY", name_adv, new_scrip_sale=(action == "SELL")
            )["total"]

        pct_adv = np.nan
        note = ""
        if adv is not None and isin in adv.index and adv[isin] > 0:
            pct_adv = abs(delta_value) / float(adv[isin])
            if pct_adv > 0.10:
                note = f"large order: {pct_adv:.0%} of ADV - consider splitting over days"

        rows.append(_row(isin, symbol_map, cur_q, tgt_q, price, cur_w, tgt_w,
                         action=action, delta_q=delta_q, delta_value=delta_value,
                         est_cost=est_cost, pct_adv=pct_adv, note=note))

    orders = pd.DataFrame(rows, columns=ORDER_COLUMNS)
    if not orders.empty:
        order_rank = {"SELL": 0, "BUY": 1, "HOLD": 2}
        orders = orders.sort_values(
            by=["action", "delta_value"],
            key=lambda c: c.map(order_rank) if c.name == "action" else c.abs(),
            ascending=[True, False],
        ).reset_index(drop=True)

    actionable = orders[orders["action"].isin(["BUY", "SELL"])] if not orders.empty else orders
    summary = {
        "as_of": str(as_of.date()),
        "portfolio_value": round(portfolio_value, 2),
        "n_orders": int(len(actionable)),
        "n_buys": int((orders["action"] == "BUY").sum()) if not orders.empty else 0,
        "n_sells": int((orders["action"] == "SELL").sum()) if not orders.empty else 0,
        "n_holds": int((orders["action"] == "HOLD").sum()) if not orders.empty else 0,
        "buy_value": float(actionable.loc[actionable["action"] == "BUY", "delta_value"].sum())
        if not actionable.empty else 0.0,
        "sell_value": float(-actionable.loc[actionable["action"] == "SELL", "delta_value"].sum())
        if not actionable.empty else 0.0,
        "est_total_cost": float(actionable["est_cost"].sum()) if not actionable.empty else 0.0,
        "turnover_pct": float(actionable["delta_value"].abs().sum() / portfolio_value * 100)
        if not actionable.empty else 0.0,
        "note": "Decision support only - no orders are placed. Execute manually.",
    }
    return RebalancePlan(orders, summary, as_of)


def _row(
    isin: str, symbol_map: dict, cur_q: float, tgt_q: float, price: float,
    cur_w: float, tgt_w: float, action: str = "HOLD", delta_q: float = 0.0,
    delta_value: float = 0.0, est_cost: float = 0.0, pct_adv: float = np.nan,
    note: str = "",
) -> dict:
    return {
        "symbol": symbol_map.get(isin, isin),
        "isin": isin,
        "action": action,
        "current_qty": int(cur_q),
        "target_qty": int(tgt_q) if not np.isnan(tgt_q) else 0,
        "delta_qty": int(delta_q),
        "price": round(price, 2) if not np.isnan(price) else None,
        "current_weight": round(cur_w, 4),
        "target_weight": round(tgt_w, 4),
        "delta_value": round(delta_value, 2),
        "est_cost": round(est_cost, 2),
        "pct_of_adv": round(pct_adv, 4) if not np.isnan(pct_adv) else None,
        "notes": note,
    }


def _current_quantities(holdings: pd.DataFrame) -> dict[str, float]:
    if holdings is None or holdings.empty:
        return {}
    qty_col = next((c for c in ("quantity", "qty", "net_quantity") if c in holdings.columns), None)
    isin_col = next((c for c in ("isin", "ISIN") if c in holdings.columns), None)
    if qty_col is None or isin_col is None:
        log.warning("Holdings frame lacks isin/quantity columns; treating as empty")
        return {}
    out: dict[str, float] = {}
    for _, r in holdings.iterrows():
        isin = str(r[isin_col])
        out[isin] = out.get(isin, 0.0) + float(r[qty_col] or 0)
    return {k: v for k, v in out.items() if v > 0}


def _symbol_map(securities: pd.DataFrame | None, holdings: pd.DataFrame | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if securities is not None and not securities.empty and "isin" in securities.columns:
        out.update(dict(zip(securities["isin"], securities.get("symbol", securities["isin"]))))
    if holdings is not None and not holdings.empty:
        isin_col = next((c for c in ("isin", "ISIN") if c in holdings.columns), None)
        sym_col = next((c for c in ("tradingsymbol", "symbol") if c in holdings.columns), None)
        if isin_col and sym_col:
            out.update(dict(zip(holdings[isin_col], holdings[sym_col])))
    return out


def _cash_from_holdings(holdings: pd.DataFrame | None) -> float:
    if holdings is None or holdings.empty:
        return 0.0
    for col in ("cash", "available_cash"):
        if col in holdings.columns:
            try:
                return float(holdings[col].iloc[0])
            except (ValueError, TypeError, IndexError):
                return 0.0
    return 0.0
