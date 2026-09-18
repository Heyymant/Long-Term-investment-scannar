"""Indian equity transaction costs (Zerodha delivery).

Getting this wrong is how paper strategies die in production. The components:

  brokerage      - zero for delivery at Zerodha (capped elsewhere)
  STT            - 0.1% on BOTH buy and sell for delivery
  exchange txn   - NSE charge on turnover
  SEBI           - tiny, on turnover
  stamp duty     - BUY side only
  GST            - 18% on (brokerage + exchange + SEBI)
  DP charges     - flat per scrip per day on SELLS; material for small books
  slippage       - baseline spread cost
  market impact  - scales with order size vs ADV (square-root law)

Verify the rate card before live use; these are configurable in config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AppConfig


@dataclass
class CostModel:
    brokerage_pct: float = 0.0
    brokerage_max_per_order: float = 20.0
    stt_buy_pct: float = 0.001
    stt_sell_pct: float = 0.001
    exchange_txn_pct: float = 0.0000297
    sebi_charges_pct: float = 0.000001
    stamp_duty_buy_pct: float = 0.00015
    gst_pct: float = 0.18
    dp_charge_per_scrip_sell: float = 15.34
    slippage_bps: float = 5.0
    impact_model: str = "sqrt"        # none | linear | sqrt
    impact_coefficient: float = 0.1

    @classmethod
    def from_config(cls, cfg: AppConfig) -> CostModel:
        c = cfg.costs or {}
        return cls(
            brokerage_pct=float(c.get("brokerage_pct", 0.0)),
            brokerage_max_per_order=float(c.get("brokerage_max_per_order", 20.0)),
            stt_buy_pct=float(c.get("stt_buy_pct", 0.001)),
            stt_sell_pct=float(c.get("stt_sell_pct", 0.001)),
            exchange_txn_pct=float(c.get("exchange_txn_pct", 0.0000297)),
            sebi_charges_pct=float(c.get("sebi_charges_pct", 0.000001)),
            stamp_duty_buy_pct=float(c.get("stamp_duty_buy_pct", 0.00015)),
            gst_pct=float(c.get("gst_pct", 0.18)),
            dp_charge_per_scrip_sell=float(c.get("dp_charge_per_scrip_sell", 15.34)),
            slippage_bps=float(c.get("slippage_bps", 5.0)),
            impact_model=str(c.get("impact_model", "sqrt")),
            impact_coefficient=float(c.get("impact_coefficient", 0.1)),
        )

    # -- components ------------------------------------------------------------
    def statutory_charges(self, turnover: float, is_buy: bool) -> dict[str, float]:
        """Regulatory + exchange charges on one order's turnover."""
        if turnover <= 0:
            return {"brokerage": 0.0, "stt": 0.0, "exchange": 0.0, "sebi": 0.0,
                    "stamp": 0.0, "gst": 0.0, "total": 0.0}

        brokerage = min(turnover * self.brokerage_pct, self.brokerage_max_per_order)
        stt = turnover * (self.stt_buy_pct if is_buy else self.stt_sell_pct)
        exchange = turnover * self.exchange_txn_pct
        sebi = turnover * self.sebi_charges_pct
        stamp = turnover * self.stamp_duty_buy_pct if is_buy else 0.0
        gst = (brokerage + exchange + sebi) * self.gst_pct

        total = brokerage + stt + exchange + sebi + stamp + gst
        return {"brokerage": brokerage, "stt": stt, "exchange": exchange,
                "sebi": sebi, "stamp": stamp, "gst": gst, "total": total}

    def slippage_cost(self, turnover: float, adv: float | None = None) -> float:
        """Spread cost plus size-dependent market impact.

        Square-root impact is the standard empirical form: pushing 10% of ADV
        costs far more than 1%, and ignoring that makes big backtests lie.
        """
        if turnover <= 0:
            return 0.0

        base = turnover * (self.slippage_bps / 10_000.0)
        if self.impact_model == "none" or not adv or adv <= 0:
            return base

        participation = turnover / adv
        if self.impact_model == "linear":
            impact_bps = self.impact_coefficient * participation * 10_000.0
        else:  # sqrt
            impact_bps = self.impact_coefficient * np.sqrt(participation) * 10_000.0

        return base + turnover * (impact_bps / 10_000.0)

    # -- aggregate -------------------------------------------------------------
    def trade_cost(
        self, turnover: float, is_buy: bool, adv: float | None = None,
        new_scrip_sale: bool = False,
    ) -> dict[str, float]:
        """Total cost of a single trade, itemized."""
        charges = self.statutory_charges(turnover, is_buy)
        slippage = self.slippage_cost(turnover, adv)
        dp = self.dp_charge_per_scrip_sell if (not is_buy and new_scrip_sale) else 0.0

        out = dict(charges)
        out["slippage"] = slippage
        out["dp"] = dp
        out["total"] = charges["total"] + slippage + dp
        out["bps"] = (out["total"] / turnover * 10_000.0) if turnover > 0 else 0.0
        return out

    def rebalance_cost(
        self,
        trades: pd.Series,                 # signed INR value per isin (+buy, -sell)
        adv: pd.Series | None = None,
    ) -> dict[str, float]:
        """Total cost of a rebalance across many names."""
        if trades.empty:
            return {"total": 0.0, "buy_value": 0.0, "sell_value": 0.0, "bps": 0.0}

        totals = {"brokerage": 0.0, "stt": 0.0, "exchange": 0.0, "sebi": 0.0,
                  "stamp": 0.0, "gst": 0.0, "slippage": 0.0, "dp": 0.0, "total": 0.0}
        buy_value = sell_value = 0.0

        for isin, value in trades.items():
            if abs(value) < 1e-6:
                continue
            is_buy = value > 0
            turnover = abs(float(value))
            name_adv = float(adv[isin]) if (adv is not None and isin in adv.index) else None

            c = self.trade_cost(turnover, is_buy, name_adv, new_scrip_sale=not is_buy)
            for k in totals:
                totals[k] += c.get(k, 0.0)
            if is_buy:
                buy_value += turnover
            else:
                sell_value += turnover

        gross = buy_value + sell_value
        totals["buy_value"] = buy_value
        totals["sell_value"] = sell_value
        totals["bps"] = (totals["total"] / gross * 10_000.0) if gross > 0 else 0.0
        return totals


# --------------------------------------------------------------------------- #
# Capacity / breakeven analysis
# --------------------------------------------------------------------------- #
def breakeven_cost_bps(
    gross_annual_return: float, benchmark_return: float, annual_turnover: float
) -> float:
    """Round-trip cost (bps) at which the strategy's edge disappears.

    If breakeven is 40bps and realistic costs are 60bps, the edge is not real.
    """
    if annual_turnover <= 0:
        return np.inf
    excess = gross_annual_return - benchmark_return
    if excess <= 0:
        return 0.0
    return (excess / (annual_turnover * 2.0)) * 10_000.0


def capacity_estimate(
    weights: pd.Series, adv: pd.Series, max_participation: float = 0.10,
    rebalance_days: int = 5,
) -> dict[str, float]:
    """Largest AUM tradeable within participation limits.

    The binding constraint is the *smallest, least liquid* position, which is
    usually far below what a naive average-ADV calculation suggests.
    """
    if weights.empty or adv.empty:
        return {"capacity_inr": np.nan, "binding_isin": None}

    common = [i for i in weights.index if i in adv.index and weights[i] > 0]
    if not common:
        return {"capacity_inr": np.nan, "binding_isin": None}

    # weight * AUM <= adv * participation * days  =>  AUM <= adv*p*d / weight
    per_name = {
        i: float(adv[i]) * max_participation * rebalance_days / float(weights[i])
        for i in common
    }
    binding = min(per_name, key=per_name.get)
    return {
        "capacity_inr": per_name[binding],
        "capacity_cr": per_name[binding] / 1e7,
        "binding_isin": binding,
        "binding_weight": float(weights[binding]),
        "binding_adv": float(adv[binding]),
    }


def cost_sensitivity(
    base_returns: pd.Series, annual_turnover: float,
    cost_levels_bps: list[float] | None = None, periods_per_year: int = 252,
) -> pd.DataFrame:
    """Sharpe/CAGR as assumed costs rise - how fragile is the edge?"""
    levels = cost_levels_bps or [0, 10, 20, 30, 50, 75, 100, 150]
    rows = []
    for bps in levels:
        annual_drag = annual_turnover * 2.0 * (bps / 10_000.0)
        adjusted = base_returns - annual_drag / periods_per_year
        vol = float(adjusted.std() * np.sqrt(periods_per_year))
        cagr = float((1 + adjusted).prod() ** (periods_per_year / max(len(adjusted), 1)) - 1)
        rows.append({
            "cost_bps": bps,
            "annual_drag": round(annual_drag, 4),
            "cagr": round(cagr, 4),
            "sharpe": round(float(adjusted.mean() / adjusted.std() * np.sqrt(periods_per_year)), 3)
            if adjusted.std() > 0 else np.nan,
            "vol": round(vol, 4),
        })
    return pd.DataFrame(rows)
