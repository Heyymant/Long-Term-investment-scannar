"""Import and reconcile actual fills - closing the loop on manual execution.

Because execution is manual, the portfolio you hold will drift from the
portfolio the model asked for: orders get skipped, partially filled, or filled
at worse prices. Without importing the real tradebook, the live results are
unexplainable. This measures the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

TRADE_COLUMNS = [
    "trade_date", "symbol", "isin", "side", "quantity", "price", "value", "order_id",
]


def import_kite_trades(trades: pd.DataFrame, securities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize Kite's trades payload into our schema (read-only)."""
    if trades is None or trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    df = trades.copy()
    out = pd.DataFrame({
        "trade_date": pd.to_datetime(
            df.get("fill_timestamp", df.get("exchange_timestamp", df.get("trade_date"))),
            errors="coerce",
        ),
        "symbol": df.get("tradingsymbol", ""),
        "side": df.get("transaction_type", "").astype(str).str.upper(),
        "quantity": pd.to_numeric(df.get("quantity"), errors="coerce"),
        "price": pd.to_numeric(df.get("average_price", df.get("price")), errors="coerce"),
        "order_id": df.get("order_id", ""),
    })
    out["value"] = out["quantity"] * out["price"]

    out["isin"] = None
    if securities is not None and not securities.empty and "symbol" in securities.columns:
        sym_to_isin = dict(zip(securities["symbol"], securities["isin"]))
        out["isin"] = out["symbol"].map(sym_to_isin)

    return out.dropna(subset=["trade_date", "quantity", "price"])[TRADE_COLUMNS]


def load_trades(data_dir: str | Path) -> pd.DataFrame:
    p = Path(data_dir) / "trades.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=TRADE_COLUMNS)


def save_trades(trades: pd.DataFrame, data_dir: str | Path) -> Path:
    p = Path(data_dir) / "trades.parquet"
    existing = load_trades(data_dir)
    merged = pd.concat([existing, trades], ignore_index=True)
    if "order_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["order_id", "trade_date", "quantity"], keep="last")
    merged.to_parquet(p, index=False)
    return p


@dataclass
class ReconciliationResult:
    matched: pd.DataFrame = field(default_factory=pd.DataFrame)
    unfilled: pd.DataFrame = field(default_factory=pd.DataFrame)
    unplanned: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        frames = []
        for name, df in (("matched", self.matched), ("unfilled", self.unfilled),
                         ("unplanned", self.unplanned)):
            if df is not None and not df.empty:
                d = df.copy()
                d["reconciliation"] = name
                frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reconcile(
    planned_orders: pd.DataFrame,
    actual_trades: pd.DataFrame,
    tolerance_days: int = 5,
) -> ReconciliationResult:
    """Match the checklist against what was actually executed.

    Reports three buckets: filled as planned, planned but never filled, and
    trades taken that were not on the checklist (discretionary overrides).
    """
    if planned_orders is None or planned_orders.empty:
        return ReconciliationResult(unplanned=actual_trades,
                                    summary={"note": "no planned orders to reconcile"})
    if actual_trades is None or actual_trades.empty:
        return ReconciliationResult(unfilled=planned_orders,
                                    summary={"note": "no trades executed", "fill_rate": 0.0})

    planned = planned_orders[planned_orders["action"].isin(["BUY", "SELL"])].copy()
    actual = actual_trades.copy()
    actual["trade_date"] = pd.to_datetime(actual["trade_date"])
    actual["_used"] = False

    matched_rows, unfilled_rows = [], []

    for _, order in planned.iterrows():
        isin = order.get("isin")
        side = order["action"]
        candidates = actual[
            (actual["isin"] == isin) & (actual["side"] == side) & (~actual["_used"])
        ]
        if candidates.empty:
            unfilled_rows.append(order.to_dict())
            continue

        fill = candidates.iloc[0]
        actual.loc[fill.name, "_used"] = True

        planned_price = float(order.get("price") or np.nan)
        fill_price = float(fill["price"])
        # Slippage is signed against you: paying more on a buy is negative.
        slippage_bps = np.nan
        if planned_price and not np.isnan(planned_price) and planned_price > 0:
            raw = (fill_price - planned_price) / planned_price
            slippage_bps = float((-raw if side == "BUY" else raw) * 10_000)

        matched_rows.append({
            **order.to_dict(),
            "filled_qty": float(fill["quantity"]),
            "fill_price": fill_price,
            "fill_date": fill["trade_date"],
            "slippage_bps": slippage_bps,
            "fill_ratio": float(fill["quantity"]) / abs(float(order["delta_qty"]))
            if order.get("delta_qty") else np.nan,
        })

    unplanned = actual[~actual["_used"]].drop(columns=["_used"])
    matched = pd.DataFrame(matched_rows)
    unfilled = pd.DataFrame(unfilled_rows)

    fill_rate = len(matched) / len(planned) if len(planned) else 0.0
    summary = {
        "n_planned": int(len(planned)),
        "n_matched": int(len(matched)),
        "n_unfilled": int(len(unfilled)),
        "n_unplanned": int(len(unplanned)),
        "fill_rate": round(fill_rate, 4),
        "avg_slippage_bps": round(float(matched["slippage_bps"].mean()), 2)
        if not matched.empty and matched["slippage_bps"].notna().any() else None,
        "median_slippage_bps": round(float(matched["slippage_bps"].median()), 2)
        if not matched.empty and matched["slippage_bps"].notna().any() else None,
    }
    if fill_rate < 0.8:
        summary["warning"] = (
            f"Only {fill_rate:.0%} of planned orders were executed - live results "
            "will diverge from the backtest."
        )
    return ReconciliationResult(matched, unfilled, unplanned, summary)
