"""Live vs backtest drift, realized slippage and after-tax realized P&L.

The backtest is a hypothesis. This module checks it against reality: if live
tracking error is large or realized slippage far exceeds the model, the
backtest's conclusions do not transfer and the design needs revisiting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import get_logger
from ..backtest.taxes import TaxConfig, TaxLotBook

log = get_logger(__name__)


@dataclass
class DriftReport:
    live_return: float = np.nan
    expected_return: float = np.nan
    tracking_error: float = np.nan
    implementation_shortfall: float = np.nan
    correlation: float = np.nan
    n_days: int = 0
    realized_slippage_bps: float = np.nan
    modelled_slippage_bps: float = np.nan
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: (None if isinstance(v, float) and np.isnan(v) else v)
             for k, v in self.__dict__.items()}
        return d


def build_live_equity_curve(
    trades: pd.DataFrame, prices: pd.DataFrame, starting_cash: float,
) -> tuple[pd.Series, TaxLotBook]:
    """Reconstruct the live portfolio value from actual fills.

    Also populates a tax lot book from real trades, so realized gains and the
    tax bill are computed on what you actually did.
    """
    if trades.empty or prices.empty:
        return pd.Series(dtype=float), TaxLotBook(TaxConfig())

    trades = trades.sort_values("trade_date").copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()

    book = TaxLotBook(TaxConfig())
    positions: dict[str, float] = {}
    cash = float(starting_cash)

    start = trades["trade_date"].min()
    dates = prices.index[prices.index >= start]
    equity: dict[pd.Timestamp, float] = {}

    by_date = dict(tuple(trades.groupby("trade_date")))

    for dt in dates:
        for _, t in by_date.get(dt, pd.DataFrame()).iterrows():
            isin = t.get("isin")
            if not isin or pd.isna(isin):
                continue
            qty, price = float(t["quantity"]), float(t["price"])
            if t["side"] == "BUY":
                cash -= qty * price
                positions[isin] = positions.get(isin, 0.0) + qty
                book.buy(isin, qty, price, dt)
            else:
                cash += qty * price
                positions[isin] = positions.get(isin, 0.0) - qty
                book.sell(isin, qty, price, dt)
                if positions[isin] <= 1e-9:
                    positions.pop(isin, None)

        row = prices.loc[dt]
        mv = sum(q * float(row.get(i, np.nan)) for i, q in positions.items()
                 if i in row.index and not np.isnan(row.get(i, np.nan)))
        equity[dt] = cash + mv

    return pd.Series(equity).sort_index(), book


def compare_to_backtest(
    live_equity: pd.Series,
    backtest_returns: pd.Series,
    tracking_error_threshold: float = 0.05,
) -> DriftReport:
    """Quantify how far live has drifted from the modelled expectation."""
    report = DriftReport()
    if live_equity.empty or len(live_equity) < 5:
        report.alerts.append("Not enough live history to assess drift")
        return report

    live_rets = live_equity.pct_change().dropna()
    expected = backtest_returns.reindex(live_rets.index).dropna()
    common = live_rets.index.intersection(expected.index)
    if len(common) < 5:
        report.alerts.append("Insufficient overlap between live and backtest periods")
        report.live_return = float((1 + live_rets).prod() - 1)
        report.n_days = len(live_rets)
        return report

    lr, er = live_rets.loc[common], expected.loc[common]
    active = lr - er

    report.live_return = float((1 + lr).prod() - 1)
    report.expected_return = float((1 + er).prod() - 1)
    report.implementation_shortfall = report.live_return - report.expected_return
    report.tracking_error = float(active.std() * np.sqrt(252))
    report.correlation = float(lr.corr(er))
    report.n_days = len(common)

    if report.tracking_error > tracking_error_threshold:
        report.alerts.append(
            f"Tracking error {report.tracking_error:.1%} exceeds "
            f"{tracking_error_threshold:.0%} - execution is diverging from the model"
        )
    if report.implementation_shortfall < -0.02:
        report.alerts.append(
            f"Implementation shortfall {report.implementation_shortfall:.1%} - "
            "live is materially behind the backtest"
        )
    if not np.isnan(report.correlation) and report.correlation < 0.8:
        report.alerts.append(
            f"Live/backtest correlation only {report.correlation:.2f} - "
            "the portfolio being held differs from the modelled one"
        )
    return report


def slippage_analysis(matched_trades: pd.DataFrame) -> dict:
    """Realized slippage from reconciled fills, split by side."""
    if matched_trades.empty or "slippage_bps" not in matched_trades.columns:
        return {"note": "no matched trades with slippage data"}

    s = matched_trades["slippage_bps"].dropna()
    if s.empty:
        return {"note": "no slippage measurements"}

    out = {
        "mean_bps": round(float(s.mean()), 2),
        "median_bps": round(float(s.median()), 2),
        "std_bps": round(float(s.std()), 2),
        "worst_bps": round(float(s.min()), 2),
        "best_bps": round(float(s.max()), 2),
        "n_trades": int(len(s)),
    }
    for side in ("BUY", "SELL"):
        sub = matched_trades[matched_trades["action"] == side]["slippage_bps"].dropna()
        if not sub.empty:
            out[f"{side.lower()}_mean_bps"] = round(float(sub.mean()), 2)
    return out


def after_tax_realized(book: TaxLotBook) -> dict:
    """Realized gains and tax accrual by financial year, from real fills."""
    realized = book.realized_frame()
    if realized.empty:
        return {"note": "no realized gains yet"}

    total = book.total_tax()
    st = realized[~realized["is_long_term"]]["gain"].sum()
    lt = realized[realized["is_long_term"]]["gain"].sum()

    return {
        "realized_gain_total": round(float(realized["gain"].sum()), 2),
        "short_term_gain": round(float(st), 2),
        "long_term_gain": round(float(lt), 2),
        "estimated_tax": round(float(total["total"]), 2),
        "stcg_tax": round(float(total["stcg_tax"]), 2),
        "ltcg_tax": round(float(total["ltcg_tax"]), 2),
        "n_realizations": int(len(realized)),
        "avg_holding_days": round(float(realized["holding_days"].mean()), 1),
    }
