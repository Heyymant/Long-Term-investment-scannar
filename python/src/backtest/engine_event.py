"""Event-driven backtest engine - the source of truth for a final design.

Slower than the vectorized engine but materially more realistic:

  * integer share quantities (you cannot buy 3.7 shares)
  * explicit cash balance; trades are funded, not assumed
  * per-order costs with DP charges applied per scrip
  * lot-level FIFO tax tracking, so after-tax results are path-correct
  * tax-aware rebalancing can defer sales nearing the LTCG threshold
  * t+1 execution - signals computed on the close of t trade on t+1

Used to validate the chosen configuration after the vectorized engine has
done the searching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import get_logger
from ..factors.composite import compute_composite
from ..portfolio.construction import construct_portfolio, rebalance_dates
from ..portfolio.regime import compute_exposure_series, exposure_on
from ..risk.portfolio_risk import apply_risk_limits
from ..schema import StrategyConfig
from .costs import CostModel
from .engine_vectorized import BacktestResult
from .taxes import TaxConfig, TaxLotBook, after_tax_equity_curve, tax_aware_sell_priority

log = get_logger(__name__)


@dataclass
class Fill:
    date: pd.Timestamp
    isin: str
    side: str            # BUY | SELL
    quantity: float
    price: float
    value: float
    cost: float


@dataclass
class EventState:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)   # isin -> shares

    def market_value(self, prices: pd.Series) -> float:
        total = 0.0
        for isin, qty in self.positions.items():
            px = prices.get(isin)
            if px is not None and not np.isnan(px):
                total += qty * px
        return total

    def equity(self, prices: pd.Series) -> float:
        return self.cash + self.market_value(prices)


def run_event_backtest(
    config: StrategyConfig,
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    sectors: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    adv: pd.DataFrame | None = None,
    cost_model: CostModel | None = None,
    tax_config: TaxConfig | None = None,
    sleeve_signals: dict[pd.Timestamp, pd.Series] | None = None,
) -> BacktestResult:
    """Replay a design order-by-order with cash, shares and tax lots."""
    np.random.seed(config.seed)

    cost_model = cost_model or CostModel()
    tax_config = tax_config or TaxConfig(enabled=config.apply_taxes)
    book = TaxLotBook(tax_config)

    sleeve_a = config.sleeve_a
    exposure_series = (
        compute_exposure_series(benchmark, sleeve_a.regime)
        if benchmark is not None and not benchmark.empty
        else pd.Series(1.0, index=prices.index)
    )

    warmup = max(sleeve_a.factors.low_vol_lookback_days, 252) + 21
    rebals = set(rebalance_dates(prices.index, sleeve_a.rebalance.value, warmup))
    if not rebals:
        return BacktestResult(returns=pd.Series(dtype=float), equity_curve=pd.Series(dtype=float),
                              engine="event")

    state = EventState(cash=float(config.window.initial_capital))
    equity_values: dict[pd.Timestamp, float] = {}
    fills: list[Fill] = []
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    turnover_history: dict[pd.Timestamp, float] = {}
    cost_rows: list[dict] = []

    pending_target: pd.Series | None = None    # decided at t, executed at t+1
    derisked = False
    start_date = min(rebals)
    dates = prices.index[prices.index >= start_date]

    for dt in dates:
        px_today = prices.loc[dt]

        # --- 1) execute yesterday's decision (t+1 execution) ----------------
        if pending_target is not None:
            equity_now = state.equity(px_today)
            new_fills, costs = _execute(
                state, book, pending_target, px_today, dt, equity_now,
                cost_model, tax_config, _adv_on(adv, dt), config.apply_costs,
            )
            fills.extend(new_fills)
            if costs["total"] > 0:
                cost_rows.append({"date": dt, **costs})
            pending_target = None

        equity_values[dt] = state.equity(px_today)

        # --- 2) decide target weights on rebalance dates --------------------
        if dt in rebals:
            if sleeve_signals is not None:
                target = sleeve_signals.get(dt, pd.Series(dtype=float))
            else:
                universe = _eligible(eligibility, dt)
                if not universe:
                    continue
                scores = compute_composite(
                    prices, dt, sleeve_a.factors, universe, fundamentals, sectors,
                    sleeve_a.sector_neutralize,
                ).composite
                if scores.empty:
                    continue
                exposure = exposure_on(exposure_series, dt, 1.0)
                held = {k for k, v in state.positions.items() if v > 0}
                target = construct_portfolio(
                    scores, prices, sleeve_a, dt, held, sectors, exposure
                ).weights

            equity_now = equity_values[dt]
            recent = pd.Series(equity_values).pct_change().dropna()
            target, risk_state = apply_risk_limits(
                target, config.risk, sectors,
                portfolio_value=equity_now, adv=_adv_on(adv, dt),
                equity_curve=pd.Series(equity_values),
                returns=recent.iloc[-config.risk.vol_lookback_days:],
                currently_derisked=derisked, date=dt,
            )
            derisked = risk_state.exposure_multiplier < 1.0

            current_w = _current_weights(state, px_today, equity_now)
            turnover_history[dt] = _turnover(current_w, target)
            weights_history[dt] = target
            pending_target = target

    equity_curve = pd.Series(equity_values).sort_index()
    returns = equity_curve.pct_change().fillna(0.0)
    after_tax = after_tax_equity_curve(equity_curve, book)
    total_costs = float(sum(r.get("total", 0.0) for r in cost_rows))

    return BacktestResult(
        returns=returns,
        gross_returns=returns,
        equity_curve=equity_curve,
        after_tax_equity=after_tax,
        weights_history=pd.DataFrame(weights_history).T.fillna(0.0) if weights_history else pd.DataFrame(),
        turnover_history=pd.Series(turnover_history),
        cost_history=pd.DataFrame(cost_rows),
        exposure_history=exposure_series.reindex(equity_curve.index).ffill(),
        rebalance_dates=sorted(rebals),
        total_costs=total_costs,
        total_tax=float(book.total_tax()["total"]),
        engine="event",
        diagnostics={
            "n_fills": len(fills),
            "n_rebalances": len(weights_history),
            "final_cash": state.cash,
            "final_positions": len(state.positions),
            "realized_gains": len(book.realized),
        },
    )


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _execute(
    state: EventState,
    book: TaxLotBook,
    target: pd.Series,
    prices: pd.Series,
    dt: pd.Timestamp,
    equity: float,
    cost_model: CostModel,
    tax_config: TaxConfig,
    adv: pd.Series | None,
    apply_costs: bool,
) -> tuple[list[Fill], dict[str, float]]:
    """Turn target weights into integer-share orders; sells first to fund buys."""
    fills: list[Fill] = []
    totals = {"brokerage": 0.0, "stt": 0.0, "exchange": 0.0, "sebi": 0.0, "stamp": 0.0,
              "gst": 0.0, "slippage": 0.0, "dp": 0.0, "total": 0.0}

    if equity <= 0:
        return fills, totals

    target_value = {i: float(w) * equity for i, w in target.items() if w > 0}
    all_isins = set(target_value) | set(state.positions)

    sells: dict[str, float] = {}
    buys: dict[str, float] = {}
    for isin in all_isins:
        px = prices.get(isin)
        if px is None or np.isnan(px) or px <= 0:
            continue
        current_value = state.positions.get(isin, 0.0) * px
        desired = target_value.get(isin, 0.0)
        diff = desired - current_value
        if diff < -1.0:
            sells[isin] = -diff
        elif diff > 1.0:
            buys[isin] = diff

    # Defer sales about to qualify for LTCG.
    if tax_config.enabled and tax_config.tax_aware_rebalance and sells:
        price_map = {i: float(prices[i]) for i in sells if i in prices.index and not np.isnan(prices[i])}
        sells = tax_aware_sell_priority(book, sells, price_map, dt)

    # SELLS first - they fund the buys.
    for isin, value in sells.items():
        px = float(prices[isin])
        held = state.positions.get(isin, 0.0)
        qty = min(held, np.floor(value / px))
        if qty <= 0:
            continue
        proceeds = qty * px
        cost = _charge(cost_model, proceeds, False, adv, isin, apply_costs, totals)

        state.positions[isin] = held - qty
        if state.positions[isin] <= 1e-9:
            state.positions.pop(isin, None)
        state.cash += proceeds - cost
        book.sell(isin, qty, px, dt)
        fills.append(Fill(dt, isin, "SELL", qty, px, proceeds, cost))

    # BUYS, scaled down if cash is short.
    total_buy = sum(buys.values())
    available = max(0.0, state.cash * 0.995)   # small buffer for costs
    scale = min(1.0, available / total_buy) if total_buy > 0 else 0.0

    for isin, value in buys.items():
        px = float(prices[isin])
        qty = np.floor(value * scale / px)
        if qty <= 0:
            continue
        gross = qty * px
        cost = _charge(cost_model, gross, True, adv, isin, apply_costs, totals)
        if gross + cost > state.cash:
            continue
        state.positions[isin] = state.positions.get(isin, 0.0) + qty
        state.cash -= gross + cost
        book.buy(isin, qty, px, dt)
        fills.append(Fill(dt, isin, "BUY", qty, px, gross, cost))

    return fills, totals


def _charge(
    cost_model: CostModel, turnover: float, is_buy: bool, adv: pd.Series | None,
    isin: str, apply_costs: bool, totals: dict[str, float],
) -> float:
    if not apply_costs:
        return 0.0
    name_adv = float(adv[isin]) if (adv is not None and isin in adv.index) else None
    c = cost_model.trade_cost(turnover, is_buy, name_adv, new_scrip_sale=not is_buy)
    for k in totals:
        totals[k] += c.get(k, 0.0)
    return float(c["total"])


def _current_weights(state: EventState, prices: pd.Series, equity: float) -> pd.Series:
    if equity <= 0 or not state.positions:
        return pd.Series(dtype=float)
    vals = {}
    for isin, qty in state.positions.items():
        px = prices.get(isin)
        if px is not None and not np.isnan(px):
            vals[isin] = qty * px / equity
    return pd.Series(vals)


def _turnover(previous: pd.Series, current: pd.Series) -> float:
    if previous.empty:
        return float(current.sum())
    idx = previous.index.union(current.index)
    p = previous.reindex(idx).fillna(0.0)
    c = current.reindex(idx).fillna(0.0)
    return float((c - p).abs().sum() / 2.0)


def _eligible(eligibility: pd.DataFrame, dt: pd.Timestamp) -> list[str]:
    from ..data.universe import eligible_on

    return eligible_on(eligibility, dt)


def _adv_on(adv: pd.DataFrame | None, dt: pd.Timestamp) -> pd.Series | None:
    if adv is None or adv.empty:
        return None
    prior = adv.loc[adv.index <= dt]
    return None if prior.empty else prior.iloc[-1]
