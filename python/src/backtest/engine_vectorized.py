"""Fast vectorized backtest engine (for parameter sweeps and walk-forward).

Holds weights constant between rebalances and computes returns by matrix
multiplication. Costs are charged on turnover at each rebalance; taxes are
approximated at the portfolio level rather than tracked lot-by-lot.

Trades off exactness for speed - thousands of these run during optimization.
The event-driven engine (`engine_event.py`) is the source of truth for the
final chosen design, and `consistency.py` checks they agree.

Look-ahead safety: weights decided on date t are applied to returns from t+1
onward. That single shift is the difference between honest and fantasy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import get_logger
from ..factors.composite import compute_composite
from ..portfolio.construction import Portfolio, construct_portfolio, rebalance_dates
from ..portfolio.regime import compute_exposure_series, exposure_on
from ..risk.portfolio_risk import RiskState, apply_risk_limits
from ..schema import StrategyConfig
from .costs import CostModel

log = get_logger(__name__)


@dataclass
class BacktestResult:
    """Everything a single backtest produces."""

    returns: pd.Series                                   # net daily returns
    equity_curve: pd.Series
    gross_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    after_tax_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover_history: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cost_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    exposure_history: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    risk_states: list[RiskState] = field(default_factory=list)
    total_costs: float = 0.0
    total_tax: float = 0.0
    engine: str = "vectorized"
    diagnostics: dict = field(default_factory=dict)

    @property
    def annual_turnover(self) -> float:
        if self.turnover_history.empty or self.equity_curve.empty:
            return 0.0
        years = len(self.equity_curve) / 252
        return float(self.turnover_history.sum() / years) if years > 0 else 0.0


def run_vectorized_backtest(
    config: StrategyConfig,
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    sectors: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    adv: pd.DataFrame | None = None,
    cost_model: CostModel | None = None,
    sleeve_signals: dict[pd.Timestamp, pd.Series] | None = None,
) -> BacktestResult:
    """Replay a strategy design over history.

    `sleeve_signals` optionally injects externally computed target weights
    (used by Sleeves B and C, which have their own signal logic).
    """
    np.random.seed(config.seed)

    returns_panel = prices.pct_change().fillna(0.0)
    sleeve_a = config.sleeve_a

    # Regime overlay exposure series.
    if benchmark is not None and not benchmark.empty:
        exposure_series = compute_exposure_series(benchmark, sleeve_a.regime)
    else:
        exposure_series = pd.Series(1.0, index=prices.index)

    warmup = max(sleeve_a.factors.low_vol_lookback_days, 252) + 21
    rebals = rebalance_dates(prices.index, sleeve_a.rebalance.value, warmup)
    if len(rebals) == 0:
        log.warning("No rebalance dates available (need > %d days of history)", warmup)
        return BacktestResult(returns=pd.Series(dtype=float), equity_curve=pd.Series(dtype=float))

    weights_history: dict[pd.Timestamp, pd.Series] = {}
    turnover_history: dict[pd.Timestamp, float] = {}
    cost_rows: list[dict] = []
    risk_states: list[RiskState] = []

    current_weights = pd.Series(dtype=float)
    current_holdings: set[str] = set()
    equity = float(config.window.initial_capital)
    equity_track = pd.Series(1.0, index=prices.index[:1])
    total_costs = 0.0
    derisked = False

    daily_net = pd.Series(0.0, index=prices.index)
    daily_gross = pd.Series(0.0, index=prices.index)

    for i, rd in enumerate(rebals):
        # --- build target weights (only data up to rd) ----------------------
        if sleeve_signals is not None:
            target = sleeve_signals.get(rd, pd.Series(dtype=float))
            portfolio = Portfolio(weights=target, date=rd)
        else:
            universe = _eligible(eligibility, rd)
            if not universe:
                continue
            scores = compute_composite(
                prices, rd, sleeve_a.factors, universe, fundamentals, sectors,
                sleeve_a.sector_neutralize,
            ).composite
            if scores.empty:
                continue
            exposure = exposure_on(exposure_series, rd, 1.0)
            portfolio = construct_portfolio(
                scores, prices, sleeve_a, rd, current_holdings, sectors, exposure
            )

        target_weights = portfolio.weights

        # --- portfolio-level risk limits ------------------------------------
        equity_so_far = equity_track * equity
        recent_returns = daily_net.loc[:rd].iloc[-config.risk.vol_lookback_days:]
        adv_slice = _adv_on(adv, rd)
        target_weights, state = apply_risk_limits(
            target_weights, config.risk, sectors,
            portfolio_value=equity, adv=adv_slice,
            equity_curve=equity_so_far, returns=recent_returns,
            currently_derisked=derisked, date=rd,
        )
        derisked = state.exposure_multiplier < 1.0
        risk_states.append(state)

        # --- costs on turnover ----------------------------------------------
        if config.apply_costs and cost_model is not None:
            idx = current_weights.index.union(target_weights.index)
            delta = (target_weights.reindex(idx).fillna(0.0)
                     - current_weights.reindex(idx).fillna(0.0))
            trade_values = delta * equity
            costs = cost_model.rebalance_cost(trade_values, adv_slice)
            cost_fraction = costs["total"] / equity if equity > 0 else 0.0
            total_costs += costs["total"]
            cost_rows.append({"date": rd, **{k: v for k, v in costs.items() if k != "bps"},
                              "bps": costs["bps"]})
        else:
            cost_fraction = 0.0

        turn = _turnover(current_weights, target_weights)
        turnover_history[rd] = turn
        weights_history[rd] = target_weights
        current_weights = target_weights
        current_holdings = set(target_weights.index)

        # --- accrue returns until the next rebalance ------------------------
        next_rd = rebals[i + 1] if i + 1 < len(rebals) else prices.index[-1]
        period = returns_panel.loc[(returns_panel.index > rd) & (returns_panel.index <= next_rd)]
        if period.empty:
            continue

        cols = [c for c in target_weights.index if c in period.columns]
        if cols:
            w = target_weights.reindex(cols).fillna(0.0)
            gross_period = period[cols].fillna(0.0) @ w
        else:
            gross_period = pd.Series(0.0, index=period.index)

        net_period = gross_period.copy()
        if len(net_period):
            # Charge the whole rebalance cost on the first day of the period.
            net_period.iloc[0] -= cost_fraction

        daily_gross.loc[gross_period.index] = gross_period.values
        daily_net.loc[net_period.index] = net_period.values
        equity_track = (1 + daily_net.loc[:next_rd]).cumprod()

    start = rebals[0]
    net = daily_net.loc[daily_net.index > start]
    gross = daily_gross.loc[daily_gross.index > start]
    equity_curve = (1 + net).cumprod() * config.window.initial_capital

    return BacktestResult(
        returns=net,
        gross_returns=gross,
        equity_curve=equity_curve,
        weights_history=pd.DataFrame(weights_history).T.fillna(0.0) if weights_history else pd.DataFrame(),
        turnover_history=pd.Series(turnover_history),
        cost_history=pd.DataFrame(cost_rows),
        exposure_history=exposure_series.reindex(net.index).ffill(),
        rebalance_dates=list(rebals),
        risk_states=risk_states,
        total_costs=total_costs,
        engine="vectorized",
        diagnostics={
            "n_rebalances": len(weights_history),
            "avg_positions": float(np.mean([len(w) for w in weights_history.values()]))
            if weights_history else 0.0,
            "warmup_days": warmup,
        },
    )


def _eligible(eligibility: pd.DataFrame, dt: pd.Timestamp) -> list[str]:
    from ..data.universe import eligible_on

    return eligible_on(eligibility, dt)


def _turnover(previous: pd.Series, current: pd.Series) -> float:
    if previous.empty:
        return float(current.sum())
    idx = previous.index.union(current.index)
    p = previous.reindex(idx).fillna(0.0)
    c = current.reindex(idx).fillna(0.0)
    return float((c - p).abs().sum() / 2.0)


def _adv_on(adv: pd.DataFrame | None, dt: pd.Timestamp) -> pd.Series | None:
    if adv is None or adv.empty:
        return None
    prior = adv.loc[adv.index <= dt]
    return None if prior.empty else prior.iloc[-1]
