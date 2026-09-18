"""Sleeve C: long-only post-earnings-announcement drift (PEAD).

Buy high positive-SUE names shortly after results and hold through the drift
window (~40-65 trading days), then exit. Long-only, so we capture the
positive-surprise leg only - the short leg of the classic PEAD spread is not
available to us.

Honest caveats built into the defaults:
  * drift is weak in NSE large caps, so `exclude_top_mcap_pct` can trim them
  * turnover is high relative to Sleeve A, so the cost model is decisive
  * entries are lagged past the announcement (post-market results trade next day)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import get_logger
from ..schema import SleeveCConfig
from .sue import earnings_day_reaction, rank_sue_cross_section

log = get_logger(__name__)


@dataclass
class EventPosition:
    isin: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    sue: float
    entry_reason: str
    abnormal_return: float = np.nan

    def is_open(self, dt: pd.Timestamp) -> bool:
        return self.entry_date <= pd.Timestamp(dt) < self.exit_date


@dataclass
class EventSignals:
    weights_by_date: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    positions: list[EventPosition] = field(default_factory=list)
    signal_log: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_frame(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()
        return pd.DataFrame([{
            "isin": p.isin, "entry_date": p.entry_date, "exit_date": p.exit_date,
            "sue": p.sue, "entry_reason": p.entry_reason,
            "abnormal_return": p.abnormal_return,
        } for p in self.positions])


def build_event_signals(
    prices: pd.DataFrame,
    sue_panel: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    config: SleeveCConfig,
    benchmark: pd.Series | None = None,
    adv: pd.DataFrame | None = None,
    market_caps: pd.DataFrame | None = None,
) -> EventSignals:
    """Generate long-only PEAD target weights across the backtest."""
    from ..data.universe import eligible_on

    result = EventSignals()
    if sue_panel.empty:
        log.info("No SUE data; Sleeve C produces no signals")
        return result

    open_positions: list[EventPosition] = []
    log_rows: list[dict] = []

    for dt in rebalance_dates:
        dt = pd.Timestamp(dt)
        open_positions = [p for p in open_positions if p.is_open(dt)]

        universe = eligible_on(eligibility, dt)
        if not universe:
            continue

        universe = _apply_event_filters(universe, dt, config, adv, market_caps)
        ranked = rank_sue_cross_section(sue_panel, dt, universe, lookback_days=90)
        held = {p.isin for p in open_positions}
        room = config.n_positions - len(open_positions)

        if room > 0 and not ranked.empty:
            for isin, sue in ranked.items():
                if room <= 0:
                    break
                if isin in held or sue < config.min_sue:
                    continue

                announce = _last_announce(sue_panel, isin, dt)
                if announce is None:
                    continue
                # Respect the entry lag - results are not tradeable same-day.
                if (dt - announce).days < config.entry_lag_days:
                    continue

                reaction = earnings_day_reaction(prices, benchmark, isin, announce)
                abn = reaction.get("abnormal_return", np.nan)
                if config.require_positive_reaction and (np.isnan(abn) or abn <= 0):
                    log_rows.append({"date": dt, "isin": isin, "sue": sue,
                                     "action": "SKIP", "reason": "no positive reaction",
                                     "abnormal_return": abn})
                    continue

                exit_date = _exit_date(prices.index, dt, config.drift_window_days)
                pos = EventPosition(isin, dt, exit_date, float(sue), "positive_sue", abn)
                open_positions.append(pos)
                result.positions.append(pos)
                held.add(isin)
                room -= 1
                log_rows.append({"date": dt, "isin": isin, "sue": sue,
                                 "action": "ENTER", "reason": "positive SUE + reaction",
                                 "abnormal_return": abn, "exit_date": exit_date})

        if open_positions:
            names = [p.isin for p in open_positions]
            w = pd.Series(1.0 / len(names), index=names).clip(upper=config.max_weight_per_stock)
            if w.sum() > 0:
                result.weights_by_date[dt] = w / w.sum() if w.sum() > 1 else w

    result.signal_log = pd.DataFrame(log_rows) if log_rows else pd.DataFrame()
    log.info("Sleeve C: %d event positions across %d dates",
             len(result.positions), len(result.weights_by_date))
    return result


def _apply_event_filters(
    universe: list[str],
    dt: pd.Timestamp,
    config: SleeveCConfig,
    adv: pd.DataFrame | None,
    market_caps: pd.DataFrame | None,
) -> list[str]:
    """Liquidity floor and optional large-cap exclusion (drift is weak there)."""
    out = list(universe)

    if adv is not None and not adv.empty and config.min_adv_inr > 0:
        prior = adv.loc[adv.index <= dt]
        if not prior.empty:
            row = prior.iloc[-1]
            out = [i for i in out if i in row.index and row[i] >= config.min_adv_inr]

    if market_caps is not None and not market_caps.empty and config.exclude_top_mcap_pct > 0:
        prior = market_caps.loc[market_caps.index <= dt]
        if not prior.empty:
            caps = prior.iloc[-1].reindex(out).dropna()
            if len(caps) > 10:
                cutoff = caps.quantile(1 - config.exclude_top_mcap_pct)
                out = [i for i in out if i not in caps[caps >= cutoff].index]

    return out


def _last_announce(sue_panel: pd.DataFrame, isin: str, dt: pd.Timestamp) -> pd.Timestamp | None:
    rows = sue_panel[sue_panel["isin"] == isin]
    if rows.empty:
        return None
    d = pd.to_datetime(rows["announce_date"])
    visible = d[d <= pd.Timestamp(dt)]
    return None if visible.empty else pd.Timestamp(visible.max())


def _exit_date(index: pd.DatetimeIndex, entry: pd.Timestamp, window_days: int) -> pd.Timestamp:
    pos = index.searchsorted(pd.Timestamp(entry))
    target = min(pos + window_days, len(index) - 1)
    return pd.Timestamp(index[target])


def event_rebalance_dates(
    index: pd.DatetimeIndex, frequency_days: int = 5, warmup: int = 252
) -> pd.DatetimeIndex:
    """Sleeve C checks for new events more often than the factor sleeve."""
    usable = index[warmup:] if len(index) > warmup else index[-1:]
    return pd.DatetimeIndex(usable[::max(1, frequency_days)])
