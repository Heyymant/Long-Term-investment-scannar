"""Sleeve B signal generation (long-only) and the pairs variant.

Long-only pairs trading is a compromise: we take only the cheap leg of a
cointegrated relationship and skip the short. That keeps the statistical
machinery but gives up market neutrality - the resulting position carries
beta, which the shared regime gate and position sizing have to manage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import get_logger
from ..schema import SleeveBConfig, StatArbMode
from .cointegration import PairCandidate, screen_pairs
from .spread import build_spread, rolling_zscore, zscore_window_from_half_life

log = get_logger(__name__)


@dataclass
class PairSignal:
    date: pd.Timestamp
    pair: tuple[str, str]
    z_score: float
    action: str                  # ENTER_LONG_Y | ENTER_LONG_X | EXIT | HOLD | STOP
    leg: str | None              # which ISIN to buy
    half_life: float
    hedge_ratio: float

    def to_dict(self) -> dict:
        return {
            "date": self.date, "y": self.pair[0], "x": self.pair[1],
            "z_score": self.z_score, "action": self.action, "leg": self.leg,
            "half_life": self.half_life, "hedge_ratio": self.hedge_ratio,
        }


def evaluate_pair(
    prices: pd.DataFrame,
    pair: PairCandidate,
    dt: pd.Timestamp,
    config: SleeveBConfig,
    in_position: bool = False,
) -> PairSignal | None:
    """Decide what to do with one pair as of `dt`.

    The spread is y - b*x. A strongly negative z means y is cheap relative to
    x, so the long-only trade is to buy y. A strongly positive z means x is
    the cheap leg.
    """
    dt = pd.Timestamp(dt)
    px = prices.loc[prices.index <= dt]
    if pair.y not in px.columns or pair.x not in px.columns:
        return None

    window = px.iloc[-config.pair_formation_days:]
    model = build_spread(window[pair.y], window[pair.x],
                         "kalman" if config.use_kalman else "ols")
    if model.spread.empty or not np.isfinite(model.half_life):
        return None
    if not (config.min_half_life_days <= model.half_life <= config.max_half_life_days):
        return PairSignal(dt, (pair.y, pair.x), np.nan, "STOP", None,
                          model.half_life, model.hedge_ratio)

    z_window = zscore_window_from_half_life(model.half_life)
    z_series = rolling_zscore(model.spread, z_window, shift=1)
    if z_series.dropna().empty:
        return None
    z = float(z_series.dropna().iloc[-1])

    if abs(z) >= config.pair_stop_z:
        action, leg = "STOP", None
    elif in_position and abs(z) <= config.pair_exit_z:
        action, leg = "EXIT", None
    elif not in_position and z <= -config.pair_entry_z:
        action, leg = "ENTER_LONG_Y", pair.y       # y is cheap
    elif not in_position and z >= config.pair_entry_z:
        action, leg = "ENTER_LONG_X", pair.x       # x is cheap
    else:
        action, leg = ("HOLD" if in_position else "NONE"), None

    return PairSignal(dt, (pair.y, pair.x), z, action, leg,
                      model.half_life, model.hedge_ratio)


def pairs_signals(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    sectors: pd.Series,
    config: SleeveBConfig,
    refresh_every: int = 4,
) -> tuple[dict[pd.Timestamp, pd.Series], pd.DataFrame, pd.DataFrame]:
    """Long-only pairs weights over time, plus the pair and signal logs.

    Pairs are re-screened periodically (not every date) because cointegration
    testing is expensive and relationships do not change daily.
    """
    from ..data.universe import eligible_on

    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    signal_rows: list[dict] = []
    pair_rows: list[dict] = []

    active_pairs: list[PairCandidate] = []
    open_positions: dict[tuple[str, str], str] = {}   # pair -> leg held

    for i, dt in enumerate(rebalance_dates):
        universe = eligible_on(eligibility, dt)
        if len(universe) < 4:
            continue

        if i % refresh_every == 0:
            hist = prices.loc[prices.index <= dt, [c for c in universe if c in prices.columns]]
            active_pairs = screen_pairs(
                hist, sectors,
                formation_window=config.pair_formation_days,
                max_pvalue=config.coint_pvalue,
                fdr_alpha=config.fdr_alpha,
                min_half_life=config.min_half_life_days,
                max_half_life=config.max_half_life_days,
            )
            for p in active_pairs:
                pair_rows.append({**p.to_dict(), "as_of": dt})
            # Drop open positions whose pair no longer qualifies.
            valid = {(p.y, p.x) for p in active_pairs}
            open_positions = {k: v for k, v in open_positions.items() if k in valid}

        legs: dict[str, float] = {}
        for pair in active_pairs:
            key = (pair.y, pair.x)
            sig = evaluate_pair(prices, pair, dt, config, key in open_positions)
            if sig is None:
                continue
            signal_rows.append(sig.to_dict())

            if sig.action in ("EXIT", "STOP"):
                open_positions.pop(key, None)
            elif sig.action.startswith("ENTER") and sig.leg:
                open_positions[key] = sig.leg
            if key in open_positions:
                leg = open_positions[key]
                legs[leg] = legs.get(leg, 0.0) + 1.0

        if legs:
            s = pd.Series(legs)
            w = (s / s.sum()).clip(upper=config.max_weight_per_stock)
            if w.sum() > 0:
                weights_by_date[dt] = w / w.sum()

    return (
        weights_by_date,
        pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame(),
        pd.DataFrame(signal_rows) if signal_rows else pd.DataFrame(),
    )


def volatility_sized_weights(
    selected: list[str], prices: pd.DataFrame, dt: pd.Timestamp,
    target_vol: float = 0.15, max_weight: float = 0.05, lookback: int = 60,
) -> pd.Series:
    """Size positions inversely to their own volatility."""
    if not selected:
        return pd.Series(dtype=float)
    px = prices.loc[prices.index <= dt]
    cols = [s for s in selected if s in px.columns]
    if not cols:
        return pd.Series(dtype=float)

    vol = px[cols].pct_change().iloc[-lookback:].std() * np.sqrt(252)
    vol = vol.replace(0, np.nan).fillna(vol.median())
    if vol.isna().all():
        return pd.Series(1.0 / len(cols), index=cols)

    raw = (target_vol / vol).clip(upper=max_weight / 0.01)
    w = (raw / raw.sum()).clip(upper=max_weight)
    return w / w.sum() if w.sum() > 0 else w


def build_sleeve_b_signals(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    config: SleeveBConfig,
    sectors: pd.Series | None = None,
    benchmark: pd.Series | None = None,
) -> tuple[dict[pd.Timestamp, pd.Series], pd.DataFrame, pd.DataFrame]:
    """Entry point: dispatch to residual reversion (default) or pairs."""
    if config.mode == StatArbMode.PAIRS:
        if sectors is None or sectors.empty:
            log.warning("Pairs mode needs sector data; falling back to residual reversion")
        else:
            return pairs_signals(prices, rebalance_dates, eligibility, sectors, config)

    from .residual_reversion import residual_reversion_signals

    weights = residual_reversion_signals(
        prices, rebalance_dates, eligibility, sectors, benchmark,
        config.residual_lookback_days, config.n_positions,
        config.entry_z, config.max_weight_per_stock,
    )
    signals = pd.DataFrame([
        {"date": dt, "isin": isin, "weight": w}
        for dt, ws in weights.items() for isin, w in ws.items()
    ])
    return weights, pd.DataFrame(), signals
