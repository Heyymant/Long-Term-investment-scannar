"""Corporate actions and total-return series construction.

Unadjusted prices lie: a 1:1 bonus looks like a -50% crash and will poison
momentum, volatility and any return calculation. This module applies split /
bonus / rights adjustments and folds dividends back in to build the
total-return series the whole framework uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

ACTION_COLUMNS = ["isin", "ex_date", "action_type", "ratio_from", "ratio_to", "amount"]


class ActionType(str, Enum):
    SPLIT = "split"        # ratio_from:ratio_to  (e.g. 1:5 -> 5x shares)
    BONUS = "bonus"        # ratio_to bonus shares for ratio_from held
    RIGHTS = "rights"
    DIVIDEND = "dividend"  # amount per share


@dataclass
class CorporateActions:
    actions: pd.DataFrame

    @classmethod
    def empty(cls) -> CorporateActions:
        return cls(pd.DataFrame(columns=ACTION_COLUMNS))

    @classmethod
    def load(cls, data_dir: str | Path) -> CorporateActions:
        p = Path(data_dir) / "corporate_actions.parquet"
        if not p.exists():
            return cls.empty()
        return cls(pd.read_parquet(p))

    def save(self, data_dir: str | Path) -> None:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self.actions.to_parquet(Path(data_dir) / "corporate_actions.parquet", index=False)

    # -- factors ---------------------------------------------------------------
    def price_adjustment_factors(self, isin: str, index: pd.DatetimeIndex) -> pd.Series:
        """Cumulative *price* adjustment factor (splits/bonuses only).

        Historical prices before an ex-date are multiplied by this factor so the
        series is continuous. Factor is 1.0 from the last ex-date onwards.
        """
        factor = pd.Series(1.0, index=index)
        acts = self._for(isin, {ActionType.SPLIT.value, ActionType.BONUS.value})
        if acts.empty:
            return factor

        for _, a in acts.iterrows():
            ex = pd.Timestamp(a["ex_date"])
            r = _share_multiplier(a)
            if r is None or r <= 0:
                continue
            # Prices strictly before the ex-date are divided by the multiplier.
            factor.loc[factor.index < ex] /= r
        return factor

    def dividends(self, isin: str, index: pd.DatetimeIndex) -> pd.Series:
        """Per-share dividend amounts aligned to the price index (ex-date)."""
        out = pd.Series(0.0, index=index)
        divs = self._for(isin, {ActionType.DIVIDEND.value})
        for _, a in divs.iterrows():
            ex = pd.Timestamp(a["ex_date"])
            amt = float(a.get("amount") or 0.0)
            if amt <= 0:
                continue
            pos = index.searchsorted(ex)
            if 0 <= pos < len(index):
                out.iloc[pos] += amt
        return out

    def _for(self, isin: str, types: set[str]) -> pd.DataFrame:
        if self.actions.empty:
            return self.actions
        m = (self.actions["isin"] == isin) & (self.actions["action_type"].isin(types))
        return self.actions[m].sort_values("ex_date")

    def add(
        self, isin: str, ex_date: str | pd.Timestamp, action_type: ActionType | str,
        ratio_from: float = 1.0, ratio_to: float = 1.0, amount: float = 0.0,
    ) -> None:
        row = {
            "isin": isin, "ex_date": pd.Timestamp(ex_date),
            "action_type": ActionType(action_type).value if not isinstance(action_type, str) else action_type,
            "ratio_from": ratio_from, "ratio_to": ratio_to, "amount": amount,
        }
        self.actions = pd.concat([self.actions, pd.DataFrame([row])], ignore_index=True)


def _share_multiplier(action: pd.Series) -> float | None:
    """How many shares you end up holding per share previously held."""
    a_type = action["action_type"]
    rf = float(action.get("ratio_from") or 0)
    rt = float(action.get("ratio_to") or 0)
    if rf <= 0 or rt <= 0:
        return None
    if a_type == ActionType.SPLIT.value:
        # face value 10 -> 2 means 1 old share becomes 5: ratio_from:ratio_to
        return rt / rf
    if a_type == ActionType.BONUS.value:
        # "ratio_to : ratio_from" bonus => 1 + to/from shares
        return 1.0 + (rt / rf)
    return None


# --------------------------------------------------------------------------- #
# Adjusted / total-return panels
# --------------------------------------------------------------------------- #
def adjust_prices(
    prices: pd.DataFrame, ca: CorporateActions, columns: list[str] | None = None
) -> pd.DataFrame:
    """Apply split/bonus adjustment to a wide price panel (index=date, cols=isin)."""
    if ca.actions.empty:
        return prices.copy()
    out = prices.copy()
    for isin in (columns or prices.columns):
        if isin not in out.columns:
            continue
        out[isin] = out[isin] * ca.price_adjustment_factors(isin, prices.index)
    return out


def total_return_panel(
    prices: pd.DataFrame, ca: CorporateActions | None = None
) -> pd.DataFrame:
    """Build a total-return index (base 100) from adjusted prices + dividends.

    Dividends are treated as reinvested on the ex-date, which is the standard
    convention and matches how TRI benchmarks are computed.
    """
    adj = adjust_prices(prices, ca) if ca is not None else prices.copy()
    simple = adj.pct_change()

    if ca is not None and not ca.actions.empty:
        for isin in adj.columns:
            divs = ca.dividends(isin, adj.index)
            if divs.abs().sum() == 0:
                continue
            prev_close = adj[isin].shift(1)
            simple[isin] = simple[isin] + (divs / prev_close).fillna(0.0)

    tr = (1.0 + simple.fillna(0.0)).cumprod() * 100.0
    tr.iloc[0] = 100.0
    return tr


def returns_from_panel(panel: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    """Simple (default) or log returns from a price/TR panel."""
    if log_returns:
        return np.log(panel / panel.shift(1))
    return panel.pct_change()


def detect_unflagged_actions(
    prices: pd.DataFrame, threshold: float = -0.35, max_flags: int = 500
) -> pd.DataFrame:
    """Heuristic: find suspicious one-day drops that look like missing splits.

    A genuine -50% single-day move in a liquid large/mid cap is rare; a 1:2
    split is exactly -50%. Anything landing near a clean ratio gets flagged for
    review rather than silently corrupting the backtest.
    """
    rets = prices.pct_change()
    flags = []
    common = {0.5: "1:2", 1 / 3: "1:3", 0.25: "1:4", 0.2: "1:5", 0.1: "1:10", 2 / 3: "2:3"}
    for isin in rets.columns:
        hits = rets[isin][rets[isin] <= threshold]
        for dt, r in hits.items():
            implied = 1.0 + r  # price ratio
            near = min(common, key=lambda c: abs(c - implied))
            if abs(near - implied) < 0.03:
                flags.append({
                    "isin": isin, "date": dt, "return": r,
                    "implied_ratio": implied, "suspected_action": common[near],
                })
            if len(flags) >= max_flags:
                break
    return pd.DataFrame(flags)
