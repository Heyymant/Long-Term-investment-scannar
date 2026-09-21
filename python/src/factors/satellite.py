"""ETF / index-product scores.

Listed ETFs do not have usable QMJ or value prints, so this sleeve follows
the price-based papers:

  * 12–1 and 6–1 momentum (Jegadeesh–Titman; skip-month for India)
  * volatility-adjusted momentum (Barroso & Santa-Clara crash dampener)
  * low-vol (Ang / Indian Nifty-500 evidence)
  * dual momentum (Antonacci): relative rank AND absolute 12-month return > 0
  * time-series momentum (Moskowitz, Ooi, Pedersen): trailing return sign

Mutual-fund folios are not scored here — they have no exchange price panel
in this repo. The dashboard treats them as a satellite HOLD versus the
equity/ETF book.
"""

from __future__ import annotations

import pandas as pd

from ..data.asset_class import classify
from .base import combine_scores
from .lowvol import compute_low_vol_score
from .momentum import absolute_momentum, compute_momentum_score


ETF_TOP_N = 8
ETF_ADV_FLOOR = 20_000_000.0  # ₹2 crore — ETFs are thinner than NSE 500 names


def score_etfs(
    prices: pd.DataFrame,
    securities: pd.DataFrame | None = None,
    adv: pd.DataFrame | None = None,
    min_adv_inr: float = ETF_ADV_FLOOR,
) -> pd.DataFrame:
    """Cross-section of liquid NSE ETFs as of the last price date."""
    if prices is None or prices.empty:
        return pd.DataFrame()

    symbols = {}
    sectors = {}
    if securities is not None and not securities.empty and "isin" in securities.columns:
        for _, row in securities.iterrows():
            isin = str(row["isin"])
            symbols[isin] = str(row.get("symbol") or isin)
            sectors[isin] = str(row.get("sector") or "ETF")

    candidates: list[str] = []
    for col in prices.columns:
        isin = str(col)
        sym = symbols.get(isin, isin)
        if classify(isin, sym) != "etf":
            continue
        series = prices[col].dropna()
        if len(series) < 130:
            continue
        if adv is not None and not adv.empty and col in adv.columns:
            last_adv = adv[col].dropna()
            if not last_adv.empty and float(last_adv.iloc[-1]) < min_adv_inr:
                continue
        candidates.append(isin)

    if len(candidates) < 3:
        return pd.DataFrame()

    px = prices[candidates]
    momentum = compute_momentum_score(px, universe=candidates)
    low_vol = compute_low_vol_score(px, universe=candidates)
    abs_mom = absolute_momentum(px, lookback_months=12)
    composite = combine_scores(
        {"momentum": momentum, "low_vol": low_vol},
        {"momentum": 0.6, "low_vol": 0.4},
    )
    frame = pd.DataFrame({
        "isin": candidates,
        "symbol": [symbols.get(i, i) for i in candidates],
        "sector": [sectors.get(i, "ETF") for i in candidates],
        "composite": [composite.get(i) for i in candidates],
        "momentum": [momentum.get(i) for i in candidates],
        "low_vol": [low_vol.get(i) for i in candidates],
        "quality": None,
        "value": None,
        "abs_momentum": [abs_mom.get(i) for i in candidates],
        "asset_class": "etf",
        "sleeve": "ETF",
    })
    frame = frame.dropna(subset=["composite"])
    if frame.empty:
        return frame
    frame["dual_momentum"] = (frame["momentum"] > 0) & (frame["abs_momentum"] > 0)
    frame["rank"] = frame["composite"].rank(ascending=False)
    return frame.sort_values("rank").reset_index(drop=True)
