"""Synthetic Indian-equity market generator.

Purpose: let the entire pipeline run end-to-end with no Kite subscription, and
give the test-suite a *known answer* to check against. Each stock is assigned
latent factor loadings; returns are generated so that documented effects are
genuinely present in the data:

  * quality  -> positive drift
  * low vol  -> lower idiosyncratic vol, mild positive drift
  * momentum -> short-horizon return autocorrelation
  * value    -> positive drift, but only when quality is decent (value traps)
  * PEAD     -> positive drift after positive earnings surprises

Because we know the planted premia, `tests/` can assert the factor engine
recovers them - and that the leak detectors stay silent on honest signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = [
    "FINANCIAL SERVICES", "IT", "OIL & GAS", "FMCG", "AUTOMOBILE",
    "PHARMA", "METALS", "POWER", "CONSTRUCTION", "CONSUMER DURABLES",
]


@dataclass
class SyntheticMarket:
    """A complete synthetic dataset mirroring the real data contracts."""

    securities: pd.DataFrame        # isin, symbol, sector, kite_token, ...
    prices: pd.DataFrame            # wide: index=date, cols=isin (total-return adjusted)
    volumes: pd.DataFrame           # wide: index=date, cols=isin (shares)
    benchmark: pd.Series            # index=date, TRI level
    fundamentals: pd.DataFrame      # long: isin, date (announce), metric columns
    earnings: pd.DataFrame          # long: isin, announce_date, fiscal_quarter, eps
    risk_free: pd.Series            # index=date, annualized
    true_scores: pd.DataFrame       # latent loadings (for test assertions)


def generate_market(
    n_stocks: int = 120,
    start: str = "2012-01-01",
    end: str = "2024-12-31",
    seed: int = 7,
) -> SyntheticMarket:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n_days = len(dates)

    # ---- identities ------------------------------------------------------- #
    isins = [f"INE{i:04d}A01010" for i in range(n_stocks)]
    symbols = [f"SYN{i:03d}" for i in range(n_stocks)]
    sectors = [SECTORS[i % len(SECTORS)] for i in range(n_stocks)]
    tokens = [100000 + i for i in range(n_stocks)]

    # ---- latent factor loadings ------------------------------------------- #
    quality = rng.normal(0, 1, n_stocks)
    value = rng.normal(0, 1, n_stocks)
    lowvol_pref = rng.normal(0, 1, n_stocks)          # higher => lower vol
    beta = np.clip(rng.normal(1.0, 0.3, n_stocks), 0.3, 2.0)
    size = rng.normal(0, 1, n_stocks)

    # Idiosyncratic vol: low-vol names really are calmer.
    idio_vol = np.clip(0.022 - 0.006 * lowvol_pref, 0.006, 0.055)

    # Annual premia (in daily terms) - deliberately modest and realistic.
    d = 252.0
    drift = (
        0.04 * quality / d          # quality premium
        + 0.02 * lowvol_pref / d    # low-vol premium
        + 0.03 * value * (quality > -0.5) / d   # value works only if not junk
        - 0.01 * np.maximum(size, 0) / d
        + 0.06 / d                  # baseline equity drift
    )

    # ---- common factors ---------------------------------------------------- #
    market = rng.normal(0.055 / d, 0.011, n_days)
    # Plant a crash so drawdown/regime logic is exercised.
    crash = int(n_days * 0.45)
    market[crash:crash + 60] -= 0.006
    sector_ids = np.array([SECTORS.index(s) for s in sectors])
    sector_rets = rng.normal(0, 0.005, (n_days, len(SECTORS)))

    # ---- return generation with momentum autocorrelation ------------------- #
    returns = np.zeros((n_days, n_stocks))
    trailing = np.zeros(n_stocks)
    mom_strength = 0.02
    for t in range(n_days):
        idio = rng.normal(0, 1, n_stocks) * idio_vol
        mom_kick = mom_strength * np.tanh(trailing) / d
        returns[t] = drift + beta * market[t] + sector_rets[t, sector_ids] + idio + mom_kick
        # 120-day trailing momentum state (EWMA proxy)
        trailing = 0.99 * trailing + returns[t] * 20.0

    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=isins
    )
    prices.index.name = "date"

    # ---- volumes: bigger names trade more --------------------------------- #
    base_vol = np.exp(rng.normal(12.5, 1.1, n_stocks) - 0.5 * size)
    noise = rng.lognormal(0, 0.35, (n_days, n_stocks))
    volumes = pd.DataFrame(base_vol * noise, index=dates, columns=isins).round()
    volumes.index.name = "date"

    # ---- benchmark (cap-weighted proxy, total return) ---------------------- #
    w = np.exp(-size) / np.exp(-size).sum()
    bench_ret = returns @ w
    benchmark = pd.Series(1000.0 * np.exp(np.cumsum(bench_ret)), index=dates, name="NIFTY50_TRI")
    benchmark.index.name = "date"

    # ---- risk-free --------------------------------------------------------- #
    rf = pd.Series(
        np.clip(0.065 + np.cumsum(rng.normal(0, 0.00004, n_days)), 0.03, 0.09),
        index=dates, name="risk_free",
    )
    rf.index.name = "date"

    fundamentals = _generate_fundamentals(isins, dates, quality, value, rng)
    earnings = _generate_earnings(isins, dates, quality, rng)

    securities = pd.DataFrame({
        "isin": isins,
        "symbol": symbols,
        "name": [f"Synthetic Co {i}" for i in range(n_stocks)],
        "kite_token": tokens,
        "exchange": "NSE",
        "sector": sectors,
        "industry": sectors,
        "listing_date": pd.Timestamp(start),
        "delisting_date": pd.NaT,
        "fundamentals_id": symbols,
        "status": "active",
    })

    true_scores = pd.DataFrame({
        "isin": isins, "quality": quality, "value": value,
        "lowvol_pref": lowvol_pref, "beta": beta, "size": size,
        "idio_vol": idio_vol, "annual_drift": drift * d,
    }).set_index("isin")

    return SyntheticMarket(
        securities=securities, prices=prices, volumes=volumes, benchmark=benchmark,
        fundamentals=fundamentals, earnings=earnings, risk_free=rf, true_scores=true_scores,
    )


def _generate_fundamentals(
    isins: list[str], dates: pd.DatetimeIndex, quality: np.ndarray,
    value: np.ndarray, rng: np.random.Generator,
) -> pd.DataFrame:
    """Quarterly fundamentals stamped with an *announcement* date.

    Announce dates are the point-in-time guard: a metric only becomes usable
    on the day it is published, roughly 45 days after quarter end.
    """
    rows = []
    quarter_ends = pd.date_range(dates[0], dates[-1], freq="QE")
    for qi, qe in enumerate(quarter_ends):
        announce = qe + pd.Timedelta(days=45)
        if announce > dates[-1]:
            break
        for i, isin in enumerate(isins):
            q, v = quality[i], value[i]
            wobble = rng.normal(0, 0.15)
            roic = 0.14 + 0.06 * q + wobble * 0.03
            gross_prof = 0.30 + 0.10 * q + wobble * 0.04
            rows.append({
                "isin": isin,
                "announce_date": announce,
                "period_end": qe,
                "fiscal_quarter": f"{qe.year}Q{qe.quarter}",
                "roic": roic,
                "gross_profitability": gross_prof,
                "cfo_to_pat": 1.0 + 0.35 * q + rng.normal(0, 0.12),
                "fcf_to_assets": 0.07 + 0.045 * q + rng.normal(0, 0.02),
                "debt_to_assets": np.clip(0.32 - 0.11 * q + rng.normal(0, 0.05), 0.0, 0.95),
                "payout": np.clip(0.28 + 0.14 * q + rng.normal(0, 0.06), 0.0, 0.95),
                # Valuation: high `value` == cheap
                "earnings_yield": np.clip(0.055 + 0.030 * v + rng.normal(0, 0.008), 0.001, 0.30),
                "fcf_yield": np.clip(0.042 + 0.026 * v + rng.normal(0, 0.008), -0.05, 0.30),
                "ev_ebitda": np.clip(15.0 - 4.2 * v + rng.normal(0, 1.5), 2.0, 60.0),
                "pb": np.clip(3.2 - 1.0 * v + rng.normal(0, 0.4), 0.2, 20.0),
                "pe": np.clip(24.0 - 7.0 * v + rng.normal(0, 3.0), 3.0, 120.0),
                "total_assets": float(np.exp(rng.normal(10, 0.6)) * 1e7),
            })
    return pd.DataFrame(rows)


def _generate_earnings(
    isins: list[str], dates: pd.DatetimeIndex, quality: np.ndarray, rng: np.random.Generator,
) -> pd.DataFrame:
    """Quarterly EPS with persistent surprises (so SUE is informative)."""
    rows = []
    quarter_ends = pd.date_range(dates[0], dates[-1], freq="QE")
    base_eps = {isin: 4.0 + 2.0 * quality[i] for i, isin in enumerate(isins)}
    growth = {isin: 0.02 + 0.014 * quality[i] for i, isin in enumerate(isins)}
    surprise_state = dict.fromkeys(isins, 0.0)

    for qi, qe in enumerate(quarter_ends):
        announce = qe + pd.Timedelta(days=45)
        if announce > dates[-1]:
            break
        for isin in isins:
            # Surprises are autocorrelated - that persistence is what PEAD trades.
            surprise_state[isin] = 0.45 * surprise_state[isin] + rng.normal(0, 1.0)
            trend = base_eps[isin] * ((1 + growth[isin]) ** qi)
            seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * qe.quarter / 4)
            eps = trend * seasonal * (1 + 0.06 * surprise_state[isin])
            rows.append({
                "isin": isin,
                "announce_date": announce,
                "period_end": qe,
                "fiscal_quarter": f"{qe.year}Q{qe.quarter}",
                "eps": eps,
                "_true_surprise": surprise_state[isin],  # test-only column
            })
    return pd.DataFrame(rows)


def write_synthetic_dataset(market: SyntheticMarket, data_dir: str | Path) -> None:
    """Persist a synthetic market into the normal cache layout."""
    s = Path(data_dir) / "synthetic"
    s.mkdir(parents=True, exist_ok=True)
    market.securities.to_parquet(s / "securities.parquet", index=False)
    market.prices.to_parquet(s / "prices.parquet")
    market.volumes.to_parquet(s / "volumes.parquet")
    market.benchmark.to_frame().to_parquet(s / "benchmark.parquet")
    market.fundamentals.to_parquet(s / "fundamentals.parquet", index=False)
    market.earnings.to_parquet(s / "earnings.parquet", index=False)
    market.risk_free.to_frame().to_parquet(s / "risk_free.parquet")


def load_synthetic_dataset(data_dir: str | Path) -> SyntheticMarket | None:
    s = Path(data_dir) / "synthetic"
    if not (s / "prices.parquet").exists():
        return None
    return SyntheticMarket(
        securities=pd.read_parquet(s / "securities.parquet"),
        prices=pd.read_parquet(s / "prices.parquet"),
        volumes=pd.read_parquet(s / "volumes.parquet"),
        benchmark=pd.read_parquet(s / "benchmark.parquet").iloc[:, 0],
        fundamentals=pd.read_parquet(s / "fundamentals.parquet"),
        earnings=pd.read_parquet(s / "earnings.parquet"),
        risk_free=pd.read_parquet(s / "risk_free.parquet").iloc[:, 0],
        true_scores=pd.DataFrame(),
    )
