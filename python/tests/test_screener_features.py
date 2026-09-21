"""Screener table must carry rupee closes and filing fundamentals."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.runner import _add_screener_features


def test_screener_features_fill_close_and_live_pe(monkeypatch):
    """Closing price is the latest raw NSE close; PE is close / TTM EPS."""
    monkeypatch.setattr("src.backtest.runner._live_close_panel", lambda: pd.DataFrame())
    monkeypatch.setattr("src.backtest.runner._live_volume_panel", lambda: pd.DataFrame())

    isin = "INETEST01018"
    dates = pd.bdate_range("2024-01-02", periods=10)
    raw = pd.DataFrame({isin: np.linspace(100.0, 109.0, len(dates))}, index=dates)
    fund = pd.DataFrame([{
        "isin": isin,
        "announce_date": pd.Timestamp("2024-01-05"),
        "period_end": pd.Timestamp("2023-12-31"),
        "eps": 2.0,
        "eps_ttm": 10.0,
        "net_income": 200.0,
        "revenue": 1000.0,
        "pe": np.nan,
        "market_cap": np.nan,
        "earnings_yield": np.nan,
        "pb": np.nan,
        "payout": np.nan,
        "roic": 0.15,
        "debt_to_assets": np.nan,
        "net_margin": 0.2,
        "pretax_margin": 0.22,
    }])
    ds = SimpleNamespace(
        prices=raw,
        raw_prices=raw,
        volumes=pd.DataFrame(),
        fundamentals=fund,
        securities=pd.DataFrame([{"isin": isin, "name": "Test Ltd", "industry": "Banks"}]),
    )
    ranks = pd.DataFrame([{"isin": isin, "symbol": "TEST", "sector": "FINANCIALS", "rank": 1}])
    out = _add_screener_features(ranks, ds)

    assert out.loc[0, "last_price"] == 109.0
    assert out.loc[0, "close"] == 109.0
    assert out.loc[0, "prev_close"] == 108.0
    assert out.loc[0, "pe"] == 10.9
    assert out.loc[0, "eps"] == 2.0
    assert out.loc[0, "revenue"] == 1000.0
    assert out.loc[0, "market_cap"] == 109.0 * (200.0 / 2.0)
    assert out.loc[0, "name"] == "Test Ltd"
    assert out.loc[0, "industry"] == "Banks"
    assert out.loc[0, "pretax_margin"] == pytest.approx(0.22)


def test_write_price_sparks(tmp_path, monkeypatch):
    from src.backtest.runner import write_price_sparks

    isin = "INETEST01018"
    dates = pd.bdate_range("2024-01-02", periods=30)
    px = pd.DataFrame({isin: np.linspace(100.0, 129.0, len(dates))}, index=dates)
    monkeypatch.setattr("src.backtest.runner._live_close_panel", lambda: px)
    df = pd.DataFrame([{"isin": isin, "symbol": "TEST"}])
    n = write_price_sparks(tmp_path, df, px)
    assert n == 1
    payload = json.loads((tmp_path / "sparks.json").read_text(encoding="utf-8"))
    assert payload["closes"]["TEST"][-1] == 129.0
    assert len(payload["closes"]["TEST"]) == 30



def test_screener_iima_profitability_from_filings(monkeypatch):
    """IIMA profitability z-scores attach when several names print margins."""
    monkeypatch.setattr("src.backtest.runner._live_close_panel", lambda: pd.DataFrame())
    monkeypatch.setattr("src.backtest.runner._live_volume_panel", lambda: pd.DataFrame())

    n = 8
    isins = [f"INE{i:03d}01018" for i in range(n)]
    dates = pd.bdate_range("2024-01-02", periods=10)
    raw = pd.DataFrame({i: np.linspace(100.0, 109.0, len(dates)) for i in isins}, index=dates)
    fund = pd.DataFrame({
        "isin": isins,
        "announce_date": pd.Timestamp("2024-01-05"),
        "period_end": pd.Timestamp("2023-12-31"),
        "net_margin": np.linspace(0.04, 0.20, n),
        "pretax_margin": np.linspace(0.05, 0.22, n),
        "roic": np.linspace(0.08, 0.18, n),
        "gross_profitability": np.linspace(0.12, 0.30, n),
        "eps_ttm": 10.0,
        "eps": 2.0,
        "net_income": 200.0,
        "payout": np.nan,
        "debt_to_assets": np.nan,
    })
    ds = SimpleNamespace(
        prices=raw,
        raw_prices=raw,
        volumes=pd.DataFrame(),
        fundamentals=fund,
        securities=pd.DataFrame([{"isin": i, "name": i, "industry": "Banks"} for i in isins]),
    )
    ranks = pd.DataFrame({"isin": isins, "symbol": [f"T{i}" for i in range(n)], "sector": "FINANCIALS", "rank": range(1, n + 1)})
    out = _add_screener_features(ranks, ds)
    assert out["profitability"].notna().sum() == n
    assert out.loc[out["isin"] == isins[-1], "profitability"].iloc[0] > out.loc[out["isin"] == isins[0], "profitability"].iloc[0]
    assert out["payout_z"].isna().all()
    assert out["pe"].notna().all()
    assert (out["last_price"] == 109.0).all()
