"""Shared fixtures built from recorded NSE market data.

Every price, fundamental and benchmark used here is a slice of the real
exchange archive (see ``tests/fixtures/manifest.json``). Simulated markets
are not part of this suite.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                    # noqa: E402
from src.data.loader import load_dataset              # noqa: E402
from src.schema import UniverseConfig                 # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass
class MarketSlice:
    """The recorded NSE slice the behavioural tests assert against."""

    prices: pd.DataFrame
    volumes: pd.DataFrame
    securities: pd.DataFrame
    fundamentals: pd.DataFrame
    earnings: pd.DataFrame
    benchmark: pd.Series
    risk_free: pd.Series
    manifest: dict


def _read_panel(name: str) -> pd.DataFrame:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"Recorded NSE fixture missing: {path.name}. Run scripts/build_fixtures.py")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def _read_table(name: str) -> pd.DataFrame:
    path = FIXTURE_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def market() -> MarketSlice:
    """Liquid NSE names, 2022-2024, carved from the bhavcopy archive."""
    manifest_path = FIXTURE_DIR / "manifest.json"
    if not manifest_path.exists() or not (FIXTURE_DIR / "panel_close.parquet").exists():
        pytest.skip("Recorded NSE fixtures are missing. Run scripts/build_fixtures.py")

    prices = _read_panel("panel_close.parquet")
    volumes = _read_panel("panel_volume.parquet") if (FIXTURE_DIR / "panel_volume.parquet").exists() else pd.DataFrame()
    securities = _read_table("securities.parquet")
    if securities.empty:
        pytest.skip("Fixture security master is missing")

    benches = _read_panel("benchmarks.parquet")
    benchmark = benches["NIFTY50_TRI"] if "NIFTY50_TRI" in benches.columns else benches.iloc[:, 0]

    rf = _read_panel("risk_free.parquet") if (FIXTURE_DIR / "risk_free.parquet").exists() else pd.DataFrame()
    risk_free = (
        rf.iloc[:, 0].reindex(prices.index).ffill().bfill()
        if not rf.empty else pd.Series(0.065, index=prices.index, name="risk_free")
    )

    return MarketSlice(
        prices=prices,
        volumes=volumes,
        securities=securities,
        fundamentals=_read_table("fundamentals.parquet"),
        earnings=_read_table("earnings.parquet"),
        benchmark=benchmark.reindex(prices.index).ffill(),
        risk_free=risk_free.rename("risk_free"),
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
    )


@pytest.fixture(scope="session")
def dataset(cfg, market):
    """Full Dataset assembled from the same recorded NSE slice."""
    return load_dataset(
        cfg,
        universe_config=UniverseConfig(
            min_financial_quarters=0,
            min_listing_days=0,
            min_adv_inr=0.0,
            min_price_inr=0.0,
            point_in_time=False,
        ),
        source="fixtures",
        start=str(market.prices.index.min().date()),
        end=str(market.prices.index.max().date()),
        run_health_check=False,
    )


@pytest.fixture(scope="session")
def prices(market):
    return market.prices


@pytest.fixture(scope="session")
def returns(market):
    return market.prices.pct_change().dropna()
