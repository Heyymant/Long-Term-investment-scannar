"""Shared fixtures. Synthetic data is generated once per session - it is the
known-answer dataset the behavioural tests assert against."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                    # noqa: E402
from src.data.synthetic import generate_market        # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def market():
    """Small, fast synthetic market with planted factor premia."""
    return generate_market(n_stocks=60, start="2015-01-01", end="2022-12-31", seed=11)


@pytest.fixture(scope="session")
def dataset(cfg, market, tmp_path_factory):
    """A full Dataset built from the synthetic market."""
    from src.data.loader import load_dataset

    return load_dataset(
        cfg, source="synthetic", n_synthetic=60,
        start="2015-01-01", end="2022-12-31", run_health_check=False,
    )


@pytest.fixture(scope="session")
def prices(market):
    return market.prices


@pytest.fixture(scope="session")
def returns(market):
    return market.prices.pct_change().dropna()
