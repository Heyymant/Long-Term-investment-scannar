"""Pluggable fundamentals providers."""

from __future__ import annotations

from typing import Any

from ...config import AppConfig, get_logger
from .base import (
    FUNDAMENTAL_COLUMNS,
    QUALITY_METRICS,
    VALUE_METRICS,
    FundamentalsCache,
    FundamentalsProvider,
    as_of,
    build_metric_panel,
    staleness_days,
)

log = get_logger(__name__)

__all__ = [
    "FUNDAMENTAL_COLUMNS",
    "QUALITY_METRICS",
    "VALUE_METRICS",
    "FundamentalsCache",
    "FundamentalsProvider",
    "as_of",
    "build_metric_panel",
    "staleness_days",
    "get_provider",
]


def get_provider(cfg: AppConfig, **kwargs: Any) -> FundamentalsProvider:
    """Instantiate the provider named in config.yaml."""
    name = str(cfg.get("data_sources.fundamentals.provider", "free")).lower()
    lag = int(cfg.get("data_sources.fundamentals.reporting_lag_days", 45))

    if name == "nse":
        from .nse_provider import NSEProvider

        return NSEProvider(
            cache_dir=kwargs.pop("cache_dir", cfg.paths.data_dir),
            reporting_lag_days=lag,
            **kwargs,
        )

    if name == "eodhd":
        from .eodhd import EODHDProvider

        return EODHDProvider(reporting_lag_days=lag, **kwargs)

    if name == "prowess":
        from .prowess import ProwessProvider

        export_dir = kwargs.pop("export_dir", cfg.paths.data_dir / "prowess")
        return ProwessProvider(export_dir=export_dir, reporting_lag_days=lag, **kwargs)

    if name == "synthetic":
        from .free import SyntheticProvider

        return SyntheticProvider(reporting_lag_days=lag, **kwargs)

    if name in ("free", "yfinance"):
        from .free import YFinanceProvider

        return YFinanceProvider(reporting_lag_days=lag, **kwargs)

    if name == "csv":
        from .free import CSVProvider

        path = kwargs.pop("path", cfg.paths.data_dir / "fundamentals.csv")
        return CSVProvider(path=path, reporting_lag_days=lag, **kwargs)

    raise ValueError(f"Unknown fundamentals provider '{name}'")
