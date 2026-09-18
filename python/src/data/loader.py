"""Assemble the full dataset a backtest needs, from whichever source is configured.

One entry point (`load_dataset`) so scripts, tests and the runner all see the
same shapes and the same point-in-time guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import AppConfig, get_logger
from ..schema import UniverseConfig
from .corporate_actions import CorporateActions, total_return_panel
from .quality import DataHealthReport, build_report
from .reference import load_risk_free, sector_series
from .universe import (
    IndexMembership,
    UniverseBuilder,
    UniverseFilters,
    average_daily_value,
    fundamentals_coverage_matrix,
)

log = get_logger(__name__)


@dataclass
class Dataset:
    """Everything the engines and sleeves consume."""

    prices: pd.DataFrame                  # total-return adjusted, wide
    raw_prices: pd.DataFrame              # unadjusted (for order sizing)
    volumes: pd.DataFrame
    adv: pd.DataFrame                     # rolling average daily value (INR)
    eligibility: pd.DataFrame             # boolean date x isin
    securities: pd.DataFrame
    sectors: pd.Series
    benchmark: pd.Series
    risk_free: pd.Series
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)
    earnings: pd.DataFrame = field(default_factory=pd.DataFrame)
    health: DataHealthReport | None = None

    @property
    def isins(self) -> list[str]:
        return list(self.prices.columns)

    def slice(self, start: str | pd.Timestamp | None, end: str | pd.Timestamp | None) -> Dataset:
        """Restrict to a date window (used by walk-forward folds)."""
        idx = self.prices.index
        mask = pd.Series(True, index=idx)
        if start is not None:
            mask &= idx >= pd.Timestamp(start)
        if end is not None:
            mask &= idx <= pd.Timestamp(end)
        keep = idx[mask.values]

        return Dataset(
            prices=self.prices.loc[keep],
            raw_prices=self.raw_prices.loc[keep],
            volumes=self.volumes.loc[keep] if not self.volumes.empty else self.volumes,
            adv=self.adv.loc[keep] if not self.adv.empty else self.adv,
            eligibility=self.eligibility.loc[keep] if not self.eligibility.empty else self.eligibility,
            securities=self.securities,
            sectors=self.sectors,
            benchmark=self.benchmark.reindex(keep).ffill() if not self.benchmark.empty else self.benchmark,
            risk_free=self.risk_free.reindex(keep).ffill() if not self.risk_free.empty else self.risk_free,
            fundamentals=self.fundamentals,
            earnings=self.earnings,
            health=self.health,
        )


def load_dataset(
    cfg: AppConfig,
    universe_config: UniverseConfig | None = None,
    source: str | None = None,
    n_synthetic: int = 120,
    start: str | None = None,
    end: str | None = None,
    run_health_check: bool = True,
) -> Dataset:
    """Build a Dataset from the configured source.

    source: 'synthetic' (offline) or 'kite' (cached real data).
    """
    source = source or str(cfg.get("data_sources.prices.provider", "synthetic"))
    uc = universe_config or UniverseConfig()

    if source == "synthetic":
        ds = _load_synthetic(cfg, uc, n_synthetic, start, end)
    elif source == "nse":
        ds = _load_nse(cfg, uc, start, end)
    else:
        ds = _load_cached(cfg, uc, start, end)

    if run_health_check:
        ds.health = build_report(ds.prices, ds.volumes, ds.fundamentals)
        if ds.health.n_errors:
            log.warning("Data health: %d errors, %d warnings",
                        ds.health.n_errors, ds.health.n_warnings)
    return ds


def _load_synthetic(
    cfg: AppConfig, uc: UniverseConfig, n: int, start: str | None, end: str | None
) -> Dataset:
    from .synthetic import generate_market

    market = generate_market(
        n_stocks=n, start=start or "2012-01-01", end=end or "2024-12-31", seed=7
    )
    # Synthetic ISINs are not real, so any index membership cached from live
    # NSE data would match nothing and silently empty the universe. Build
    # membership from the synthetic universe itself.
    membership = IndexMembership.from_static(list(market.prices.columns), uc.index)

    return _assemble(
        cfg, uc,
        raw_prices=market.prices,
        volumes=market.volumes,
        securities=market.securities,
        benchmark=market.benchmark,
        fundamentals=market.fundamentals,
        earnings=market.earnings,
        risk_free=market.risk_free,
        corporate_actions=CorporateActions.empty(),
        membership=membership,
    )


def _load_nse(
    cfg: AppConfig, uc: UniverseConfig, start: str | None, end: str | None
) -> Dataset:
    """Load the panels built from the NSE bhavcopy archive.

    This is the survivorship-bias-free path: the panels come from replaying
    every daily file, so delisted companies are present for the period they
    actually traded.
    """
    from .fundamentals.base import FundamentalsCache
    from .reference import load_benchmark
    from .security_master import SecurityMaster

    d = cfg.paths.data_dir
    close_path = d / "panel_close.parquet"
    if not close_path.exists():
        raise RuntimeError(
            "No NSE price panels found. Run:\n"
            "  python scripts/fetch_nse.py --all --start 2015-01-01"
        )

    raw_prices = pd.read_parquet(close_path)
    raw_prices.index = pd.to_datetime(raw_prices.index)

    volumes = _read_optional_panel(d / "panel_volume.parquet")
    traded_value = _read_optional_panel(d / "panel_traded_value.parquet")

    if start:
        raw_prices = raw_prices.loc[raw_prices.index >= pd.Timestamp(start)]
    if end:
        raw_prices = raw_prices.loc[raw_prices.index <= pd.Timestamp(end)]
    if not volumes.empty:
        volumes = volumes.reindex(raw_prices.index)
    if not traded_value.empty:
        traded_value = traded_value.reindex(raw_prices.index)

    sm = SecurityMaster.load(d)
    if sm.securities.empty:
        raise RuntimeError("No security master. Run: python scripts/fetch_nse.py --master")

    securities = sm.securities
    ds = _assemble(
        cfg, uc,
        raw_prices=raw_prices,
        volumes=volumes,
        securities=securities,
        benchmark=load_benchmark(cfg, "NIFTY50_TRI", raw_prices.index),
        fundamentals=FundamentalsCache(d).read(),
        earnings=_read_optional(d / "earnings.parquet"),
        risk_free=load_risk_free(cfg, raw_prices.index),
        corporate_actions=CorporateActions.load(d),
    )

    # NSE reports traded value directly, which is a better ADV measure than
    # price x volume (it accounts for intraday price variation).
    if not traded_value.empty:
        ds.adv = traded_value.rolling(63, min_periods=20).mean()

    if ds.benchmark.empty:
        # No TRI cached: fall back to an equal-weight proxy so the regime
        # overlay and benchmark comparisons still function.
        log.warning(
            "No NIFTY50_TRI benchmark cached; using an equal-weight universe proxy. "
            "Comparisons against a real Total Return index will differ."
        )
        proxy = (1 + ds.prices.pct_change().mean(axis=1).fillna(0)).cumprod() * 1000
        ds.benchmark = proxy.rename("EW_PROXY")

    return ds


def _read_optional_panel(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def _load_cached(
    cfg: AppConfig, uc: UniverseConfig, start: str | None, end: str | None
) -> Dataset:
    from .fundamentals.base import FundamentalsCache
    from .kite_client import PriceStore, build_price_panel
    from .reference import load_benchmark
    from .security_master import SecurityMaster

    sm = SecurityMaster.load(cfg.paths.data_dir)
    if sm.securities.empty:
        raise RuntimeError(
            "No security master found. Run `python scripts/fetch_data.py --build-master` first."
        )

    store = PriceStore(cfg.paths.data_dir)
    token_to_isin = {
        int(r.kite_token): r.isin
        for r in sm.securities.itertuples()
        if pd.notna(r.kite_token)
    }
    raw_prices = build_price_panel(store, token_to_isin, "close")
    volumes = build_price_panel(store, token_to_isin, "volume")
    if raw_prices.empty:
        raise RuntimeError("No cached price data. Run `python scripts/fetch_data.py` first.")

    if start:
        raw_prices = raw_prices.loc[raw_prices.index >= pd.Timestamp(start)]
        volumes = volumes.loc[volumes.index >= pd.Timestamp(start)]
    if end:
        raw_prices = raw_prices.loc[raw_prices.index <= pd.Timestamp(end)]
        volumes = volumes.loc[volumes.index <= pd.Timestamp(end)]

    return _assemble(
        cfg, uc,
        raw_prices=raw_prices,
        volumes=volumes,
        securities=sm.securities,
        benchmark=load_benchmark(cfg, "NIFTY50_TRI", raw_prices.index),
        fundamentals=FundamentalsCache(cfg.paths.data_dir).read(),
        earnings=_read_optional(cfg.paths.data_dir / "earnings.parquet"),
        risk_free=load_risk_free(cfg, raw_prices.index),
        corporate_actions=CorporateActions.load(cfg.paths.data_dir),
    )


def _assemble(
    cfg: AppConfig,
    uc: UniverseConfig,
    raw_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    securities: pd.DataFrame,
    benchmark: pd.Series,
    fundamentals: pd.DataFrame,
    earnings: pd.DataFrame,
    risk_free: pd.Series,
    corporate_actions: CorporateActions,
    membership: IndexMembership | None = None,
) -> Dataset:
    """Shared assembly: adjust prices, compute ADV, build the eligibility mask."""
    prices = total_return_panel(raw_prices, corporate_actions)
    # total_return_panel rebases to 100; keep it comparable to raw levels.
    prices = prices / prices.iloc[0] * raw_prices.iloc[0].fillna(100.0)

    adv = average_daily_value(raw_prices, volumes) if not volumes.empty else pd.DataFrame()
    sectors = sector_series(securities)

    filters = UniverseFilters.from_config(cfg)
    filters.min_adv_inr = uc.min_adv_inr
    filters.min_price_inr = uc.min_price_inr
    filters.min_listing_days = uc.min_listing_days
    filters.min_financial_quarters = uc.min_financial_quarters

    if membership is None:
        membership = IndexMembership.load(cfg.paths.data_dir)
    if membership.df.empty and uc.point_in_time:
        membership = IndexMembership.from_static(list(raw_prices.columns), uc.index)

    # Guard against membership from a different data source (e.g. real NSE
    # constituents loaded while running on synthetic data). Zero overlap means
    # the mask would silently empty the universe.
    if not membership.df.empty:
        overlap = len(set(membership.df["isin"]) & set(raw_prices.columns))
        if overlap == 0:
            log.warning(
                "Cached index membership shares no ISINs with the loaded prices; "
                "ignoring it and using all available names. (Mixing data sources?)"
            )
            membership = IndexMembership.from_static(list(raw_prices.columns), uc.index)

    custom = None
    if uc.index == "custom" and uc.custom_list_file:
        custom = _read_custom_list(uc.custom_list_file)

    coverage = (
        fundamentals_coverage_matrix(fundamentals, raw_prices.index, list(raw_prices.columns))
        if not fundamentals.empty and filters.min_financial_quarters > 0
        else None
    )

    builder = UniverseBuilder(filters, membership, uc.index, custom)
    eligibility = builder.build(raw_prices, volumes, coverage)

    return Dataset(
        prices=prices,
        raw_prices=raw_prices,
        volumes=volumes,
        adv=adv,
        eligibility=eligibility,
        securities=securities,
        sectors=sectors,
        benchmark=benchmark,
        risk_free=risk_free,
        fundamentals=fundamentals,
        earnings=earnings,
    )


def _read_optional(path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _read_custom_list(path: str) -> list[str]:
    p = pd.read_csv(path)
    col = "isin" if "isin" in p.columns else p.columns[0]
    return list(p[col].astype(str))
