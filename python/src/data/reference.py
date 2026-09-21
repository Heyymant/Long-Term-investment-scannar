"""Reference data: risk-free rate, TRI benchmarks, sector classification.

Small module, but it matters: Sharpe/Sortino/attribution are all wrong if the
risk-free series is wrong, and comparing a total-return strategy against a
price-return index flatters the strategy by roughly the dividend yield.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import AppConfig, get_logger

log = get_logger(__name__)

# Benchmarks must be TOTAL RETURN to be a fair comparison.
KNOWN_BENCHMARKS = {
    "NIFTY50_TRI": "NIFTY 50 Total Return Index",
    "NIFTY200_MOMENTUM30_TRI": "NIFTY200 Momentum 30 Total Return Index",
    "NIFTY500_TRI": "NIFTY 500 Total Return Index",
}


# --------------------------------------------------------------------------- #
# Risk-free rate
# --------------------------------------------------------------------------- #
def load_risk_free(
    cfg: AppConfig, index: pd.DatetimeIndex | None = None
) -> pd.Series:
    """Annualized risk-free rate aligned to `index`.

    Prefers a cached 91-day T-bill series; falls back to the configured
    constant so nothing breaks offline.
    """
    source = cfg.get("market.risk_free.source", "constant")
    const = float(cfg.get("market.risk_free.constant_annual", 0.065))

    path = cfg.paths.data_dir / "risk_free.parquet"
    if source != "constant":
        series = None
        if path.exists():
            df = pd.read_parquet(path)
            series = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
        else:
            history = cfg.paths.data_dir / "index_history.parquet"
            if history.exists():
                series = build_risk_free_from_index_history(pd.read_parquet(history))
                if not series.empty:
                    save_risk_free(series, cfg)
        if series is not None and not series.empty:
            series.index = pd.to_datetime(series.index)
            series = series.sort_index()
            if index is not None:
                series = series.reindex(index).ffill().bfill()
            return series.rename("risk_free")
        log.warning("Risk-free source '%s' unavailable; using constant %.3f", source, const)

    if index is None:
        return pd.Series(dtype=float, name="risk_free")
    return pd.Series(const, index=index, name="risk_free")


def daily_risk_free(annual: pd.Series, trading_days: int = 252) -> pd.Series:
    """Convert an annualized rate to a daily compounding rate."""
    return (1.0 + annual) ** (1.0 / trading_days) - 1.0


def save_risk_free(series: pd.Series, cfg: AppConfig) -> Path:
    p = cfg.paths.data_dir / "risk_free.parquet"
    series.to_frame("risk_free").to_parquet(p)
    return p


def build_risk_free_from_index_history(index_history: pd.DataFrame) -> pd.Series:
    """Annualized overnight rate implied by NSE's Nifty 1D Rate Index.

    That index is a total-return series of Indian overnight money-market rates
    (MIBOR). The daily percentage change is the overnight rate itself, which
    we scale by 365 to an annualized series for Sharpe / attribution.
    """
    if index_history.empty or "index_name" not in index_history.columns:
        return pd.Series(dtype=float, name="risk_free")

    sel = index_history[
        index_history["index_name"].astype(str).str.strip() == "Nifty 1D Rate Index"
    ]
    if sel.empty:
        return pd.Series(dtype=float, name="risk_free")

    levels = (
        sel.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")["close"]
        .astype(float)
        .sort_index()
    )
    daily = levels.pct_change()
    # A handful of published restatements create 1-4% one-day jumps. Overnight
    # rates do not move like that; clip before annualizing.
    daily = daily.clip(lower=-0.005, upper=0.005)
    annual = (daily * 365.0).clip(lower=0.0, upper=0.25).bfill().ffill()
    return annual.rename("risk_free")


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def load_benchmark(
    cfg: AppConfig, name: str = "NIFTY50_TRI", index: pd.DatetimeIndex | None = None
) -> pd.Series:
    """Load a TRI benchmark level series from cache."""
    path = cfg.paths.data_dir / "benchmarks" / f"{name}.parquet"
    if not path.exists():
        log.warning("Benchmark %s not cached at %s", name, path)
        return pd.Series(dtype=float, name=name)
    df = pd.read_parquet(path)
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().rename(name)
    if index is not None:
        s = s.reindex(index).ffill()
    return s


def save_benchmark(series: pd.Series, name: str, cfg: AppConfig) -> Path:
    d = cfg.paths.data_dir / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.parquet"
    series.to_frame(name).to_parquet(p)
    return p


def benchmark_returns(levels: pd.Series) -> pd.Series:
    return levels.pct_change().fillna(0.0)


# --------------------------------------------------------------------------- #
# Sector classification
# --------------------------------------------------------------------------- #
def load_sectors(cfg: AppConfig) -> pd.DataFrame:
    """isin/symbol -> sector, industry."""
    p = cfg.paths.data_dir / "sectors.parquet"
    if p.exists():
        return pd.read_parquet(p)
    csv = cfg.paths.data_dir / "sectors.csv"
    if csv.exists():
        return pd.read_csv(csv)
    log.warning("No sector classification found; sector-neutralization will be a no-op")
    return pd.DataFrame(columns=["isin", "symbol", "sector", "industry"])


def save_sectors(df: pd.DataFrame, cfg: AppConfig) -> Path:
    p = cfg.paths.data_dir / "sectors.parquet"
    df.to_parquet(p, index=False)
    return p


def sector_series(securities: pd.DataFrame) -> pd.Series:
    """isin -> sector, with a safe UNKNOWN default."""
    if securities.empty or "sector" not in securities.columns:
        return pd.Series(dtype=object)
    return securities.set_index("isin")["sector"].fillna("UNKNOWN")
