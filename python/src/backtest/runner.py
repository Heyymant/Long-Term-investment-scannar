"""The orchestrator: strategy_config.json in, artifacts out.

Sequence:
  load data -> build sleeve signals -> allocate -> backtest -> metrics
  -> attribution -> validation -> write artifacts -> register the run

This is what `scripts/run_backtest.py` and the dashboard's "Run" button both
call, so the dashboard and the CLI can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..allocator import combined_signals
from ..config import AppConfig, get_logger, load_config
from ..data.loader import Dataset, load_dataset
from ..factors.composite import build_score_panel
from ..factors.ic_icir import analyze_factors
from ..portfolio.construction import rebalance_dates
from ..reports.artifacts import ArtifactWriter
from ..schema import StrategyConfig
from .attribution import capm_alpha_beta, run_attribution
from .costs import CostModel, breakeven_cost_bps, capacity_estimate, cost_sensitivity
from .engine_event import run_event_backtest
from .engine_vectorized import BacktestResult, run_vectorized_backtest
from .metrics import (
    compute_metrics,
    drawdown_series,
    monthly_return_table,
    regime_breakdown,
    rolling_metrics,
)
from .registry import RunRegistry, data_hash
from .statistics import validate_strategy
from .taxes import TaxConfig

log = get_logger(__name__)


@dataclass
class RunOutput:
    run_id: str
    result: BacktestResult
    metrics: dict[str, Any] = field(default_factory=dict)
    attribution: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    artifacts_dir: str = ""

    def headline(self) -> str:
        m = self.metrics.get("combined", {})
        return (
            f"CAGR {_pct(m.get('cagr'))}  Sharpe {_num(m.get('sharpe'))}  "
            f"MaxDD {_pct(m.get('max_drawdown'))}  Turnover {_num(m.get('turnover'))}"
        )


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{float(v) * 100:.2f}%"


def _num(v: Any) -> str:
    return "n/a" if v is None else f"{float(v):.2f}"


def run_strategy(
    config: StrategyConfig,
    cfg: AppConfig,
    dataset: Dataset | None = None,
    engine: str = "vectorized",          # vectorized | event | both
    validate: bool = True,
    write_artifacts: bool = True,
    study: str = "adhoc",
    source: str | None = None,
    set_latest: bool = True,
) -> RunOutput:
    """Execute one strategy design end to end."""
    ds = dataset or load_dataset(
        cfg, config.universe, source=source,
        start=config.window.start, end=config.window.end,
    )

    registry = RunRegistry(cfg.paths.artifacts_dir)
    record = registry.start(
        name=config.name,
        config_hash=config.config_hash(),
        data_hash=data_hash(ds.prices, ds.fundamentals),
        seed=config.seed,
        study=study,
        kind="backtest",
    )

    try:
        out = _execute(
            config, cfg, ds, engine, validate, write_artifacts, registry, record.run_id,
            set_latest=set_latest,
        )
        registry.complete(record, out.metrics.get("combined", {}))
        if write_artifacts and set_latest:
            registry.set_latest(record.run_id)
        return out
    except Exception as exc:
        registry.fail(record, str(exc))
        raise


def _execute(
    config: StrategyConfig,
    cfg: AppConfig,
    ds: Dataset,
    engine: str,
    validate: bool,
    write_artifacts: bool,
    registry: RunRegistry,
    run_id: str,
    set_latest: bool = True,
) -> RunOutput:
    cost_model = CostModel.from_config(cfg)
    tax_config = TaxConfig.from_config(cfg)
    tax_config.enabled = config.apply_taxes

    sleeves = config.active_sleeves()
    log.info("Running '%s' (run %s) with sleeves %s", config.name, run_id, sleeves)

    sleeve_signals, sleeve_artifacts = _build_sleeve_signals(config, ds)

    # Multi-sleeve runs pre-merge into combined weights; a pure Sleeve A run
    # lets the engine compute scores itself (cheaper, and keeps one code path).
    merged = None
    if len(sleeves) > 1 or sleeves in (["B"], ["C"]):
        merged = combined_signals(
            config,
            sleeve_a=sleeve_signals.get("A"),
            sleeve_b=sleeve_signals.get("B"),
            sleeve_c=sleeve_signals.get("C"),
            sectors=ds.sectors,
        )

    engine_kwargs = dict(
        config=config, prices=ds.prices, eligibility=ds.eligibility,
        fundamentals=ds.fundamentals, sectors=ds.sectors,
        benchmark=ds.benchmark, adv=ds.adv, cost_model=cost_model,
        sleeve_signals=merged,
    )

    if engine == "event":
        result = run_event_backtest(tax_config=tax_config, **engine_kwargs)
    else:
        result = run_vectorized_backtest(**engine_kwargs)

    consistency = {}
    if engine == "both":
        event_result = run_event_backtest(tax_config=tax_config, **engine_kwargs)
        consistency = compare_engines(result, event_result)
        log.info("Engine consistency: %s", consistency.get("verdict"))

    if result.returns.empty:
        raise RuntimeError(
            "Backtest produced no returns - check the date window, universe filters "
            "and that enough history exists for the warm-up period."
        )

    bench_returns = ds.benchmark.pct_change().reindex(result.returns.index)
    metrics_obj = compute_metrics(
        result.returns, bench_returns, ds.risk_free,
        turnover=result.annual_turnover,
        after_tax_returns=result.after_tax_equity.pct_change()
        if not result.after_tax_equity.empty else None,
        total_tax=result.total_tax or None,
        exposure=result.exposure_history,
    )
    metrics = {"combined": metrics_obj.to_dict()}

    attribution = _attribution(config, cfg, ds, result, bench_returns)
    validation = _validation(config, result, bench_returns, registry) if validate else {}

    extras = _extras(config, ds, result, metrics_obj, cost_model, bench_returns)

    if write_artifacts:
        _write_artifacts(
            cfg, registry, run_id, config, ds, result, metrics,
            attribution, validation, extras, sleeve_artifacts, consistency,
            set_latest=set_latest,
        )

    return RunOutput(
        run_id=run_id, result=result, metrics=metrics,
        attribution=attribution, validation=validation,
        artifacts_dir=str(registry.run_dir(run_id)),
    )


# --------------------------------------------------------------------------- #
# Sleeve signals
# --------------------------------------------------------------------------- #
def _build_sleeve_signals(
    config: StrategyConfig, ds: Dataset
) -> tuple[dict[str, dict], dict[str, pd.DataFrame]]:
    signals: dict[str, dict] = {}
    artifacts: dict[str, pd.DataFrame] = {}

    warmup = max(config.sleeve_a.factors.low_vol_lookback_days, 252) + 21

    if config.sleeve_a.enabled and config.allocation.sleeve_a > 0:
        from ..factors.composite import compute_composite
        from ..portfolio.construction import construct_portfolio
        from ..portfolio.regime import compute_exposure_series, exposure_on
        from ..data.universe import eligible_on

        dates = rebalance_dates(ds.prices.index, config.sleeve_a.rebalance.value, warmup)
        exposure = compute_exposure_series(ds.benchmark, config.sleeve_a.regime)
        weights: dict[pd.Timestamp, pd.Series] = {}
        holdings: set[str] = set()

        for dt in dates:
            universe = eligible_on(ds.eligibility, dt)
            if not universe:
                continue
            scores = compute_composite(
                ds.prices, dt, config.sleeve_a.factors, universe,
                ds.fundamentals, ds.sectors, config.sleeve_a.sector_neutralize,
            ).composite
            if scores.empty:
                continue
            p = construct_portfolio(
                scores, ds.prices, config.sleeve_a, dt, holdings, ds.sectors,
                exposure_on(exposure, dt, 1.0),
            )
            if not p.weights.empty:
                weights[dt] = p.weights
                holdings = set(p.weights.index)
        signals["A"] = weights

    if config.sleeve_b.enabled and config.allocation.sleeve_b > 0:
        from ..statarb.signals import build_sleeve_b_signals

        dates = pd.DatetimeIndex(
            ds.prices.index[warmup::max(1, config.sleeve_b.reversion_horizon_days)]
        )
        w, pairs, sigs = build_sleeve_b_signals(
            ds.prices, dates, ds.eligibility, config.sleeve_b, ds.sectors, ds.benchmark
        )
        signals["B"] = w
        artifacts["statarb_pairs"] = pairs
        artifacts["statarb_signals"] = sigs

    if config.sleeve_c.enabled and config.allocation.sleeve_c > 0:
        from ..data.earnings import EarningsCalendar, build_earnings_from_fundamentals
        from ..events.signals import build_event_signals, event_rebalance_dates
        from ..events.sue import compute_sue_panel

        calendar = (
            EarningsCalendar.from_frame(ds.earnings) if not ds.earnings.empty
            else build_earnings_from_fundamentals(ds.fundamentals)
        )
        sue_panel = compute_sue_panel(calendar.data, config.sleeve_c.sue_lookback_quarters)
        dates = event_rebalance_dates(ds.prices.index, 5, warmup)
        ev = build_event_signals(
            ds.prices, sue_panel, dates, ds.eligibility, config.sleeve_c,
            ds.benchmark, ds.adv,
        )
        signals["C"] = ev.weights_by_date
        artifacts["earnings_calendar"] = calendar.data
        artifacts["event_signals"] = ev.to_frame()

    return signals, artifacts


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _attribution(
    config: StrategyConfig, cfg: AppConfig, ds: Dataset,
    result: BacktestResult, bench_returns: pd.Series,
) -> dict[str, Any]:
    from ..data.iima_factors import align_factors, load_iima_factors

    out: dict[str, Any] = {}
    factors = load_iima_factors(cfg)

    if not factors.empty:
        port, fac = align_factors(result.returns, factors, "M")
        if len(port) >= 12:
            out["factor_model"] = run_attribution(port, fac, "monthly").to_dict()

    out["capm"] = capm_alpha_beta(result.returns, bench_returns, ds.risk_free)
    if not out.get("factor_model"):
        out["note"] = (
            "IIMA factor library not found in data/. Only CAPM alpha vs the benchmark "
            "is reported; that is NOT true alpha net of factor exposure."
        )
    return out


def _validation(
    config: StrategyConfig, result: BacktestResult,
    bench_returns: pd.Series, registry: RunRegistry,
) -> dict[str, Any]:
    # Trial count drives the Deflated Sharpe Ratio.
    n_trials = max(1, registry.trial_count(config_hash=config.config_hash()))
    report = validate_strategy(
        result.returns, n_trials=n_trials, benchmark_returns=bench_returns, seed=config.seed
    )
    return report.to_dict()


def _extras(
    config: StrategyConfig, ds: Dataset, result: BacktestResult,
    metrics_obj, cost_model: CostModel, bench_returns: pd.Series,
) -> dict[str, Any]:
    from ..portfolio.regime import overlay_diagnostics, regime_label

    extras: dict[str, Any] = {}

    extras["monthly_returns"] = monthly_return_table(result.returns)
    extras["rolling"] = rolling_metrics(result.returns, 252, bench_returns)

    if not ds.benchmark.empty:
        regimes = regime_label(ds.benchmark).reindex(result.returns.index)
        extras["regime"] = regime_breakdown(result.returns, regimes)
        if config.sleeve_a.regime.enabled:
            extras["overlay"] = overlay_diagnostics(ds.benchmark, result.exposure_history)

    if metrics_obj.cagr is not None and not np.isnan(metrics_obj.cagr):
        extras["breakeven_cost_bps"] = breakeven_cost_bps(
            metrics_obj.cagr, metrics_obj.benchmark_cagr or 0.0, result.annual_turnover
        )
    extras["cost_sensitivity"] = cost_sensitivity(result.returns, result.annual_turnover)

    if not result.weights_history.empty and not ds.adv.empty:
        last_w = result.weights_history.iloc[-1]
        last_w = last_w[last_w > 0]
        extras["capacity"] = capacity_estimate(last_w, ds.adv.iloc[-1])

    return extras


def compare_engines(
    vectorized: BacktestResult, event: BacktestResult, tolerance: float = 0.15
) -> dict[str, Any]:
    """Cross-check the two engines.

    They will not match exactly - the event engine uses integer shares, real
    cash and lot-level tax - but a large gap means one of them is wrong.
    """
    from .metrics import cagr, sharpe_ratio

    v_sharpe, e_sharpe = sharpe_ratio(vectorized.returns), sharpe_ratio(event.returns)
    v_cagr, e_cagr = cagr(vectorized.returns), cagr(event.returns)

    sharpe_gap = abs(v_sharpe - e_sharpe) if not (np.isnan(v_sharpe) or np.isnan(e_sharpe)) else np.nan
    cagr_gap = abs(v_cagr - e_cagr) if not (np.isnan(v_cagr) or np.isnan(e_cagr)) else np.nan

    ok = (not np.isnan(sharpe_gap)) and sharpe_gap <= tolerance
    return {
        "vectorized_sharpe": _f(v_sharpe), "event_sharpe": _f(e_sharpe),
        "vectorized_cagr": _f(v_cagr), "event_cagr": _f(e_cagr),
        "sharpe_gap": _f(sharpe_gap), "cagr_gap": _f(cagr_gap),
        "tolerance": tolerance,
        "consistent": bool(ok),
        "verdict": (
            "engines agree within tolerance" if ok
            else "ENGINES DISAGREE - investigate before trusting these results"
        ),
    }


def _f(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _write_artifacts(
    cfg: AppConfig, registry: RunRegistry, run_id: str, config: StrategyConfig,
    ds: Dataset, result: BacktestResult, metrics: dict, attribution: dict,
    validation: dict, extras: dict, sleeve_artifacts: dict, consistency: dict,
    set_latest: bool = True,
) -> None:
    writer = ArtifactWriter(registry.run_dir(run_id), run_id)

    writer.write_config(config)
    writer.write_metrics(metrics)
    writer.write_equity_curve(
        result.equity_curve,
        after_tax=result.after_tax_equity if not result.after_tax_equity.empty else None,
        benchmark=ds.benchmark,
        exposure=result.exposure_history,
        drawdown=drawdown_series(result.returns),
    )
    writer.write_json("attribution.json", attribution)
    if validation:
        writer.write_json("validation_stats.json", validation)

    # Risk snapshot from the last rebalance.
    if result.risk_states:
        last = result.risk_states[-1]
        writer.write_json("risk.json", {
            "latest": last.to_dict(),
            "history": [s.to_dict() for s in result.risk_states[-24:]],
            "capacity": extras.get("capacity", {}),
            "breakeven_cost_bps": extras.get("breakeven_cost_bps"),
            "overlay": extras.get("overlay", {}),
        })

    # Current rankings and targets.
    if not result.weights_history.empty:
        last_w = result.weights_history.iloc[-1]
        last_w = last_w[last_w > 0].sort_values(ascending=False)
        targets = pd.DataFrame({
            "isin": last_w.index,
            "weight": last_w.values,
            "symbol": [_symbol(ds, i) for i in last_w.index],
            "sector": [ds.sectors.get(i, "UNKNOWN") for i in last_w.index],
        })
        writer.write_csv("holdings_target.csv", targets)

    rankings_df = _write_rankings(writer, config, ds)
    _write_trade_signals(
        writer, config, ds, result, rankings_df, sleeve_artifacts,
    )

    for name, df in sleeve_artifacts.items():
        if df is not None and not df.empty:
            writer.write_parquet(f"{name}.parquet", df)

    if isinstance(extras.get("monthly_returns"), pd.DataFrame):
        writer.write_parquet("monthly_returns.parquet", extras["monthly_returns"], index=True)
    if isinstance(extras.get("cost_sensitivity"), pd.DataFrame):
        writer.write_parquet("cost_sensitivity.parquet", extras["cost_sensitivity"])
    if isinstance(extras.get("regime"), pd.DataFrame):
        writer.write_parquet("regime_breakdown.parquet", extras["regime"])

    if ds.health is not None:
        writer.write_json("data_health.json", ds.health.to_dict())
    if consistency:
        writer.write_json("engine_consistency.json", consistency)

    writer.finalize(cfg.paths.artifacts_dir, update_latest=set_latest)


def _write_rankings(writer: ArtifactWriter, config: StrategyConfig, ds: Dataset) -> pd.DataFrame:
    """Current factor scores - what the dashboard's Rankings view shows."""
    from ..data.universe import eligible_on
    from ..factors.composite import compute_composite

    dt = ds.prices.index[-1]
    universe = eligible_on(ds.eligibility, dt)
    if not universe:
        return pd.DataFrame()

    cs = compute_composite(
        ds.prices, dt, config.sleeve_a.factors, universe,
        ds.fundamentals, ds.sectors, config.sleeve_a.sector_neutralize,
    )
    if cs.composite.empty:
        return pd.DataFrame()

    df = cs.to_frame()
    df["symbol"] = [_symbol(ds, i) for i in df.index]
    df["sector"] = [ds.sectors.get(i, "UNKNOWN") for i in df.index]
    df["rank"] = df["composite"].rank(ascending=False)
    df = df.sort_values("composite", ascending=False).reset_index().rename(columns={"index": "isin"})
    from ..data.asset_class import classify
    df["asset_class"] = [
        classify(str(i), str(s)) for i, s in zip(df["isin"], df["symbol"])
    ]

    etf_df = _etf_rankings(ds)
    if not etf_df.empty:
        writer.write_parquet("etf_rankings.parquet", etf_df)
        have = set(df["isin"].astype(str))
        extra = etf_df[~etf_df["isin"].astype(str).isin(have)]
        if not extra.empty:
            df = pd.concat([df, extra], ignore_index=True, sort=False)

    df = _add_screener_features(df, ds)
    writer.write_parquet("rankings.parquet", df)
    return df


def _read_parquet_panel(path: Path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - a corrupt cache must not abort rankings
        return pd.DataFrame()
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _panel_row(px: pd.DataFrame, loc: int) -> pd.Series:
    if not isinstance(px, pd.DataFrame) or px.empty or abs(loc) > len(px):
        return pd.Series(dtype=float)
    row = px.iloc[loc]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0] if len(row) else pd.Series(dtype=float)
    if not isinstance(row, pd.Series):
        return pd.Series(dtype=float)
    vals = pd.to_numeric(row, errors="coerce")
    vals.index = vals.index.map(str)
    return vals.dropna()


def _overlay_row(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """Prefer `primary` (live bhavcopy) and fill holes from `fallback`."""
    out: dict[str, float] = {}
    for series in (fallback, primary):
        if series is None or series.empty:
            continue
        for key, val in series.items():
            if pd.notna(val):
                try:
                    out[str(key)] = float(val)
                except (TypeError, ValueError):
                    continue
    return pd.Series(out, dtype=float)


def _map_by_isin(isins: pd.Series, series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(np.nan, index=isins.index)
    s = series.copy()
    s.index = s.index.astype(str)
    mapped = isins.astype(str).map(s)
    return pd.to_numeric(mapped, errors="coerce")


def _session_returns(px: pd.DataFrame, isins: pd.Series) -> dict[str, pd.Series]:
    windows = {"ret_1w": 5, "ret_1m": 21, "ret_3m": 63, "ret_6m": 126, "ret_1y": 252, "ret_3y": 756}
    empty = {c: pd.Series(np.nan, index=isins.index) for c in windows}
    if not isinstance(px, pd.DataFrame) or px.empty:
        return empty
    last = _panel_row(px, -1)
    out = {}
    for col, n in windows.items():
        if len(px) <= n:
            out[col] = pd.Series(np.nan, index=isins.index)
            continue
        past = _panel_row(px, -1 - n)
        vals = []
        for i in isins.astype(str):
            a, b = last.get(i), past.get(i)
            if pd.notna(a) and pd.notna(b) and b > 0:
                vals.append(float(a / b - 1.0))
            else:
                vals.append(np.nan)
        out[col] = pd.Series(vals, index=isins.index)
    return out


def _live_close_panel() -> pd.DataFrame:
    """Most recent NSE rupee close file on disk (not TR-adjusted backtest prices)."""
    try:
        root = load_config().paths.data_dir
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    live = _read_parquet_panel(root / "panels" / "close.parquet")
    if not live.empty:
        return live
    return _read_parquet_panel(root / "panel_close.parquet")


def _live_volume_panel() -> pd.DataFrame:
    try:
        root = load_config().paths.data_dir
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    return _read_parquet_panel(root / "panels" / "volume.parquet")


def _add_screener_features(df: pd.DataFrame, ds) -> pd.DataFrame:
    """Macrotrends-style identity, EOD close, performance and latest fundamentals.

    Closing price comes from the latest NSE bhavcopy (rupee close), not from
    total-return-adjusted backtest prices and not from Kite. Missing legs stay
    null so the dashboard never invents PE, market cap or returns.
    """
    from ..data.fundamentals.base import as_of

    out = df.copy()
    if "symbol" in out.columns:
        out["name"] = out["symbol"]
    elif "isin" in out.columns:
        out["name"] = out["isin"]
    out["exchange"] = "NSE"
    out["country"] = "India"
    if "sector" in out.columns:
        out["industry"] = out["sector"]
    else:
        out["industry"] = "UNKNOWN"

    sec = getattr(ds, "securities", None)
    if isinstance(sec, pd.DataFrame) and not sec.empty and "isin" in out.columns:
        keyed = sec.copy()
        if "isin" in keyed.columns:
            keyed = keyed.drop_duplicates(subset=["isin"]).set_index("isin")
        keyed.index = keyed.index.astype(str)
        if "name" in keyed.columns:
            mapped = out["isin"].astype(str).map(keyed["name"])
            out["name"] = mapped.where(mapped.notna() & (mapped.astype(str).str.len() > 0), out["name"])
        for src in ("industry", "sector"):
            if src in keyed.columns:
                mapped = out["isin"].astype(str).map(keyed[src])
                out["industry"] = mapped.where(mapped.notna(), out["industry"])
                break

    isins = out["isin"].astype(str) if "isin" in out.columns else pd.Series(dtype=str, index=out.index)
    raw = getattr(ds, "raw_prices", None)
    raw = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()
    live = _live_close_panel()

    last = _overlay_row(_panel_row(live, -1), _panel_row(raw, -1))
    prev = _overlay_row(_panel_row(live, -2), _panel_row(raw, -2))
    if not last.empty:
        out["close"] = _map_by_isin(isins, last)
        out["last_price"] = out["close"]
        out["prev_close"] = _map_by_isin(isins, prev) if not prev.empty else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            out["day_chg"] = np.where(
                (out["prev_close"] > 0) & out["last_price"].notna(),
                (out["last_price"] / out["prev_close"] - 1.0) * 100.0,
                np.nan,
            )

    vol_live = _live_volume_panel()
    vol_ds = getattr(ds, "volumes", None)
    vol_ds = vol_ds if isinstance(vol_ds, pd.DataFrame) else pd.DataFrame()
    vol_row = _overlay_row(_panel_row(vol_live, -1), _panel_row(vol_ds, -1))
    if not vol_row.empty:
        out["volume"] = _map_by_isin(isins, vol_row)

    live_ret = _session_returns(live if not live.empty else raw, isins)
    hist_px = getattr(ds, "prices", None)
    hist_ret = _session_returns(hist_px if isinstance(hist_px, pd.DataFrame) else pd.DataFrame(), isins)
    for col in ("ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_3y"):
        a = live_ret.get(col, pd.Series(np.nan, index=out.index))
        b = hist_ret.get(col, pd.Series(np.nan, index=out.index))
        out[col] = a.where(a.notna(), b)

    fund = getattr(ds, "fundamentals", None)
    if isinstance(fund, pd.DataFrame) and not fund.empty and "isin" in out.columns:
        latest = as_of(fund, pd.Timestamp.now().normalize())
        if latest is not None and not latest.empty:
            for col in ("pe", "pb", "market_cap", "roic", "debt_to_assets", "payout",
                        "revenue", "eps", "net_income", "earnings_yield",
                        "net_margin", "pretax_margin", "eps_ttm"):
                if col not in latest.columns:
                    continue
                mapped = out["isin"].astype(str).map(latest[col])
                dest = {"roic": "roe", "debt_to_assets": "debt_equity"}.get(col, col)
                out[dest] = pd.to_numeric(mapped, errors="coerce")

            px_now = pd.to_numeric(out["last_price"], errors="coerce") if "last_price" in out.columns \
                else pd.Series(np.nan, index=out.index)
            eps_ttm = pd.to_numeric(out["eps_ttm"], errors="coerce") if "eps_ttm" in out.columns \
                else pd.Series(np.nan, index=out.index)
            live_pe = np.where((eps_ttm > 0) & (px_now > 0), px_now / eps_ttm, np.nan)
            live_pe = pd.Series(live_pe, index=out.index)
            if "pe" in out.columns:
                out["pe"] = live_pe.where(live_pe.notna(), pd.to_numeric(out["pe"], errors="coerce"))
            else:
                out["pe"] = live_pe
            live_ey = np.where(out["pe"] > 0, 1.0 / out["pe"], np.nan)
            live_ey = pd.Series(live_ey, index=out.index)
            if "earnings_yield" in out.columns:
                out["earnings_yield"] = live_ey.where(
                    live_ey.notna(), pd.to_numeric(out["earnings_yield"], errors="coerce")
                )
            else:
                out["earnings_yield"] = live_ey

            ni = pd.to_numeric(out["net_income"], errors="coerce") if "net_income" in out.columns \
                else pd.Series(np.nan, index=out.index)
            eps_q = pd.to_numeric(out["eps"], errors="coerce") if "eps" in out.columns \
                else pd.Series(np.nan, index=out.index)
            shares = np.where((eps_q > 0) & ni.notna(), ni / eps_q, np.nan)
            live_mcap = np.where((px_now > 0) & np.isfinite(shares) & (shares > 0), px_now * shares, np.nan)
            live_mcap = pd.Series(live_mcap, index=out.index)
            existing = pd.to_numeric(out["market_cap"], errors="coerce").replace(0, np.nan) \
                if "market_cap" in out.columns else pd.Series(np.nan, index=out.index)
            out["market_cap"] = live_mcap.where(live_mcap.notna(), existing)

            from ..factors.quality import compute_iima_dimensions
            dims = compute_iima_dimensions(
                fund, pd.Timestamp.now().normalize(), list(dict.fromkeys(isins.tolist())),
            )
            if not dims.empty:
                for col in ("profitability", "growth", "safety", "payout_z"):
                    if col in dims.columns:
                        out[col] = out["isin"].astype(str).map(dims[col])
            if "low_vol" in out.columns:
                lv = pd.to_numeric(out["low_vol"], errors="coerce")
                if "safety" in out.columns:
                    out["safety"] = pd.to_numeric(out["safety"], errors="coerce").fillna(lv)
                else:
                    out["safety"] = lv
    return out


def _etf_rankings(ds) -> pd.DataFrame:
    from ..factors.satellite import score_etfs
    try:
        return score_etfs(ds.prices, ds.securities, ds.adv)
    except Exception:
        return pd.DataFrame()


def _write_trade_signals(
    writer: ArtifactWriter,
    config: StrategyConfig,
    ds: Dataset,
    result,
    rankings: pd.DataFrame,
    sleeve_artifacts: dict[str, pd.DataFrame],
) -> None:
    """BUY/SELL list from the same rank-buffer rules the backtest used."""
    from ..signals.trading_rules import build_trade_signals

    if rankings is None or rankings.empty:
        return

    held: set[str] = set()
    weights = None
    if not result.weights_history.empty:
        last_w = result.weights_history.iloc[-1]
        last_w = last_w[last_w > 0]
        held = set(last_w.index.astype(str))
        weights = last_w

    report = build_trade_signals(
        rankings, held, config, weights,
        event_signals=sleeve_artifacts.get("event_signals"),
        statarb_signals=sleeve_artifacts.get("statarb_signals"),
    )
    writer.write_parquet("trade_signals.parquet", report.frame)
    writer.write_json("trade_signals_summary.json", report.summary)


def _symbol(ds: Dataset, isin: str) -> str:
    if ds.securities.empty or "isin" not in ds.securities.columns:
        return isin
    row = ds.securities.loc[ds.securities["isin"] == isin, "symbol"]
    return str(row.iloc[0]) if not row.empty else isin
