#!/usr/bin/env python
"""Horse-race each research-paper signal over the same NSE window.

    python scripts/run_paper_signal_horse_race.py --start 2022-01-01 --end 2024-12-31

Does not replace the dashboard's `latest` pointer. Writes
`artifacts/paper_signal_horse_race.json` and `dashboard/assets/paper_signals.json`
so the screener can turn passing signals into one-click filters.

Quality / value / PEAD are marked `no_data` when no filings were announced
inside the window - we never invent ratios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest.runner import run_strategy
from src.config import load_config
from src.data.loader import load_dataset
from src.presets import get_preset
from src.schema import UniverseConfig

STUDY = "paper_signal_horse_race"

# Isolated paper recipes. `needs_fund` is true when the signal is empty
# without point-in-time filings dated inside the backtest window.
CATALOG: list[dict[str, Any]] = [
    {
        "id": "momentum_only",
        "label": "Momentum 12-1 / 6-1",
        "paper": "Jegadeesh–Titman; Agarwalla, Jacob & Varma (IIMA)",
        "needs_fund": False,
        "filter": {"mom-min": "0", "rank-max": "30"},
        "colset": "factors",
    },
    {
        "id": "momentum_trend",
        "label": "Momentum + 200d trend",
        "paper": "Moskowitz time-series momentum overlay on 12-1",
        "needs_fund": False,
        "filter": {"mom-min": "0", "rank-max": "30"},
        "colset": "factors",
    },
    {
        "id": "low_vol_only",
        "label": "Low volatility",
        "paper": "Indian low-vol / betting-against-beta (long-only)",
        "needs_fund": False,
        "filter": {"lv-min": "0", "rank-max": "50"},
        "colset": "factors",
    },
    {
        "id": "price_only_core",
        "label": "Momentum 60% + low-vol 40%",
        "paper": "IIMA price factors + 200d trend overlay",
        "needs_fund": False,
        "filter": {"mom-min": "0", "lv-min": "0", "rank-max": "40"},
        "colset": "factors",
    },
    {
        "id": "residual_only",
        "label": "Residual reversion",
        "paper": "Long-only residual mean reversion (Sleeve B)",
        "needs_fund": False,
        "filter": {"rank-max": "20"},
        "colset": "factors",
    },
    {
        "id": "quality_only",
        "label": "Quality / QMJ",
        "paper": "Asness–Frazzini–Pedersen; Jacob, Pradeep & Varma (IIMA)",
        "needs_fund": True,
        "filter": {"q-min": "0", "rank-max": "50"},
        "colset": "factors",
    },
    {
        "id": "value_only",
        "label": "Earnings yield / cheapness",
        "paper": "IIMA HML-style value (long-only)",
        "needs_fund": True,
        "filter": {"v-min": "0", "rank-max": "50"},
        "colset": "factors",
    },
    {
        "id": "value_quality",
        "label": "Value × Quality",
        "paper": "Cheap AND good — drop cheap-and-junk",
        "needs_fund": True,
        "filter": {"v-min": "0", "q-min": "0"},
        "junk": True,
        "colset": "factors",
    },
    {
        "id": "conservative_formula",
        "label": "Conservative Formula",
        "paper": "Low-vol + payout + momentum (India conservative formula)",
        "needs_fund": True,
        "filter": {"lv-min": "0", "mom-min": "0", "rank-max": "50"},
        "colset": "factors",
    },
    {
        "id": "full_composite",
        "label": "4-factor composite",
        "paper": "0.35Q + 0.30M + 0.20V + 0.15LV with trend overlay",
        "needs_fund": True,
        "filter": {"q-min": "0", "mom-min": "0", "v-min": "0", "lv-min": "0", "rank-max": "50"},
        "colset": "factors",
    },
    {
        "id": "pead_only",
        "label": "PEAD / SUE",
        "paper": "Bernard & Thomas; Indian post-earnings announcement drift",
        "needs_fund": True,
        "filter": {"rank-max": "20"},
        "colset": "overview",
    },
    {
        "id": "profitability_only",
        "label": "IIMA profitability",
        "paper": "Jacob, Pradeep & Varma: QMJ profitability (margins, ROIC, gross profit)",
        "needs_fund": True,
        "filter": {"prof-min": "0", "rank-max": "50"},
        "colset": "iima",
    },
    {
        "id": "growth_only",
        "label": "IIMA growth",
        "paper": "Jacob, Pradeep & Varma: YoY change in profitability (paper uses 5-year)",
        "needs_fund": True,
        "filter": {"growth-min": "0", "rank-max": "50"},
        "colset": "iima",
    },
    {
        "id": "payout_only",
        "label": "IIMA payout",
        "paper": "Jacob, Pradeep & Varma: cash returned to shareholders (tunnelling)",
        "needs_fund": True,
        "filter": {"payoutz-min": "0", "rank-max": "50"},
        "colset": "iima",
    },
    {
        "id": "profitability_payout",
        "label": "Profitability + payout",
        "paper": "IIMA: the two QMJ legs that drove Indian alpha",
        "needs_fund": True,
        "filter": {"prof-min": "0", "rank-max": "50"},
        "colset": "iima",
    },
]


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def _pit_coverage(fundamentals: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    if fundamentals is None or fundamentals.empty:
        return {"n_rows": 0, "n_isins": 0, "n_eps": 0, "n_margin": 0, "n_pe": 0}
    ad = pd.to_datetime(fundamentals["announce_date"], errors="coerce")
    pit = fundamentals[(ad >= pd.Timestamp(start)) & (ad <= pd.Timestamp(end))]
    return {
        "n_rows": int(len(pit)),
        "n_isins": int(pit["isin"].nunique()) if not pit.empty else 0,
        "n_eps": int(pit["eps"].notna().sum()) if "eps" in pit.columns else 0,
        "n_margin": int(pit["net_margin"].notna().sum()) if "net_margin" in pit.columns else 0,
        "n_pe": int(pit["pe"].notna().sum()) if "pe" in pit.columns else 0,
        "announce_min": str(ad.min().date()) if ad.notna().any() else None,
        "announce_max": str(ad.max().date()) if ad.notna().any() else None,
    }


def _status(metrics: dict[str, Any], traded: bool) -> str:
    if not traded:
        return "no_data"
    sharpe = _f(metrics.get("sharpe"))
    excess = _f(metrics.get("excess_return"))
    cagr = _f(metrics.get("cagr"))
    if sharpe is None or excess is None or cagr is None:
        return "no_data"
    if sharpe > 0 and excess > 0 and cagr > 0:
        return "pass"
    return "fail"


def _refresh_latest_rankings(cfg, ds) -> None:
    """Fill quality/value/PE columns on the dashboard's current run.

    Does not change that run's equity curve or `latest` pointer.
    """
    from src.backtest.registry import RunRegistry
    from src.backtest.runner import _add_screener_features, _symbol, write_price_sparks
    from src.data.universe import eligible_on
    from src.factors.composite import compute_composite
    from src.schema import StrategyConfig

    rid = RunRegistry(cfg.paths.artifacts_dir).latest_run_id()
    if not rid:
        return
    run_dir = Path(cfg.paths.artifacts_dir) / "runs" / rid
    cfg_path = run_dir / "strategy_config.json"
    rank_path = run_dir / "rankings.parquet"
    if not cfg_path.exists() or not rank_path.exists():
        return
    config = StrategyConfig.load(cfg_path)
    dt = ds.prices.index[-1]
    universe = eligible_on(ds.eligibility, dt)
    if not universe:
        return
    cs = compute_composite(
        ds.prices, dt, config.sleeve_a.factors, universe,
        ds.fundamentals, ds.sectors, config.sleeve_a.sector_neutralize,
    )
    if cs.composite.empty:
        return
    df = cs.to_frame()
    df["symbol"] = [_symbol(ds, i) for i in df.index]
    df["sector"] = [ds.sectors.get(i, "UNKNOWN") for i in df.index]
    df["rank"] = df["composite"].rank(ascending=False)
    df = df.sort_values("composite", ascending=False).reset_index().rename(columns={"index": "isin"})
    df = _add_screener_features(df, ds)
    df.to_parquet(rank_path, index=False)
    from src.reports.artifacts import ArtifactWriter
    ArtifactWriter(run_dir, rid)._write_table_json("rankings.json", df, False)
    n_spark = write_price_sparks(run_dir, df, publish_dashboard=True)
    print(f"  refreshed rankings for latest run {rid} ({len(df)} names, {n_spark} 1y charts)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Horse-race paper signals on one NSE window")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--source", default="nse")
    ap.add_argument("--skip", nargs="*", default=[], help="preset ids to skip")
    ap.add_argument("--only", nargs="*", default=[], help="run only these preset ids")
    args = ap.parse_args()

    cfg = load_config()
    print(f"\nPaper-signal horse race  {args.start} -> {args.end}")
    print("Loading NSE dataset once...\n")
    ds = load_dataset(
        cfg,
        UniverseConfig(min_financial_quarters=0),
        source=args.source,
        start=args.start,
        end=args.end,
        run_health_check=False,
    )
    pit = _pit_coverage(ds.fundamentals, args.start, args.end)
    print(
        f"  prices {ds.prices.shape}  {ds.prices.index.min().date()}..{ds.prices.index.max().date()}"
    )
    print(
        f"  PIT filings in window: {pit['n_rows']} rows / {pit['n_isins']} ISINs "
        f"(EPS {pit['n_eps']}, net margin {pit['n_margin']}, PE {pit['n_pe']})"
    )
    if pit.get("announce_min"):
        print(f"  all cached announce dates: {pit['announce_min']} .. {pit['announce_max']}")

    fund_ok = pit["n_isins"] >= 30 and (pit["n_margin"] >= 30 or pit["n_pe"] >= 30 or pit["n_eps"] >= 30)

    previous: dict[str, dict[str, Any]] = {}
    dash_path = Path(__file__).resolve().parents[2] / "dashboard" / "assets" / "paper_signals.json"
    if dash_path.exists():
        try:
            prev_payload = json.loads(dash_path.read_text(encoding="utf-8"))
            previous = {s["id"]: s for s in prev_payload.get("signals", []) if "id" in s}
        except Exception:  # noqa: BLE001
            previous = {}

    rows: list[dict[str, Any]] = []
    for spec in CATALOG:
        sid = spec["id"]
        if args.only and sid not in args.only:
            if sid in previous:
                rows.append(previous[sid])
            continue
        if sid in args.skip:
            if sid in previous:
                rows.append(previous[sid])
            continue

        record = {
            "id": sid,
            "label": spec["label"],
            "paper": spec["paper"],
            "filter": spec["filter"],
            "colset": spec.get("colset", "factors"),
            "junk": bool(spec.get("junk")),
            "needs_fund": spec["needs_fund"],
            "status": "no_data",
            "note": "",
            "run_id": None,
            "cagr": None,
            "sharpe": None,
            "excess": None,
            "max_dd": None,
            "turnover": None,
        }

        if spec["needs_fund"] and not fund_ok:
            record["note"] = (
                "No point-in-time NSE filings inside this window "
                "(announce_date must be on or before the backtest end)."
            )
            rows.append(record)
            print(f"  {sid:<24} no_data  (no PIT filings)")
            continue

        strategy = get_preset(sid)
        strategy.window.start = args.start
        strategy.window.end = args.end
        # Shared eligibility so the horse race is apples-to-apples.
        strategy.universe.min_financial_quarters = 0

        print(f"  running {sid}...")
        try:
            out = run_strategy(
                strategy, cfg, dataset=ds, engine="vectorized",
                validate=False, write_artifacts=True, study=STUDY,
                set_latest=False,
            )
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["note"] = str(exc)[:300]
            rows.append(record)
            print(f"  {sid:<24} error  {exc}")
            continue

        m = out.metrics.get("combined", {})
        traded = bool(getattr(out.result, "returns", pd.Series(dtype=float)).notna().any())
        record.update({
            "run_id": out.run_id,
            "cagr": _f(m.get("cagr")),
            "sharpe": _f(m.get("sharpe")),
            "excess": _f(m.get("excess_return")),
            "max_dd": _f(m.get("max_drawdown")),
            "turnover": _f(m.get("turnover")),
            "status": _status(m, traded),
        })
        if record["status"] == "no_data":
            record["note"] = "Engine produced no invested returns (missing factor legs)."
        elif record["status"] == "fail":
            record["note"] = "Ran on real data but Sharpe/excess/CAGR did not all clear zero."
        else:
            record["note"] = "Positive Sharpe, excess vs Nifty 50 TRI, and CAGR on this window."
        rows.append(record)
        print(
            f"  {sid:<24} {record['status']:<7}  "
            f"CAGR {_pct(record['cagr'])}  Sharpe {_num(record['sharpe'])}  "
            f"excess {_pct(record['excess'])}"
        )

    payload = {
        "window": {"start": args.start, "end": args.end},
        "benchmark": "NIFTY50_TRI",
        "study": STUDY,
        "bar": {"sharpe": 0.0, "excess": 0.0, "cagr": 0.0},
        "fundamentals": pit,
        "signals": rows,
        "passed": [r["id"] for r in rows if r["status"] == "pass"],
        "failed": [r["id"] for r in rows if r["status"] == "fail"],
        "untested": [r["id"] for r in rows if r["status"] in ("no_data", "error")],
    }

    art = Path(cfg.paths.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    out_art = art / "paper_signal_horse_race.json"
    out_art.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    dash = Path(__file__).resolve().parents[2] / "dashboard" / "assets" / "paper_signals.json"
    dash.parent.mkdir(parents=True, exist_ok=True)
    dash.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_art}")
    print(f"Wrote {dash}")

    try:
        _refresh_latest_rankings(cfg, ds)
    except Exception as exc:  # noqa: BLE001
        print(f"  ranking refresh skipped: {exc}")

    print("\nPassed :", payload["passed"] or "(none)")
    print("Failed :", payload["failed"] or "(none)")
    print("Untested:", payload["untested"] or "(none)")
    return 0


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
