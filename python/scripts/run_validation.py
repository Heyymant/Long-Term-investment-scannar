#!/usr/bin/env python
"""Statistical validation of a strategy, including leak detection.

    python scripts/run_validation.py --preset full_composite
    python scripts/run_validation.py --preset full_composite --compare-presets

Answers: is the observed Sharpe distinguishable from luck, does it survive
correction for how many configurations were tried, and is there evidence of
look-ahead leaking into the signals?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from src.backtest.costs import CostModel                        # noqa: E402
from src.backtest.engine_vectorized import run_vectorized_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics                # noqa: E402
from src.backtest.registry import RunRegistry                   # noqa: E402
from src.backtest.statistics import (                           # noqa: E402
    detect_lookahead, probability_backtest_overfitting,
    reality_check, shifted_signal_test, spa_test, validate_strategy,
)
from src.config import load_config                              # noqa: E402
from src.data.loader import load_dataset                        # noqa: E402
from src.presets import get_preset, list_presets                # noqa: E402
from src.schema import StrategyConfig                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a strategy statistically")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=list_presets())
    src.add_argument("--config", type=str)

    ap.add_argument("--source", choices=["nse", "kite"], default=None)
    ap.add_argument("--compare-presets", action="store_true",
                    help="run all presets and apply Reality Check / SPA across them")
    ap.add_argument("--trials", type=int, default=None,
                    help="override the trial count used for the Deflated Sharpe")
    args = ap.parse_args()

    cfg = load_config()
    base = get_preset(args.preset) if args.preset else StrategyConfig.load(args.config)
    ds = load_dataset(cfg, base.universe, source=args.source,
                      start=base.window.start, end=base.window.end, run_health_check=False)
    cost_model = CostModel.from_config(cfg)
    registry = RunRegistry(cfg.paths.artifacts_dir)

    def backtest(strategy) -> pd.Series:
        res = run_vectorized_backtest(
            strategy, ds.prices, ds.eligibility, ds.fundamentals,
            ds.sectors, ds.benchmark, ds.adv, cost_model,
        )
        return res.returns

    print(f"\nValidating '{base.name}'...")
    returns = backtest(base)
    if returns.empty:
        print("Backtest produced no returns.")
        return 1

    bench = ds.benchmark.pct_change().reindex(returns.index)
    m = compute_metrics(returns, bench, ds.risk_free)
    print(f"  CAGR {m.cagr:.2%}   Sharpe {m.sharpe:.2f}   MaxDD {m.max_drawdown:.1%}\n")

    candidates = None
    if args.compare_presets:
        print("Running all presets for data-snooping tests...")
        curves = {}
        for name in list_presets():
            try:
                r = backtest(get_preset(name))
                if not r.empty:
                    curves[name] = r
                    print(f"  {name:<22} done")
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<22} failed: {exc}")
        if len(curves) >= 2:
            candidates = pd.DataFrame(curves).dropna(how="all")
        print()

    n_trials = args.trials or max(
        1, registry.trial_count(config_hash=base.config_hash()),
        len(candidates.columns) if candidates is not None else 1,
    )

    report = validate_strategy(
        returns, n_trials=n_trials, benchmark_returns=bench,
        candidate_returns=candidates, seed=base.seed,
    )

    print("=" * 74)
    print(f"STATISTICAL VALIDATION   (trials counted: {n_trials})")
    print("=" * 74)

    st = report.sharpe_test
    print(f"\nSharpe ratio        {st.get('sharpe'):.3f}" if st.get("sharpe") else "\nSharpe: n/a")
    if st.get("ci_lower") is not None:
        print(f"  95% CI            [{st['ci_lower']:.3f}, {st['ci_upper']:.3f}]")
        print(f"  t-stat / p-value  {st.get('t_stat'):.2f} / {st.get('p_value'):.4f}")
        print(f"  significant (5%)  {st.get('significant_5pct')}")

    bs = report.bootstrap
    if bs.get("ci_lower") is not None:
        print(f"\nBootstrap (block)   CI [{bs['ci_lower']:.3f}, {bs['ci_upper']:.3f}]")
        print(f"  P(Sharpe > 0)     {bs.get('prob_positive'):.1%}")

    ds_r = report.deflated_sharpe
    if ds_r.get("dsr") is not None:
        print(f"\nDeflated Sharpe     {ds_r['dsr']:.3f}  (>0.95 desired)")
        print(f"  expected max SR   {ds_r.get('expected_max_sharpe'):.3f} from {n_trials} trials")

    hc = report.haircut
    if hc.get("haircut_sharpe") is not None:
        print(f"\nHaircut Sharpe      {hc['haircut_sharpe']:.3f} "
              f"(cut {hc.get('haircut_pct', 0):.0f}%)")

    trl = report.min_trl
    if trl.get("min_track_record_years") is not None:
        years = trl["min_track_record_years"]
        have = len(returns) / 252
        print(f"\nMinTRL              {years:.1f} years needed, {have:.1f} available")

    if report.pbo.get("pbo") is not None:
        print(f"\nPBO                 {report.pbo['pbo']:.3f} ({report.pbo.get('interpretation')})")
    if report.reality_check.get("p_value") is not None:
        print(f"Reality Check       p = {report.reality_check['p_value']:.4f}")
    if report.spa.get("p_value") is not None:
        print(f"SPA test            p = {report.spa['p_value']:.4f}")

    # Leak detection: delaying signals should HURT.
    print("\nLeak detection")
    leak = _shifted_test(base, ds, cost_model)
    print(f"  {leak.get('note', 'n/a')}")
    if leak.get("base_sharpe") is not None:
        print(f"  base {leak['base_sharpe']:.3f} -> delayed {leak['shifted_sharpe']:.3f} "
              f"({leak['degradation_pct']:+.1f}%)")
    report.leaks = leak

    print("\n" + "=" * 74)
    print(f"VERDICT: {report.verdict}")
    for flag in report.flags:
        print(f"  - {flag}")
    print("=" * 74 + "\n")

    out = cfg.paths.artifacts_dir / "validation_stats.json"
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"Saved to {out}\n")
    return 0


def _shifted_test(base, ds, cost_model) -> dict:
    """Re-run with the rebalance calendar pushed back; performance should fall."""
    def run(shift: int) -> float:
        strategy = base.model_copy(deep=True)
        # A longer low-vol lookback delays every signal by construction.
        strategy.sleeve_a.factors.low_vol_lookback_days += shift * 21
        res = run_vectorized_backtest(
            strategy, ds.prices, ds.eligibility, ds.fundamentals,
            ds.sectors, ds.benchmark, ds.adv, cost_model,
        )
        if res.returns.empty:
            return float("nan")
        return compute_metrics(res.returns).sharpe

    try:
        return shifted_signal_test(run, shift_days=1)
    except Exception as exc:  # noqa: BLE001
        return {"note": f"shifted-signal test failed: {exc}"}


if __name__ == "__main__":
    raise SystemExit(main())
