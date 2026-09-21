#!/usr/bin/env python
"""Run a backtest from a strategy config or a named preset.

    python scripts/run_backtest.py --preset full_composite
    python scripts/run_backtest.py --config ../artifacts/strategy_config.json --engine both
    python scripts/run_backtest.py --preset all_sleeves --source nse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.runner import run_strategy          # noqa: E402
from src.config import load_config                    # noqa: E402
from src.presets import get_preset, list_presets      # noqa: E402
from src.schema import StrategyConfig                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an Indian equity strategy backtest")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=list_presets(), help="named literature preset")
    src.add_argument("--config", type=str, help="path to strategy_config.json")

    ap.add_argument("--engine", choices=["vectorized", "event", "both"], default="vectorized")
    ap.add_argument("--source", choices=["nse", "kite"], default=None,
                    help="data source (defaults to config.yaml)")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--no-validate", action="store_true", help="skip statistical validation")
    ap.add_argument("--no-artifacts", action="store_true", help="do not write artifacts")
    ap.add_argument("--study", type=str, default="adhoc")
    args = ap.parse_args()

    cfg = load_config()
    strategy = get_preset(args.preset) if args.preset else StrategyConfig.load(args.config)

    if args.start:
        strategy.window.start = args.start
    if args.end:
        strategy.window.end = args.end

    print(f"\nStrategy : {strategy.name}")
    print(f"Sleeves  : {', '.join(strategy.active_sleeves()) or 'none'}")
    print(f"Window   : {strategy.window.start} -> {strategy.window.end or 'latest'}")
    print(f"Engine   : {args.engine}\n")

    out = run_strategy(
        strategy, cfg,
        engine=args.engine,
        validate=not args.no_validate,
        write_artifacts=not args.no_artifacts,
        study=args.study,
        source=args.source,
    )

    m = out.metrics.get("combined", {})
    print("=" * 66)
    print(f"RESULTS  (run {out.run_id})")
    print("=" * 66)
    for label, key, pct in [
        ("CAGR", "cagr", True), ("Volatility", "volatility", True),
        ("Sharpe", "sharpe", False), ("Sortino", "sortino", False),
        ("Calmar", "calmar", False), ("Max drawdown", "max_drawdown", True),
        ("Recovery (days)", "recovery_days", False),
        ("Annual turnover", "turnover", False),
        ("Benchmark CAGR", "benchmark_cagr", True),
        ("Excess return", "excess_return", True),
        ("Information ratio", "information_ratio", False),
        ("After-tax CAGR", "after_tax_cagr", True),
    ]:
        v = m.get(key)
        if v is None:
            continue
        print(f"  {label:<20} {f'{v * 100:.2f}%' if pct else f'{v:.2f}'}")

    fm = out.attribution.get("factor_model")
    if fm:
        print("\n  True alpha (vs IIMA factors)")
        print(f"    annualized alpha   {fm.get('alpha_annualized', 0) * 100:.2f}%")
        print(f"    t-stat / p-value   {fm.get('alpha_tstat')} / {fm.get('alpha_pvalue')}")
        print(f"    betas              {fm.get('betas')}")
    elif out.attribution.get("note"):
        print(f"\n  {out.attribution['note']}")

    if out.validation:
        print(f"\n  Validation: {out.validation.get('verdict', 'n/a')}")
        for flag in out.validation.get("flags", []):
            print(f"    - {flag}")

    if not args.no_artifacts:
        print(f"\n  Artifacts: {out.artifacts_dir}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
