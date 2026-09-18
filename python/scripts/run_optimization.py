#!/usr/bin/env python
"""Walk-forward parameter optimization with overfitting controls.

    python scripts/run_optimization.py --preset full_composite --method random --n 40

Reports the in-sample best AND the plateau choice, plus PBO. The plateau
configuration is usually what you should actually use - an isolated peak in
the parameter surface is noise.
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
from src.backtest.optimization import (                         # noqa: E402
    ObjectiveConfig, apply_params, default_space_sleeve_a,
    grid_search, robust_objective, sensitivity_table,
)
from src.backtest.registry import RunRegistry                   # noqa: E402
from src.backtest.statistics import probability_backtest_overfitting  # noqa: E402
from src.backtest.walkforward import generate_folds             # noqa: E402
from src.config import load_config                              # noqa: E402
from src.data.loader import load_dataset                        # noqa: E402
from src.presets import get_preset, list_presets                # noqa: E402
from src.schema import StrategyConfig                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimize strategy parameters robustly")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=list_presets())
    src.add_argument("--config", type=str)

    ap.add_argument("--method", choices=["grid", "random", "optuna"], default="random")
    ap.add_argument("--n", type=int, default=30, help="number of trials")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--source", choices=["synthetic", "nse", "kite"], default=None)
    ap.add_argument("--turnover-penalty", type=float, default=0.10)
    ap.add_argument("--study", type=str, default="sweep")
    args = ap.parse_args()

    cfg = load_config()
    base = get_preset(args.preset) if args.preset else StrategyConfig.load(args.config)

    ds = load_dataset(cfg, base.universe, source=args.source,
                      start=base.window.start, end=base.window.end, run_health_check=False)
    cost_model = CostModel.from_config(cfg)
    registry = RunRegistry(cfg.paths.artifacts_dir)

    folds = generate_folds(ds.prices.index, n_folds=args.folds, mode="rolling",
                           train_years=3.0, test_years=1.0)
    if not folds:
        print("Not enough history for walk-forward folds; widen the date window.")
        return 1

    print(f"\nOptimizing '{base.name}' - {args.method} search, {args.n} trials, "
          f"{len(folds)} walk-forward folds")
    print("Selection is out-of-sample and penalizes turnover.\n")

    space = default_space_sleeve_a()
    objective_cfg = ObjectiveConfig(turnover_penalty=args.turnover_penalty)
    oos_curves: dict[str, pd.Series] = {}

    def evaluate(params: dict) -> dict:
        candidate = apply_params(base, params)
        fold_metrics, turnovers, chunks = [], [], []

        for fold in folds:
            test = ds.slice(fold.train_start, fold.test_end)
            res = run_vectorized_backtest(
                candidate, test.prices, test.eligibility, test.fundamentals,
                test.sectors, test.benchmark, test.adv, cost_model,
            )
            # Score ONLY the out-of-sample segment.
            oos = res.returns.loc[res.returns.index >= fold.test_start]
            if oos.empty:
                continue
            m = compute_metrics(oos, turnover=res.annual_turnover)
            fold_metrics.append({"sharpe": m.sharpe, "cagr": m.cagr})
            turnovers.append(res.annual_turnover)
            chunks.append(oos)

        if not fold_metrics:
            return {"score": -np.inf, "sharpe": np.nan, "turnover": np.nan}

        avg_turnover = float(np.mean(turnovers))
        score = robust_objective(fold_metrics, avg_turnover, space.degrees_of_freedom, objective_cfg)

        key = json.dumps(params, sort_keys=True, default=str)
        oos_curves[key] = pd.concat(chunks).sort_index()

        sharpes = [f["sharpe"] for f in fold_metrics if not np.isnan(f["sharpe"])]
        return {
            "score": float(score),
            "sharpe": float(np.median(sharpes)) if sharpes else np.nan,
            "mean_sharpe": float(np.mean(sharpes)) if sharpes else np.nan,
            "cagr": float(np.median([f["cagr"] for f in fold_metrics])),
            "turnover": avg_turnover,
            "n_folds": len(fold_metrics),
            "consistency": float(np.mean([s > 0 for s in sharpes])) if sharpes else np.nan,
        }

    result = grid_search(
        space, evaluate, method=args.method, n_samples=args.n,
        seed=base.seed, study=args.study, registry=registry, base_config=base,
    )

    print("=" * 78)
    print(f"OPTIMIZATION RESULTS  ({result.n_trials} trials)")
    print("=" * 78)
    print("\nTop 5 by robust OOS score:")
    print(result.top(5).to_string(index=False))

    print(f"\nIn-sample best  score={result.best_score:.3f}")
    print(f"  {result.best_params}")
    print(f"\nPlateau choice  score={result.plateau_score:.3f}   <-- prefer this")
    print(f"  {result.plateau_params}")
    print("  (a parameter set whose neighbours also perform well is far more "
          "likely to survive out-of-sample)")

    # PBO across candidate OOS curves.
    if len(oos_curves) >= 4:
        matrix = pd.DataFrame(oos_curves).dropna(how="all")
        if matrix.shape[1] >= 2:
            pbo = probability_backtest_overfitting(matrix, n_splits=6)
            print(f"\nProbability of backtest overfitting: {pbo.get('pbo')} "
                  f"({pbo.get('interpretation')})")

    for param in list(space.params)[:3]:
        t = sensitivity_table(result.results, param)
        if not t.empty:
            print(f"\nSensitivity - {param}")
            print(t.to_string(index=False))

    out = cfg.paths.artifacts_dir / "optimization_results.parquet"
    result.results.to_parquet(out, index=False)
    (cfg.paths.artifacts_dir / "optimization_summary.json").write_text(
        json.dumps(result.summary(), indent=2, default=str), encoding="utf-8"
    )
    print(f"\nSaved to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
