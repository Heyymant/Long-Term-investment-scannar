#!/usr/bin/env python
"""Generate the manual rebalance checklist.

Reads your live Zerodha holdings (read-only), compares them to the strategy's
target weights, and writes `rebalance_orders.csv`.

    python scripts/generate_rebalance.py --preset full_composite
    python scripts/generate_rebalance.py --preset full_composite --source synthetic --dry-run

NOTHING IS TRADED. This produces a checklist for you to execute yourself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                             # noqa: E402

from src.backtest.costs import CostModel                        # noqa: E402
from src.config import load_config                              # noqa: E402
from src.data.loader import load_dataset                        # noqa: E402
from src.data.universe import eligible_on                       # noqa: E402
from src.factors.composite import compute_composite             # noqa: E402
from src.portfolio.construction import construct_portfolio      # noqa: E402
from src.portfolio.regime import compute_exposure_series, exposure_on  # noqa: E402
from src.presets import get_preset, list_presets                # noqa: E402
from src.rebalance.generate_orders import generate_rebalance    # noqa: E402
from src.schema import StrategyConfig                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a manual rebalance checklist")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=list_presets())
    src.add_argument("--config", type=str)

    ap.add_argument("--source", choices=["synthetic", "nse", "kite"], default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="use simulated holdings instead of querying Kite")
    ap.add_argument("--capital", type=float, default=None,
                    help="portfolio value to size against (default: from holdings)")
    ap.add_argument("--out", type=str, default=None, help="output CSV path")
    args = ap.parse_args()

    cfg = load_config()
    strategy = get_preset(args.preset) if args.preset else StrategyConfig.load(args.config)

    ds = load_dataset(cfg, strategy.universe, source=args.source, run_health_check=False)
    dt = ds.prices.index[-1]

    universe = eligible_on(ds.eligibility, dt)
    if not universe:
        print("No eligible names in the tradeable universe on the latest date.")
        return 1

    scores = compute_composite(
        ds.prices, dt, strategy.sleeve_a.factors, universe,
        ds.fundamentals, ds.sectors, strategy.sleeve_a.sector_neutralize,
    ).composite
    if scores.empty:
        print("Could not compute factor scores (insufficient data).")
        return 1

    exposure = exposure_on(
        compute_exposure_series(ds.benchmark, strategy.sleeve_a.regime), dt, 1.0
    )

    holdings = _load_holdings(cfg, args.dry_run, ds, scores)
    current = set(holdings["isin"]) if not holdings.empty else set()

    portfolio = construct_portfolio(
        scores, ds.prices, strategy.sleeve_a, dt, current, ds.sectors, exposure
    )

    prices_now = ds.raw_prices.loc[dt]
    adv_now = ds.adv.loc[dt] if not ds.adv.empty else None

    plan = generate_rebalance(
        target_weights=portfolio.weights,
        current_holdings=holdings,
        prices=prices_now,
        portfolio_value=args.capital,
        securities=ds.securities,
        adv=adv_now,
        cost_model=CostModel.from_config(cfg),
        tradeable_universe=set(universe),
        as_of=dt,
    )

    out_path = Path(args.out) if args.out else cfg.paths.artifacts_dir / "rebalance_orders.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan.orders.to_csv(out_path, index=False)
    (out_path.parent / "rebalance_summary.json").write_text(
        json.dumps(plan.summary, indent=2), encoding="utf-8"
    )

    print(f"\nRebalance checklist as of {dt.date()}   (regime exposure {exposure:.0%})")
    print("=" * 78)
    actionable = plan.actionable()
    if actionable.empty:
        print("  No trades required - the portfolio is already on target.")
    else:
        cols = ["symbol", "action", "current_qty", "target_qty", "delta_qty", "price", "delta_value"]
        print(actionable[cols].to_string(index=False))

    print("\nSummary")
    for k, v in plan.summary.items():
        print(f"  {k:<18} {v}")
    print(f"\nWritten to {out_path}")
    print("\n*** Decision support only. No orders have been placed. Execute manually. ***\n")
    return 0


def _load_holdings(cfg, dry_run: bool, ds, scores) -> pd.DataFrame:
    """Live Kite holdings, or a simulated book for --dry-run."""
    if dry_run:
        top = scores.nlargest(min(10, len(scores))).index
        rows = []
        for isin in top:
            price = ds.raw_prices.iloc[-1].get(isin)
            if price and not pd.isna(price):
                sym = ds.securities.loc[ds.securities["isin"] == isin, "symbol"]
                rows.append({
                    "isin": isin,
                    "tradingsymbol": sym.iloc[0] if not sym.empty else isin,
                    "quantity": int(500_000 / float(price)),
                })
        print("[dry-run] using simulated holdings")
        return pd.DataFrame(rows)

    try:
        from src.data.kite_client import KiteDataClient

        holdings = KiteDataClient(cfg).holdings()
        print(f"Loaded {len(holdings)} live holdings from Kite (read-only)")
        return holdings
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load Kite holdings ({exc}); falling back to an empty book.")
        print("Use --dry-run to test with simulated holdings.")
        return pd.DataFrame()


if __name__ == "__main__":
    raise SystemExit(main())
