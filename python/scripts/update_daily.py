#!/usr/bin/env python
"""Daily incremental refresh: prices, fundamentals, earnings, signals.

Schedule this after market close. It only fetches what is missing, then
regenerates the artifacts the dashboard reads.

    python scripts/update_daily.py
    python scripts/update_daily.py --skip-prices --preset full_composite
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                             # noqa: E402

from src.config import load_config                              # noqa: E402
from src.data.quality import build_report                       # noqa: E402
from src.presets import get_preset, list_presets                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily incremental data + signal refresh")
    ap.add_argument("--preset", choices=list_presets(), default="full_composite")
    ap.add_argument("--source", choices=["synthetic", "nse", "kite"], default=None)
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--skip-fundamentals", action="store_true")
    ap.add_argument("--skip-signals", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    today = date.today().isoformat()
    log: dict[str, object] = {"date": today, "steps": {}}
    print(f"Daily update - {today}\n")

    # 1) Prices
    if not args.skip_prices and (args.source or cfg.get("data_sources.prices.provider")) != "synthetic":
        try:
            from src.data.kite_client import KiteDataClient
            from src.data.security_master import SecurityMaster

            master = SecurityMaster.load(cfg.paths.data_dir)
            if master.securities.empty:
                log["steps"]["prices"] = "skipped - no security master"
                print("  prices        skipped (run fetch_data.py --build-master)")
            else:
                client = KiteDataClient(cfg)
                tokens = [int(t) for t in master.securities["kite_token"].dropna()]
                data = client.fetch_many(tokens, "2010-01-01", None)
                log["steps"]["prices"] = f"{len(data)} instruments refreshed"
                print(f"  prices        {len(data)} instruments refreshed")
        except Exception as exc:  # noqa: BLE001
            log["steps"]["prices"] = f"failed: {exc}"
            print(f"  prices        FAILED: {exc}")
    else:
        print("  prices        skipped")

    # 2) Fundamentals
    if not args.skip_fundamentals:
        try:
            from src.data.fundamentals import FundamentalsCache, get_provider
            from src.data.security_master import SecurityMaster

            master = SecurityMaster.load(cfg.paths.data_dir)
            if not master.securities.empty:
                provider = get_provider(cfg)
                isins = list(master.securities["isin"])
                symbols = dict(zip(master.securities["isin"], master.securities["symbol"]))
                try:
                    df = provider.fetch(isins, "2010-01-01", today, symbols=symbols)
                except TypeError:
                    df = provider.fetch(isins, "2010-01-01", today)
                if not df.empty:
                    FundamentalsCache(cfg.paths.data_dir).upsert(df)
                log["steps"]["fundamentals"] = f"{len(df)} rows from {provider.name}"
                print(f"  fundamentals  {len(df)} rows ({provider.name})")
            else:
                print("  fundamentals  skipped (no security master)")
        except Exception as exc:  # noqa: BLE001
            log["steps"]["fundamentals"] = f"failed: {exc}"
            print(f"  fundamentals  FAILED: {exc}")

    # 3) Health + signals
    if not args.skip_signals:
        try:
            from src.backtest.runner import run_strategy
            from src.data.loader import load_dataset

            strategy = get_preset(args.preset)
            ds = load_dataset(cfg, strategy.universe, source=args.source)

            health = build_report(ds.prices, ds.volumes, ds.fundamentals)
            health.save(cfg.paths.artifacts_dir / "data_health.json")
            log["steps"]["health"] = {"errors": health.n_errors, "warnings": health.n_warnings}
            print(f"  data health   {health.n_errors} errors, {health.n_warnings} warnings")

            out = run_strategy(strategy, cfg, dataset=ds, validate=False, study="daily")
            log["steps"]["signals"] = {"run_id": out.run_id, **out.metrics.get("combined", {})}
            print(f"  signals       run {out.run_id} | {out.headline()}")
        except Exception as exc:  # noqa: BLE001
            log["steps"]["signals"] = f"failed: {exc}"
            print(f"  signals       FAILED: {exc}")

    path = cfg.paths.artifacts_dir / "update_log.json"
    path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"\nLog written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
