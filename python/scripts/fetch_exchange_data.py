#!/usr/bin/env python
"""Build the data layer from NSE/BSE public data - no broker account needed.

    # 1. Security master (ISIN, sectors, BSE cross-reference)
    python scripts/fetch_exchange_data.py --master

    # 2. Price history from the bhavcopy archive (INCLUDES DELISTED NAMES)
    python scripts/fetch_exchange_data.py --prices --start 2015-01-01

    # 3. Corporate actions -> split/bonus/dividend adjustment
    python scripts/fetch_exchange_data.py --actions --start 2015-01-01

    # 4. Fundamentals + earnings with REAL announcement timestamps
    python scripts/fetch_exchange_data.py --fundamentals --start 2020-01-01

    # everything
    python scripts/fetch_exchange_data.py --all --start 2018-01-01

Why this path is better than a broker feed:
  * bhavcopy keeps companies that later delisted  -> no survivorship bias
  * filings carry the exchange dissemination time -> no look-ahead guess

Steps 1-3 use static archive files and plain HTTP. Step 4 uses NSE's JSON
APIs, which require browser impersonation (curl_cffi) and are a grey area
under NSE's terms - personal research only, low request rates, no
redistribution.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                            # noqa: E402

from src.config import load_config                             # noqa: E402
from src.data.bse_client import BSEClient                      # noqa: E402
from src.data.exchange_ingest import (                         # noqa: E402
    build_corporate_actions, build_index_membership, build_price_history,
    build_security_master, infer_delistings, save_price_panels,
)
from src.data.nse_client import NSEClient                      # noqa: E402
from src.data.security_master import SecurityMaster            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch NSE/BSE public market data")
    ap.add_argument("--master", action="store_true", help="build the security master")
    ap.add_argument("--prices", action="store_true", help="bhavcopy price history")
    ap.add_argument("--actions", action="store_true", help="corporate actions")
    ap.add_argument("--fundamentals", action="store_true", help="filings + XBRL (needs the API)")
    ap.add_argument("--membership", action="store_true", help="index membership")
    ap.add_argument("--all", action="store_true", help="run every step")

    ap.add_argument("--start", type=str, default="2018-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--index", type=str, default="NIFTY500")
    ap.add_argument("--no-api", action="store_true",
                    help="never use the Akamai-protected JSON APIs")
    ap.add_argument("--no-xbrl", action="store_true", help="skip XBRL parsing (much faster)")
    args = ap.parse_args()

    cfg = load_config()
    end = args.end or date.today().isoformat()

    allow_api = (not args.no_api) and bool(cfg.get("data_sources.exchange.allow_nse_api", True))
    rate = float(cfg.get("data_sources.exchange.rate_limit_per_sec", 1.0))

    nse = NSEClient(cache_dir=cfg.paths.data_dir / "nse", allow_api=allow_api, rate_limit_per_sec=rate)
    bse = BSEClient(cache_dir=cfg.paths.data_dir / "bse")

    steps = {
        "master": args.master or args.all,
        "prices": args.prices or args.all,
        "actions": args.actions or args.all,
        "fundamentals": args.fundamentals or args.all,
        "membership": args.membership or args.all,
    }
    if not any(steps.values()):
        ap.print_help()
        return 0

    print(f"\nNSE/BSE ingest  {args.start} -> {end}")
    print(f"  JSON APIs: {'enabled' if allow_api else 'disabled (archives only)'}\n")

    # ---------------------------------------------------------------- master
    if steps["master"]:
        print("[1] Security master")
        master = build_security_master(cfg, nse, bse, args.index)
        master.save(cfg.paths.data_dir)
        print(f"    {len(master.securities)} securities, "
              f"{master.securities['sector'].notna().sum()} with sectors\n")

    # ---------------------------------------------------------------- prices
    if steps["prices"]:
        print("[2] Price history from bhavcopy (this is the survivorship fix)")
        panels = build_price_history(cfg, args.start, end, nse)
        save_price_panels(panels, cfg)

        close = panels["close"]
        gone = infer_delistings(close)
        print(f"    {close.shape[1]} securities x {close.shape[0]} days")
        print(f"    {len(gone)} stopped trading before the end of the window")
        print("    (a live broker dump would have silently omitted these)\n")

        if not gone.empty:
            sm = SecurityMaster.load(cfg.paths.data_dir)
            if not sm.securities.empty:
                for row in gone.itertuples(index=False):
                    sm.record_delisting(row.isin, row.delisting_date)
                sm.save(cfg.paths.data_dir)
                print(f"    Recorded {len(gone)} delistings in the security master\n")

    # --------------------------------------------------------------- actions
    if steps["actions"]:
        print("[3] Corporate actions")
        if not allow_api:
            print("    SKIPPED - needs the JSON API (drop --no-api)\n")
        else:
            sm = SecurityMaster.load(cfg.paths.data_dir)
            sym_map = sm.symbol_to_isin_map() if not sm.securities.empty else {}
            actions = build_corporate_actions(cfg, args.start, end, nse, sym_map)
            actions.save(cfg.paths.data_dir)
            n = len(actions.actions)
            print(f"    {n} actions parsed"
                  + (f" ({actions.actions['action_type'].value_counts().to_dict()})" if n else "")
                  + "\n")

    # ---------------------------------------------------------- fundamentals
    if steps["fundamentals"]:
        print("[4] Fundamentals with real announcement timestamps")
        if not allow_api:
            print("    SKIPPED - needs the JSON API (drop --no-api)\n")
        else:
            from src.data.fundamentals.base import FundamentalsCache
            from src.data.fundamentals.nse import NSEFundamentalsProvider, build_eps_history

            sm = SecurityMaster.load(cfg.paths.data_dir)
            isins = list(sm.securities["isin"]) if not sm.securities.empty else []

            provider = NSEFundamentalsProvider(
                client=nse,
                basis=str(cfg.get("data_sources.fundamentals.basis", "consolidated")),
                fetch_xbrl=not args.no_xbrl,
            )
            df = provider.fetch(isins, args.start, end)

            if df.empty:
                print("    No filings retrieved\n")
            else:
                FundamentalsCache(cfg.paths.data_dir).upsert(df)
                eps = build_eps_history(df)
                if not eps.empty:
                    eps.to_parquet(cfg.paths.data_dir / "earnings.parquet", index=False)

                with_eps = int(df["eps"].notna().sum()) if "eps" in df.columns else 0
                print(f"    {len(df)} filings, {df['isin'].nunique()} companies")
                print(f"    {with_eps} with EPS (feeds Sleeve C SUE)")
                print("    All dated by exchange dissemination time - no look-ahead\n")

    # ------------------------------------------------------------ membership
    if steps["membership"]:
        print("[5] Index membership")
        from src.data.exchange_ingest import load_price_panels

        panels = load_price_panels(cfg)
        membership = build_index_membership(cfg, args.index, nse, panels.get("close"))
        if not membership.df.empty:
            membership.save(cfg.paths.data_dir)
            print(f"    {len(membership.df)} securities in {args.index}\n")

    print("Done. Run a backtest with:")
    print("  python scripts/run_backtest.py --preset full_composite --source nse\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
