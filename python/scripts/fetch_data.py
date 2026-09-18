#!/usr/bin/env python
"""Fetch and cache market data (read-only Kite access).

    # one-time: authenticate (token is valid for the trading day)
    python scripts/fetch_data.py --login
    python scripts/fetch_data.py --request-token <token_from_redirect>

    # build the security master, then pull price history
    python scripts/fetch_data.py --build-master
    python scripts/fetch_data.py --prices --start 2010-01-01

    # generate an offline synthetic dataset instead
    python scripts/fetch_data.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                            # noqa: E402

from src.config import load_config                             # noqa: E402
from src.data.kite_client import KiteDataClient, KiteSession    # noqa: E402
from src.data.security_master import SecurityMaster            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Indian equity market data")
    ap.add_argument("--login", action="store_true", help="print the Kite login URL")
    ap.add_argument("--request-token", type=str, help="exchange a request token for an access token")
    ap.add_argument("--build-master", action="store_true", help="build the security master")
    ap.add_argument("--prices", action="store_true", help="fetch historical prices")
    ap.add_argument("--synthetic", action="store_true", help="generate an offline dataset")
    ap.add_argument("--start", type=str, default="2010-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only fetch N instruments (testing)")
    ap.add_argument("--n-stocks", type=int, default=120, help="synthetic universe size")
    args = ap.parse_args()

    cfg = load_config()

    if args.synthetic:
        from src.data.synthetic import generate_market, write_synthetic_dataset

        print(f"Generating synthetic market ({args.n_stocks} stocks)...")
        market = generate_market(args.n_stocks, args.start, args.end or "2024-12-31")
        write_synthetic_dataset(market, cfg.paths.data_dir)
        print(f"  prices       {market.prices.shape}")
        print(f"  fundamentals {market.fundamentals.shape}")
        print(f"  earnings     {market.earnings.shape}")
        print(f"Saved to {cfg.paths.data_dir / 'synthetic'}")
        return 0

    if args.login:
        session = KiteSession.from_env(cfg.paths.data_dir)
        print("\n1) Open this URL and log in:")
        print(f"   {session.login_url()}")
        print("\n2) Copy `request_token` from the redirect URL")
        print("3) Run: python scripts/fetch_data.py --request-token <token>\n")
        return 0

    if args.request_token:
        session = KiteSession.from_env(cfg.paths.data_dir)
        token = session.exchange_request_token(args.request_token)
        print(f"Access token obtained and cached (valid for today): {token[:8]}...")
        return 0

    client = KiteDataClient(cfg)

    if args.build_master:
        print("Fetching NSE instruments...")
        instruments = client.fetch_instruments("NSE", refresh=True)
        master = SecurityMaster.from_kite_instruments(instruments)
        master.save(cfg.paths.data_dir)
        print(f"Security master: {len(master.securities)} instruments")
        if master.securities["isin"].astype(str).str.startswith("SYN:").any():
            print("  NOTE: some rows have synthetic ISINs (Kite's dump lacks ISIN).")
            print("        Enrich with a real ISIN source for robust long-term joins.")
        return 0

    if args.prices:
        master = SecurityMaster.load(cfg.paths.data_dir)
        if master.securities.empty:
            print("No security master. Run --build-master first.")
            return 1

        tokens = [
            int(t) for t in master.securities["kite_token"].dropna().tolist()
        ][: args.limit]
        print(f"Fetching daily candles for {len(tokens)} instruments from {args.start}...")
        print("(Kite caps each request at 2000 days; requests are paginated and cached.)")
        data = client.fetch_many(tokens, args.start, args.end)
        total = sum(len(v) for v in data.values())
        print(f"Done: {len(data)} instruments, {total} rows cached in {cfg.paths.data_dir / 'prices'}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
