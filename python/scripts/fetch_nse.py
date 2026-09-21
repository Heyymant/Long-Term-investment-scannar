#!/usr/bin/env python
"""Bootstrap the whole dataset from NSE's free public data.

NSE publishes everything the framework needs except live ticks:

    python scripts/fetch_nse.py --all --start 2015-01-01

    # or piece by piece
    python scripts/fetch_nse.py --master                     # ISINs + sectors
    python scripts/fetch_nse.py --prices --start 2015-01-01  # bhavcopy archive
    python scripts/fetch_nse.py --actions                    # splits/bonus/dividends
    python scripts/fetch_nse.py --earnings --max-filings 300 # results + XBRL

The bhavcopy archive is the important one: replaying it reconstructs a
survivorship-bias-free history, because each historical file lists exactly
what was trading that day - including companies that later delisted.

Please be considerate: this is a free public service. The client is
rate-limited and caches everything to disk.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                  # noqa: E402

from src.config import load_config                                   # noqa: E402
from src.data.corporate_actions import ACTION_COLUMNS, CorporateActions  # noqa: E402
from src.data.nse import (                                           # noqa: E402
    NSEClient, bhavcopy_to_panels, build_symbol_history,
    infer_delistings, normalize_corporate_actions,
)
from src.data.security_master import build_from_nse                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Indian market data from NSE (free)")
    ap.add_argument("--all", action="store_true", help="run every step")
    ap.add_argument("--master", action="store_true", help="security master (ISINs + sectors)")
    ap.add_argument("--prices", action="store_true", help="bhavcopy archive -> price panels")
    ap.add_argument("--actions", action="store_true", help="corporate actions")
    ap.add_argument("--earnings", action="store_true", help="results calendar + XBRL fundamentals")
    ap.add_argument("--benchmarks", action="store_true", help="index closes -> Total Return series")

    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--index", type=str, default="NSE500")
    ap.add_argument("--max-filings", type=int, default=200,
                    help="cap XBRL downloads per run (they are fetched one by one)")
    ap.add_argument("--rate-limit", type=float, default=2.0, help="requests per second")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = cfg.paths.data_dir
    client = NSEClient(cache_dir=data_dir, rate_limit_per_sec=args.rate_limit)

    end = args.end or date.today().isoformat()
    run_all = args.all or not any(
        [args.master, args.prices, args.actions, args.earnings, args.benchmarks]
    )

    equity_list = pd.DataFrame()
    bhav = pd.DataFrame()

    # ---------------------------------------------------------------- master
    if run_all or args.master:
        print("\n[1] Security master (ISINs, listing dates, sectors)")
        equity_list = client.fetch_equity_list(refresh=True)
        if equity_list.empty:
            print("    FAILED - could not download EQUITY_L.csv")
        else:
            constituents = client.fetch_index_constituents(args.index, refresh=True)
            kite_instruments = _load_kite_instruments(data_dir)
            master = build_from_nse(equity_list, constituents, kite_instruments)
            master.save(data_dir)
            print(f"    {len(master.securities)} instruments, "
                  f"{int(master.securities['sector'].notna().sum())} with sectors, "
                  f"{int(master.securities['kite_token'].notna().sum())} matched to Kite tokens")

            if not constituents.empty:
                _save_membership(constituents, args.index, data_dir)
                print(f"    {args.index} membership saved ({len(constituents)} names)")

    # ---------------------------------------------------------------- prices
    if run_all or args.prices:
        print(f"\n[2] Bhavcopy archive {args.start} -> {end}")
        print("    (one request per trading day; cached, so re-runs are fast)")
        bhav = client.fetch_bhavcopy_range(args.start, end)
        if bhav.empty:
            print("    FAILED - no bhavcopy data retrieved")
        else:
            bhav.to_parquet(data_dir / "bhavcopy.parquet", index=False)
            panels = bhavcopy_to_panels(bhav)
            for name, panel in panels.items():
                panel.to_parquet(data_dir / f"panel_{name}.parquet")
            print(f"    {len(bhav):,} rows | {bhav['isin'].nunique():,} ISINs | "
                  f"{bhav['date'].nunique():,} trading days")
            print(f"    panels: {', '.join(sorted(panels))}")

            history = build_symbol_history(bhav)
            delisted = infer_delistings(bhav)
            history.to_parquet(data_dir / "symbol_history.parquet", index=False)
            print(f"    {len(history)} ISIN/symbol pairs, {len(delisted)} delisted or suspended")
            print("    NOTE: replaying the archive gives survivorship-bias-free history.")

            # Refresh the master now that we know symbol history and delistings.
            if not equity_list.empty:
                master = build_from_nse(
                    equity_list,
                    client.fetch_index_constituents(args.index),
                    _load_kite_instruments(data_dir),
                    symbol_history=history,
                    delistings=delisted,
                )
                master.save(data_dir)

    # ------------------------------------------------------------ benchmarks
    if run_all or args.benchmarks:
        print(f"\n[2b] Index history -> Total Return benchmarks ({args.start} -> {end})")
        history = client.fetch_index_history(args.start, end)
        if history.empty:
            print("    FAILED - no index data retrieved")
        else:
            history.to_parquet(data_dir / "index_history.parquet", index=False)
            wanted = {
                "NIFTY50_TRI": "Nifty 50",
                "NIFTY500_TRI": "Nifty 500",
                "NIFTY200_MOMENTUM30_TRI": "Nifty200 Momentum 30",
            }
            from src.data.nse import build_total_return_index
            from src.data.reference import (
                build_risk_free_from_index_history,
                save_benchmark,
                save_risk_free,
            )

            for label, nse_name in wanted.items():
                tri = build_total_return_index(history, nse_name)
                if tri.empty:
                    print(f"    {label:26s} not published in this window")
                    continue
                save_benchmark(tri, label, cfg)
                years = max(len(tri) / 252, 1e-9)
                cagr = (tri.iloc[-1] / tri.iloc[0]) ** (1 / years) - 1
                print(f"    {label:26s} {len(tri):>5} days | CAGR {cagr:6.2%}")
            print("    TRI reconstructed from NSE price levels + published dividend yield")

            rf = build_risk_free_from_index_history(history)
            if rf.empty:
                print("    risk-free                Nifty 1D Rate Index not in this window")
            else:
                save_risk_free(rf, cfg)
                print(f"    risk-free (Nifty 1D)     {len(rf):>5} days | median {rf.median():6.2%}")

    # ------------------------------------------------------- corporate actions
    if run_all or args.actions:
        print("\n[3] Corporate actions (splits, bonuses, dividends)")
        raw = client.fetch_corporate_actions()
        if raw.empty:
            print("    none returned (the API window is short; re-run periodically)")
        else:
            parsed, unparsed = normalize_corporate_actions(raw)
            if parsed.empty:
                print(f"    {len(raw)} rows, none parsed into structured actions")
            else:
                ca = CorporateActions.load(data_dir)
                combined = pd.concat([ca.actions, parsed[ACTION_COLUMNS]], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["isin", "ex_date", "action_type"], keep="last"
                )
                CorporateActions(combined).save(data_dir)
                counts = parsed["action_type"].value_counts().to_dict()
                print(f"    parsed {len(parsed)} actions: {counts}")
            if not unparsed.empty:
                unparsed.to_csv(data_dir / "corporate_actions_unparsed.csv", index=False)
                print(f"    {len(unparsed)} unparsed rows saved for review "
                      "(a missed split corrupts returns, so these are kept, not dropped)")

    # -------------------------------------------------------------- earnings
    if run_all or args.earnings:
        print("\n[4] Results calendar + XBRL fundamentals")
        meetings = client.fetch_board_meetings()
        if not meetings.empty:
            meetings.to_parquet(data_dir / "board_meetings.parquet", index=False)
            n_results = int(meetings["is_results"].sum())
            print(f"    {len(meetings)} forthcoming board meetings ({n_results} results-related)")

        from src.data.fundamentals.nse_provider import NSEProvider, build_earnings_from_nse

        provider = NSEProvider(client=client, cache_dir=data_dir, max_filings=args.max_filings)
        prices = _load_price_panel(data_dir)
        isins = list(equity_list["isin"]) if not equity_list.empty else []

        fundamentals = provider.fetch(isins, args.start, end, prices=prices)
        if fundamentals.empty:
            print("    no fundamentals parsed")
        else:
            from src.data.fundamentals.base import FundamentalsCache

            FundamentalsCache(data_dir).upsert(fundamentals)
            earnings = build_earnings_from_nse(fundamentals)
            if not earnings.empty:
                earnings.to_parquet(data_dir / "earnings.parquet", index=False)
            print(f"    {len(fundamentals)} company-quarters "
                  f"({fundamentals['isin'].nunique()} companies)")
            print(f"    {len(earnings)} EPS observations for SUE")
            print("    announcement timestamps are real (broadCastDate) - point-in-time safe")

    print("\nDone. Next:")
    print("  python scripts/run_backtest.py --preset full_composite --source nse")
    return 0


def _load_kite_instruments(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "instruments_NSE.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_price_panel(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "panel_close.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _save_membership(constituents: pd.DataFrame, index: str, data_dir: Path) -> None:
    """Persist current membership.

    This is a *snapshot*, not history: NSE does not publish a machine-readable
    inclusion/exclusion log. Accumulating these snapshots over time builds the
    point-in-time record; until then the universe carries survivorship bias,
    which the universe builder warns about.
    """
    from src.data.universe import IndexMembership

    membership = IndexMembership.load(data_dir)
    today = pd.Timestamp.today().normalize()
    rows = [
        {"isin": isin, "index_name": index, "start_date": today, "end_date": pd.NaT}
        for isin in constituents["isin"].dropna().unique()
    ]
    snapshot = pd.DataFrame(rows)

    if membership.df.empty:
        # First snapshot: backdate so existing history is usable.
        snapshot["start_date"] = pd.Timestamp("1990-01-01")
        combined = snapshot
    else:
        combined = pd.concat([membership.df, snapshot], ignore_index=True)
        combined = combined.drop_duplicates(subset=["isin", "index_name"], keep="first")

    IndexMembership(combined).save(data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
