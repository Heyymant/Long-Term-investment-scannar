#!/usr/bin/env python
"""Build the committed real-market test fixtures.

The test suite runs exclusively on **real NSE market data**, not simulated
prices. This script carves a small, deterministic slice out of the full
downloaded archive and writes it to ``tests/fixtures/`` so the suite stays
fast and offline while still exercising genuine market behaviour - real
gaps, real corporate actions, real fat tails, real delistings.

Run once after fetching data:

    python scripts/fetch_nse.py --all --start 2022-01-01
    python scripts/build_fixtures.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                     # noqa: E402

from src.config import load_config                      # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build real-data test fixtures")
    ap.add_argument("--n-stocks", type=int, default=60,
                    help="most liquid N names to include")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config()
    d = cfg.paths.data_dir
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    close_path = d / "panel_close.parquet"
    if not close_path.exists():
        print("No price panels found. Run: python scripts/fetch_nse.py --all --start 2022-01-01")
        return 1

    close = _read_panel(close_path)
    traded_value = _read_panel(d / "panel_traded_value.parquet")
    volume = _read_panel(d / "panel_volume.parquet")

    if args.start:
        close = close.loc[close.index >= pd.Timestamp(args.start)]
    if args.end:
        close = close.loc[close.index <= pd.Timestamp(args.end)]

    # Pick the most liquid names with near-complete history. Liquid large/mid
    # caps keep the fixture representative of what the strategy would actually
    # trade, and complete history keeps tests deterministic.
    coverage = close.notna().mean()
    candidates = coverage[coverage > 0.98].index
    if traded_value.empty:
        ranked = close[candidates].mean().sort_values(ascending=False)
    else:
        ranked = traded_value[candidates].mean().sort_values(ascending=False)
    chosen = list(ranked.head(args.n_stocks).index)
    print(f"Selected {len(chosen)} liquid names with >98% history coverage")

    close = close[chosen].dropna(how="all")
    _write(close, "panel_close.parquet", index=True)
    for name, panel in (("panel_volume.parquet", volume),
                        ("panel_traded_value.parquet", traded_value)):
        if not panel.empty:
            _write(panel.reindex(index=close.index, columns=chosen), name, index=True)

    # Security master rows for the chosen names.
    sm_path = d / "security_master" / "securities.parquet"
    if sm_path.exists():
        sec = pd.read_parquet(sm_path)
        _write(sec[sec["isin"].isin(chosen)], "securities.parquet")

    hist_path = d / "symbol_history.parquet"
    if hist_path.exists():
        hist = pd.read_parquet(hist_path)
        _write(hist[hist["isin"].isin(chosen)], "symbol_history.parquet")

    # Benchmarks (real NSE index closes -> TRI) and overnight risk-free.
    bench_dir = d / "benchmarks"
    if bench_dir.exists():
        out = {}
        for f in bench_dir.glob("*.parquet"):
            series = pd.read_parquet(f).iloc[:, 0]
            series.index = pd.to_datetime(series.index)
            out[f.stem] = series.reindex(close.index).ffill()
        if out:
            _write(pd.DataFrame(out), "benchmarks.parquet", index=True)
            print(f"  benchmarks: {', '.join(out)}")

    rf_path = d / "risk_free.parquet"
    if rf_path.exists():
        rf = pd.read_parquet(rf_path).iloc[:, 0]
        rf.index = pd.to_datetime(rf.index)
        _write(rf.reindex(close.index).ffill().bfill().to_frame("risk_free"),
               "risk_free.parquet", index=True)

    # Corporate actions and fundamentals for these names.
    ca_path = d / "corporate_actions.parquet"
    if ca_path.exists():
        ca = pd.read_parquet(ca_path)
        _write(ca, "corporate_actions.parquet")   # keep all: few rows, all real

    f_path = d / "fundamentals.parquet"
    if f_path.exists():
        fund = pd.read_parquet(f_path)
        _write(fund, "fundamentals.parquet")

    e_path = d / "earnings.parquet"
    if e_path.exists():
        earn = pd.read_parquet(e_path)
        _write(earn, "earnings.parquet")

    # A real XBRL document, for the parser tests.
    _save_sample_xbrl(cfg, d)

    manifest = {
        "source": "NSE India public data (bhavcopy archive, index archive, XBRL filings)",
        "note": "Real market data only - no simulated prices anywhere in the test suite.",
        "n_stocks": len(chosen),
        "date_range": [str(close.index.min().date()), str(close.index.max().date())],
        "n_days": int(len(close)),
        "isins": chosen,
    }
    (FIXTURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in FIXTURE_DIR.iterdir() if f.is_file())
    print(f"\nFixtures written to {FIXTURE_DIR}")
    print(f"  {len(chosen)} stocks | {len(close)} days | "
          f"{manifest['date_range'][0]} -> {manifest['date_range'][1]}")
    print(f"  total size {total / 1e6:.2f} MB")
    return 0


def _read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def _write(df: pd.DataFrame, name: str, index: bool = False) -> None:
    if df is None or df.empty:
        return
    df.to_parquet(FIXTURE_DIR / name, index=index)
    print(f"  {name:32s} {df.shape}")


def _save_sample_xbrl(cfg, data_dir: Path) -> None:
    """Keep one real XBRL filing so the parser is tested against reality."""
    target = FIXTURE_DIR / "sample_filing.xml"
    if target.exists():
        return

    from src.data.nse import NSEClient

    client = NSEClient(cache_dir=data_dir)
    results = client.fetch_financial_results("Quarterly")
    if results.empty:
        print("  (no results available to sample an XBRL filing)")
        return

    for _, row in results.head(20).iterrows():
        content = client.fetch_xbrl(row.get("xbrl_url"))
        if content and "RevenueFromOperations" in content:
            target.write_text(content, encoding="utf-8")
            meta = {
                "isin": row.get("isin"), "symbol": row.get("symbol"),
                "period_end": str(row.get("period_end")),
                "announce_date": str(row.get("announce_date")),
                "source_url": row.get("xbrl_url"),
            }
            (FIXTURE_DIR / "sample_filing.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print(f"  sample_filing.xml                real filing for {row.get('symbol')}")
            return
    print("  (could not download a usable XBRL sample)")


if __name__ == "__main__":
    raise SystemExit(main())
