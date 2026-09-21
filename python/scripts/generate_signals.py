"""Rebuild the BUY/SELL signal list from an existing run's artifacts.

    python scripts/generate_signals.py
    python scripts/generate_signals.py --run-id 20260919-033403-22f514

No market reload. Reads rankings + holdings + strategy_config and writes
trade_signals.parquet / .json plus trade_signals_summary.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                                          # noqa: E402
from src.reports.artifacts import ArtifactWriter, read_latest_run_id        # noqa: E402
from src.signals.trading_rules import signals_from_artifacts                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate BUY/SELL signals from a run")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg = load_config()
    root = cfg.paths.artifacts_dir
    run_id = args.run_id or read_latest_run_id(root)
    if not run_id:
        print("No run found. Run a backtest first.")
        return 1

    run_dir = Path(root) / "runs" / run_id
    if not run_dir.is_dir():
        print(f"Run directory missing: {run_dir}")
        return 1

    report = signals_from_artifacts(run_dir)
    writer = ArtifactWriter(run_dir, run_id)
    writer.write_parquet("trade_signals.parquet", report.frame)
    writer.write_json("trade_signals_summary.json", report.summary)
    print(
        f"run {run_id}  BUY {report.summary['n_buys']}  "
        f"SELL {report.summary['n_sells']}  HOLD {report.summary['n_holds']}  "
        f"WATCH {report.summary['n_watch']}"
    )
    if not report.frame.empty:
        show = report.frame[report.frame["action"].isin(["BUY", "SELL"])]
        for _, r in show.iterrows():
            print(f"  {r['action']:<4} {r['symbol']:<12} rank {r['rank']}  {r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
