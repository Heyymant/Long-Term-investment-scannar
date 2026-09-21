"""Copy the dashboard + latest research JSON into a Netlify publish folder.

Run locally (or as the Netlify build command):

    python python/scripts/build_netlify.py

The hosted site is static: it does not run the Rust server or Python jobs.
Rankings, horse-race results and 1-year NSE charts are baked in from the
latest artifact run. Portfolio upload lives in the browser (localStorage).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "dashboard" / "assets"
PUBLISH = ROOT / "dashboard" / "assets"  # Netlify publish directory
ARTIFACTS = ROOT / "artifacts"
DATA = PUBLISH / "data"

SNAPSHOT = [
    "rankings.json",
    "etf_rankings.json",
    "backtest.json",
    "equity_curve.json",
    "strategy_config.json",
    "validation_stats.json",
    "attribution.json",
    "risk.json",
    "monthly_returns.json",
    "cost_sensitivity.json",
    "regime_breakdown.json",
    "holdings_target.json",
    "rebalance_orders.json",
    "rebalance_summary.json",
    "trade_signals.json",
    "trade_signals_summary.json",
    "data_health.json",
    "sparks.json",
    "ic_icir.json",
]


def latest_run_id() -> str | None:
    pointer = ARTIFACTS / "latest.json"
    if not pointer.exists():
        return None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    return payload.get("run_id")


def copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def main() -> int:
    if not ASSETS.exists():
        print(f"missing dashboard assets at {ASSETS}", file=sys.stderr)
        return 1

    DATA.mkdir(parents=True, exist_ok=True)

    run_id = latest_run_id()
    run_dir = ARTIFACTS / "runs" / run_id if run_id else None
    copied = 0
    if run_dir and run_dir.is_dir():
        for name in SNAPSHOT:
            if copy_if_exists(run_dir / name, DATA / name):
                copied += 1
        copy_if_exists(ARTIFACTS / "paper_signal_horse_race.json", DATA / "paper_signal_horse_race.json")
        copy_if_exists(ARTIFACTS / "nightly_research.json", DATA / "nightly_research.json")
        copy_if_exists(ARTIFACTS / "rebalance_summary.json", DATA / "rebalance_summary.json")
        print(f"snapshot from run {run_id}: {copied} files")
    else:
        print("no artifacts/latest.json — keeping existing dashboard/assets/data/")

    # Assets already live in the publish dir. Refresh the copies that the
    # horse-race / nightly scripts write next to the UI.
    for name in ("paper_signals.json", "sparks.json", "nightly.json"):
        src = ASSETS / name
        if src.exists() and src.resolve() != (PUBLISH / name).resolve():
            shutil.copy2(src, PUBLISH / name)

    # Prefer the run's sparks (same file the dashboard serves) if present.
    if (DATA / "sparks.json").exists():
        shutil.copy2(DATA / "sparks.json", PUBLISH / "sparks.json")

    manifest = {
        "run_id": run_id,
        "hosted": True,
        "note": "Static Netlify snapshot. Jobs, Kite login and live ticks stay on the local dashboard.",
        "files": sorted(p.name for p in DATA.glob("*.json")),
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {DATA / 'manifest.json'}")
    print(f"publish directory: {PUBLISH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
