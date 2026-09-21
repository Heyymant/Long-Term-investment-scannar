#!/usr/bin/env python
"""Nightly research refresh so the horse-race filters stay current.

    python scripts/nightly_research.py --quick   # pytest + ranking refresh
    python scripts/nightly_research.py --full    # also re-run the paper horse race

Does not replace the dashboard `latest` pointer. Writes
`artifacts/nightly_research.json` and `dashboard/assets/nightly.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out[-4000:]


def main() -> int:
    ap = argparse.ArgumentParser(description="Nightly metric tests + ranking refresh")
    ap.add_argument("--full", action="store_true", help="re-run the paper horse race")
    ap.add_argument("--quick", action="store_true", help="pytest + rankings only (default)")
    args = ap.parse_args()
    full = bool(args.full)

    tests = [
        "tests/test_factors.py",
        "tests/test_screener_features.py",
        "tests/test_trading_signals.py",
        "tests/test_contracts.py",
    ]
    code, test_log = _run([sys.executable, "-m", "pytest", *tests, "-q", "--tb=line"])
    pytest_ok = code == 0

    horse = None
    if full:
        hcode, hlog = _run([
            sys.executable, "scripts/run_paper_signal_horse_race.py",
            "--start", "2022-01-01", "--end", "2024-12-31",
        ])
        horse = {"ok": hcode == 0, "log_tail": hlog[-1500:]}
    else:
        from src.config import load_config
        from src.data.loader import load_dataset
        from src.schema import UniverseConfig
        from run_paper_signal_horse_race import _refresh_latest_rankings

        cfg = load_config()
        ds = load_dataset(
            cfg, UniverseConfig(min_financial_quarters=0),
            source="nse", run_health_check=False,
        )
        _refresh_latest_rankings(cfg, ds)

    dash = REPO / "dashboard" / "assets" / "paper_signals.json"
    passed, failed = [], []
    if dash.exists():
        try:
            payload = json.loads(dash.read_text(encoding="utf-8"))
            passed = payload.get("passed") or []
            failed = payload.get("failed") or []
        except json.JSONDecodeError:
            pass

    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full" if full else "quick",
        "pytest_ok": pytest_ok,
        "pytest_log": test_log[-1200:],
        "horse_race": horse,
        "passed": passed,
        "failed": failed,
        "n_passed": len(passed),
        "n_failed": len(failed),
        "note": "Decision support only. Rankings refreshed; latest pointer unchanged.",
    }
    art = REPO / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    (art / "nightly_research.json").write_text(text, encoding="utf-8")
    out_dash = REPO / "dashboard" / "assets" / "nightly.json"
    out_dash.write_text(text, encoding="utf-8")
    print(f"nightly {report['mode']} pytest={'PASS' if pytest_ok else 'FAIL'} "
          f"passed={len(passed)} failed={len(failed)}")
    print(f"Wrote {out_dash}")
    return 0 if pytest_ok and (horse is None or horse["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
