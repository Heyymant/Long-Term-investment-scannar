"""Nightly research CLI stays importable and documents the expected flags."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_nightly_research_help():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nightly_research.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0
    assert "--full" in proc.stdout
    assert "--quick" in proc.stdout
