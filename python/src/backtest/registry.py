"""Run registry - reproducibility and honest trial counting.

Two jobs:

1. **Reproducibility.** Every run records its config hash, data-snapshot hash,
   seed and git commit, so any result can be regenerated exactly.

2. **Trial counting.** The Deflated Sharpe Ratio needs to know how many
   configurations were actually tried. If you test 500 variants and report the
   best, its Sharpe is inflated by selection. The registry is the only honest
   source for that count - which is precisely why it must log *every* trial,
   including the disappointing ones.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

REGISTRY_FILE = "runs_registry.parquet"


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    name: str
    config_hash: str
    data_hash: str
    seed: int
    git_commit: str
    study: str = "adhoc"            # groups trials belonging to one search
    kind: str = "backtest"          # backtest | optimization_trial | validation
    status: str = "running"         # running | completed | failed
    sharpe: float | None = None
    cagr: float | None = None
    max_drawdown: float | None = None
    turnover: float | None = None
    notes: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = json.dumps(self.params, sort_keys=True, default=str)
        return d


class RunRegistry:
    """Append-only ledger of every run, stored as parquet."""

    def __init__(self, artifacts_dir: str | Path):
        self.root = Path(artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / REGISTRY_FILE

    # -- read ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=[f.name for f in RunRecord.__dataclass_fields__.values()])
        return pd.read_parquet(self.path)

    def trial_count(self, study: str | None = None, config_hash: str | None = None) -> int:
        """Number of trials run - feeds the Deflated Sharpe Ratio."""
        df = self.load()
        if df.empty:
            return 0
        if study:
            df = df[df["study"] == study]
        if config_hash:
            df = df[df["config_hash"] == config_hash]
        return int(len(df))

    def best(self, study: str, metric: str = "sharpe") -> pd.Series | None:
        df = self.load()
        if df.empty:
            return None
        df = df[(df["study"] == study) & (df["status"] == "completed")]
        if df.empty or metric not in df.columns:
            return None
        return df.loc[df[metric].idxmax()]

    def study_summary(self) -> pd.DataFrame:
        df = self.load()
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby("study")
            .agg(n_trials=("run_id", "count"),
                 best_sharpe=("sharpe", "max"),
                 median_sharpe=("sharpe", "median"),
                 last_run=("created_at", "max"))
            .reset_index()
            .sort_values("last_run", ascending=False)
        )

    # -- write -----------------------------------------------------------------
    def start(
        self, name: str, config_hash: str, data_hash: str, seed: int,
        study: str = "adhoc", kind: str = "backtest", params: dict[str, Any] | None = None,
    ) -> RunRecord:
        record = RunRecord(
            run_id=_new_run_id(),
            created_at=datetime.now().isoformat(timespec="seconds"),
            name=name, config_hash=config_hash, data_hash=data_hash,
            seed=seed, git_commit=git_commit(), study=study, kind=kind,
            params=params or {},
        )
        self._append(record)
        return record

    def complete(
        self, record: RunRecord, metrics: dict[str, Any] | None = None, notes: str = ""
    ) -> None:
        record.status = "completed"
        record.notes = notes
        if metrics:
            record.sharpe = _get(metrics, "sharpe")
            record.cagr = _get(metrics, "cagr")
            record.max_drawdown = _get(metrics, "max_drawdown")
            record.turnover = _get(metrics, "turnover")
        self._update(record)

    def fail(self, record: RunRecord, error: str) -> None:
        record.status = "failed"
        record.notes = error[:500]
        self._update(record)

    def _append(self, record: RunRecord) -> None:
        df = self.load()
        df = pd.concat([df, pd.DataFrame([record.to_row()])], ignore_index=True)
        df.to_parquet(self.path, index=False)

    def _update(self, record: RunRecord) -> None:
        df = self.load()
        if df.empty or record.run_id not in set(df["run_id"]):
            self._append(record)
            return
        idx = df.index[df["run_id"] == record.run_id][0]
        for k, v in record.to_row().items():
            df.at[idx, k] = v
        df.to_parquet(self.path, index=False)

    def run_dir(self, run_id: str) -> Path:
        d = self.root / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def set_latest(self, run_id: str) -> None:
        """Pointer the dashboard follows to find the most recent run."""
        (self.root / "latest.json").write_text(
            json.dumps({"run_id": run_id, "updated_at": datetime.now().isoformat(timespec="seconds")}),
            encoding="utf-8",
        )

    def latest_run_id(self) -> str | None:
        p = self.root / "latest.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("run_id")
        except (json.JSONDecodeError, OSError):
            return None


def _get(d: dict[str, Any], key: str) -> float | None:
    v = d.get(key)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _new_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def data_hash(*frames: pd.DataFrame | pd.Series | None) -> str:
    """Fingerprint the input data so a run can be tied to its snapshot."""
    h = hashlib.sha256()
    for f in frames:
        if f is None:
            continue
        try:
            h.update(str(getattr(f, "shape", len(f))).encode())
            if hasattr(f, "index") and len(f.index):
                h.update(str(f.index[0]).encode())
                h.update(str(f.index[-1]).encode())
            if isinstance(f, pd.DataFrame) and len(f.columns):
                h.update(",".join(map(str, sorted(f.columns)[:50])).encode())
        except (TypeError, ValueError, IndexError):
            continue
    return h.hexdigest()[:16]
