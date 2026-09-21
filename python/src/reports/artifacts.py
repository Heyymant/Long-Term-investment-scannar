"""Artifacts contract - the seam between Python analytics and the Rust dashboard.

Python writes, Rust reads. Everything lands under
``artifacts/runs/<run_id>/`` with a ``latest.json`` pointer, and every file
carries ``schema_version`` so the two sides can detect a mismatch instead of
silently misreading columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = "1.0.0"

# Filename -> description. The Rust side mirrors this list.
ARTIFACT_FILES = {
    "strategy_config.json": "the design under test",
    "backtest.json": "headline metrics per sleeve and combined",
    "equity_curve.parquet": "date, equity, after_tax_equity, drawdown, benchmark, exposure",
    "attribution.json": "true alpha, factor betas, t-stats",
    "ic_icir.parquet": "per-factor IC/ICIR diagnostics",
    "validation_stats.json": "Sharpe CI, bootstrap, DSR, haircut, PBO, SPA, MinTRL",
    "optimization_results.parquet": "per-config metrics + OOS results",
    "risk.json": "exposure, drawdown, sector weights, correlations, liquidity",
    "rankings.parquet": "current factor scores per stock",
    "trade_signals.parquet": "BUY/SELL/HOLD/WATCH list from the trading rules",
    "holdings_target.csv": "target weights",
    "rebalance_orders.csv": "manual execution checklist",
    "statarb_pairs.parquet": "Sleeve B cointegrated pairs / residual candidates",
    "statarb_signals.parquet": "Sleeve B current signals",
    "earnings_calendar.parquet": "upcoming and recent results dates",
    "event_signals.parquet": "Sleeve C SUE rankings and PEAD signals",
    "journal.parquet": "executed fills vs checklist, realized slippage",
    "data_health.json": "data quality report",
    "monthly_returns.parquet": "year x month return table",
    "walkforward.parquet": "per-fold out-of-sample results",
}


@dataclass
class ArtifactWriter:
    """Writes one run's artifacts atomically-ish and updates the latest pointer."""

    run_dir: Path
    run_id: str

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # -- primitives ------------------------------------------------------------
    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / name
        body = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **payload}
        # NaN/Infinity are not valid JSON and break strict parsers (including
        # the Rust dashboard), so they are normalized to null.
        path.write_text(
            json.dumps(_sanitize(body), indent=2, default=_json_default, allow_nan=False),
            encoding="utf-8",
        )
        return path

    def write_parquet(self, name: str, df: pd.DataFrame, index: bool = False) -> Path | None:
        """Write Parquet for analysis, plus a compact JSON sibling.

        The JSON is what the dashboard reads. Keeping the contract in JSON means
        the Rust side needs no dataframe library at all - these tables are small
        (hundreds to a few thousand rows) and the simplicity is worth far more
        than the bytes saved.
        """
        if df is None or df.empty:
            return None
        path = self.run_dir / name
        out = df.copy()
        if index and out.index.name is None:
            out.index.name = "index"
        out.to_parquet(path, index=index)
        self._write_table_json(Path(name).with_suffix(".json").name, out, index)
        return path

    def write_csv(self, name: str, df: pd.DataFrame, index: bool = False) -> Path | None:
        if df is None or df.empty:
            return None
        path = self.run_dir / name
        df.to_csv(path, index=index)
        self._write_table_json(Path(name).with_suffix(".json").name, df, index)
        return path

    def _write_table_json(self, name: str, df: pd.DataFrame, index: bool = False) -> Path:
        """Columnar JSON: {schema_version, columns, rows}."""
        frame = df.reset_index() if index else df
        columns = [str(c) for c in frame.columns]
        rows = [
            [_jsonable(v) for v in record]
            for record in frame.itertuples(index=False, name=None)
        ]
        path = self.run_dir / name
        path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "columns": columns,
                    "rows": rows,
                    "n_rows": len(rows),
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return path

    # -- typed writers ---------------------------------------------------------
    def write_config(self, config) -> Path:
        path = self.run_dir / "strategy_config.json"
        path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_equity_curve(
        self,
        equity: pd.Series,
        after_tax: pd.Series | None = None,
        benchmark: pd.Series | None = None,
        exposure: pd.Series | None = None,
        drawdown: pd.Series | None = None,
    ) -> Path | None:
        if equity is None or equity.empty:
            return None
        df = pd.DataFrame({"date": equity.index, "equity": equity.values})
        if after_tax is not None and not after_tax.empty:
            df["after_tax_equity"] = after_tax.reindex(equity.index).values
        if benchmark is not None and not benchmark.empty:
            b = benchmark.reindex(equity.index).ffill()
            # Rebase the benchmark so both curves start at the same level.
            first = b.dropna().iloc[0] if b.notna().any() else np.nan
            if first and not np.isnan(first):
                df["benchmark"] = (b / first * equity.iloc[0]).values
        if exposure is not None and not exposure.empty:
            df["exposure"] = exposure.reindex(equity.index).ffill().values
        if drawdown is not None and not drawdown.empty:
            df["drawdown"] = drawdown.reindex(equity.index).values
        else:
            df["drawdown"] = (equity / equity.cummax() - 1.0).values
        return self.write_parquet("equity_curve.parquet", df)

    def write_metrics(self, metrics_by_sleeve: dict[str, dict]) -> Path:
        return self.write_json("backtest.json", {"sleeves": metrics_by_sleeve})

    def write_rankings(self, scores: pd.DataFrame) -> Path | None:
        if scores is None or scores.empty:
            return None
        df = scores.reset_index().rename(columns={"index": "isin"})
        return self.write_parquet("rankings.parquet", df)

    def finalize(self, artifacts_root: Path, update_latest: bool = True) -> None:
        """Record a manifest; optionally point `latest` at this run."""
        manifest = {
            "run_id": self.run_id,
            "schema_version": SCHEMA_VERSION,
            "files": sorted(p.name for p in self.run_dir.iterdir() if p.is_file()),
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if update_latest:
            (Path(artifacts_root) / "latest.json").write_text(
                json.dumps({"run_id": self.run_id, "schema_version": SCHEMA_VERSION}), encoding="utf-8"
            )
        log.info("Artifacts written to %s (%d files)", self.run_dir, len(manifest["files"]))


def _jsonable(v: Any) -> Any:
    """Coerce one cell into a JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, pd.Period)):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v, (int, str)):
        return v
    if pd.isna(v):
        return None
    return str(v)


def _sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so the JSON is valid."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _json_default(o: Any) -> Any:
    if isinstance(o, (pd.Timestamp, pd.Period)):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def read_latest_run_id(artifacts_root: str | Path) -> str | None:
    p = Path(artifacts_root) / "latest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("run_id")
    except (json.JSONDecodeError, OSError):
        return None
