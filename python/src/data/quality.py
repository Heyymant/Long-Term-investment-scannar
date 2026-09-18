"""Data-quality checks -> `data_health.json`.

Bad data produces confident, wrong backtests. These checks run before any
research so problems surface as an explicit report (and a dashboard view)
rather than as mysterious alpha.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = "1.0.0"


@dataclass
class HealthIssue:
    severity: str          # info | warning | error
    check: str
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class DataHealthReport:
    generated_at: str
    schema_version: str
    n_dates: int
    n_instruments: int
    date_range: list[str]
    coverage_pct: float
    issues: list[HealthIssue] = field(default_factory=list)
    per_instrument: dict[str, Any] = field(default_factory=dict)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_errors"] = self.n_errors
        d["n_warnings"] = self.n_warnings
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        log.info("Data health report -> %s (%d errors, %d warnings)", path, self.n_errors, self.n_warnings)
        return path


def check_prices(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    max_stale_days: int = 10,
    extreme_return: float = 0.5,
    max_examples: int = 10,
) -> DataHealthReport:
    """Run the standard battery of price-panel checks."""
    issues: list[HealthIssue] = []

    if prices.empty:
        return DataHealthReport(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            schema_version=SCHEMA_VERSION, n_dates=0, n_instruments=0,
            date_range=[], coverage_pct=0.0,
            issues=[HealthIssue("error", "empty_panel", "Price panel is empty")],
        )

    n_dates, n_inst = prices.shape
    coverage = float(prices.notna().sum().sum() / (n_dates * n_inst) * 100)

    # 1) Non-monotonic / duplicate dates
    if not prices.index.is_monotonic_increasing:
        issues.append(HealthIssue("error", "index_order", "Date index is not sorted ascending"))
    dupes = int(prices.index.duplicated().sum())
    if dupes:
        issues.append(HealthIssue("error", "duplicate_dates", f"{dupes} duplicate dates in the index", dupes))

    # 2) Non-positive prices
    nonpos = (prices <= 0).sum()
    bad = nonpos[nonpos > 0]
    if len(bad):
        issues.append(HealthIssue(
            "error", "non_positive_price",
            f"{len(bad)} instruments have non-positive prices",
            int(bad.sum()), list(bad.index[:max_examples].astype(str)),
        ))

    # 3) Stale prices (repeated identical close)
    stale = _max_stale_run(prices)
    stale_bad = stale[stale > max_stale_days]
    if len(stale_bad):
        issues.append(HealthIssue(
            "warning", "stale_prices",
            f"{len(stale_bad)} instruments have an unchanged price for >{max_stale_days} days",
            len(stale_bad), list(stale_bad.index[:max_examples].astype(str)),
        ))

    # 4) Extreme single-day returns (possible unadjusted corporate actions)
    rets = prices.pct_change()
    extreme = (rets.abs() > extreme_return).sum()
    ext_bad = extreme[extreme > 0]
    if len(ext_bad):
        issues.append(HealthIssue(
            "warning", "extreme_returns",
            f"{len(ext_bad)} instruments have |1-day return| > {extreme_return:.0%} "
            "(check for unadjusted splits/bonuses)",
            int(ext_bad.sum()), list(ext_bad.index[:max_examples].astype(str)),
        ))

    # 5) Interior gaps (missing data between first and last observation)
    gaps = _interior_gaps(prices)
    gap_bad = gaps[gaps > 0]
    if len(gap_bad):
        issues.append(HealthIssue(
            "warning", "interior_gaps",
            f"{len(gap_bad)} instruments have missing days inside their listed history",
            int(gap_bad.sum()), list(gap_bad.index[:max_examples].astype(str)),
        ))

    # 6) Sparse names
    obs = prices.notna().sum()
    sparse = obs[obs < max(20, n_dates * 0.05)]
    if len(sparse):
        issues.append(HealthIssue(
            "info", "sparse_history",
            f"{len(sparse)} instruments have very little history",
            len(sparse), list(sparse.index[:max_examples].astype(str)),
        ))

    # 7) Zero-volume days
    if volumes is not None and not volumes.empty:
        common = prices.columns.intersection(volumes.columns)
        if len(common):
            zero = (volumes[common].fillna(0) <= 0).sum()
            zero_bad = zero[zero > n_dates * 0.10]
            if len(zero_bad):
                issues.append(HealthIssue(
                    "warning", "zero_volume",
                    f"{len(zero_bad)} instruments have zero volume on >10% of days",
                    len(zero_bad), list(zero_bad.index[:max_examples].astype(str)),
                ))
        missing_vol = [c for c in prices.columns if c not in volumes.columns]
        if missing_vol:
            issues.append(HealthIssue(
                "info", "missing_volume",
                f"{len(missing_vol)} instruments have no volume data (liquidity filters will skip them)",
                len(missing_vol), [str(x) for x in missing_vol[:max_examples]],
            ))

    return DataHealthReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        schema_version=SCHEMA_VERSION,
        n_dates=n_dates,
        n_instruments=n_inst,
        date_range=[str(prices.index[0].date()), str(prices.index[-1].date())],
        coverage_pct=round(coverage, 2),
        issues=issues,
        per_instrument={
            "observations": {str(k): int(v) for k, v in obs.items()},
            "max_stale_run": {str(k): int(v) for k, v in stale.items()},
        },
    )


def check_fundamentals(
    fundamentals: pd.DataFrame, required: list[str] | None = None
) -> list[HealthIssue]:
    """Coverage/sanity checks on the fundamentals table."""
    issues: list[HealthIssue] = []
    if fundamentals.empty:
        return [HealthIssue("warning", "fundamentals_empty",
                            "No fundamentals loaded; Quality/Value factors unavailable")]

    if "announce_date" not in fundamentals.columns:
        issues.append(HealthIssue(
            "error", "missing_announce_date",
            "fundamentals lack announce_date - point-in-time alignment is impossible (look-ahead risk)",
        ))

    required = required or [
        "roic", "gross_profitability", "cfo_to_pat", "fcf_to_assets",
        "debt_to_assets", "payout", "earnings_yield", "fcf_yield",
        "ev_ebitda", "pb", "pe",
    ]
    for col in required:
        if col not in fundamentals.columns:
            issues.append(HealthIssue("warning", "missing_metric", f"fundamentals missing '{col}'"))
            continue
        null_pct = float(fundamentals[col].isna().mean() * 100)
        if null_pct > 30:
            issues.append(HealthIssue(
                "warning", "sparse_metric",
                f"'{col}' is {null_pct:.0f}% null - factor scores will be unreliable",
                int(fundamentals[col].isna().sum()),
            ))

    # Announcement dates in the future are a red flag.
    if "announce_date" in fundamentals.columns:
        future = pd.to_datetime(fundamentals["announce_date"]) > pd.Timestamp.today()
        if future.any():
            issues.append(HealthIssue(
                "error", "future_announce_date",
                f"{int(future.sum())} fundamental rows are dated in the future",
                int(future.sum()),
            ))
    return issues


def _max_stale_run(prices: pd.DataFrame) -> pd.Series:
    """Longest run of an unchanged price per instrument."""
    out = {}
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) < 2:
            out[col] = 0
            continue
        unchanged = (s.diff() == 0).astype(int)
        # Length of the longest consecutive run of 1s.
        grp = (unchanged != unchanged.shift()).cumsum()
        runs = unchanged.groupby(grp).sum()
        out[col] = int(runs.max()) if len(runs) else 0
    return pd.Series(out)


def _interior_gaps(prices: pd.DataFrame) -> pd.Series:
    """Missing observations between an instrument's first and last data point."""
    out = {}
    for col in prices.columns:
        s = prices[col]
        valid = s.notna()
        if not valid.any():
            out[col] = 0
            continue
        first, last = valid.idxmax(), valid[::-1].idxmax()
        window = s.loc[first:last]
        out[col] = int(window.isna().sum())
    return pd.Series(out)


def build_report(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> DataHealthReport:
    report = check_prices(prices, volumes)
    if fundamentals is not None:
        report.issues.extend(check_fundamentals(fundamentals))
    return report
