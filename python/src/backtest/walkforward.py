"""Walk-forward analysis with purged, embargoed cross-validation.

Two things make time-series validation different from ordinary k-fold:

1. **Purging** - if a signal predicts a 60-day forward return, training rows
   within 60 days of the test set overlap it and leak. Those rows are dropped.
2. **Embargo** - autocorrelation means samples immediately after the test
   window are still contaminated, so a buffer is excluded too.

Nested CV is available so the reported out-of-sample figure is not itself
contaminated by the selection that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    validation_start: pd.Timestamp | None = None
    validation_end: pd.Timestamp | None = None

    def describe(self) -> str:
        return (f"fold {self.fold_id}: train {self.train_start.date()}..{self.train_end.date()} "
                f"-> test {self.test_start.date()}..{self.test_end.date()}")


def generate_folds(
    index: pd.DatetimeIndex,
    n_folds: int = 5,
    mode: str = "rolling",              # rolling | anchored
    train_years: float = 4.0,
    test_years: float = 1.0,
    purge_days: int = 21,
    embargo_days: int = 10,
    validation_fraction: float = 0.0,   # >0 carves a validation slice off train
) -> list[Fold]:
    """Build walk-forward folds with purge + embargo gaps."""
    if len(index) < 100:
        return []

    ppy = 252
    train_len = int(train_years * ppy)
    test_len = int(test_years * ppy)
    gap = purge_days + embargo_days

    folds: list[Fold] = []
    total = len(index)
    start = 0

    for k in range(n_folds):
        train_lo = 0 if mode == "anchored" else start
        train_hi = train_lo + train_len if mode != "anchored" else train_len + k * test_len
        test_lo = train_hi + gap
        test_hi = test_lo + test_len

        if test_hi > total:
            break

        val_start = val_end = None
        eff_train_hi = train_hi
        if validation_fraction > 0:
            val_len = int((train_hi - train_lo) * validation_fraction)
            if val_len > 20:
                eff_train_hi = train_hi - val_len
                val_start = index[eff_train_hi]
                val_end = index[train_hi - 1]

        folds.append(Fold(
            fold_id=k,
            train_start=index[train_lo],
            train_end=index[eff_train_hi - 1],
            test_start=index[test_lo],
            test_end=index[test_hi - 1],
            validation_start=val_start,
            validation_end=val_end,
        ))
        start += test_len

    if not folds:
        log.warning(
            "No walk-forward folds fit in %d observations (need ~%d per fold)",
            total, train_len + test_len + gap,
        )
    return folds


def purged_indices(
    index: pd.DatetimeIndex, test_start: pd.Timestamp, test_end: pd.Timestamp,
    purge_days: int, embargo_days: int,
) -> np.ndarray:
    """Training positions with the test window (plus purge/embargo) removed."""
    purge_lo = test_start - pd.Timedelta(days=purge_days)
    embargo_hi = test_end + pd.Timedelta(days=embargo_days)
    mask = (index < purge_lo) | (index > embargo_hi)
    return np.where(mask)[0]


@dataclass
class WalkForwardResult:
    folds: list[Fold] = field(default_factory=list)
    fold_results: list[dict] = field(default_factory=list)
    oos_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    selected_params: list[dict] = field(default_factory=list)

    @property
    def median_oos_sharpe(self) -> float:
        vals = [f.get("test_sharpe", np.nan) for f in self.fold_results]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.median(vals)) if vals else np.nan

    @property
    def mean_oos_sharpe(self) -> float:
        vals = [f.get("test_sharpe", np.nan) for f in self.fold_results]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    @property
    def consistency(self) -> float:
        """Fraction of folds with a positive OOS Sharpe."""
        vals = [f.get("test_sharpe", np.nan) for f in self.fold_results]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean([v > 0 for v in vals])) if vals else np.nan

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.fold_results)

    def summary(self) -> dict:
        return {
            "n_folds": len(self.fold_results),
            "median_oos_sharpe": self.median_oos_sharpe,
            "mean_oos_sharpe": self.mean_oos_sharpe,
            "consistency": self.consistency,
            "oos_sharpe_std": float(np.std([
                f.get("test_sharpe", np.nan) for f in self.fold_results
            ])) if self.fold_results else np.nan,
        }


def run_walk_forward(
    index: pd.DatetimeIndex,
    evaluate: Callable[[pd.Timestamp, pd.Timestamp, dict | None], dict],
    optimize: Callable[[pd.Timestamp, pd.Timestamp], dict] | None = None,
    n_folds: int = 5,
    mode: str = "rolling",
    train_years: float = 4.0,
    test_years: float = 1.0,
    purge_days: int = 21,
    embargo_days: int = 10,
) -> WalkForwardResult:
    """Optimize on train, evaluate only on test, report OOS results.

    `optimize(train_start, train_end) -> params`
    `evaluate(start, end, params) -> {"sharpe": ..., "returns": pd.Series, ...}`
    """
    folds = generate_folds(
        index, n_folds, mode, train_years, test_years, purge_days, embargo_days
    )
    result = WalkForwardResult(folds=folds)
    oos_chunks: list[pd.Series] = []

    for fold in folds:
        params = optimize(fold.train_start, fold.train_end) if optimize else None

        train_eval = evaluate(fold.train_start, fold.train_end, params)
        test_eval = evaluate(fold.test_start, fold.test_end, params)

        row = {
            "fold": fold.fold_id,
            "train_start": fold.train_start, "train_end": fold.train_end,
            "test_start": fold.test_start, "test_end": fold.test_end,
            "train_sharpe": train_eval.get("sharpe", np.nan),
            "test_sharpe": test_eval.get("sharpe", np.nan),
            "train_cagr": train_eval.get("cagr", np.nan),
            "test_cagr": test_eval.get("cagr", np.nan),
            "test_max_dd": test_eval.get("max_drawdown", np.nan),
            "params": params,
        }
        # How much of the in-sample edge survived?
        if row["train_sharpe"] and not np.isnan(row["train_sharpe"]) and row["train_sharpe"] != 0:
            row["degradation"] = 1 - (row["test_sharpe"] / row["train_sharpe"])
        result.fold_results.append(row)
        result.selected_params.append(params or {})

        ret = test_eval.get("returns")
        if isinstance(ret, pd.Series) and not ret.empty:
            oos_chunks.append(ret)

        log.info("%s | train SR %.2f -> test SR %.2f", fold.describe(),
                 row["train_sharpe"] or np.nan, row["test_sharpe"] or np.nan)

    if oos_chunks:
        result.oos_returns = pd.concat(oos_chunks).sort_index()
        result.oos_returns = result.oos_returns[~result.oos_returns.index.duplicated()]

    return result


def combinatorial_splits(
    n_groups: int = 6, n_test_groups: int = 2
) -> Iterator[tuple[list[int], list[int]]]:
    """Combinatorial purged CV splits (Lopez de Prado).

    Yields many train/test partitions instead of one path, giving a
    distribution of OOS outcomes rather than a single lucky sequence.
    """
    from itertools import combinations

    groups = list(range(n_groups))
    for test in combinations(groups, n_test_groups):
        train = [g for g in groups if g not in test]
        yield train, list(test)
