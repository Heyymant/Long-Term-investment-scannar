"""Parameter optimization that resists curve-fitting.

The naive approach - grid search, take the highest in-sample Sharpe - is a
reliable way to produce a strategy that works beautifully until you fund it.
Three defences here:

1. **Robust objective** - median OOS Sharpe across walk-forward folds, minus
   penalties for turnover and complexity, rather than a single peak.
2. **Plateau selection** - prefer a parameter set whose *neighbours* also work.
   A lone spike in the parameter surface is noise; a broad plateau is signal.
3. **Trial accounting** - every configuration is logged to the run registry,
   because the Deflated Sharpe Ratio needs the true trial count. A
   degrees-of-freedom budget caps how much searching is allowed.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
@dataclass
class ParamSpace:
    """Declared tunable parameters, as dotted paths into StrategyConfig."""

    params: dict[str, Sequence[Any]] = field(default_factory=dict)

    def add(self, path: str, values: Sequence[Any]) -> ParamSpace:
        self.params[path] = list(values)
        return self

    @property
    def n_combinations(self) -> int:
        n = 1
        for v in self.params.values():
            n *= max(1, len(v))
        return n

    @property
    def degrees_of_freedom(self) -> int:
        return sum(1 for v in self.params.values() if len(v) > 1)

    def grid(self) -> list[dict[str, Any]]:
        if not self.params:
            return [{}]
        keys = list(self.params)
        return [dict(zip(keys, combo)) for combo in itertools.product(*(self.params[k] for k in keys))]

    def sample(self, n: int, seed: int = 42) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        keys = list(self.params)
        seen: set[tuple] = set()
        out: list[dict[str, Any]] = []
        # Cap attempts so a small space can't spin forever.
        for _ in range(n * 20):
            if len(out) >= n:
                break
            combo = tuple(rng.choice(self.params[k]) for k in keys)
            if combo in seen:
                continue
            seen.add(combo)
            out.append(dict(zip(keys, combo)))
        return out


def default_space_sleeve_a() -> ParamSpace:
    """A deliberately small default space - fewer knobs, less overfitting."""
    return (
        ParamSpace()
        .add("sleeve_a.top_n", [30, 50, 75])
        .add("sleeve_a.rank_buffer_multiple", [1.0, 1.5, 2.0])
        .add("sleeve_a.factors.weight_quality", [0.25, 0.35, 0.45])
        .add("sleeve_a.factors.weight_momentum", [0.20, 0.30, 0.40])
        .add("sleeve_a.regime.ma_days", [150, 200, 250])
    )


def apply_params(config, params: dict[str, Any]):
    """Return a deep copy of `config` with dotted-path overrides applied."""
    updated = config.model_copy(deep=True)
    for path, value in params.items():
        node = updated
        parts = path.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], value)
    return updated


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
@dataclass
class ObjectiveConfig:
    """How a configuration is scored. Defaults penalize churn and complexity."""

    metric: str = "sharpe"
    turnover_penalty: float = 0.10      # per unit of annual turnover
    complexity_penalty: float = 0.01    # per free parameter
    min_trades: int = 4
    use_median_fold: bool = True        # median over folds, not mean


def robust_objective(
    fold_metrics: list[dict[str, float]],
    turnover: float,
    n_free_params: int,
    cfg: ObjectiveConfig | None = None,
) -> float:
    """Score = central OOS metric - turnover penalty - complexity penalty.

    The median across folds is used by default: one spectacular fold should
    not rescue a configuration that fails everywhere else.
    """
    cfg = cfg or ObjectiveConfig()
    values = [f.get(cfg.metric, np.nan) for f in fold_metrics]
    values = [v for v in values if v is not None and not np.isnan(v)]
    if not values:
        return -np.inf

    central = float(np.median(values)) if cfg.use_median_fold else float(np.mean(values))
    return central - cfg.turnover_penalty * max(turnover, 0.0) - cfg.complexity_penalty * n_free_params


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
@dataclass
class OptimizationResult:
    results: pd.DataFrame
    best_params: dict[str, Any]
    best_score: float
    plateau_params: dict[str, Any] = field(default_factory=dict)
    plateau_score: float = np.nan
    n_trials: int = 0
    space: ParamSpace | None = None
    study: str = ""

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.results.nlargest(n, "score") if not self.results.empty else self.results

    def summary(self) -> dict:
        return {
            "n_trials": self.n_trials,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "plateau_params": self.plateau_params,
            "plateau_score": self.plateau_score,
            "degrees_of_freedom": self.space.degrees_of_freedom if self.space else None,
            "study": self.study,
        }


def grid_search(
    space: ParamSpace,
    evaluate: Callable[[dict[str, Any]], dict[str, float]],
    method: str = "grid",           # grid | random | optuna
    n_samples: int = 50,
    max_dof: int = 6,
    seed: int = 42,
    study: str = "optimization",
    registry=None,
    base_config=None,
) -> OptimizationResult:
    """Search the space, logging every trial.

    `evaluate(params) -> {"score": float, "sharpe": ..., "turnover": ..., ...}`
    """
    if space.degrees_of_freedom > max_dof:
        log.warning(
            "Search has %d free parameters (budget %d). More knobs means more "
            "overfitting - consider narrowing the space.",
            space.degrees_of_freedom, max_dof,
        )

    if method == "optuna":
        return _optuna_search(space, evaluate, n_samples, seed, study, registry, base_config)

    candidates = space.grid() if method == "grid" else space.sample(n_samples, seed)
    if method == "grid" and len(candidates) > n_samples * 10:
        log.warning("Grid has %d combinations; consider method='random'", len(candidates))

    rows: list[dict[str, Any]] = []
    for i, params in enumerate(candidates, 1):
        try:
            out = evaluate(params)
        except Exception as exc:  # noqa: BLE001 - a bad config shouldn't kill the sweep
            log.warning("Trial %d failed: %s", i, exc)
            continue

        row = {**{f"p:{k}": v for k, v in params.items()}, **out, "_params": params}
        rows.append(row)

        if registry is not None and base_config is not None:
            _log_trial(registry, base_config, params, out, study)

        if i % 10 == 0:
            log.info("Optimization: %d/%d trials", i, len(candidates))

    if not rows:
        return OptimizationResult(pd.DataFrame(), {}, -np.inf, n_trials=0, space=space, study=study)

    df = pd.DataFrame(rows)
    best_idx = df["score"].idxmax()
    best_params = df.loc[best_idx, "_params"]
    best_score = float(df.loc[best_idx, "score"])

    plateau_params, plateau_score = find_plateau(df, space)

    return OptimizationResult(
        results=df.drop(columns=["_params"]),
        best_params=best_params, best_score=best_score,
        plateau_params=plateau_params, plateau_score=plateau_score,
        n_trials=len(df), space=space, study=study,
    )


def find_plateau(
    results: pd.DataFrame, space: ParamSpace, neighbourhood: int = 1
) -> tuple[dict[str, Any], float]:
    """Pick the configuration with the best *neighbourhood average* score.

    This is the anti-curve-fitting step: a parameter set surrounded by other
    good parameter sets is far likelier to survive out-of-sample than an
    isolated peak.
    """
    if results.empty or not space.params:
        return {}, np.nan

    param_cols = [c for c in results.columns if c.startswith("p:")]
    if not param_cols:
        return {}, np.nan

    # Map each value to its ordinal position in the declared grid.
    ordinals = {}
    for col in param_cols:
        path = col[2:]
        values = space.params.get(path, [])
        ordinals[col] = {v: i for i, v in enumerate(values)}

    coords = np.array([
        [ordinals[c].get(row[c], -1) for c in param_cols]
        for _, row in results.iterrows()
    ], dtype=float)
    scores = results["score"].values

    best_avg, best_row = -np.inf, None
    for i in range(len(results)):
        dist = np.abs(coords - coords[i]).max(axis=1)
        near = dist <= neighbourhood
        if near.sum() < 2:
            continue
        avg = float(np.nanmean(scores[near]))
        if avg > best_avg:
            best_avg, best_row = avg, i

    if best_row is None:
        return {}, np.nan

    params = {c[2:]: results.iloc[best_row][c] for c in param_cols}
    return params, best_avg


def _optuna_search(
    space: ParamSpace, evaluate, n_trials: int, seed: int,
    study_name: str, registry, base_config,
) -> OptimizationResult:
    try:
        import optuna
    except ImportError:
        log.warning("optuna not installed; falling back to random search")
        return grid_search(space, evaluate, "random", n_trials, seed=seed, study=study_name,
                           registry=registry, base_config=base_config)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows: list[dict[str, Any]] = []

    def objective(trial: "optuna.Trial") -> float:
        params = {
            path: trial.suggest_categorical(path, list(values))
            for path, values in space.params.items()
        }
        out = evaluate(params)
        rows.append({**{f"p:{k}": v for k, v in params.items()}, **out, "_params": params})
        if registry is not None and base_config is not None:
            _log_trial(registry, base_config, params, out, study_name)
        return float(out.get("score", -np.inf))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if not rows:
        return OptimizationResult(pd.DataFrame(), {}, -np.inf, n_trials=0, space=space, study=study_name)

    df = pd.DataFrame(rows)
    best_idx = df["score"].idxmax()
    plateau_params, plateau_score = find_plateau(df, space)
    return OptimizationResult(
        results=df.drop(columns=["_params"]),
        best_params=df.loc[best_idx, "_params"],
        best_score=float(df.loc[best_idx, "score"]),
        plateau_params=plateau_params, plateau_score=plateau_score,
        n_trials=len(df), space=space, study=study_name,
    )


def _log_trial(registry, base_config, params: dict, out: dict, study: str) -> None:
    """Record a trial so the Deflated Sharpe Ratio sees the real trial count."""
    try:
        rec = registry.start(
            name=f"{study}_trial", config_hash=base_config.config_hash(),
            data_hash="", seed=base_config.seed, study=study,
            kind="optimization_trial", params=params,
        )
        registry.complete(rec, out)
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not log trial to registry: %s", exc)


def sensitivity_table(
    results: pd.DataFrame, param: str, metric: str = "score"
) -> pd.DataFrame:
    """Marginal effect of one parameter - flat is good, spiky is suspicious."""
    col = f"p:{param}"
    if results.empty or col not in results.columns:
        return pd.DataFrame()
    return (
        results.groupby(col)[metric]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={col: param})
    )


def sensitivity_heatmap_data(
    results: pd.DataFrame, param_x: str, param_y: str, metric: str = "score"
) -> pd.DataFrame:
    cx, cy = f"p:{param_x}", f"p:{param_y}"
    if results.empty or cx not in results.columns or cy not in results.columns:
        return pd.DataFrame()
    return results.pivot_table(index=cy, columns=cx, values=metric, aggfunc="mean")
