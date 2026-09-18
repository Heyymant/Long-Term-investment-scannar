"""Statistical validation - is the edge real, or luck and overfitting?

A backtest Sharpe of 1.2 means very little on its own. If it is the best of
500 configurations tried on 8 years of data, the honest expectation is far
lower. This module implements the standard defences:

  * Lo (2002)       - autocorrelation-corrected Sharpe standard errors
  * block bootstrap - confidence intervals preserving time dependence
  * Deflated Sharpe - discounts for the number of trials (Bailey & Lopez de Prado)
  * Haircut Sharpe  - multiple-testing haircut (Harvey, Liu & Zhu)
  * PBO via CSCV    - probability the in-sample best underperforms out-of-sample
  * White / Hansen  - Reality Check and SPA for data snooping
  * MinBTL / MinTRL - how much track record is needed to trust the Sharpe
  * leak detectors  - shifted-signal and shuffled-future sanity tests
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..config import get_logger

log = get_logger(__name__)

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Sharpe inference
# --------------------------------------------------------------------------- #
@dataclass
class SharpeTest:
    sharpe: float
    standard_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_obs: int
    autocorrelation: float = np.nan
    method: str = "lo2002"

    @property
    def significant(self) -> bool:
        return bool(self.p_value < 0.05) if not np.isnan(self.p_value) else False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant_5pct"] = self.significant
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def sharpe_standard_error(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> tuple[float, float]:
    """Lo (2002) standard error, adjusted for autocorrelation and non-normality.

    Naive SE = sqrt((1 + SR^2/2)/n) assumes iid normal returns. Real strategy
    returns are skewed, fat-tailed and autocorrelated, all of which inflate
    apparent significance.
    """
    r = returns.dropna()
    n = len(r)
    if n < 20:
        return np.nan, np.nan

    sd = r.std(ddof=1)
    if sd == 0:
        return np.nan, np.nan

    sr = float(r.mean() / sd)                 # per period
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    # Non-normality correction (Mertens / Lo).
    var = (1 + 0.5 * sr**2 - skew * sr + (kurt - 3) / 4 * sr**2) / n
    se = float(np.sqrt(max(var, 0.0)))

    rho1 = float(r.autocorr(lag=1)) if n > 2 else 0.0
    if not np.isnan(rho1) and abs(rho1) > 0.01:
        # Positive autocorrelation understates risk -> widen the SE.
        se *= float(np.sqrt((1 + rho1) / (1 - rho1))) if abs(rho1) < 0.99 else 1.0

    return se * np.sqrt(periods_per_year), rho1


def sharpe_test(
    returns: pd.Series, benchmark_sharpe: float = 0.0,
    periods_per_year: int = TRADING_DAYS, confidence: float = 0.95,
) -> SharpeTest:
    """Test whether the annualized Sharpe exceeds `benchmark_sharpe`."""
    r = returns.dropna()
    n = len(r)
    if n < 20 or r.std(ddof=1) == 0:
        return SharpeTest(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n)

    sr_ann = float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))
    se, rho1 = sharpe_standard_error(r, periods_per_year)
    if np.isnan(se) or se == 0:
        return SharpeTest(sr_ann, np.nan, np.nan, np.nan, np.nan, np.nan, n, rho1)

    t_stat = (sr_ann - benchmark_sharpe) / se
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
    z = stats.norm.ppf(0.5 + confidence / 2)

    return SharpeTest(
        sharpe=sr_ann, standard_error=se, t_stat=float(t_stat), p_value=p_value,
        ci_lower=sr_ann - z * se, ci_upper=sr_ann + z * se,
        n_obs=n, autocorrelation=rho1,
    )


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap with geometric block lengths."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            idx[i] = rng.integers(0, n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def bootstrap_metric(
    returns: pd.Series,
    metric_fn=None,
    n_boot: int = 1000,
    mean_block: int = 21,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for a statistic of the return series.

    Block bootstrap (not iid) because resampling day-by-day would destroy the
    autocorrelation and volatility clustering that drive real risk.
    """
    r = returns.dropna()
    if len(r) < 30:
        return {"point": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_boot": 0}

    if metric_fn is None:
        def metric_fn(x: np.ndarray) -> float:
            sd = x.std(ddof=1)
            return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan

    rng = np.random.default_rng(seed)
    values = np.asarray(r.values)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(len(values), mean_block, rng)
        samples[b] = metric_fn(values[idx])

    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return {"point": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_boot": 0}

    lo = (1 - confidence) / 2 * 100
    return {
        "point": float(metric_fn(values)),
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "ci_lower": float(np.percentile(samples, lo)),
        "ci_upper": float(np.percentile(samples, 100 - lo)),
        "prob_positive": float((samples > 0).mean()),
        "n_boot": int(len(samples)),
    }


def permutation_test(
    returns: pd.Series, n_perm: int = 1000, seed: int = 42
) -> dict[str, float]:
    """Is the Sharpe better than random orderings of the same returns?

    Shuffling destroys any timing/compounding structure while keeping the
    return distribution, so it isolates sequencing effects.
    """
    r = returns.dropna()
    if len(r) < 30:
        return {"observed": np.nan, "p_value": np.nan}

    def _sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    rng = np.random.default_rng(seed)
    values = np.asarray(r.values)
    observed = _sharpe(values)
    perm = np.array([_sharpe(rng.permutation(values)) for _ in range(n_perm)])
    return {
        "observed": observed,
        "permuted_mean": float(perm.mean()),
        "p_value": float((perm >= observed).mean()),
        "n_perm": n_perm,
    }


# --------------------------------------------------------------------------- #
# Multiple-testing corrections
# --------------------------------------------------------------------------- #
def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_variance: float | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    Probability the true Sharpe is positive once you account for having
    selected the best of `n_trials`. Testing many variants guarantees some
    will look good by chance; DSR prices that in.
    """
    if n_obs < 20 or np.isnan(observed_sharpe):
        return {"dsr": np.nan, "expected_max_sharpe": np.nan, "n_trials": n_trials}

    # Expected maximum Sharpe under the null of zero true skill.
    var_sr = sharpe_variance if sharpe_variance is not None else 1.0
    n_trials = max(1, int(n_trials))
    if n_trials == 1:
        expected_max = 0.0
    else:
        e = EULER_MASCHERONI
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        expected_max = np.sqrt(var_sr) * ((1 - e) * z1 + e * z2)

    sr_per_period = observed_sharpe / np.sqrt(periods_per_year)
    emax_per_period = expected_max / np.sqrt(periods_per_year)

    denom = np.sqrt(
        max(1 - skew * sr_per_period + (kurtosis - 1) / 4 * sr_per_period**2, 1e-12)
    )
    numer = (sr_per_period - emax_per_period) * np.sqrt(n_obs - 1)
    dsr = float(stats.norm.cdf(numer / denom))

    return {
        "dsr": dsr,
        "observed_sharpe": float(observed_sharpe),
        "expected_max_sharpe": float(expected_max),
        "n_trials": n_trials,
        "n_obs": int(n_obs),
        "passes_95": bool(dsr > 0.95),
    }


def haircut_sharpe(
    observed_sharpe: float, n_trials: int, n_obs: int,
    periods_per_year: int = TRADING_DAYS, method: str = "bonferroni",
) -> dict[str, float]:
    """Harvey-Liu-Zhu style haircut: what survives multiple-testing correction."""
    if np.isnan(observed_sharpe) or n_obs < 20:
        return {"haircut_sharpe": np.nan, "haircut_pct": np.nan}

    se = np.sqrt((1 + 0.5 * (observed_sharpe / np.sqrt(periods_per_year)) ** 2) / n_obs)
    se_ann = se * np.sqrt(periods_per_year)
    t_stat = observed_sharpe / se_ann if se_ann > 0 else np.nan
    p_single = float(2 * (1 - stats.norm.cdf(abs(t_stat)))) if not np.isnan(t_stat) else np.nan

    n_trials = max(1, int(n_trials))
    if method == "bonferroni":
        p_adj = min(1.0, p_single * n_trials)
    elif method == "holm":
        p_adj = min(1.0, p_single * n_trials)   # worst-case rank
    else:  # BHY
        c = sum(1.0 / i for i in range(1, n_trials + 1))
        p_adj = min(1.0, p_single * n_trials * c)

    z_adj = stats.norm.ppf(1 - p_adj / 2) if 0 < p_adj < 1 else 0.0
    adjusted = float(max(0.0, z_adj * se_ann))

    return {
        "observed_sharpe": float(observed_sharpe),
        "haircut_sharpe": adjusted,
        "haircut_pct": float((1 - adjusted / observed_sharpe) * 100) if observed_sharpe > 0 else np.nan,
        "p_value_single": p_single,
        "p_value_adjusted": float(p_adj),
        "method": method,
        "n_trials": n_trials,
    }


def min_track_record_length(
    observed_sharpe: float, target_sharpe: float = 0.0,
    skew: float = 0.0, kurtosis: float = 3.0, confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """MinTRL - observations needed to be confident SR > target.

    Answers "how long must the track record be before I believe this?"
    """
    if np.isnan(observed_sharpe) or observed_sharpe <= target_sharpe:
        return {"min_track_record_obs": np.inf, "min_track_record_years": np.inf}

    sr = observed_sharpe / np.sqrt(periods_per_year)
    tgt = target_sharpe / np.sqrt(periods_per_year)
    z = stats.norm.ppf(confidence)

    numer = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    n = 1 + numer * (z / (sr - tgt)) ** 2

    return {
        "min_track_record_obs": float(n),
        "min_track_record_years": float(n / periods_per_year),
        "confidence": confidence,
    }


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------- #
def probability_backtest_overfitting(
    returns_matrix: pd.DataFrame, n_splits: int = 8, seed: int = 42
) -> dict[str, float]:
    """PBO via Combinatorially Symmetric Cross-Validation.

    Split the timeline into S chunks, form every balanced IS/OOS partition,
    pick the best configuration in-sample and see where it ranks out-of-sample.
    If the in-sample winner is routinely a below-median performer OOS, the
    selection process is fitting noise.

    `returns_matrix`: columns = configurations, rows = time.
    """
    if returns_matrix.shape[1] < 2 or returns_matrix.shape[0] < n_splits * 4:
        return {"pbo": np.nan, "n_combinations": 0,
                "note": "need >= 2 configs and enough observations"}

    n_splits = n_splits if n_splits % 2 == 0 else n_splits - 1
    chunks = np.array_split(np.arange(len(returns_matrix)), n_splits)

    logits: list[float] = []
    n_configs = returns_matrix.shape[1]

    for is_idx in itertools.combinations(range(n_splits), n_splits // 2):
        oos_idx = [i for i in range(n_splits) if i not in is_idx]
        is_rows = np.concatenate([chunks[i] for i in is_idx])
        oos_rows = np.concatenate([chunks[i] for i in oos_idx])

        is_sr = _sharpe_matrix(returns_matrix.iloc[is_rows])
        oos_sr = _sharpe_matrix(returns_matrix.iloc[oos_rows])
        if is_sr.isna().all() or oos_sr.isna().all():
            continue

        best = is_sr.idxmax()
        # Relative rank of the IS winner in the OOS ranking.
        rank = oos_sr.rank(pct=True).get(best, np.nan)
        if np.isnan(rank):
            continue
        rank = min(max(float(rank), 1e-6), 1 - 1e-6)
        logits.append(float(np.log(rank / (1 - rank))))

    if not logits:
        return {"pbo": np.nan, "n_combinations": 0}

    arr = np.array(logits)
    pbo = float((arr <= 0).mean())     # winner landed below OOS median
    return {
        "pbo": pbo,
        "n_combinations": len(arr),
        "n_configs": n_configs,
        "median_logit": float(np.median(arr)),
        "interpretation": _pbo_verdict(pbo),
    }


def _pbo_verdict(pbo: float) -> str:
    if pbo < 0.1:
        return "low overfitting risk"
    if pbo < 0.3:
        return "moderate overfitting risk"
    if pbo < 0.5:
        return "high overfitting risk"
    return "severe - selection is likely fitting noise"


def _sharpe_matrix(df: pd.DataFrame) -> pd.Series:
    mean = df.mean()
    sd = df.std(ddof=1)
    return (mean / sd.replace(0, np.nan)) * np.sqrt(TRADING_DAYS)


# --------------------------------------------------------------------------- #
# Data-snooping tests
# --------------------------------------------------------------------------- #
def reality_check(
    strategy_returns: pd.DataFrame, benchmark_returns: pd.Series,
    n_boot: int = 1000, mean_block: int = 21, seed: int = 42,
) -> dict[str, float]:
    """White's Reality Check: is the BEST of many strategies genuinely better
    than the benchmark, given that we searched?"""
    if strategy_returns.empty:
        return {"p_value": np.nan, "best_strategy": None}

    aligned = strategy_returns.join(benchmark_returns.rename("_bench"), how="inner").dropna()
    if len(aligned) < 30:
        return {"p_value": np.nan, "best_strategy": None}

    bench = aligned["_bench"]
    strategies = aligned.drop(columns=["_bench"])
    excess = strategies.sub(bench, axis=0)

    mean_excess = excess.mean()
    n = len(excess)
    v_obs = float(np.sqrt(n) * mean_excess.max())
    best = str(mean_excess.idxmax())

    rng = np.random.default_rng(seed)
    values = excess.values
    stats_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        resampled = values[idx]
        centered = resampled.mean(axis=0) - mean_excess.values
        stats_boot[b] = np.sqrt(n) * centered.max()

    return {
        "test_statistic": v_obs,
        "p_value": float((stats_boot >= v_obs).mean()),
        "best_strategy": best,
        "best_mean_excess": float(mean_excess.max()),
        "n_strategies": int(strategies.shape[1]),
        "n_boot": n_boot,
    }


def spa_test(
    strategy_returns: pd.DataFrame, benchmark_returns: pd.Series,
    n_boot: int = 1000, mean_block: int = 21, seed: int = 42,
) -> dict[str, float]:
    """Hansen's Superior Predictive Ability test.

    More powerful than the Reality Check because it studentizes and discards
    hopeless strategies rather than letting them dilute the null.
    """
    if strategy_returns.empty:
        return {"p_value": np.nan}

    aligned = strategy_returns.join(benchmark_returns.rename("_bench"), how="inner").dropna()
    if len(aligned) < 30:
        return {"p_value": np.nan}

    bench = aligned["_bench"]
    excess = aligned.drop(columns=["_bench"]).sub(bench, axis=0)
    n = len(excess)

    mean_excess = excess.mean().values
    sd = excess.std(ddof=1).replace(0, np.nan).values
    with np.errstate(invalid="ignore"):
        t_obs = np.sqrt(n) * mean_excess / sd
    t_max = float(np.nanmax(t_obs))

    # Recentring threshold: drop strategies too poor to matter.
    threshold = -np.sqrt(2 * np.log(np.log(max(n, 3)))) * sd / np.sqrt(n)
    recentre = np.where(mean_excess >= threshold, mean_excess, 0.0)

    rng = np.random.default_rng(seed)
    values = excess.values
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        resampled = values[idx]
        centered = resampled.mean(axis=0) - recentre
        with np.errstate(invalid="ignore"):
            t_boot = np.sqrt(n) * centered / sd
        boot[b] = np.nanmax(t_boot)

    return {
        "test_statistic": t_max,
        "p_value": float((boot >= t_max).mean()),
        "n_strategies": int(excess.shape[1]),
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------------- #
# Leak detectors
# --------------------------------------------------------------------------- #
def detect_lookahead(
    signal_panel: pd.DataFrame, forward_returns: pd.DataFrame,
    threshold: float = 0.15,
) -> dict[str, object]:
    """Flag suspiciously strong *contemporaneous* signal/return correlation.

    A signal correlating with the return it is supposed to predict, at the
    same timestamp, usually means future information leaked in. Real
    cross-sectional IC in equities is single-digit percent; 0.15+ is a red
    flag, not a discovery.
    """
    from ..factors.ic_icir import compute_ic

    ic = compute_ic(signal_panel, forward_returns)
    if ic.empty:
        return {"suspicious": False, "mean_ic": np.nan, "note": "no overlapping data"}

    mean_ic = float(ic.mean())
    suspicious = abs(mean_ic) > threshold
    return {
        "suspicious": bool(suspicious),
        "mean_ic": mean_ic,
        "max_ic": float(ic.max()),
        "threshold": threshold,
        "note": (
            f"Mean IC {mean_ic:.3f} exceeds {threshold} - inspect for look-ahead"
            if suspicious else "IC within a plausible range"
        ),
    }


def shifted_signal_test(
    backtest_fn, shift_days: int = 1, tolerance: float = 0.5
) -> dict[str, object]:
    """Delaying signals by a day should *reduce* performance, not leave it
    unchanged. If it does not, execution timing is probably not being honoured.

    `backtest_fn(shift_days) -> Sharpe`.
    """
    base = backtest_fn(0)
    shifted = backtest_fn(shift_days)
    if np.isnan(base) or np.isnan(shifted) or base == 0:
        return {"suspicious": False, "note": "insufficient data"}

    degradation = (base - shifted) / abs(base)
    suspicious = degradation < -tolerance   # got notably BETTER when delayed
    return {
        "base_sharpe": float(base),
        "shifted_sharpe": float(shifted),
        "degradation_pct": float(degradation * 100),
        "suspicious": bool(suspicious),
        "note": (
            "Performance improved when signals were delayed - check timing logic"
            if suspicious else "Behaves as expected under delay"
        ),
    }


def shuffled_future_test(
    returns: pd.Series, signal: pd.Series, n_perm: int = 200, seed: int = 42
) -> dict[str, float]:
    """After shuffling the signal, any edge should vanish."""
    df = pd.concat([returns.rename("r"), signal.rename("s")], axis=1).dropna()
    if len(df) < 30:
        return {"p_value": np.nan}

    observed = float(df["r"].corr(df["s"]))
    rng = np.random.default_rng(seed)
    perm = np.array([
        float(pd.Series(rng.permutation(df["s"].values)).corr(pd.Series(df["r"].values)))
        for _ in range(n_perm)
    ])
    return {
        "observed_corr": observed,
        "permuted_mean": float(np.nanmean(perm)),
        "p_value": float((np.abs(perm) >= abs(observed)).mean()),
        "leak_suspected": bool(abs(float(np.nanmean(perm))) > 0.05),
    }


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    sharpe_test: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)
    permutation: dict = field(default_factory=dict)
    deflated_sharpe: dict = field(default_factory=dict)
    haircut: dict = field(default_factory=dict)
    min_trl: dict = field(default_factory=dict)
    pbo: dict = field(default_factory=dict)
    reality_check: dict = field(default_factory=dict)
    spa: dict = field(default_factory=dict)
    leaks: dict = field(default_factory=dict)
    verdict: str = ""
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_strategy(
    returns: pd.Series,
    n_trials: int = 1,
    benchmark_returns: pd.Series | None = None,
    candidate_returns: pd.DataFrame | None = None,
    periods_per_year: int = TRADING_DAYS,
    seed: int = 42,
) -> ValidationReport:
    """Run the full validation battery and produce a plain-language verdict."""
    r = returns.dropna()
    report = ValidationReport()
    if len(r) < 30:
        report.verdict = "Insufficient data for statistical validation"
        return report

    sh = sharpe_test(r, 0.0, periods_per_year)
    report.sharpe_test = sh.to_dict()

    report.bootstrap = bootstrap_metric(r, n_boot=1000, seed=seed)
    report.permutation = permutation_test(r, n_perm=500, seed=seed)

    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    report.deflated_sharpe = deflated_sharpe_ratio(
        sh.sharpe, n_trials, len(r), skew, kurt, periods_per_year=periods_per_year
    )
    report.haircut = haircut_sharpe(sh.sharpe, n_trials, len(r), periods_per_year)
    report.min_trl = min_track_record_length(
        sh.sharpe, 0.0, skew, kurt, periods_per_year=periods_per_year
    )

    if candidate_returns is not None and candidate_returns.shape[1] >= 2:
        report.pbo = probability_backtest_overfitting(candidate_returns, seed=seed)
        if benchmark_returns is not None:
            report.reality_check = reality_check(candidate_returns, benchmark_returns, seed=seed)
            report.spa = spa_test(candidate_returns, benchmark_returns, seed=seed)

    report.flags, report.verdict = _verdict(report, sh, n_trials, len(r), periods_per_year)
    return report


def _verdict(
    report: ValidationReport, sh: SharpeTest, n_trials: int, n_obs: int, ppy: int
) -> tuple[list[str], str]:
    flags: list[str] = []

    if not sh.significant:
        flags.append(f"Sharpe not significant at 5% (p={sh.p_value:.3f})")
    if report.bootstrap.get("ci_lower") is not None and report.bootstrap["ci_lower"] < 0:
        flags.append("Bootstrap 95% CI for Sharpe includes zero")

    dsr = report.deflated_sharpe.get("dsr")
    if dsr is not None and not np.isnan(dsr) and dsr < 0.95:
        flags.append(f"Deflated Sharpe {dsr:.2f} < 0.95 after {n_trials} trials")

    hc = report.haircut.get("haircut_sharpe")
    if hc is not None and not np.isnan(hc) and hc <= 0:
        flags.append("Sharpe does not survive multiple-testing haircut")

    trl = report.min_trl.get("min_track_record_years")
    if trl is not None and np.isfinite(trl) and trl > n_obs / ppy:
        flags.append(
            f"Track record too short: needs ~{trl:.1f}y, have {n_obs / ppy:.1f}y"
        )

    pbo = report.pbo.get("pbo")
    if pbo is not None and not np.isnan(pbo) and pbo > 0.3:
        flags.append(f"High probability of backtest overfitting (PBO={pbo:.2f})")

    spa_p = report.spa.get("p_value")
    if spa_p is not None and not np.isnan(spa_p) and spa_p > 0.05:
        flags.append(f"Fails SPA data-snooping test (p={spa_p:.3f})")

    if not flags:
        verdict = "PASS - the edge survives every statistical check applied."
    elif len(flags) <= 2:
        verdict = "CAUTION - mostly holds up, but some checks failed. Treat with scepticism."
    else:
        verdict = "FAIL - multiple checks failed; most likely overfitting or noise."
    return flags, verdict
