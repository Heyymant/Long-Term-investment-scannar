"""IIMA Indian Fama-French-Momentum factor library.

Agarwalla, Jacob & Varma publish survivorship-adjusted Indian factor returns
(MKT, SMB, HML, WML and the risk-free rate). These are the benchmark factors
for the true-alpha attribution regression.

The library does not publish a QMJ series, so when quality attribution is
wanted we construct QMJ in-repo from our own fundamentals following the
Asness-Frazzini-Pedersen recipe as adapted by the IIMA quality paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig, get_logger

log = get_logger(__name__)

FACTOR_COLUMNS = ["MKT", "SMB", "HML", "WML", "RF"]
# Header spellings vary between the library's CSV releases.
COLUMN_ALIASES = {
    "mkt": "MKT", "mkt-rf": "MKT", "mktrf": "MKT", "market": "MKT", "rmrf": "MKT",
    "smb": "SMB", "hml": "HML",
    "wml": "WML", "mom": "WML", "umd": "WML",
    "rf": "RF", "riskfree": "RF", "risk_free": "RF",
    "qmj": "QMJ",
    "date": "date", "month": "date", "ym": "date",
}


def load_iima_factors(cfg: AppConfig, as_decimal: bool = True) -> pd.DataFrame:
    """Load the cached IIMA factor file (monthly or daily).

    Returns a DataFrame indexed by date with MKT/SMB/HML/WML/RF columns
    expressed as decimals (0.01 == 1%).
    """
    local = cfg.get("data_sources.iima_factors.local_file", "iima_factors.csv")
    path = cfg.paths.data_dir / local
    if not path.exists():
        log.warning(
            "IIMA factor file not found at %s. Download from %s and save it there; "
            "attribution will fall back to a market-only regression.",
            path, cfg.get("data_sources.iima_factors.url"),
        )
        return pd.DataFrame(columns=FACTOR_COLUMNS)

    raw = pd.read_csv(path)
    raw.columns = [COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().upper())
                   for c in raw.columns]

    date_col = "date" if "date" in raw.columns else raw.columns[0]
    raw[date_col] = _parse_dates(raw[date_col])
    df = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"

    keep = [c for c in (*FACTOR_COLUMNS, "QMJ") if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")

    # The library publishes percentages; convert to decimals unless already tiny.
    if as_decimal and not df.empty:
        scale = df[[c for c in keep if c != "RF"]].abs().median().median()
        if scale is not None and scale > 0.05:
            df = df / 100.0
            log.info("IIMA factors appeared to be in percent; scaled to decimals")

    log.info("Loaded IIMA factors: %d rows, columns %s", len(df), list(df.columns))
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    """Handle YYYYMM, YYYYMMDD and normal date strings."""
    txt = s.astype(str).str.strip()
    if txt.str.fullmatch(r"\d{6}").all():
        return pd.to_datetime(txt, format="%Y%m", errors="coerce") + pd.offsets.MonthEnd(0)
    if txt.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(txt, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(txt, errors="coerce", dayfirst=True)


def to_monthly(factors: pd.DataFrame) -> pd.DataFrame:
    """Compound daily factor returns to month-end."""
    if factors.empty:
        return factors
    inferred = pd.infer_freq(factors.index[:20]) if len(factors) > 20 else None
    if inferred and inferred.startswith("M"):
        return factors
    return (1 + factors).resample("ME").prod() - 1


def align_factors(
    portfolio_returns: pd.Series, factors: pd.DataFrame, frequency: str = "M"
) -> tuple[pd.Series, pd.DataFrame]:
    """Align portfolio returns and factors onto a common frequency/index."""
    if factors.empty:
        return portfolio_returns, factors

    if frequency.upper().startswith("M"):
        port = (1 + portfolio_returns).resample("ME").prod() - 1
        fac = to_monthly(factors)
    else:
        port, fac = portfolio_returns, factors

    idx = port.index.intersection(fac.index)
    return port.reindex(idx).dropna(), fac.reindex(idx).dropna()


# --------------------------------------------------------------------------- #
# In-repo QMJ construction
# --------------------------------------------------------------------------- #
def build_qmj_factor(
    returns: pd.DataFrame,
    quality_scores: pd.DataFrame,
    market_caps: pd.DataFrame | None = None,
    rebalance: str = "ME",
    quantile: float = 0.3,
) -> pd.Series:
    """Construct a Quality-Minus-Junk return series from our own data.

    Standard 2x3-style construction, simplified to a single quality sort:
    long the top `quantile` of quality, short the bottom, value-weighted when
    market caps are supplied (equal-weighted otherwise), rebalanced monthly.

    Note this is a *factor-mimicking* series for attribution. It is not a
    tradeable strategy (we are long-only) - it exists so the regression can
    strip out quality exposure when measuring true alpha.
    """
    if returns.empty or quality_scores.empty:
        return pd.Series(dtype=float, name="QMJ")

    rebal_dates = returns.resample(rebalance).last().index
    out = pd.Series(0.0, index=returns.index, name="QMJ")

    for i, rd in enumerate(rebal_dates[:-1]):
        formation = quality_scores.loc[quality_scores.index <= rd]
        if formation.empty:
            continue
        scores = formation.iloc[-1].dropna()
        if len(scores) < 10:
            continue

        hi_cut = scores.quantile(1 - quantile)
        lo_cut = scores.quantile(quantile)
        longs = scores[scores >= hi_cut].index
        shorts = scores[scores <= lo_cut].index
        if len(longs) == 0 or len(shorts) == 0:
            continue

        period = returns.loc[(returns.index > rd) & (returns.index <= rebal_dates[i + 1])]
        if period.empty:
            continue

        lw = _weights(longs, market_caps, rd)
        sw = _weights(shorts, market_caps, rd)
        long_ret = period[longs.intersection(period.columns)].mul(lw, axis=1).sum(axis=1)
        short_ret = period[shorts.intersection(period.columns)].mul(sw, axis=1).sum(axis=1)
        out.loc[period.index] = (long_ret - short_ret).values

    return out


def _weights(
    members: pd.Index, market_caps: pd.DataFrame | None, dt: pd.Timestamp
) -> pd.Series:
    if market_caps is None or market_caps.empty:
        return pd.Series(1.0 / len(members), index=members)
    hist = market_caps.loc[market_caps.index <= dt]
    if hist.empty:
        return pd.Series(1.0 / len(members), index=members)
    caps = hist.iloc[-1].reindex(members).fillna(0.0)
    total = caps.sum()
    if total <= 0:
        return pd.Series(1.0 / len(members), index=members)
    return caps / total


def ensure_qmj(
    factors: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    quality_scores: pd.DataFrame | None = None,
    market_caps: pd.DataFrame | None = None,
    enabled: bool = True,
) -> pd.DataFrame:
    """Return factors with a QMJ column, building it locally if required."""
    if "QMJ" in factors.columns or not enabled:
        return factors
    if returns is None or quality_scores is None:
        log.info("QMJ unavailable and no data to construct it; attribution will use 4 factors")
        return factors

    qmj = build_qmj_factor(returns, quality_scores, market_caps)
    if qmj.abs().sum() == 0:
        return factors

    out = factors.copy()
    qmj_aligned = to_monthly(qmj.to_frame("QMJ")) if _is_monthly(factors) else qmj.to_frame("QMJ")
    out = out.join(qmj_aligned, how="left")
    out["QMJ"] = out["QMJ"].fillna(0.0)
    log.info("Constructed QMJ factor in-repo (%d observations)", int((out['QMJ'] != 0).sum()))
    return out


def _is_monthly(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    return bool(np.median(np.diff(df.index.values).astype("timedelta64[D]").astype(int)) > 20)
