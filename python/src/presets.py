"""Literature presets - reference designs used to sanity-check the engine
against published Indian-equity results *before* any optimization.

Each preset isolates a documented effect so we can confirm the pipeline
reproduces the expected qualitative behaviour (e.g. momentum = high return with
deep drawdowns; low-vol = lower return, much shallower drawdown).
"""

from __future__ import annotations

from .schema import (
    AllocationConfig,
    FactorConfig,
    QualityWeights,
    RegimeConfig,
    SleeveAConfig,
    SleeveBConfig,
    SleeveCConfig,
    StrategyConfig,
    UniverseConfig,
    WeightScheme,
)


def _factor_only(**weights: float) -> FactorConfig:
    base = {"weight_quality": 0.0, "weight_momentum": 0.0, "weight_value": 0.0, "weight_low_vol": 0.0}
    base.update(weights)
    return FactorConfig(**base)


def _price_universe(**kwargs) -> UniverseConfig:
    """Don't drop names for missing filings when the signal is price-only."""
    base = dict(min_financial_quarters=0)
    base.update(kwargs)
    return UniverseConfig(**base)


def quality_only() -> StrategyConfig:
    """Jacob, Pradeep & Varma (IIMA): quality is the strongest Indian factor;
    low turnover, shallow drawdowns, viable long-only."""
    return StrategyConfig(
        name="quality_only",
        description="Long-only Quality (QMJ-style). Expect: solid alpha, low turnover, shallow DD.",
        universe=_price_universe(min_financial_quarters=4),
        sleeve_a=SleeveAConfig(
            top_n=50,
            factors=_factor_only(weight_quality=1.0),
            regime=RegimeConfig(enabled=False),
        ),
    )


def momentum_only() -> StrategyConfig:
    """Agarwalla, Jacob & Varma: momentum is the highest-returning Indian
    factor but crash-prone (~-70% DD in 2008)."""
    return StrategyConfig(
        name="momentum_only",
        description="Long-only Momentum (12-1, 6-1, vol-adj). Expect: high return, deep drawdowns.",
        universe=_price_universe(),
        sleeve_a=SleeveAConfig(
            top_n=30,
            weight_scheme=WeightScheme.EQUAL,
            factors=_factor_only(weight_momentum=1.0),
            regime=RegimeConfig(enabled=False),
        ),
    )


def low_vol_only() -> StrategyConfig:
    """Volatility effect in India: best risk-adjusted profile, fast recovery."""
    return StrategyConfig(
        name="low_vol_only",
        description="Long-only Low Volatility. Expect: lower return, much shallower DD, better Sharpe.",
        universe=_price_universe(),
        sleeve_a=SleeveAConfig(
            top_n=50,
            factors=_factor_only(weight_low_vol=1.0),
            regime=RegimeConfig(enabled=False),
        ),
    )


def value_quality() -> StrategyConfig:
    """Value x Quality: cheap AND good, to avoid value traps."""
    return StrategyConfig(
        name="value_quality",
        description="Value combined with Quality gate (cheap != undervalued).",
        universe=_price_universe(min_financial_quarters=4),
        sleeve_a=SleeveAConfig(
            top_n=50,
            factors=FactorConfig(
                weight_quality=0.5,
                weight_value=0.5,
                weight_momentum=0.0,
                weight_low_vol=0.0,
                value_quality_gate=True,
            ),
            regime=RegimeConfig(enabled=False),
        ),
    )


def conservative_formula() -> StrategyConfig:
    """'The Conservative Formula: Evidence from India' -
    low volatility + payout + momentum, concentrated."""
    return StrategyConfig(
        name="conservative_formula",
        description="Low-vol + payout + momentum (Conservative Formula, India).",
        universe=_price_universe(min_financial_quarters=4),
        sleeve_a=SleeveAConfig(
            top_n=50,
            weight_scheme=WeightScheme.EQUAL,
            factors=FactorConfig(
                weight_low_vol=0.5,
                weight_momentum=0.3,
                weight_quality=0.2,   # payout enters via the quality block
                weight_value=0.0,
                quality={"roic": 0.0, "gross_profitability": 0.0, "cfo_to_pat": 0.0,
                         "fcf_to_assets": 0.0, "neg_debt_to_assets": 0.0, "payout": 1.0},
            ),
            regime=RegimeConfig(enabled=False),
        ),
    )


def full_composite() -> StrategyConfig:
    """The plan's headline design: Score = 0.35Q + 0.30M + 0.20V + 0.15LV,
    sector-neutral, rank-buffered, with the trend overlay."""
    return StrategyConfig(
        name="full_composite",
        description="Quality-led 4-factor composite, sector-neutral, trend overlay. Sleeve A only.",
        universe=_price_universe(min_financial_quarters=4),
        sleeve_a=SleeveAConfig(top_n=50, factors=FactorConfig(), regime=RegimeConfig(enabled=True)),
    )


def price_only_core() -> StrategyConfig:
    """Roadmap step 2 - runs on prices alone (no fundamentals)."""
    return StrategyConfig(
        name="price_only_core",
        description="Momentum + Low-Vol + trend overlay. Requires no fundamentals data.",
        # This strategy reads no financial statements, so requiring reported
        # history would exclude names for no reason.
        universe=_price_universe(),
        sleeve_a=SleeveAConfig(
            top_n=40,
            factors=_factor_only(weight_momentum=0.6, weight_low_vol=0.4),
            regime=RegimeConfig(enabled=True),
        ),
    )


def value_only() -> StrategyConfig:
    """IIMA HML-style cheapness: earnings yield (and PE when printed)."""
    return StrategyConfig(
        name="value_only",
        description="Long-only Value (earnings yield / cheapness). Cyclical; traps without quality.",
        universe=_price_universe(min_financial_quarters=4),
        sleeve_a=SleeveAConfig(
            top_n=50,
            factors=_factor_only(weight_value=1.0),
            regime=RegimeConfig(enabled=False),
        ),
    )


def momentum_trend() -> StrategyConfig:
    """Skip-month momentum plus Moskowitz-style 200-day trend overlay."""
    return StrategyConfig(
        name="momentum_trend",
        description="Momentum 12-1/6-1 with absolute-momentum / 200d MA overlay.",
        universe=_price_universe(),
        sleeve_a=SleeveAConfig(
            top_n=30,
            weight_scheme=WeightScheme.EQUAL,
            factors=_factor_only(weight_momentum=1.0),
            regime=RegimeConfig(enabled=True),
        ),
    )


def pead_only() -> StrategyConfig:
    """Post-earnings announcement drift from SUE (Bernard & Thomas / Indian PEAD)."""
    return StrategyConfig(
        name="pead_only",
        description="Long-only PEAD: high standardized unexpected earnings, 60-day drift.",
        universe=_price_universe(min_financial_quarters=4),
        allocation=AllocationConfig(sleeve_a=0.0, sleeve_b=0.0, sleeve_c=1.0),
        sleeve_a=SleeveAConfig(enabled=False),
        sleeve_c=SleeveCConfig(enabled=True),
    )


def residual_only() -> StrategyConfig:
    """Long-only residual reversion (Sleeve B)."""
    return StrategyConfig(
        name="residual_only",
        description="Long-only residual reversion vs the market residual.",
        universe=_price_universe(),
        allocation=AllocationConfig(sleeve_a=0.0, sleeve_b=1.0, sleeve_c=0.0),
        sleeve_a=SleeveAConfig(enabled=False),
        sleeve_b=SleeveBConfig(enabled=True),
    )


def all_sleeves() -> StrategyConfig:
    """Full multi-sleeve system: factor core + residual reversion + PEAD."""
    return StrategyConfig(
        name="all_sleeves",
        description="Sleeve A (factors) 70% + Sleeve B (residual reversion) 15% + Sleeve C (PEAD) 15%.",
        allocation=AllocationConfig(sleeve_a=0.70, sleeve_b=0.15, sleeve_c=0.15),
        sleeve_a=SleeveAConfig(top_n=50, factors=FactorConfig()),
        sleeve_b=SleeveBConfig(enabled=True),
        sleeve_c=SleeveCConfig(enabled=True),
    )


def _quality_dimension(**legs: float) -> FactorConfig:
    """One IIMA QMJ dimension as the entire rank (long-only)."""
    zeros = {
        "roic": 0.0, "gross_profitability": 0.0, "cfo_to_pat": 0.0,
        "fcf_to_assets": 0.0, "neg_debt_to_assets": 0.0, "payout": 0.0,
        "net_margin": 0.0, "pretax_margin": 0.0, "margin_growth": 0.0,
    }
    zeros.update(legs)
    return FactorConfig(
        weight_quality=1.0, weight_momentum=0.0, weight_value=0.0, weight_low_vol=0.0,
        quality=QualityWeights(**zeros),
        quality_min_components=1,
        value_quality_gate=False,
    )


def profitability_only() -> StrategyConfig:
    """IIMA QMJ profitability: GPOA/ROE/margins (tunnelling — diversion of revenue)."""
    return StrategyConfig(
        name="profitability_only",
        description="Long-only IIMA profitability (net/pretax margin, ROIC, gross profitability).",
        universe=_price_universe(min_financial_quarters=4),
        sleeve_a=SleeveAConfig(
            top_n=50,
            factors=_quality_dimension(
                roic=0.35, gross_profitability=0.15, net_margin=0.35, pretax_margin=0.15,
            ),
            regime=RegimeConfig(enabled=False),
        ),
    )


def growth_only() -> StrategyConfig:
    """IIMA QMJ growth: one-year change in profitability (paper uses five years)."""
    return StrategyConfig(
        name="growth_only",
        description="Long-only IIMA growth (YoY change in P&L profitability legs).",
        universe=_price_universe(min_financial_quarters=4),
        sleeve_a=SleeveAConfig(
            top_n=50,
            factors=_quality_dimension(margin_growth=1.0),
            regime=RegimeConfig(enabled=False),
        ),
    )


def payout_only() -> StrategyConfig:
    """IIMA QMJ payout: cash returned to shareholders (tunnelling — diversion of profit)."""
    return StrategyConfig(
        name="payout_only",
        description="Long-only IIMA payout. Empty until filings print a payout ratio.",
        universe=_price_universe(min_financial_quarters=4),
        sleeve_a=SleeveAConfig(
            top_n=50,
            factors=_quality_dimension(payout=1.0),
            regime=RegimeConfig(enabled=False),
        ),
    )


def profitability_payout() -> StrategyConfig:
    """The two IIMA quality legs that actually drove Indian QMJ alpha."""
    return StrategyConfig(
        name="profitability_payout",
        description="Profitability + payout (IIMA tunnelling hypothesis, long-only).",
        universe=_price_universe(min_financial_quarters=4),
        sleeve_a=SleeveAConfig(
            top_n=50,
            factors=_quality_dimension(
                roic=0.25, gross_profitability=0.15, net_margin=0.25, pretax_margin=0.15,
                payout=0.20,
            ),
            regime=RegimeConfig(enabled=False),
        ),
    )


PRESETS: dict[str, callable] = {
    "quality_only": quality_only,
    "momentum_only": momentum_only,
    "low_vol_only": low_vol_only,
    "value_only": value_only,
    "value_quality": value_quality,
    "conservative_formula": conservative_formula,
    "full_composite": full_composite,
    "price_only_core": price_only_core,
    "momentum_trend": momentum_trend,
    "pead_only": pead_only,
    "residual_only": residual_only,
    "profitability_only": profitability_only,
    "growth_only": growth_only,
    "payout_only": payout_only,
    "profitability_payout": profitability_payout,
    "all_sleeves": all_sleeves,
}


def get_preset(name: str) -> StrategyConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return PRESETS[name]()


def list_presets() -> list[str]:
    return sorted(PRESETS)
