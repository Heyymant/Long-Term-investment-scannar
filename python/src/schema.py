"""Strategy design schema (`strategy_config.json`).

This is the single contract describing *what strategy to run*. It is validated
with pydantic here and exported to JSON Schema so the Rust dashboard validates
the identical structure before launching a run.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class RebalanceFrequency(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"


class WeightScheme(str, Enum):
    EQUAL = "equal"
    INVERSE_VOL = "inverse_vol"
    SCORE_TILTED = "score_tilted"


class StatArbMode(str, Enum):
    RESIDUAL_REVERSION = "residual_reversion"  # primary
    PAIRS = "pairs"                            # variant


# --------------------------------------------------------------------------- #
# Sleeve A - factor portfolio
# --------------------------------------------------------------------------- #
class QualityWeights(_Base):
    """Q = 0.25 Z(ROIC) + 0.20 Z(GrossProf) + 0.15 Z(CFO/PAT)
         + 0.15 Z(FCF/Assets) + 0.15 Z(-Debt/Assets) + 0.10 Z(Payout)"""

    roic: float = 0.25
    gross_profitability: float = 0.20
    cfo_to_pat: float = 0.15
    fcf_to_assets: float = 0.15
    neg_debt_to_assets: float = 0.15
    payout: float = 0.10
    # Used when balance-sheet legs are missing (NSE quarterly P&L).
    net_margin: float = 0.0
    pretax_margin: float = 0.0
    # IIMA QMJ growth dimension: YoY change in profitability (not 5-year).
    margin_growth: float = 0.0


class ValueWeights(_Base):
    """V = 0.25 Z(FCF Yield) + 0.25 Z(Earnings Yield)
         + 0.20 Z(-EV/EBITDA) + 0.15 Z(-P/B) + 0.15 Z(-P/E)"""

    fcf_yield: float = 0.25
    earnings_yield: float = 0.25
    neg_ev_ebitda: float = 0.20
    neg_pb: float = 0.15
    neg_pe: float = 0.15


class MomentumWeights(_Base):
    """M = 0.35 Z(Mom12) + 0.35 Z(Mom6) + 0.30 Z(Mom/Vol)"""

    mom_12_1: float = 0.35
    mom_6_1: float = 0.35
    vol_adjusted: float = 0.30


class FactorConfig(_Base):
    # Composite: Score = 0.35 Q + 0.30 M + 0.20 V + 0.15 LV
    weight_quality: float = 0.35
    weight_momentum: float = 0.30
    weight_value: float = 0.20
    weight_low_vol: float = 0.15

    quality: QualityWeights = Field(default_factory=QualityWeights)
    value: ValueWeights = Field(default_factory=ValueWeights)
    momentum: MomentumWeights = Field(default_factory=MomentumWeights)

    low_vol_lookback_days: int = Field(252, ge=20, le=1260)
    winsorize_pct: float = Field(0.01, ge=0.0, le=0.2)

    # Value x Quality interaction: penalize cheap-but-junk names.
    value_quality_gate: bool = True
    value_quality_gate_strength: float = Field(0.5, ge=0.0, le=1.0)

    # Optional IC/ICIR-driven dynamic weighting (research toggle).
    dynamic_weighting: bool = False
    icir_lookback_months: int = Field(36, ge=12, le=240)
    # Isolated IIMA quality legs (profitability, payout, growth) need a single
    # printed metric; the full QMJ composite still requires two.
    quality_min_components: int = Field(2, ge=1, le=8)

    def composite_weights(self) -> dict[str, float]:
        return {
            "quality": self.weight_quality,
            "momentum": self.weight_momentum,
            "value": self.weight_value,
            "low_vol": self.weight_low_vol,
        }

    @model_validator(mode="after")
    def _check_weights(self) -> FactorConfig:
        total = sum(self.composite_weights().values())
        if total <= 0:
            raise ValueError("composite factor weights must sum to > 0")
        return self


class RegimeConfig(_Base):
    """Trend/absolute-momentum overlay with hysteresis (avoids MA whipsaw)."""

    enabled: bool = True
    ma_days: int = Field(200, ge=20, le=400)
    # Hysteresis: exit below (1-band), re-enter above (1+band) of the MA.
    hysteresis_band: float = Field(0.02, ge=0.0, le=0.2)
    graded: bool = True          # scale exposure by distance from MA
    min_exposure: float = Field(0.0, ge=0.0, le=1.0)
    max_exposure: float = Field(1.0, ge=0.0, le=1.0)


class VolScalingConfig(_Base):
    """Barroso & Santa-Clara style volatility scaling."""

    enabled: bool = False
    target_vol: float = Field(0.15, gt=0.0, le=1.0)
    lookback_days: int = Field(126, ge=20, le=756)
    max_leverage: float = Field(1.0, ge=0.1, le=1.0)  # long-only, no leverage


class SleeveAConfig(_Base):
    enabled: bool = True
    top_n: int = Field(50, ge=5, le=500)
    # Rank buffer: sell only when rank falls past buffer_multiple * top_n.
    rank_buffer_multiple: float = Field(2.0, ge=1.0, le=5.0)
    weight_scheme: WeightScheme = WeightScheme.INVERSE_VOL
    max_weight_per_stock: float = Field(0.05, gt=0.0, le=1.0)
    max_weight_per_sector: float = Field(0.25, gt=0.0, le=1.0)
    sector_neutralize: bool = True
    rebalance: RebalanceFrequency = RebalanceFrequency.QUARTERLY
    factors: FactorConfig = Field(default_factory=FactorConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    vol_scaling: VolScalingConfig = Field(default_factory=VolScalingConfig)


# --------------------------------------------------------------------------- #
# Sleeve B - statistical mean reversion
# --------------------------------------------------------------------------- #
class SleeveBConfig(_Base):
    enabled: bool = False
    mode: StatArbMode = StatArbMode.RESIDUAL_REVERSION

    # Residual reversion
    residual_lookback_days: int = Field(60, ge=20, le=504)
    reversion_horizon_days: int = Field(5, ge=1, le=60)
    n_positions: int = Field(20, ge=1, le=200)
    entry_z: float = Field(-1.5, le=0.0)
    exit_z: float = Field(-0.25, le=0.0)

    # Pairs variant
    pair_formation_days: int = Field(252, ge=60, le=1260)
    coint_pvalue: float = Field(0.05, gt=0.0, le=0.5)
    fdr_alpha: float = Field(0.05, gt=0.0, le=0.5)
    use_kalman: bool = False
    min_half_life_days: float = Field(1.0, gt=0.0)
    max_half_life_days: float = Field(60.0, gt=0.0)
    pair_entry_z: float = Field(2.0, gt=0.0)
    pair_exit_z: float = Field(0.5, ge=0.0)
    pair_stop_z: float = Field(3.0, gt=0.0)

    max_weight_per_stock: float = Field(0.05, gt=0.0, le=1.0)
    execution_lag_days: int = Field(1, ge=1, le=5)

    @model_validator(mode="after")
    def _check(self) -> SleeveBConfig:
        if self.min_half_life_days >= self.max_half_life_days:
            raise ValueError("min_half_life_days must be < max_half_life_days")
        if self.entry_z > self.exit_z:
            raise ValueError("entry_z must be <= exit_z (both negative, entry deeper)")
        return self


# --------------------------------------------------------------------------- #
# Sleeve C - event-driven earnings (PEAD)
# --------------------------------------------------------------------------- #
class SleeveCConfig(_Base):
    enabled: bool = False
    sue_lookback_quarters: int = Field(8, ge=4, le=20)
    min_sue: float = Field(1.0)               # enter above this SUE
    n_positions: int = Field(20, ge=1, le=200)
    drift_window_days: int = Field(60, ge=10, le=120)
    entry_lag_days: int = Field(1, ge=1, le=10)
    require_positive_reaction: bool = True    # earnings-day abnormal return > 0
    min_adv_inr: float = Field(50_000_000.0, ge=0.0)
    exclude_top_mcap_pct: float = Field(0.0, ge=0.0, le=1.0)  # drift weak in large caps
    max_weight_per_stock: float = Field(0.03, gt=0.0, le=1.0)
    stop_loss_pct: float | None = Field(-0.15, le=0.0)


# --------------------------------------------------------------------------- #
# Portfolio-level risk
# --------------------------------------------------------------------------- #
class RiskConfig(_Base):
    vol_target_enabled: bool = False
    target_vol: float = Field(0.15, gt=0.0, le=1.0)
    vol_lookback_days: int = Field(126, ge=20, le=756)

    max_gross_exposure: float = Field(1.0, gt=0.0, le=1.0)   # long-only, unlevered
    min_cash: float = Field(0.0, ge=0.0, lt=1.0)

    # Staged de-risking on strategy drawdown.
    drawdown_derisk_enabled: bool = True
    drawdown_thresholds: list[float] = Field(default_factory=lambda: [-0.15, -0.25, -0.35])
    drawdown_exposures: list[float] = Field(default_factory=lambda: [0.75, 0.50, 0.25])
    drawdown_recovery_buffer: float = Field(0.05, ge=0.0, le=0.5)
    # Drawdown is measured against a ROLLING peak, not the all-time high.
    # Using the all-time peak causes a permanent lockout: once exposure is cut,
    # a partially-invested portfolio can never climb back to its old high, so
    # the de-risk never releases and the strategy is dead from then on.
    drawdown_lookback_days: int = Field(504, ge=63, le=2520)

    max_pct_of_adv: float = Field(0.10, gt=0.0, le=1.0)
    max_weight_per_stock_total: float = Field(0.07, gt=0.0, le=1.0)
    max_weight_per_sector_total: float = Field(0.30, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> RiskConfig:
        if len(self.drawdown_thresholds) != len(self.drawdown_exposures):
            raise ValueError("drawdown_thresholds and drawdown_exposures must be same length")
        if list(self.drawdown_thresholds) != sorted(self.drawdown_thresholds, reverse=True):
            raise ValueError("drawdown_thresholds must be descending, e.g. [-0.15, -0.25]")
        return self


# --------------------------------------------------------------------------- #
# Universe / backtest window / allocation
# --------------------------------------------------------------------------- #
class UniverseConfig(_Base):
    index: Literal["NSE500", "NIFTY200", "NIFTY50", "custom"] = "NSE500"
    custom_list_file: str | None = None
    min_adv_inr: float = Field(50_000_000.0, ge=0.0)
    min_price_inr: float = Field(20.0, ge=0.0)
    min_listing_days: int = Field(365, ge=0)
    min_financial_quarters: int = Field(8, ge=0)
    point_in_time: bool = True


class BacktestWindow(_Base):
    start: str = "2010-01-01"
    end: str | None = None
    initial_capital: float = Field(10_000_000.0, gt=0.0)
    benchmark: str = "NIFTY50_TRI"


class AllocationConfig(_Base):
    sleeve_a: float = Field(1.0, ge=0.0, le=1.0)
    sleeve_b: float = Field(0.0, ge=0.0, le=1.0)
    sleeve_c: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> AllocationConfig:
        total = self.sleeve_a + self.sleeve_b + self.sleeve_c
        if total <= 0:
            raise ValueError("at least one sleeve must have a positive allocation")
        if total > 1.0 + 1e-9:
            raise ValueError(f"sleeve allocations must sum to <= 1.0 (got {total:.4f})")
        return self


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
class StrategyConfig(_Base):
    """The complete strategy design replayed by the backtester."""

    schema_version: str = SCHEMA_VERSION
    name: str = "untitled"
    description: str = ""
    seed: int = 42

    window: BacktestWindow = Field(default_factory=BacktestWindow)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)

    sleeve_a: SleeveAConfig = Field(default_factory=SleeveAConfig)
    sleeve_b: SleeveBConfig = Field(default_factory=SleeveBConfig)
    sleeve_c: SleeveCConfig = Field(default_factory=SleeveCConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    apply_costs: bool = True
    apply_taxes: bool = True

    # -- helpers ---------------------------------------------------------------
    def config_hash(self) -> str:
        """Stable hash of the design (excludes name/description)."""
        payload = self.model_dump(mode="json", exclude={"name", "description"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> StrategyConfig:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def json_schema_dict(cls) -> dict[str, Any]:
        return cls.model_json_schema()

    def active_sleeves(self) -> list[str]:
        out = []
        if self.sleeve_a.enabled and self.allocation.sleeve_a > 0:
            out.append("A")
        if self.sleeve_b.enabled and self.allocation.sleeve_b > 0:
            out.append("B")
        if self.sleeve_c.enabled and self.allocation.sleeve_c > 0:
            out.append("C")
        return out


def export_json_schema(path: str | Path) -> Path:
    """Write the shared JSON Schema consumed by the Rust dashboard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = StrategyConfig.json_schema_dict()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "StrategyConfig"
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return path
