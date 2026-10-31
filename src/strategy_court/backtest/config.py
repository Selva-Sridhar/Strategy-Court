"""Execution and cost assumptions. Recorded in every trace so a verdict is reproducible."""

from enum import StrEnum

from pydantic import Field

from strategy_court.schemas._base import FrozenModel


class ExecutionTiming(StrEnum):
    NEXT_OPEN = "next_open"  # trade at open t+1 (falls back to close t+1 where open is missing)
    NEXT_CLOSE = "next_close"  # trade at close t+1


class CostModel(FrozenModel):
    """Transaction-cost model, charged on the currency value traded.

    * Linear: commission + half the bid-ask spread, per unit traded.
    * Market impact (square-root law): cost_i = Y * sigma_i * (|trade_i| / ADV_i)^delta * |trade_i|
      with sigma_i the daily volatility and ADV_i the average daily dollar volume, both estimated
      from data up to the previous close (no look-ahead in the cost model).
      Tóth et al. (2011), Phys. Rev. X 1, 021006; Almgren, Thum, Hauptmann & Li (2005), Risk 18(7).
    * Borrow: annual fee on the value of short positions, accrued per bar.
    """

    commission_bps: float = Field(default=1.0, ge=0)
    half_spread_bps: float = Field(default=2.0, ge=0)
    impact_coefficient: float = Field(default=0.0, ge=0, description="Y; 0 disables impact.")
    impact_exponent: float = Field(default=0.5, gt=0, le=1, description="delta; 0.5 = sqrt law.")
    impact_window: int = Field(default=20, ge=2, description="Bars for sigma / ADV estimates.")
    impact_fallback_bps: float = Field(
        default=50.0, ge=0, description="Charged when sigma/ADV are not yet estimable."
    )
    borrow_bps_annual: float = Field(default=0.0, ge=0)

    @property
    def linear_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps

    @classmethod
    def zero(cls) -> "CostModel":
        return cls(commission_bps=0, half_spread_bps=0, impact_fallback_bps=0)


class BacktestConfig(FrozenModel):
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN
    costs: CostModel = CostModel()
    risk_free_rate: float = Field(default=0.0, ge=-0.1, le=1.0, description="Annual, on cash.")
    delisting_return: float = Field(
        default=0.0,
        ge=-1,
        le=1,
        description="Return applied when a held asset delists (Shumway 1997 suggests ~-0.30 "
        "for performance-related delistings).",
    )
    max_gross_leverage: float | None = Field(default=None, gt=0)
    periods_per_year: int = Field(default=252, ge=1)

    @property
    def rf_per_period(self) -> float:
        return float((1 + self.risk_free_rate) ** (1 / self.periods_per_year) - 1)
