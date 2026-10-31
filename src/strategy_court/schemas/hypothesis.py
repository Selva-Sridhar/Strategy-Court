from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from strategy_court.schemas._base import FrozenModel


class SourceType(StrEnum):
    CODE = "code"
    TEXT = "text"
    SCREENSHOT = "screenshot"


class RebalanceFrequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class CurvePoint(FrozenModel):
    date: date
    value: float = Field(gt=0, description="Equity level (any base, e.g. 1.0 or 100).")


class ClaimedPerformance(FrozenModel):
    """Performance as claimed by the source. Every field is optional: claims are often partial."""

    sharpe: float | None = Field(default=None, description="Annualised Sharpe ratio.")
    cagr: float | None = Field(default=None, gt=-1, description="Fraction, e.g. 0.25 = 25%/yr.")
    max_drawdown: float | None = Field(
        default=None, ge=0, le=1, description="Positive fraction, e.g. 0.20 = -20% peak-to-trough."
    )
    period_start: date | None = None
    period_end: date | None = None

    @model_validator(mode="after")
    def _period_order(self) -> Self:
        if self.period_start and self.period_end and self.period_start >= self.period_end:
            raise ValueError("period_start must be before period_end")
        return self


class Hypothesis(FrozenModel):
    """Testable restatement of a strategy claim (Intake output)."""

    source_type: SourceType
    universe: list[str] = Field(min_length=1, description="Tickers or a named universe.")
    signal_description: str = Field(min_length=1)
    holding_period_days: int | None = Field(default=None, ge=1)
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    claimed: ClaimedPerformance = ClaimedPerformance()
    claimed_curve: list[CurvePoint] | None = Field(
        default=None, description="Equity curve digitised from a screenshot, if any."
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Gaps in the claim that Intake had to fill."
    )

    @model_validator(mode="after")
    def _curve_sorted(self) -> Self:
        if self.claimed_curve:
            dates = [p.date for p in self.claimed_curve]
            if dates != sorted(set(dates)):
                raise ValueError("claimed_curve dates must be strictly increasing")
        return self
