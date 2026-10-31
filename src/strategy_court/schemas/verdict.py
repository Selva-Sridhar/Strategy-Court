from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from strategy_court.schemas._base import FrozenModel


class VerdictLabel(StrEnum):
    ACQUITTED = "acquitted"
    NOT_PROVEN = "not_proven"
    CONVICTED = "convicted"


class Penalty(FrozenModel):
    evidence_id: str = Field(min_length=1)
    rule: str = Field(min_length=1, description="Rubric rule name from rubric.yaml.")
    points: float = Field(ge=0)


class Verdict(FrozenModel):
    """Rubric output. `score` and `label` are computed by code; only `opinion` is LLM-written."""

    strategy_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=100, description="Credibility score 0-100.")
    label: VerdictLabel
    rubric_version: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    penalties: list[Penalty] = Field(default_factory=list)
    family_penalties: dict[str, float] = Field(
        default_factory=dict, description="Points deducted per evidence family after caps."
    )
    notes: list[str] = Field(
        default_factory=list, description="Deterministic reasons, e.g. why acquittal was withheld."
    )
    auto_conviction_reason: str | None = None
    opinion: str | None = None
    opinion_validated: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.auto_conviction_reason and self.label is not VerdictLabel.CONVICTED:
            raise ValueError("auto-conviction requires label 'convicted'")
        unknown = {p.evidence_id for p in self.penalties} - set(self.evidence_ids)
        if unknown:
            raise ValueError(f"penalties reference unknown evidence: {sorted(unknown)}")
        if self.opinion_validated and not self.opinion:
            raise ValueError("cannot validate a missing opinion")
        return self
