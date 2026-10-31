import hashlib
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from strategy_court.schemas._base import FrozenModel

ParamValue = int | float | str | bool


class Provenance(StrEnum):
    USER = "user"
    CODER = "coder"
    DEFENSE_PATCH = "defense_patch"
    BENCHMARK = "benchmark"


class StrategySpec(FrozenModel):
    """Source code of one strategy implementing the fixed `Strategy` interface."""

    id: str = Field(min_length=1)
    source_code: str = Field(min_length=1)
    class_name: str = Field(default="Strategy", min_length=1)
    params: dict[str, ParamValue] = Field(default_factory=dict)
    provenance: Provenance
    interpretation_index: int = Field(default=1, ge=1)
    n_interpretations: int = Field(
        default=1, ge=1, description="K versions of the same claim; feeds the DSR trial count."
    )
    parent_id: str | None = Field(default=None, description="Strategy this one was patched from.")

    @property
    def code_sha256(self) -> str:
        return hashlib.sha256(self.source_code.encode()).hexdigest()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.interpretation_index > self.n_interpretations:
            raise ValueError("interpretation_index cannot exceed n_interpretations")
        if self.provenance is Provenance.DEFENSE_PATCH and self.parent_id is None:
            raise ValueError("a defense patch must reference its parent strategy")
        return self
