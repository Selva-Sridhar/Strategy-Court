from enum import IntEnum

from pydantic import Field, JsonValue

from strategy_court.schemas._base import FrozenModel


class Severity(IntEnum):
    """Ordered so the rubric can compare (`sev >= Severity.HIGH`)."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Citation(FrozenModel):
    authors: str = Field(min_length=1, description="e.g. 'Bailey & López de Prado'")
    year: int = Field(ge=1900, le=2100)
    title: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    doi: str | None = None
    url: str | None = None

    def short(self) -> str:
        return f"{self.authors} ({self.year})"


class Evidence(FrozenModel):
    """Output of one audit tool run. The Judge may only cite evidence by `id`."""

    id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    statistic_name: str | None = None
    statistic: float | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    flag: bool = Field(description="True if the tool found a flaw.")
    severity: Severity
    summary: str = Field(min_length=1, description="One deterministic sentence, no LLM text.")
    details: dict[str, JsonValue] = Field(default_factory=dict)
    plot_data: dict[str, JsonValue] | None = None
    citations: list[Citation] = Field(default_factory=list)

    def numbers(self) -> list[float]:
        """Every numeric value a Judge opinion may legitimately quote from this evidence."""
        found: list[float] = [x for x in (self.statistic, self.p_value) if x is not None]

        def walk(v: JsonValue) -> None:
            if isinstance(v, bool):
                return
            if isinstance(v, int | float):
                found.append(float(v))
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)

        walk(self.details)
        return found
