from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, JsonValue

from strategy_court.schemas._base import FrozenModel, MutableModel
from strategy_court.schemas.verdict import Verdict


class Stage(StrEnum):
    INTAKE = "intake"
    CODER = "coder"
    ARRAIGNMENT = "arraignment"
    PROSECUTION = "prosecution"
    DEFENSE = "defense"
    RETRIAL = "retrial"
    VERDICT = "verdict"
    OPINION = "opinion"


class EventKind(StrEnum):
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    EVIDENCE = "evidence"
    ERROR = "error"
    VERDICT = "verdict"


def _now() -> datetime:
    return datetime.now(UTC)


class TraceEvent(FrozenModel):
    seq: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=_now)
    stage: Stage
    kind: EventKind
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class LLMCall(FrozenModel):
    provider: str
    model: str
    prompt_sha256: str = Field(min_length=64, max_length=64)
    response_text: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    cached: bool = False


class TrialTrace(MutableModel):
    """Append-only record of a trial; persisted as JSON and replayable."""

    trial_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=_now)
    data_hashes: dict[str, str] = Field(default_factory=dict)
    events: list[TraceEvent] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)

    def emit(
        self, stage: Stage, kind: EventKind, payload: dict[str, JsonValue] | None = None
    ) -> TraceEvent:
        event = TraceEvent(seq=len(self.events), stage=stage, kind=kind, payload=payload or {})
        self.events.append(event)
        return event

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)
