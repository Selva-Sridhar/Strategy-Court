from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable contract: unknown fields are rejected so malformed LLM output fails loudly."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class MutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
