import hashlib
from typing import Literal, Protocol

from pydantic import Field

from strategy_court.schemas._base import FrozenModel


class ImageInput(FrozenModel):
    media_type: str = Field(pattern=r"^image/(png|jpeg|webp|gif)$")
    data_b64: str = Field(min_length=1)


class Message(FrozenModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: list[ImageInput] = Field(default_factory=list)


class LLMRequest(FrozenModel):
    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    json_mode: bool = False

    def sha256(self) -> str:
        """Cache key: identical requests (incl. images and sampling params) share one entry."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def with_messages(self, *extra: Message) -> "LLMRequest":
        return self.model_copy(update={"messages": [*self.messages, *extra]})


class LLMResponse(FrozenModel):
    text: str
    model: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ModelPrice(FrozenModel):
    input_per_mtok: float = Field(ge=0, description="USD per 1M input tokens.")
    output_per_mtok: float = Field(ge=0, description="USD per 1M output tokens.")

    def cost(self, response: LLMResponse) -> float:
        return (
            response.input_tokens * self.input_per_mtok
            + response.output_tokens * self.output_per_mtok
        ) / 1e6


class LLMProvider(Protocol):
    """Anything that can answer an `LLMRequest`. Meta Model API, fakes, etc. implement this."""

    @property
    def name(self) -> str: ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...
