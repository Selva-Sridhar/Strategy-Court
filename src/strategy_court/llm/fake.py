from collections.abc import Callable, Iterable

from strategy_court.llm.types import LLMRequest, LLMResponse


class FakeProvider:
    """Deterministic stand-in for a real model: scripted replies or a responder function."""

    def __init__(
        self,
        replies: Iterable[str] | Callable[[LLMRequest], str],
        model: str | None = None,
    ) -> None:
        self._responder = replies if callable(replies) else None
        self._queue = None if callable(replies) else list(replies)
        self._model = model
        self.requests: list[LLMRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._responder is not None:
            text = self._responder(request)
        else:
            assert self._queue is not None
            if not self._queue:
                raise RuntimeError("FakeProvider ran out of scripted replies")
            text = self._queue.pop(0)
        prompt_chars = sum(len(m.content) for m in request.messages)
        return LLMResponse(
            text=text,
            model=self._model or request.model,
            input_tokens=max(1, prompt_chars // 4),
            output_tokens=max(1, len(text) // 4),
        )
