import re
from collections.abc import Mapping

from pydantic import BaseModel, ValidationError

from strategy_court.llm.cache import CacheMode, ReplayCache
from strategy_court.llm.errors import (
    BudgetExceededError,
    CacheMissError,
    StructuredOutputError,
    UnpricedModelError,
)
from strategy_court.llm.types import LLMProvider, LLMRequest, LLMResponse, Message, ModelPrice
from strategy_court.schemas import LLMCall, TrialTrace

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull the JSON payload out of a model reply (strips code fences / surrounding prose)."""
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text.strip()
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    return text[start : end + 1] if end > start else text[start:]


class LLMClient:
    """Single gateway for every LLM call: cache, cost meter, budget and trace logging."""

    def __init__(
        self,
        provider: LLMProvider,
        cache: ReplayCache | None = None,
        mode: CacheMode = CacheMode.AUTO,
        pricing: Mapping[str, ModelPrice] | None = None,
        budget_usd: float | None = None,
        trace: TrialTrace | None = None,
    ) -> None:
        if mode is not CacheMode.AUTO and cache is None:
            raise ValueError(f"cache mode '{mode}' requires a cache")
        self.provider = provider
        self.cache = cache
        self.mode = mode
        self.pricing = dict(pricing or {})
        self.budget_usd = budget_usd
        self.trace = trace
        self.spent_usd = 0.0

    def _price(self, response: LLMResponse) -> float:
        price = self.pricing.get(response.model)
        if price is None:
            raise UnpricedModelError(f"no price configured for model '{response.model}'")
        return price.cost(response)

    def _live(self, request: LLMRequest) -> LLMResponse:
        if self.budget_usd is not None and self.spent_usd >= self.budget_usd:
            raise BudgetExceededError(
                f"spent ${self.spent_usd:.4f} of ${self.budget_usd:.2f} budget"
            )
        if request.model not in self.pricing:
            raise UnpricedModelError(f"no price configured for model '{request.model}'")
        return self.provider.complete(request)

    def complete(self, request: LLMRequest) -> LLMCall:
        key = request.sha256()
        response = None
        if self.cache is not None and self.mode is not CacheMode.RECORD:
            response = self.cache.get(key)
        cached = response is not None
        if response is None:
            if self.mode is CacheMode.REPLAY:
                raise CacheMissError(f"no recorded response for request {key[:12]}")
            response = self._live(request)
            if self.cache is not None:
                self.cache.put(key, request, response)
        cost = 0.0 if cached else self._price(response)
        self.spent_usd += cost
        call = LLMCall(
            provider=self.provider.name,
            model=response.model,
            prompt_sha256=key,
            response_text=response.text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost,
            cached=cached,
        )
        if self.trace is not None:
            self.trace.llm_calls.append(call)
        return call

    def complete_structured[M: BaseModel](
        self, request: LLMRequest, schema: type[M], max_attempts: int = 3
    ) -> M:
        """Ask for JSON matching `schema`; on failure, feed the validation error back and retry."""
        last_error = ""
        for _ in range(max_attempts):
            call = self.complete(request)
            try:
                return schema.model_validate_json(extract_json(call.response_text))
            except ValidationError as exc:
                last_error = str(exc)
                request = request.with_messages(
                    Message(role="assistant", content=call.response_text),
                    Message(
                        role="user",
                        content=(
                            "Your reply failed schema validation:\n"
                            f"{last_error}\nReturn only the corrected JSON object."
                        ),
                    ),
                )
        raise StructuredOutputError(
            f"{schema.__name__} not produced after {max_attempts} attempts: {last_error}"
        )
