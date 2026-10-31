from pathlib import Path

import pytest
from pydantic import BaseModel

from strategy_court.llm import (
    BudgetExceededError,
    CacheMissError,
    CacheMode,
    FakeProvider,
    LLMClient,
    LLMRequest,
    Message,
    ModelPrice,
    ReplayCache,
    StructuredOutputError,
    UnpricedModelError,
    extract_json,
)
from strategy_court.schemas import TrialTrace

PRICING = {"m": ModelPrice(input_per_mtok=1_000_000, output_per_mtok=1_000_000)}  # $1/token


def req(text: str = "hello", model: str = "m") -> LLMRequest:
    return LLMRequest(model=model, messages=[Message(role="user", content=text)])


class Answer(BaseModel):
    verdict: str
    score: int


def test_request_hash_is_stable_and_content_sensitive() -> None:
    assert req().sha256() == req().sha256()
    assert req("a").sha256() != req("b").sha256()
    assert req().sha256() != req().model_copy(update={"temperature": 0.5}).sha256()


def test_auto_mode_records_then_replays_for_free(tmp_path: Path) -> None:
    provider = FakeProvider(["hi"])
    trace = TrialTrace(trial_id="t")
    client = LLMClient(provider, ReplayCache(tmp_path), pricing=PRICING, trace=trace)
    first, second = client.complete(req()), client.complete(req())
    assert len(provider.requests) == 1
    assert (first.cached, second.cached) == (False, True)
    assert first.cost_usd > 0 and second.cost_usd == 0
    assert second.response_text == "hi"
    assert len(trace.llm_calls) == 2
    assert client.spent_usd == first.cost_usd


def test_replay_mode_never_calls_provider(tmp_path: Path) -> None:
    cache = ReplayCache(tmp_path)
    LLMClient(FakeProvider(["recorded"]), cache, pricing=PRICING).complete(req())
    replay = LLMClient(FakeProvider([]), cache, mode=CacheMode.REPLAY)
    assert replay.complete(req()).response_text == "recorded"
    with pytest.raises(CacheMissError):
        replay.complete(req("unseen"))


def test_record_mode_always_calls_provider(tmp_path: Path) -> None:
    provider = FakeProvider(["v1", "v2"])
    client = LLMClient(provider, ReplayCache(tmp_path), mode=CacheMode.RECORD, pricing=PRICING)
    client.complete(req())
    assert client.complete(req()).response_text == "v2"
    assert len(provider.requests) == 2


def test_budget_blocks_live_calls_but_not_cached(tmp_path: Path) -> None:
    client = LLMClient(
        FakeProvider(lambda r: "x"), ReplayCache(tmp_path), pricing=PRICING, budget_usd=1.0
    )
    client.complete(req("a"))
    with pytest.raises(BudgetExceededError):
        client.complete(req("b"))
    assert client.complete(req("a")).cached


def test_unpriced_model_is_rejected_before_calling() -> None:
    provider = FakeProvider(["x"])
    with pytest.raises(UnpricedModelError):
        LLMClient(provider, pricing=PRICING).complete(req(model="unknown"))
    assert provider.requests == []


def test_non_auto_mode_requires_cache() -> None:
    with pytest.raises(ValueError):
        LLMClient(FakeProvider([]), mode=CacheMode.REPLAY)


@pytest.mark.parametrize(
    "text",
    [
        '{"verdict": "convicted", "score": 5}',
        'Sure!\n```json\n{"verdict": "convicted", "score": 5}\n```\nDone.',
        'Here you go: {"verdict": "convicted", "score": 5} hope it helps',
    ],
)
def test_extract_json(text: str) -> None:
    assert Answer.model_validate_json(extract_json(text)) == Answer(verdict="convicted", score=5)


def test_structured_output_repairs_with_error_feedback() -> None:
    provider = FakeProvider(['{"verdict": "convicted"}', '{"verdict": "convicted", "score": 5}'])
    out = LLMClient(provider, pricing=PRICING).complete_structured(req(), Answer)
    assert out == Answer(verdict="convicted", score=5)
    retry = provider.requests[1].messages
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "score" in retry[-1].content


def test_structured_output_gives_up() -> None:
    client = LLMClient(FakeProvider(lambda r: "not json"), pricing=PRICING)
    with pytest.raises(StructuredOutputError):
        client.complete_structured(req(), Answer, max_attempts=2)
