from datetime import date

import pytest
from pydantic import BaseModel, ValidationError

from strategy_court.schemas import (
    Citation,
    ClaimedPerformance,
    CurvePoint,
    EventKind,
    Evidence,
    Hypothesis,
    LLMCall,
    Penalty,
    Provenance,
    Severity,
    SourceType,
    Stage,
    StrategySpec,
    TrialTrace,
    Verdict,
    VerdictLabel,
)

DSR = Citation(
    authors="Bailey & López de Prado",
    year=2014,
    title="The Deflated Sharpe Ratio",
    venue="Journal of Portfolio Management",
)


def make_evidence(**kw: object) -> Evidence:
    base: dict[str, object] = {
        "id": "ev-1",
        "tool": "deflated_sharpe",
        "strategy_id": "s1",
        "statistic_name": "dsr",
        "statistic": 0.42,
        "p_value": 0.58,
        "flag": True,
        "severity": Severity.HIGH,
        "summary": "Deflated Sharpe below threshold.",
        "details": {"n_trials": 50, "sr": 1.9, "nested": {"x": [1, 2.5]}, "ok": True},
        "citations": [DSR],
    }
    return Evidence.model_validate(base | kw)


def make_verdict(**kw: object) -> Verdict:
    base: dict[str, object] = {
        "strategy_id": "s1",
        "score": 12.0,
        "label": VerdictLabel.CONVICTED,
        "rubric_version": "1.0.0",
        "evidence_ids": ["ev-1"],
        "penalties": [Penalty(evidence_id="ev-1", rule="dsr_low", points=40)],
    }
    return Verdict.model_validate(base | kw)


def roundtrip[M: BaseModel](m: M) -> M:
    return type(m).model_validate_json(m.model_dump_json())


def test_hypothesis_roundtrip() -> None:
    h = Hypothesis(
        source_type=SourceType.SCREENSHOT,
        universe=["SPY"],
        signal_description="RSI(14) < 30 buy, > 70 sell",
        claimed=ClaimedPerformance(
            sharpe=3.1,
            max_drawdown=0.12,
            period_start=date(2020, 1, 1),
            period_end=date(2024, 1, 1),
        ),
        claimed_curve=[
            CurvePoint(date=date(2020, 1, 1), value=100),
            CurvePoint(date=date(2021, 1, 1), value=180),
        ],
    )
    assert roundtrip(h) == h


@pytest.mark.parametrize(
    "claimed",
    [
        {"max_drawdown": -0.2},
        {"max_drawdown": 1.5},
        {"period_start": date(2024, 1, 1), "period_end": date(2020, 1, 1)},
    ],
)
def test_claimed_performance_rejects_invalid(claimed: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ClaimedPerformance.model_validate(claimed)


def test_hypothesis_rejects_unsorted_curve_and_extra_fields() -> None:
    pts = [CurvePoint(date=date(2021, 1, 1), value=1), CurvePoint(date=date(2020, 1, 1), value=1)]
    with pytest.raises(ValidationError):
        Hypothesis(source_type="text", universe=["SPY"], signal_description="x", claimed_curve=pts)
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(
            {"source_type": "text", "universe": ["SPY"], "signal_description": "x", "alpha": 1}
        )


def test_strategy_spec_hash_and_rules() -> None:
    s = StrategySpec(id="s1", source_code="class Strategy: ...", provenance=Provenance.CODER)
    assert len(s.code_sha256) == 64
    assert roundtrip(s) == s
    with pytest.raises(ValidationError):
        StrategySpec(
            id="s2",
            source_code="x",
            provenance=Provenance.CODER,
            interpretation_index=3,
            n_interpretations=2,
        )
    with pytest.raises(ValidationError):
        StrategySpec(id="s3", source_code="x", provenance=Provenance.DEFENSE_PATCH)


def test_evidence_roundtrip_numbers_and_ordering() -> None:
    ev = make_evidence()
    assert roundtrip(ev) == ev
    assert sorted(ev.numbers()) == sorted([0.42, 0.58, 50.0, 1.9, 1.0, 2.5])
    assert Severity.CRITICAL > Severity.HIGH
    assert DSR.short() == "Bailey & López de Prado (2014)"
    with pytest.raises(ValidationError):
        make_evidence(p_value=1.2)


def test_evidence_is_immutable() -> None:
    ev = make_evidence()
    with pytest.raises(ValidationError):
        ev.flag = False  # type: ignore[misc]


def test_verdict_rules() -> None:
    v = make_verdict()
    assert roundtrip(v) == v
    with pytest.raises(ValidationError):
        make_verdict(score=101)
    with pytest.raises(ValidationError):
        make_verdict(label=VerdictLabel.ACQUITTED, auto_conviction_reason="causality failed")
    with pytest.raises(ValidationError):
        make_verdict(penalties=[Penalty(evidence_id="ev-unknown", rule="r", points=1)])
    with pytest.raises(ValidationError):
        make_verdict(opinion_validated=True)


def test_trial_trace_emit_and_roundtrip() -> None:
    t = TrialTrace(trial_id="t1", data_hashes={"spy": "abc"})
    t.emit(Stage.ARRAIGNMENT, EventKind.STAGE_STARTED)
    t.emit(Stage.ARRAIGNMENT, EventKind.EVIDENCE, {"evidence_id": "ev-1"})
    t.llm_calls.append(
        LLMCall(provider="fake", model="m", prompt_sha256="0" * 64, response_text="", cost_usd=0.01)
    )
    t.verdicts.append(make_verdict())
    assert [e.seq for e in t.events] == [0, 1]
    assert t.total_cost_usd == pytest.approx(0.01)
    assert roundtrip(t) == t
