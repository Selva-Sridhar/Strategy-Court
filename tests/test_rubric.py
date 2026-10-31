from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from strategy_court.audit.registry import TOOLS
from strategy_court.rubric import Rubric, load_rubric, score
from strategy_court.schemas import Evidence, Severity, VerdictLabel

RB = load_rubric()


def ev(
    tool: str, severity: Severity = Severity.INFO, flag: bool | None = None, **details: object
) -> Evidence:
    return Evidence(
        id=f"{tool}:s",
        tool=tool,
        strategy_id="s",
        flag=severity >= Severity.MEDIUM if flag is None else flag,
        severity=severity,
        summary="x",
        details=details,  # type: ignore[arg-type]
    )


def clean_set() -> list[Evidence]:
    return [ev(t) for t in TOOLS]


def test_rubric_covers_every_registered_tool_exactly_once() -> None:
    assert set(RB.tool_family) == set(TOOLS)
    assert RB.version_tag.startswith("1.2.0+") and len(RB.fingerprint) == 8


def test_clean_evidence_is_acquitted_with_full_score() -> None:
    v = score(clean_set())
    assert v.score == 100 and v.label is VerdictLabel.ACQUITTED and not v.penalties


def test_proven_lookahead_is_automatic_conviction() -> None:
    evidence = [ev(t) for t in TOOLS if t != "causality"] + [ev("causality", Severity.CRITICAL)]
    v = score(evidence)
    assert v.label is VerdictLabel.CONVICTED and v.score <= 5
    assert v.auto_conviction_reason == "look-ahead proven by the causality test"
    assert sum(p.points for p in v.penalties) == pytest.approx(100 - v.score)


def test_family_aggregation_cap_and_diminishing_returns() -> None:
    evidence = [e for e in clean_set() if e.tool not in ("deflated_sharpe", "pbo")]
    evidence += [ev("deflated_sharpe", Severity.HIGH), ev("pbo", Severity.MEDIUM)]
    v = score(evidence)
    assert v.family_penalties["selection"] == pytest.approx(35 + 0.5 * 15)  # below the cap
    for tool in ("haircut_sharpe", "purged_cv", "data_snooping"):
        evidence.append(ev(tool, Severity.HIGH).model_copy(update={"id": f"{tool}-2"}))
    v2 = score(evidence)
    cap = RB.families["selection"].cap
    assert cap < 35 + 0.5 * (15 + 25 + 25 + 25)
    assert v2.family_penalties["selection"] == pytest.approx(cap)  # capped
    assert v2.label is VerdictLabel.CONVICTED


def test_no_skill_alone_is_not_proven_not_convicted() -> None:
    skill = {"psr", "placebo_null", "bootstrap_sharpe", "subperiod_stability"}
    evidence = [ev(t, Severity.HIGH if t in skill else Severity.INFO) for t in TOOLS]
    v = score(evidence)
    assert v.score <= RB.labels.convict_below and v.label is VerdictLabel.NOT_PROVEN
    strict = RB.model_copy(update={"labels": RB.labels.model_copy(update={"convict_below": 50})})
    w = score(evidence, strict)
    assert w.label is VerdictLabel.NOT_PROVEN
    assert w.notes == ["conviction withheld: no positive evidence of a flaw, only of no skill"]
    evidence[[e.tool for e in evidence].index("cost_sensitivity")] = ev(
        "cost_sensitivity", Severity.HIGH
    )
    assert score(evidence).label is VerdictLabel.CONVICTED


def test_corroborated_selection_bias_alone_can_convict() -> None:
    selection = set(RB.families["selection"].tools)
    evidence = [ev(t, Severity.HIGH if t in selection else Severity.INFO) for t in TOOLS]
    v = score(evidence)
    assert v.label is VerdictLabel.CONVICTED and v.score < RB.labels.convict_below
    one = [ev(t, Severity.HIGH if t == "deflated_sharpe" else Severity.INFO) for t in TOOLS]
    assert score(one).label is not VerdictLabel.CONVICTED  # a single signal is not enough


def test_inconsistent_claim_is_never_acquitted() -> None:
    evidence = [ev(t, Severity.HIGH if t == "claim_consistency" else Severity.INFO) for t in TOOLS]
    v = score(evidence)
    assert v.score >= RB.labels.acquit_at_or_above and v.label is VerdictLabel.NOT_PROVEN
    assert "acquittal withheld: flagged 'claim' evidence" in v.notes


def test_acquittal_requires_conclusive_core_families() -> None:
    evidence = [e for e in clean_set() if e.tool not in ("psr", "placebo_null", "bootstrap_sharpe")]
    evidence.append(ev("psr", status="inconclusive", reason="too short"))
    v = score(evidence)
    assert v.score == 100 and v.label is VerdictLabel.NOT_PROVEN
    assert v.notes == ["acquittal withheld: no conclusive 'significance' evidence"]


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        score([])
    with pytest.raises(ValueError, match="no weight"):
        score([ev("astrology")])
    mixed = [ev("psr"), ev("pbo").model_copy(update={"strategy_id": "other", "id": "x"})]
    with pytest.raises(ValueError, match="more than one strategy"):
        score(mixed)


def test_rubric_validation_and_fingerprint(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path(__file__).parent.parent.joinpath("src/strategy_court/rubric.yaml").read_text("utf-8")
    )
    raw["families"]["claim"]["tools"]["psr"] = 5.0
    with pytest.raises(ValueError, match="families"):
        Rubric.model_validate(raw)
    raw = yaml.safe_load(
        Path(__file__).parent.parent.joinpath("src/strategy_court/rubric.yaml").read_text("utf-8")
    )
    raw["labels"]["acquit_at_or_above"] = 75
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(raw), "utf-8")
    changed = load_rubric(path)
    assert changed.version == RB.version and changed.fingerprint != RB.fingerprint


SEVERITIES = st.sampled_from(list(Severity))


@st.composite
def evidence_sets(draw: st.DrawFn) -> list[Evidence]:
    tools = draw(st.lists(st.sampled_from(list(TOOLS)), min_size=1, max_size=14, unique=True))
    out = []
    for t in tools:
        status = draw(st.sampled_from(["ok", "ok", "inconclusive"]))
        details = {"status": status} if status == "inconclusive" else {}
        out.append(ev(t, draw(SEVERITIES), **details))
    return out


@settings(max_examples=300, deadline=None)
@given(evidence_sets(), st.randoms(use_true_random=False))
def test_properties_order_invariance_identity_and_monotonicity(
    evidence: list[Evidence], rnd: object
) -> None:
    v = score(evidence)
    shuffled = list(evidence)
    rnd.shuffle(shuffled)  # type: ignore[attr-defined]
    assert score(shuffled) == v
    assert sum(p.points for p in v.penalties) == pytest.approx(100 - v.score, abs=1e-6)
    if v.label is VerdictLabel.CONVICTED and v.auto_conviction_reason is None:
        assert any(v.family_penalties[f] > 0 for f in RB.conviction_requires_any)
    if v.label is VerdictLabel.ACQUITTED:
        blocked = {f for f in RB.acquittal_blocked_by}
        assert not any(
            e.flag
            and e.details.get("status") != "inconclusive"
            and RB.tool_family[e.tool] in blocked
            for e in evidence
        )
    for i, e in enumerate(evidence):
        if e.severity is Severity.CRITICAL or e.details.get("status") == "inconclusive":
            continue
        worse = list(evidence)
        worse[i] = ev(e.tool, Severity(e.severity + 1))
        assert score(worse).score <= v.score + 1e-9
