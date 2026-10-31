"""Deterministic scoring: evidence -> credibility score -> verdict label.

The rubric is data (`rubric.yaml`), not code, so it is reviewable and versioned; its content
fingerprint is recorded in every verdict. Properties guaranteed (and property-tested):

* deterministic and order-invariant in the evidence;
* monotone: raising any evidence's severity never raises the score;
* accounting identity: sum of penalty points == 100 - score;
* no conviction without positive evidence of a flaw (or a hard auto-conviction rule);
* independent of any LLM output — only `Evidence` objects are read.
"""

import hashlib
import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, field_validator, model_validator

from strategy_court.schemas import Evidence, Penalty, Severity, Verdict, VerdictLabel
from strategy_court.schemas._base import FrozenModel


class FamilySpec(FrozenModel):
    cap: float = Field(gt=0)
    tools: dict[str, float]

    @field_validator("tools")
    @classmethod
    def _positive(cls, v: dict[str, float]) -> dict[str, float]:
        if not v or any(w <= 0 for w in v.values()):
            raise ValueError("a family needs at least one tool, all weights > 0")
        return v


class AutoConvict(FrozenModel):
    tool: str
    min_severity: Severity
    cap_score: float = Field(ge=0, le=100)
    reason: str

    @field_validator("min_severity", mode="before")
    @classmethod
    def _by_name(cls, v: object) -> object:
        return Severity[v] if isinstance(v, str) else v


class Labels(FrozenModel):
    acquit_at_or_above: float = Field(ge=0, le=100)
    convict_below: float = Field(ge=0, le=100)


class Rubric(FrozenModel):
    version: str = Field(min_length=1)
    severity_factor: dict[str, float]
    secondary_weight: float = Field(ge=0, le=1)
    inconclusive_penalty: float = Field(ge=0, le=1)
    labels: Labels
    auto_convict: list[AutoConvict] = Field(default_factory=list)
    acquittal_requires: list[str] = Field(default_factory=list)
    conviction_requires_any: list[str] = Field(default_factory=list)
    families: dict[str, FamilySpec]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.severity_factor) != {s.name for s in Severity}:
            raise ValueError("severity_factor must list exactly the Severity levels")
        factors = [self.severity_factor[s.name] for s in sorted(Severity)]
        if factors != sorted(factors) or not all(0 <= f <= 1 for f in factors):
            raise ValueError("severity factors must be in [0, 1] and non-decreasing")
        if self.labels.convict_below > self.labels.acquit_at_or_above:
            raise ValueError("convict_below must not exceed acquit_at_or_above")
        seen: dict[str, str] = {}
        for fam, spec in self.families.items():
            for tool in spec.tools:
                if tool in seen:
                    raise ValueError(f"tool '{tool}' is in families '{seen[tool]}' and '{fam}'")
                seen[tool] = fam
        unknown = (set(self.acquittal_requires) | set(self.conviction_requires_any)) - set(
            self.families
        )
        if unknown:
            raise ValueError(f"label rules name unknown families: {sorted(unknown)}")
        return self

    @property
    def tool_family(self) -> dict[str, str]:
        return {t: fam for fam, spec in self.families.items() for t in spec.tools}

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:8]

    @property
    def version_tag(self) -> str:
        return f"{self.version}+{self.fingerprint}"


def load_rubric(path: Path | None = None) -> Rubric:
    if path is None:
        text = resources.files("strategy_court").joinpath("rubric.yaml").read_text("utf-8")
    else:
        text = path.read_text("utf-8")
    return Rubric.model_validate(yaml.safe_load(text))


def _conclusive(ev: Evidence) -> bool:
    return ev.details.get("status") != "inconclusive"


def score(evidence: Sequence[Evidence], rubric: Rubric | None = None) -> Verdict:
    rb = rubric or load_rubric()
    if not evidence:
        raise ValueError("cannot score an empty evidence set")
    strategy_ids = {e.strategy_id for e in evidence}
    if len(strategy_ids) != 1:
        raise ValueError(f"evidence for more than one strategy: {sorted(strategy_ids)}")
    fam_of = rb.tool_family
    unknown = sorted({e.tool for e in evidence} - set(fam_of))
    if unknown:
        raise ValueError(f"rubric has no weight for tools: {unknown}")

    ordered = sorted(evidence, key=lambda e: e.id)
    raw: dict[str, list[tuple[Evidence, float]]] = {f: [] for f in rb.families}
    for ev in ordered:
        fam = fam_of[ev.tool]
        weight = rb.families[fam].tools[ev.tool]
        if not _conclusive(ev):
            pts = weight * rb.inconclusive_penalty
        elif ev.flag:
            pts = weight * rb.severity_factor[ev.severity.name]
        else:
            pts = 0.0
        raw[fam].append((ev, pts))

    penalties: list[Penalty] = []
    family_pen: dict[str, float] = {}
    for fam, items in raw.items():
        ranked = sorted((p for _, p in items), reverse=True)
        if not ranked or ranked[0] == 0:
            family_pen[fam] = 0.0
            continue
        total = min(rb.families[fam].cap, ranked[0] + rb.secondary_weight * sum(ranked[1:]))
        family_pen[fam] = total
        raw_sum = sum(ranked)
        for ev, p in items:
            if p > 0:
                penalties.append(
                    Penalty(evidence_id=ev.id, rule=f"{fam}:{ev.tool}", points=total * p / raw_sum)
                )

    value = max(0.0, 100.0 - sum(family_pen.values()))
    notes: list[str] = []
    reason = None
    for rule in rb.auto_convict:
        hit = next(
            (e for e in ordered if e.tool == rule.tool and e.severity >= rule.min_severity), None
        )
        if hit is not None and value > rule.cap_score:
            penalties.append(
                Penalty(
                    evidence_id=hit.id,
                    rule=f"auto_convict:{rule.tool}",
                    points=value - rule.cap_score,
                )
            )
            value = rule.cap_score
        if hit is not None:
            reason = reason or rule.reason
    overflow = sum(p.points for p in penalties) - (100.0 - value)
    if overflow > 1e-9:  # floor at 0 absorbed some points: scale so the identity holds exactly
        scale = (100.0 - value) / sum(p.points for p in penalties)
        penalties = [p.model_copy(update={"points": p.points * scale}) for p in penalties]

    flaw_shown = not rb.conviction_requires_any or any(
        family_pen[f] > 0 for f in rb.conviction_requires_any
    )
    if reason is not None:
        label = VerdictLabel.CONVICTED
    elif value < rb.labels.convict_below:
        label = VerdictLabel.CONVICTED if flaw_shown else VerdictLabel.NOT_PROVEN
        if not flaw_shown:
            notes.append("conviction withheld: no positive evidence of a flaw, only of no skill")
    elif value >= rb.labels.acquit_at_or_above:
        label = VerdictLabel.ACQUITTED
        for fam in rb.acquittal_requires:
            if not any(_conclusive(ev) for ev, _ in raw[fam]):
                label = VerdictLabel.NOT_PROVEN
                notes.append(f"acquittal withheld: no conclusive '{fam}' evidence")
    else:
        label = VerdictLabel.NOT_PROVEN
    return Verdict(
        strategy_id=next(iter(strategy_ids)),
        score=round(value, 6),
        label=label,
        rubric_version=rb.version_tag,
        evidence_ids=[e.id for e in ordered],
        penalties=penalties,
        family_penalties=family_pen,
        notes=notes,
        auto_conviction_reason=reason,
    )
