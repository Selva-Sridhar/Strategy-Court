"""Helpers shared by audit tools: JSON-safe evidence construction and severity rules."""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import JsonValue

from strategy_court.schemas import Citation, Evidence, Severity


def clean(value: object) -> JsonValue:
    """Make a value JSON-safe: NaN/inf -> None, numpy scalars/arrays -> Python, tuples -> lists."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return [clean(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [clean(v) for v in value]
    raise TypeError(f"cannot serialise {type(value).__name__} into evidence")


def finite_or_none(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def make_evidence(
    tool: str,
    strategy_id: str,
    *,
    flag: bool,
    severity: Severity,
    summary: str,
    statistic_name: str | None = None,
    statistic: float | None = None,
    p_value: float | None = None,
    details: Mapping[str, object] | None = None,
    plot_data: Mapping[str, object] | None = None,
    citations: Sequence[Citation] = (),
    evidence_id: str | None = None,
) -> Evidence:
    details_json = clean(dict(details or {}))
    plot_json = clean(dict(plot_data)) if plot_data is not None else None
    assert isinstance(details_json, dict)
    assert plot_json is None or isinstance(plot_json, dict)
    p = finite_or_none(p_value)
    return Evidence(
        id=evidence_id or f"{tool}:{strategy_id}",
        tool=tool,
        strategy_id=strategy_id,
        statistic_name=statistic_name,
        statistic=finite_or_none(statistic),
        p_value=None if p is None else min(1.0, max(0.0, p)),
        flag=flag,
        severity=severity,
        summary=summary,
        details=details_json,
        plot_data=plot_json,
        citations=list(citations),
    )


def severity_from_p(p: float, alpha: float = 0.05) -> Severity:
    """For tests whose null is 'no skill': a large p-value means the edge is not demonstrated."""
    if not math.isfinite(p):
        return Severity.MEDIUM
    if p > 0.5:
        return Severity.HIGH
    if p > alpha:
        return Severity.MEDIUM
    return Severity.INFO


def inconclusive(
    tool: str, strategy_id: str, reason: str, citations: Sequence[Citation] = ()
) -> Evidence:
    return make_evidence(
        tool,
        strategy_id,
        flag=False,
        severity=Severity.LOW,
        summary=f"Not applicable / inconclusive: {reason}.",
        details={"status": "inconclusive", "reason": reason},
        citations=citations,
    )
