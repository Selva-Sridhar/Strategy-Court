"""Planted-flaw benchmark: run the court on cases with known ground truth and report how often it
reaches the right conclusion.

Protocol (D7): rubric thresholds may be tuned on the *dev* seeds only; the *held-out* seeds are
run once the rubric is frozen, and the reported headline numbers come from held-out.

Usage:  uv run python -m strategy_court.benchmark.suite --split heldout --out benchmark_results
"""

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from strategy_court.audit.registry import TOOLS, TrialContext, applicable, run_tool
from strategy_court.backtest import summarize
from strategy_court.benchmark.flaws import (
    COURT_CONFIG,
    EXPECTED_TOOLS,
    FLAWS,
    Case,
    build_cases,
)
from strategy_court.benchmark.metrics import Rate, confusion, meets_target
from strategy_court.rubric import Rubric, load_rubric, score
from strategy_court.schemas import Evidence

DEV_SEEDS: tuple[int, ...] = tuple(range(5))
# Held-out seeds are spent once a rubric version has been evaluated on them; a new version gets
# a fresh set. 100-109: rubric 1.1.0 (results in docs/benchmark/v1.1). 200-209: rubric 1.2.0.
USED_HELDOUT_SEEDS: dict[str, tuple[int, ...]] = {"1.1.0": tuple(range(100, 110))}
HELDOUT_SEEDS: tuple[int, ...] = tuple(range(200, 210))
BUDGET: dict[str, dict[str, object]] = {
    "causality": {"n_checkpoints": 8},
    "placebo_null": {"n_placebos": 199},
    "bootstrap_sharpe": {"n_boot": 499},
    "data_snooping": {"n_boot": 299},
}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    truth: str
    target: str
    split: str
    seed: int
    label: str
    score: float
    correct: bool
    auto_conviction: bool
    flagged_tools: list[str]
    expected_tool_flagged: bool | None
    claimed_sharpe: float
    court_sharpe: float
    oos_truth_sharpe: float | None
    rubric_version: str
    seconds: float
    evidence: list[dict[str, object]]


def court_context(case: Case) -> TrialContext:
    strat = case.factory(case.params)
    return TrialContext(
        strategy_id=case.case_id,
        market=case.market,
        config=COURT_CONFIG,
        signal_fn=strat.signals,
        source_code=case.source,
        hypothesis=case.claim,
        factory=case.factory,
        params=case.params,
        trials=case.trials,
        extra_trials=case.extra_trials,
        options={k: dict(v) for k, v in BUDGET.items()},
    )


def run_case(case: Case, rubric: Rubric | None = None) -> tuple[CaseResult, list[Evidence]]:
    rb = rubric or load_rubric()
    start = time.perf_counter()
    ctx = court_context(case)
    court = summarize(ctx.result)
    ctx.replicated_sharpes = [court.sharpe] if math.isfinite(court.sharpe) else None
    evidence = [run_tool(t.name, ctx) for t in applicable(ctx)]
    verdict = score(evidence, rb)
    flagged = sorted(e.tool for e in evidence if e.flag)
    expected = EXPECTED_TOOLS[case.truth]
    result = CaseResult(
        case_id=case.case_id,
        truth=case.truth,
        target=case.target,
        split=case.split,
        seed=case.seed,
        label=verdict.label.value,
        score=verdict.score,
        correct=meets_target(case.target, verdict.label.value),
        auto_conviction=verdict.auto_conviction_reason is not None,
        flagged_tools=flagged,
        expected_tool_flagged=bool(expected & set(flagged)) if expected else None,
        claimed_sharpe=case.claimed.sharpe,
        court_sharpe=court.sharpe,
        oos_truth_sharpe=case.oos_truth_sharpe,
        rubric_version=verdict.rubric_version,
        seconds=time.perf_counter() - start,
        evidence=[
            {"tool": e.tool, "flag": e.flag, "severity": e.severity.name, "summary": e.summary}
            for e in evidence
        ],
    )
    return result, evidence


def summarise(results: Sequence[CaseResult]) -> dict[str, object]:
    by_truth: dict[str, list[CaseResult]] = {}
    for r in results:
        by_truth.setdefault(r.truth, []).append(r)

    def rate(rs: Sequence[CaseResult], pred: Callable[[CaseResult], bool]) -> Rate:
        return Rate(sum(pred(r) for r in rs), len(rs))

    flawed = [r for r in results if r.truth in FLAWS]
    per_truth = {
        t: {
            "target": rs[0].target,
            "meets_target": rate(rs, lambda r: r.correct).as_dict(),
            "convicted": rate(rs, lambda r: r.label == "convicted").as_dict(),
            "acquitted": rate(rs, lambda r: r.label == "acquitted").as_dict(),
            "expected_tool_flagged": rate(
                [r for r in rs if r.expected_tool_flagged is not None],
                lambda r: bool(r.expected_tool_flagged),
            ).as_dict(),
            "mean_claimed_sharpe": sum(r.claimed_sharpe for r in rs) / len(rs),
            "mean_score": sum(r.score for r in rs) / len(rs),
        }
        for t, rs in by_truth.items()
    }
    headline = {
        "verdict_accuracy": rate(results, lambda r: r.correct).as_dict(),
        "flaw_conviction_rate": rate(flawed, lambda r: r.label == "convicted").as_dict(),
        "flaw_rejection_rate": rate(flawed, lambda r: r.label != "acquitted").as_dict(),
        "false_conviction_rate_genuine": rate(
            by_truth.get("genuine", []), lambda r: r.label == "convicted"
        ).as_dict(),
        "false_acquittal_rate_no_edge": rate(
            by_truth.get("no_edge", []), lambda r: r.label == "acquitted"
        ).as_dict(),
    }
    return {
        "n_cases": len(results),
        "splits": sorted({r.split for r in results}),
        "rubric_versions": sorted({r.rubric_version for r in results}),
        "headline": headline,
        "per_truth": per_truth,
        "confusion": confusion([r.truth for r in results], [r.label for r in results]),
        "mean_seconds_per_case": sum(r.seconds for r in results) / max(1, len(results)),
    }


def markdown(summary: dict[str, object]) -> str:
    head = summary["headline"]
    assert isinstance(head, dict)
    lines = [
        f"# Strategy Court benchmark ({', '.join(summary['splits'])})",  # type: ignore[arg-type]
        "",
        f"Cases: {summary['n_cases']} - rubric {', '.join(summary['rubric_versions'])}",  # type: ignore[arg-type]
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for name, r in head.items():
        lines.append(f"| {name.replace('_', ' ')} | {Rate(r['k'], r['n']).fmt()} |")
    lines += [
        "",
        "| Ground truth | Target | Meets target | Convicted | Acquitted | Expected tool flagged |",
    ]
    lines.append("|---|---|---|---|---|---|")
    per = summary["per_truth"]
    assert isinstance(per, dict)
    for truth, d in per.items():
        cells = [
            Rate(d[k]["k"], d[k]["n"]).fmt()
            for k in ("meets_target", "convicted", "acquitted", "expected_tool_flagged")
        ]
        lines.append(f"| {truth} | {d['target']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Confusion (rows = ground truth, columns = verdict):",
        "",
        "| | acquitted | not_proven | convicted |",
        "|---|---|---|---|",
    ]
    conf = summary["confusion"]
    assert isinstance(conf, dict)
    for truth, row in conf.items():
        lines.append(f"| {truth} | {row['acquitted']} | {row['not_proven']} | {row['convicted']} |")
    return "\n".join(lines) + "\n"


def run_benchmark(
    split: str,
    seeds: Sequence[int] | None = None,
    kinds: Sequence[str] | None = None,
    rubric: Rubric | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[list[CaseResult], dict[str, object]]:
    if seeds is None:
        seeds = DEV_SEEDS if split == "dev" else HELDOUT_SEEDS
    cases = build_cases(seeds, split, kinds)
    results = []
    for i, case in enumerate(cases, 1):
        res, _ = run_case(case, rubric)
        results.append(res)
        if log:
            mark = "ok " if res.correct else "MISS"
            log(f"[{i}/{len(cases)}] {mark} {res.case_id:48s} {res.label:10s} {res.score:6.1f}")
    return results, summarise(results)


def write_outputs(results: Sequence[CaseResult], summary: dict[str, object], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r), default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    (out / "summary.md").write_text(markdown(summary), "utf-8")


def load_results(path: Path) -> list[CaseResult]:
    """Re-read cases.jsonl (e.g. to re-summarise without re-running the court)."""
    with path.open(encoding="utf-8") as fh:
        return [CaseResult(**json.loads(line)) for line in fh if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--kinds", nargs="*")
    p.add_argument("--out", type=Path, default=Path("benchmark_results"))
    p.add_argument("--from-cases", action="store_true", help="re-summarise existing cases.jsonl")
    a = p.parse_args(argv)
    if a.from_cases:
        results = load_results(a.out / a.split / "cases.jsonl")
        summary = summarise(results)
        write_outputs(results, summary, a.out / a.split)
        print(markdown(summary))
        return
    results, summary = run_benchmark(a.split, a.seeds, a.kinds, log=print)
    write_outputs(results, summary, a.out / a.split)
    print(markdown(summary))


assert set(BUDGET) <= set(TOOLS)

if __name__ == "__main__":
    main()
