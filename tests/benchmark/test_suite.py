from pathlib import Path

import pytest

from strategy_court.benchmark.flaws import build_cases
from strategy_court.benchmark.metrics import Rate, confusion, meets_target, wilson
from strategy_court.benchmark.suite import (
    markdown,
    run_benchmark,
    run_case,
    summarise,
    write_outputs,
)


def test_wilson_interval_known_values() -> None:
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(0, 20)[0] == 0.0 and wilson(20, 20)[1] == 1.0
    assert Rate(3, 4).value == 0.75 and "75%" in Rate(3, 4).fmt()


def test_targets_and_confusion() -> None:
    assert meets_target("convicted", "convicted") and not meets_target("convicted", "not_proven")
    assert meets_target("not_acquitted", "not_proven") and not meets_target(
        "not_acquitted", "acquitted"
    )
    assert meets_target("not_convicted", "acquitted") and not meets_target(
        "not_convicted", "convicted"
    )
    conf = confusion(["a", "a", "b"], ["convicted", "acquitted", "convicted"])
    assert conf["a"] == {"acquitted": 1, "not_proven": 0, "convicted": 1}


def test_run_case_is_deterministic_and_convicts_a_planted_leak() -> None:
    case = build_cases([0], "t", ["lookahead"])[0]
    first, evidence = run_case(case)
    second, _ = run_case(case)
    assert first.label == second.label == "convicted" and first.auto_conviction
    assert first.score == second.score and first.flagged_tools == second.flagged_tools
    assert {e.tool for e in evidence} >= {"causality", "ast_scanner", "claim_consistency"}


@pytest.mark.slow
def test_mini_benchmark_end_to_end(tmp_path: Path) -> None:
    kinds = ["genuine_trend_ma", "no_edge_trend_ma", "scaler_leak", "zero_cost"]
    results, summary = run_benchmark("dev", seeds=[0], kinds=kinds)
    assert [r.truth for r in results] == ["genuine", "no_edge", "scaler_leak", "zero_cost"]
    assert all(r.correct for r in results)
    assert summary == summarise(results)
    write_outputs(results, summary, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"cases.jsonl", "summary.json", "summary.md"}
    assert "flaw conviction rate" in markdown(summary)
