"""Ground-truth integrity: each planted flaw really is present and really makes the claim look
better than the honest view; genuine cases really have an out-of-sample edge."""

import numpy as np
import pytest

from strategy_court.audit.ast_scanner import scan_source
from strategy_court.audit.causality import causality_test
from strategy_court.backtest import run_backtest, summarize
from strategy_court.benchmark import strategies
from strategy_court.benchmark.flaws import (
    BUILDERS,
    COURT_CONFIG,
    EXPECTED_TOOLS,
    FLAWS,
    TARGET,
    break_even_check,
    build_cases,
)

SEEDS = (0, 1)


@pytest.fixture(scope="module")
def cases() -> dict[str, list]:  # type: ignore[type-arg]
    out: dict[str, list] = {}  # type: ignore[type-arg]
    for c in build_cases(SEEDS, "test"):
        out.setdefault(c.truth, []).append(c)
    return out


def court_sharpe(case) -> float:  # type: ignore[no-untyped-def]
    res = run_backtest(
        case.factory(case.params).signals(case.market.close.copy()), case.market, COURT_CONFIG
    )
    return summarize(res).sharpe


def test_every_truth_has_target_and_expected_tools() -> None:
    assert set(TARGET) == set(EXPECTED_TOOLS)
    assert set(FLAWS) < set(TARGET)
    assert len(BUILDERS) == 11


def test_templates_compile_and_are_causal_except_planted_leaks() -> None:
    prices = build_cases([0], "t", ["no_edge_trend_ma"])[0].market.close.iloc[:300]
    for name in strategies.TEMPLATES:
        cls = strategies.compile_strategy(name)
        status = causality_test(cls().signals, prices, n_checkpoints=6).status
        leaky = name in ("trend_ma_lookahead", "zscore_full_sample")
        assert status == ("fail" if leaky else "pass"), name
        assert bool(scan_source(strategies.source(name)).by_category("lookahead")) == leaky


def test_leaks_inflate_the_claim(cases: dict[str, list]) -> None:  # type: ignore[type-arg]
    for truth in ("lookahead", "scaler_leak"):
        for c in cases[truth]:
            assert c.claimed.sharpe > 0.8, c.case_id  # an attractive (fake) backtest


def test_zero_cost_claim_needs_free_trading(cases: dict[str, list]) -> None:  # type: ignore[type-arg]
    for c in cases["zero_cost"]:
        assert c.claimed.sharpe > court_sharpe(c)
        assert break_even_check(c) < 10.0  # not implementable at realistic costs


def test_survivorship_claim_beats_point_in_time_view(cases: dict[str, list]) -> None:  # type: ignore[type-arg]
    for c in cases["survivorship"]:
        assert c.claimed.sharpe > court_sharpe(c) + 0.5
        assert len(c.claim.universe) < c.market.shape[1]  # claim used survivors only


def test_best_of_n_is_selected_from_the_disclosed_search(cases: dict[str, list]) -> None:  # type: ignore[type-arg]
    for c in cases["best_of_n_disclosed"]:
        assert c.trials is not None and c.trials.n_trials == 282
        sr = c.trials.sharpes()
        assert int(np.nanargmax(sr)) == c.trials.selected
    assert all(c.trials is None for c in cases["best_of_n_hidden"])


def test_genuine_cases_have_independent_oos_edge(cases: dict[str, list]) -> None:  # type: ignore[type-arg]
    for c in cases.get("genuine", []):
        assert c.oos_truth_sharpe is not None and c.oos_truth_sharpe >= 0.3


def test_case_construction_is_deterministic() -> None:
    a = build_cases([3], "x", ["survivorship", "best_of_n_disclosed"])
    b = build_cases([3], "x", ["survivorship", "best_of_n_disclosed"])
    for x, y in zip(a, b, strict=True):
        assert x.case_id == y.case_id and x.claimed == y.claimed and x.params == y.params
    with pytest.raises(ValueError):
        build_cases([0], "x", ["astrology"])
