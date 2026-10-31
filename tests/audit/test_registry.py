import inspect
from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd
import pytest

import strategy_court.backtest.strategy as reference
from strategy_court.audit.registry import (
    TOOLS,
    TrialContext,
    always_on,
    applicable,
    catalogue,
    run_tool,
    run_tools,
)
from strategy_court.backtest import BacktestConfig, MarketData, MovingAverageCrossover
from strategy_court.data import weak_signal_prices
from strategy_court.schemas import ClaimedPerformance, Evidence, Hypothesis, ParamValue, Severity

FAST = {
    "placebo_null": {"n_placebos": 99},
    "bootstrap_sharpe": {"n_boot": 300},
    "data_snooping": {"n_boot": 200},
    "causality": {"n_checkpoints": 6},
}


def mac(params: Mapping[str, ParamValue]) -> MovingAverageCrossover:
    fast, slow = params["fast"], params["slow"]
    assert isinstance(fast, int) and isinstance(slow, int)
    return MovingAverageCrossover(fast, slow)


def full_context(signal_fn: object = None, strategy_id: str = "mac") -> TrialContext:
    prices = weak_signal_prices(n_days=700, n_assets=3, ar1=0.05, seed=9)
    return TrialContext(
        strategy_id=strategy_id,
        market=MarketData(prices),
        config=BacktestConfig(),
        signal_fn=signal_fn or MovingAverageCrossover(10, 40).signals,  # type: ignore[arg-type]
        source_code=inspect.getsource(reference.MovingAverageCrossover),
        hypothesis=Hypothesis(
            source_type="text",
            universe=["SYN000"],
            signal_description="10/40 moving-average crossover",
            claimed=ClaimedPerformance(
                sharpe=2.5, cagr=0.4, period_start=date(2010, 1, 4), period_end=date(2012, 10, 1)
            ),
        ),
        factory=mac,
        params={"fast": 10, "slow": 40},
        options=FAST,  # type: ignore[arg-type]
    )


def test_catalogue_is_complete_and_cited() -> None:
    sections = {t.section for t in TOOLS.values()}
    assert sections == {*(str(i) for i in range(1, 14)), "10a", "10b"} - {"10"}
    assert len(TOOLS) == 14
    for entry in catalogue():
        assert entry["citations"] and entry["description"]
    assert {t.name for t in TOOLS.values() if t.always_on} == {
        "causality",
        "ast_scanner",
        "claim_consistency",
    }


def test_applicability_follows_available_inputs() -> None:
    ctx = TrialContext(strategy_id="s", source_code="x = 1")
    assert [t.name for t in applicable(ctx)] == ["ast_scanner"]
    assert len(applicable(full_context())) == 14


def test_every_tool_runs_end_to_end_and_serialises() -> None:
    ctx = full_context()
    evidence = run_tools(list(TOOLS), ctx)
    assert [e.tool for e in evidence] == list(TOOLS)
    # The court's own neighbourhood is a *probe*, not the claimant's search: with no disclosed
    # trial count, selection-bias deflation is explicitly not applicable.
    undisclosed = {"deflated_sharpe", "haircut_sharpe", "data_snooping"}
    for ev in evidence:
        assert (ev.details.get("status") == "inconclusive") == (ev.tool in undisclosed), ev
        assert Evidence.model_validate_json(ev.model_dump_json()) == ev
        assert ev.citations
    ids = [e.id for e in evidence]
    assert len(set(ids)) == len(ids)
    ctx.extra_trials = 20  # claimant discloses the size (not the returns) of a 21-config search
    for ev in run_tools(["deflated_sharpe", "haircut_sharpe"], ctx):
        assert ev.details.get("status") != "inconclusive"
        assert ev.details["n_trials"] == 21


def test_leaky_strategy_is_convicted_with_honest_rerun() -> None:
    def peek(prices: pd.DataFrame) -> pd.DataFrame:
        return np.sign(prices.pct_change(fill_method=None).shift(-2)).fillna(0.0) / prices.shape[1]

    ctx = full_context(peek, "peek")
    ctx.source_code = inspect.getsource(peek)
    ctx.honest_rerun = True
    causal, scan = run_tool("causality", ctx), run_tool("ast_scanner", ctx)
    assert causal.severity is Severity.CRITICAL and causal.flag
    rerun = causal.details["honest_rerun"]
    assert isinstance(rerun, dict) and rerun["sharpe_as_submitted"] > 3  # type: ignore[operator]
    assert scan.flag and "lookahead.negative_shift" in scan.details["rules_triggered"]  # type: ignore[operator]


def test_missing_inputs_and_failures_become_inconclusive_evidence() -> None:
    bare = TrialContext(strategy_id="s")
    ev = run_tool("pbo", bare)
    assert ev.details["status"] == "inconclusive" and "trials" in str(ev.details["reason"])

    def broken(prices: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("boom")

    failing = run_tool("psr", full_context(broken))
    assert failing.details["status"] == "inconclusive" and "boom" in str(failing.details["reason"])
    with pytest.raises(KeyError):
        run_tool("astrology", bare)


def test_always_on_arraignment() -> None:
    names = [e.tool for e in always_on(full_context())]
    assert names == ["causality", "ast_scanner", "claim_consistency"]
