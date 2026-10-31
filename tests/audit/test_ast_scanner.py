import importlib.util
import inspect
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

import strategy_court.backtest.strategy as reference_strategies
from strategy_court.audit.ast_scanner import (
    rules_triggered,
    scan_source,
    security_findings,
)
from strategy_court.audit.causality import causality_test
from strategy_court.data import random_walk_prices
from strategy_court.schemas import Severity

FIXTURES = Path(__file__).parent.parent / "fixtures" / "strategies"
PRICES = random_walk_prices(n_days=260, n_assets=3, seed=31)


def source(name: str) -> str:
    return (FIXTURES / f"{name}.py").read_text("utf-8")


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, FIXTURES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # trusted, repo-authored fixture
    return module


EXPECTED = {
    "leak_negative_shift": {"lookahead.negative_shift"},
    "leak_centered_window": {"lookahead.centered_window"},
    "leak_backfill": {"lookahead.backfill"},
    "leak_full_sample_zscore": {"lookahead.full_sample_stat"},
    "leak_scaler_fit": {"lookahead.full_sample_fit"},
    "leak_reverse_time": {"lookahead.reverse_time"},
}


@pytest.mark.parametrize(("name", "rules"), EXPECTED.items())
def test_each_planted_leak_is_flagged(name: str, rules: set[str]) -> None:
    report = scan_source(source(name))
    assert rules <= rules_triggered(report.findings)
    ev = report.to_evidence("s")
    assert ev.flag and ev.severity <= Severity.HIGH  # static => suspicion, never CRITICAL


def test_clean_strategy_has_no_findings() -> None:
    report = scan_source(source("clean_momentum"))
    assert report.findings == ()
    assert not report.to_evidence("s").flag


def test_reference_strategies_have_no_lookahead_findings() -> None:
    report = scan_source(inspect.getsource(reference_strategies))
    assert not report.by_category("lookahead", "execution", "determinism")
    # Trusted court code may import court internals; untrusted strategy code may not (it could
    # monkey-patch the engine judging it), so the security rule rightly flags that import.
    assert {f.message.split("'")[1] for f in report.by_category("security")} == {
        "strategy_court.backtest.config",
        "strategy_court.backtest.engine",
        "strategy_court.backtest.market",
        "strategy_court.schemas.strategy",
    }


@pytest.mark.parametrize("name", [*EXPECTED, "clean_momentum"])
def test_static_and_dynamic_verdicts_agree(name: str) -> None:
    """Every planted leak is flagged statically AND convicted dynamically; the clean fixture
    passes both. This is the corroboration the evidence card shows."""
    fn: Callable[[pd.DataFrame], pd.DataFrame] = load(name).signals
    static_leak = bool(scan_source(source(name)).by_category("lookahead"))
    dynamic_leak = causality_test(fn, PRICES, n_checkpoints=8).status == "fail"
    assert static_leak == dynamic_leak == (name != "clean_momentum")


def test_constant_resolution_and_non_negative_shifts() -> None:
    code = (
        "N = 3\nM = N - 5\na = x.shift(N)\nb = x.shift(M)\nc = x.diff(periods=-1)\nd = x.shift(-k)"
    )
    neg = [f for f in scan_source(code).findings if f.rule == "lookahead.negative_shift"]
    assert sorted(f.line for f in neg) == [4, 5, 6]


def test_aggregation_exemptions() -> None:
    code = (
        "a = df.rolling(5).mean()\nb = df.mean(axis=1)\nc = df.ewm(10).std()\n"
        "d = df.groupby(k).sum()\nrow = df.iloc[0]\ne = row.sum()\nf = df.iloc[:, 0].max()\n"
        "g = df.mean()\n"
    )
    stats = [f.line for f in scan_source(code).findings if f.rule == "lookahead.full_sample_stat"]
    assert stats == [8]


def test_same_bar_pnl_detected_but_lagged_pnl_is_not() -> None:
    findings = scan_source(source("same_bar_pnl")).findings
    lines = [f.line for f in findings if f.rule == "execution.same_bar_pnl"]
    assert lines == [10]


def test_security_rules() -> None:
    rules = [f.rule for f in security_findings(source("unsafe"))]
    assert rules.count("security.forbidden_import") == 4  # os, subprocess, requests, yfinance
    assert rules.count("security.dangerous_call") >= 4  # open, eval, __class__, __bases__, ...
    assert security_findings(source("clean_momentum")) == []
    assert security_findings("def broken(:")[0].rule == "security.parse_error"


def test_determinism_and_hindsight() -> None:
    rules = rules_triggered(scan_source(source("unseeded_and_dates")).findings)
    assert {"determinism.unseeded", "hindsight.hardcoded_date"} <= rules
    seeded = "import numpy as np\nnp.random.seed(1)\nx = np.random.randn(3)"
    assert not scan_source(seeded).findings


def test_indented_snippets_are_dedented() -> None:
    snippet = "    def f(p):\n        return p.shift(-1)\n"
    assert rules_triggered(scan_source(snippet).findings) == {"lookahead.negative_shift"}


def test_parse_error_is_inconclusive() -> None:
    ev = scan_source("def f(:\n  pass").to_evidence("s")
    assert ev.details["status"] == "inconclusive" and not ev.flag
