"""Tool registry: the menu the Prosecutor picks from and the single entry point for running tools.

Every tool takes a `TrialContext` and returns `Evidence`. Missing inputs and tool failures never
raise into the orchestrator and are never silent: they produce *inconclusive* evidence that
states what was missing or what failed.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from strategy_court.audit import (
    ast_scanner,
    bootstrap,
    causality,
    claim_consistency,
    dsr,
    haircut,
    pbo,
    placebo,
    psr,
    purged_cv,
    robustness,
    snooping,
    stability,
)
from strategy_court.audit._common import clean, inconclusive
from strategy_court.audit.causality import SignalFn
from strategy_court.audit.citations import ARNOTT_HARVEY_MARKOWITZ_2019, KAUFMAN_2012
from strategy_court.audit.robustness import neighbourhood_grid
from strategy_court.audit.trials import Params, StrategyFactory, TrialSet, build_trial_set
from strategy_court.backtest import BacktestConfig, BacktestResult, MarketData, run_backtest
from strategy_court.schemas import Citation, Evidence, Hypothesis

Requirement = Literal["market", "signal_fn", "source", "hypothesis", "trials"]
Cost = Literal["cheap", "moderate", "expensive"]


@dataclass
class TrialContext:
    """Everything a tool may need. Derived quantities are computed once and cached."""

    strategy_id: str
    market: MarketData | None = None
    config: BacktestConfig = field(default_factory=BacktestConfig)
    signal_fn: SignalFn | None = None
    source_code: str | None = None
    hypothesis: Hypothesis | None = None
    factory: StrategyFactory | None = None
    params: Params | None = None
    trials: TrialSet | None = None
    extra_trials: int = 0  # tried but not in `trials` (e.g. K replication interpretations - 1)
    replicated_sharpes: Sequence[float] | None = None
    benchmark_returns: np.ndarray | None = None
    honest_rerun: bool = False
    options: dict[str, dict[str, object]] = field(default_factory=dict)
    _weights: pd.DataFrame | None = field(default=None, repr=False)
    _result: BacktestResult | None = field(default=None, repr=False)

    def has(self, req: Requirement) -> bool:
        if req == "trials":
            return self.trials is not None or (
                self.factory is not None and self.params is not None and self.market is not None
            )
        attr = {"source": "source_code"}.get(req, req)
        return getattr(self, attr) is not None

    @property
    def weights(self) -> pd.DataFrame:
        if self._weights is None:
            assert self.signal_fn is not None and self.market is not None
            self._weights = self.signal_fn(self.market.close.copy())
        return self._weights

    @property
    def result(self) -> BacktestResult:
        if self._result is None:
            assert self.market is not None
            self._result = run_backtest(self.weights, self.market, self.config)
        return self._result

    @property
    def returns(self) -> pd.Series:
        return self.result.returns

    def trial_set(self) -> TrialSet:
        if self.trials is None:
            assert self.factory is not None and self.params is not None and self.market is not None
            grid = neighbourhood_grid(self.params)
            self.trials = build_trial_set(self.factory, grid, self.market, self.config, selected=0)
        return self.trials

    def opts(self, tool: str) -> dict[str, object]:
        return dict(self.options.get(tool, {}))


@dataclass(frozen=True)
class ToolSpec:
    name: str
    section: str  # numbering in PROJECT_PLAN.md
    title: str
    description: str  # shown to the Prosecutor
    requires: frozenset[Requirement]
    citations: tuple[Citation, ...]
    cost: Cost
    always_on: bool
    run: Callable[[TrialContext], Evidence]

    def catalogue_entry(self) -> dict[str, object]:
        return {
            "name": self.name,
            "section": self.section,
            "title": self.title,
            "description": self.description,
            "requires": sorted(self.requires),
            "cost": self.cost,
            "always_on": self.always_on,
            "citations": [c.short() for c in self.citations],
        }


def _causality(ctx: TrialContext) -> Evidence:
    assert ctx.signal_fn is not None and ctx.market is not None
    report = causality.causality_test(ctx.signal_fn, ctx.market.close, **ctx.opts("causality"))  # type: ignore[arg-type]
    ev = report.to_evidence(ctx.strategy_id)
    if ctx.honest_rerun and report.status == "fail":
        rerun = causality.causal_rerun(ctx.signal_fn, ctx.market, ctx.config)
        details = clean({**ev.details, "honest_rerun": rerun.as_dict()})
        assert isinstance(details, dict)
        ev = ev.model_copy(update={"details": details})
    return ev


def _trials_tool(fn: Callable[..., Evidence], name: str) -> Callable[[TrialContext], Evidence]:
    def run(ctx: TrialContext) -> Evidence:
        kw = ctx.opts(name)
        if name in ("deflated_sharpe", "haircut_sharpe"):
            kw.setdefault("extra_trials", ctx.extra_trials)
        if name == "data_snooping" and ctx.benchmark_returns is not None:
            kw.setdefault("benchmark_returns", ctx.benchmark_returns)
        return fn(ctx.trial_set(), ctx.strategy_id, **kw)

    return run


def _returns_tool(fn: Callable[..., Evidence], name: str) -> Callable[[TrialContext], Evidence]:
    def run(ctx: TrialContext) -> Evidence:
        kw = ctx.opts(name)
        if name == "bootstrap_sharpe" and ctx.benchmark_returns is not None:
            kw.setdefault("benchmark_returns", ctx.benchmark_returns)
        data: object = ctx.returns if name == "subperiod_stability" else ctx.returns.to_numpy()
        return fn(data, ctx.strategy_id, **kw)

    return run


def _weights_tool(fn: Callable[..., Evidence], name: str) -> Callable[[TrialContext], Evidence]:
    def run(ctx: TrialContext) -> Evidence:
        assert ctx.market is not None
        kw = ctx.opts(name)
        kw.setdefault("config", ctx.config)
        return fn(ctx.weights, ctx.market, ctx.strategy_id, **kw)

    return run


def _parameters(ctx: TrialContext) -> Evidence:
    if ctx.factory is None or ctx.params is None:
        return inconclusive(
            robustness.PARAM_TOOL,
            ctx.strategy_id,
            "no strategy factory / parameters provided",
            robustness.PARAM_CITATIONS,
        )
    assert ctx.market is not None
    return robustness.parameter_evidence(
        ctx.factory, ctx.params, ctx.market, ctx.strategy_id, ctx.config
    )


def _ast(ctx: TrialContext) -> Evidence:
    assert ctx.source_code is not None
    return ast_scanner.scan_source(ctx.source_code).to_evidence(ctx.strategy_id)


def _claim(ctx: TrialContext) -> Evidence:
    assert ctx.hypothesis is not None
    return claim_consistency.claim_evidence(ctx.hypothesis, ctx.strategy_id, ctx.replicated_sharpes)


def _spec(
    name: str,
    section: str,
    title: str,
    description: str,
    requires: set[Requirement],
    citations: tuple[Citation, ...],
    cost: Cost,
    run: Callable[[TrialContext], Evidence],
    always_on: bool = False,
) -> ToolSpec:
    return ToolSpec(
        name, section, title, description, frozenset(requires), citations, cost, always_on, run
    )


TOOLS: dict[str, ToolSpec] = {
    s.name: s
    for s in (
        _spec(
            "causality",
            "1",
            "Causality (look-ahead) test",
            "Re-runs the strategy on truncated, mirrored and noise-perturbed futures; any change "
            "to past signals proves look-ahead. Optional honest point-in-time re-run.",
            {"market", "signal_fn"},
            (KAUFMAN_2012, ARNOTT_HARVEY_MARKOWITZ_2019),
            "moderate",
            _causality,
            True,
        ),
        _spec(
            "ast_scanner",
            "2",
            "Static look-ahead and safety scan",
            "Parses the source for negative shifts, centred windows, backfills, full-sample "
            "statistics/fits, same-bar P&L, unseeded randomness, hard-coded dates, unsafe code.",
            {"source"},
            ast_scanner.CITATIONS,
            "cheap",
            _ast,
            True,
        ),
        _spec(
            "purged_cv",
            "3",
            "Purged & embargoed CV of parameter selection",
            "Cross-validates the choice of configuration with purging, embargo and CPCV paths; "
            "reports out-of-sample Sharpe of the selection procedure.",
            {"trials"},
            purged_cv.CITATIONS,
            "moderate",
            _trials_tool(purged_cv.purged_cv_evidence, "purged_cv"),
        ),
        _spec(
            "psr",
            "4",
            "Probabilistic Sharpe ratio & minimum track record",
            "Probability the true Sharpe exceeds a benchmark given length, skew and kurtosis; "
            "years of data needed to be significant.",
            {"market", "signal_fn"},
            psr.CITATIONS,
            "cheap",
            _returns_tool(psr.psr_evidence, "psr"),
        ),
        _spec(
            "deflated_sharpe",
            "5",
            "Deflated Sharpe ratio",
            "Corrects the Sharpe for the number and dispersion of trials (selection bias), "
            "with raw and effective trial counts.",
            {"trials"},
            dsr.CITATIONS,
            "cheap",
            _trials_tool(dsr.dsr_evidence, "deflated_sharpe"),
        ),
        _spec(
            "pbo",
            "6",
            "Probability of backtest overfitting (CSCV)",
            "How often the in-sample best configuration ranks below median out-of-sample across "
            "all combinatorially symmetric splits.",
            {"trials"},
            pbo.CITATIONS,
            "moderate",
            _trials_tool(pbo.pbo_evidence, "pbo"),
        ),
        _spec(
            "haircut_sharpe",
            "7",
            "Haircut Sharpe ratio",
            "Multiple-testing-adjusted Sharpe (Bonferroni, Holm, BHY) with Lo's "
            "serial-correlation-corrected annualisation.",
            {"trials"},
            haircut.CITATIONS,
            "cheap",
            _trials_tool(haircut.haircut_evidence, "haircut_sharpe"),
        ),
        _spec(
            "placebo_null",
            "8",
            "Turnover-matched placebo null",
            "Compares the Sharpe with placebos that keep exposure, turnover and holding periods "
            "but break alignment with prices (circular shift / run-block permutation).",
            {"market", "signal_fn"},
            placebo.CITATIONS,
            "expensive",
            _weights_tool(placebo.placebo_evidence, "placebo_null"),
        ),
        _spec(
            "bootstrap_sharpe",
            "9",
            "Stationary-bootstrap Sharpe confidence interval",
            "Dependence-robust CI with automatic block length; optional Ledoit-Wolf HAC test "
            "against a benchmark.",
            {"market", "signal_fn"},
            bootstrap.CITATIONS,
            "moderate",
            _returns_tool(bootstrap.bootstrap_evidence, "bootstrap_sharpe"),
        ),
        _spec(
            "cost_sensitivity",
            "10a",
            "Cost sweep & execution-delay decay",
            "Break-even transaction cost and how fast the edge decays with execution delay.",
            {"market", "signal_fn"},
            robustness.COST_CITATIONS,
            "moderate",
            _weights_tool(robustness.cost_evidence, "cost_sensitivity"),
        ),
        _spec(
            "parameter_sensitivity",
            "10b",
            "Parameter neighbourhood",
            "Plateau or isolated peak: Sharpe of neighbouring parameter values.",
            {"market", "trials"},
            robustness.PARAM_CITATIONS,
            "moderate",
            _parameters,
        ),
        _spec(
            "subperiod_stability",
            "11",
            "Subperiod stability & structural break",
            "Sharpe by subperiod and year, HAC equal-means test and Andrews sup-Wald break test.",
            {"market", "signal_fn"},
            stability.CITATIONS,
            "cheap",
            _returns_tool(stability.stability_evidence, "subperiod_stability"),
        ),
        _spec(
            "claim_consistency",
            "12",
            "Claim consistency",
            "Checks the claim itself: Kelly growth bound, drawdown plausibility, minimum track "
            "record, digitised-curve and replication consistency.",
            {"hypothesis"},
            claim_consistency.CITATIONS,
            "cheap",
            _claim,
            True,
        ),
        _spec(
            "data_snooping",
            "13",
            "Reality Check & SPA across trials",
            "White's Reality Check and Hansen's SPA: can any tried configuration beat the "
            "benchmark beyond data-snooping luck?",
            {"trials"},
            snooping.CITATIONS,
            "expensive",
            _trials_tool(snooping.snooping_evidence, "data_snooping"),
        ),
    )
}


def list_tools() -> list[ToolSpec]:
    return list(TOOLS.values())


def catalogue() -> list[dict[str, object]]:
    """JSON-able tool menu for the Prosecutor prompt and the UI."""
    return [t.catalogue_entry() for t in TOOLS.values()]


def applicable(ctx: TrialContext) -> list[ToolSpec]:
    return [t for t in TOOLS.values() if all(ctx.has(r) for r in t.requires)]


def run_tool(name: str, ctx: TrialContext) -> Evidence:
    spec = TOOLS.get(name)
    if spec is None:
        raise KeyError(f"unknown tool '{name}'; available: {sorted(TOOLS)}")
    missing = sorted(r for r in spec.requires if not ctx.has(r))
    if missing:
        return inconclusive(
            name, ctx.strategy_id, f"missing inputs: {', '.join(missing)}", spec.citations
        )
    try:
        return spec.run(ctx)
    except Exception as exc:  # recorded as evidence, never swallowed
        return inconclusive(
            name, ctx.strategy_id, f"tool failed: {type(exc).__name__}: {exc}"[:300], spec.citations
        )


def run_tools(names: Sequence[str], ctx: TrialContext) -> list[Evidence]:
    return [run_tool(n, ctx) for n in names]


def always_on(ctx: TrialContext) -> list[Evidence]:
    """Arraignment: the tools run on every case that has their inputs."""
    return [
        run_tool(t.name, ctx)
        for t in TOOLS.values()
        if t.always_on and all(ctx.has(r) for r in t.requires)
    ]


def tool_citations() -> Mapping[str, tuple[Citation, ...]]:
    return {t.name: t.citations for t in TOOLS.values()}
