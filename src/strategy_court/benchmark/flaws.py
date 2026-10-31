"""Benchmark cases: clean strategies and planted flaws with ground truth fixed *before* the court
runs.

Ground-truth classes and what the court should conclude
-------------------------------------------------------
genuine             clean rule on data with a real, persistent edge, confirmed by an independent
                    20-year out-of-sample simulation net of costs      -> must NOT be convicted
no_edge             clean rule on zero-alpha random-walk data          -> must NOT be acquitted
lookahead           2-bar peek blended into a trend rule               -> must be convicted
scaler_leak         z-score normalised with full-sample moments        -> must be convicted
best_of_n_disclosed best of a 282-configuration trend/contrarian moving-average
                    search on zero-alpha data, disclosed as the trial set -> must be convicted
best_of_n_hidden    same search, not disclosed (court builds its own
                    neighbourhood)                                      -> must be convicted
zero_cost           high-turnover rule whose claim ignores costs and
                    whose break-even cost is below 10 bps             -> must be convicted
survivorship        'buy the losers' claimed on a survivor-only
                    universe; the court sees the point-in-time universe -> must be convicted

Design notes (D13): the engine fills at t+1, so planted leaks peek two bars ahead and planted
signals are persistent; otherwise the "flawed" backtest would not even look attractive.
The court and every ground-truth check use one cost assumption: 10 bps one-way.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from strategy_court.audit.trials import Params, StrategyFactory, TrialSet, build_trial_set
from strategy_court.backtest import (
    BacktestConfig,
    CostModel,
    MarketData,
    PerformanceSummary,
    run_backtest,
    summarize,
)
from strategy_court.benchmark import strategies
from strategy_court.data import (
    random_walk_prices,
    survivorship_universe,
    trending_signal_prices,
    weak_signal_prices,
)
from strategy_court.schemas import ClaimedPerformance, Hypothesis, SourceType

Truth = Literal[
    "genuine",
    "no_edge",
    "lookahead",
    "scaler_leak",
    "best_of_n_disclosed",
    "best_of_n_hidden",
    "zero_cost",
    "survivorship",
]
Target = Literal["convicted", "not_acquitted", "not_convicted"]
FLAWS: tuple[Truth, ...] = (
    "lookahead",
    "scaler_leak",
    "best_of_n_disclosed",
    "best_of_n_hidden",
    "zero_cost",
    "survivorship",
)
TARGET: dict[Truth, Target] = {
    "genuine": "not_convicted",
    "no_edge": "not_acquitted",
    **dict.fromkeys(FLAWS, "convicted"),
}
SELECTION_TOOLS = frozenset(
    {
        "deflated_sharpe",
        "pbo",
        "haircut_sharpe",
        "purged_cv",
        "data_snooping",
        "parameter_sensitivity",
    }
)
EXPECTED_TOOLS: dict[Truth, frozenset[str]] = {
    "genuine": frozenset(),
    "no_edge": frozenset({"psr", "placebo_null", "bootstrap_sharpe"}),
    "lookahead": frozenset({"causality"}),
    "scaler_leak": frozenset({"causality"}),
    "best_of_n_disclosed": SELECTION_TOOLS,
    "best_of_n_hidden": SELECTION_TOOLS,
    "zero_cost": frozenset({"cost_sensitivity"}),
    "survivorship": frozenset({"claim_consistency"}),
}

N_DAYS = 1260  # five years of daily bars
N_ASSETS = 5
REALISTIC_BPS = 10.0  # one-way, the cost tool's default threshold (Novy-Marx & Velikov 2016)
COURT_CONFIG = BacktestConfig(
    costs=CostModel(commission_bps=2.0, half_spread_bps=REALISTIC_BPS - 2.0),
    delisting_return=-0.3,  # Shumway (1997) performance-delisting return
)
ZERO_COST = COURT_CONFIG.model_copy(update={"costs": CostModel.zero()})
SEARCH_GRID: tuple[tuple[int, int, int], ...] = tuple(
    (f, s, d) for f in range(5, 65, 5) for s in range(20, 280, 20) if f < s for d in (1, -1)
)  # trend *and* contrarian moving-average search, 282 configurations


@dataclass(frozen=True)
class Case:
    case_id: str
    truth: Truth
    seed: int
    split: str
    template: str
    params: Params
    market: MarketData  # what the court audits on (honest, point-in-time)
    claim: Hypothesis
    claimed: PerformanceSummary
    trials: TrialSet | None = None
    extra_trials: int = 0
    oos_truth_sharpe: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def source(self) -> str:
        return strategies.source(self.template)

    @property
    def factory(self) -> StrategyFactory:
        return strategies.factory(self.template)

    @property
    def target(self) -> Target:
        return TARGET[self.truth]


def _claim(name: str, market: MarketData, summary: PerformanceSummary) -> Hypothesis:
    idx = market.index
    return Hypothesis(
        source_type=SourceType.CODE,
        universe=market.assets,
        signal_description=name.replace("_", " "),
        claimed=ClaimedPerformance(
            sharpe=summary.sharpe if math.isfinite(summary.sharpe) else 0.0,
            cagr=max(summary.cagr, -0.99) if math.isfinite(summary.cagr) else 0.0,
            max_drawdown=min(max(summary.max_drawdown, 0.0), 1.0),
            period_start=idx[0].date(),
            period_end=idx[-1].date(),
        ),
    )


def _backtest(
    template: str, params: Params, market: MarketData, cfg: BacktestConfig
) -> PerformanceSummary:
    strat = strategies.factory(template)(params)
    return summarize(run_backtest(strat.signals(market.close.copy()), market, cfg))


def _case(
    truth: Truth,
    seed: int,
    split: str,
    template: str,
    params: Params,
    court_market: MarketData,
    claim_market: MarketData | None = None,
    claim_config: BacktestConfig = COURT_CONFIG,
    **kw: object,
) -> Case:
    claimed = _backtest(template, params, claim_market or court_market, claim_config)
    return Case(
        case_id=f"{split}-{seed}-{truth}-{template}",
        truth=truth,
        seed=seed,
        split=split,
        template=template,
        params=params,
        market=court_market,
        claim=_claim(template, claim_market or court_market, claimed),
        claimed=claimed,
        **kw,  # type: ignore[arg-type]
    )


def genuine(seed: int, split: str, template: str = "trend_ma") -> Case:
    params: Params = {"fast": 20, "slow": 80} if template == "trend_ma" else {"lookback": 40}
    market = MarketData(trending_signal_prices(N_DAYS, N_ASSETS, seed=seed))
    fresh = MarketData(trending_signal_prices(20 * 252, N_ASSETS, seed=seed + 10_000))
    oos = _backtest(template, params, fresh, COURT_CONFIG).sharpe
    truth: Truth = "genuine" if oos >= 0.3 else "no_edge"
    note = (
        ()
        if truth == "genuine"
        else (f"independent OOS Sharpe {oos:.2f} < 0.3: relabelled no_edge",)
    )
    return _case(truth, seed, split, template, params, market, oos_truth_sharpe=oos, notes=note)


def no_edge(seed: int, split: str, template: str) -> Case:
    defaults: dict[str, Params] = {
        "trend_ma": {"fast": 20, "slow": 80},
        "time_momentum": {"lookback": 60},
        "loser_reversal": {"lookback": 20, "quantile": 0.2},
    }
    market = MarketData(random_walk_prices(N_DAYS, N_ASSETS, seed=seed))
    return _case("no_edge", seed, split, template, defaults[template], market, oos_truth_sharpe=0.0)


def lookahead(seed: int, split: str) -> Case:
    market = MarketData(random_walk_prices(N_DAYS, N_ASSETS, seed=seed + 1))
    return _case("lookahead", seed, split, "trend_ma_lookahead", {"fast": 20, "slow": 80}, market)


def scaler_leak(seed: int, split: str) -> Case:
    market = MarketData(random_walk_prices(N_DAYS, N_ASSETS, seed=seed + 2))
    return _case("scaler_leak", seed, split, "zscore_full_sample", {"band": 0.5}, market)


def best_of_n(seed: int, split: str, disclosed: bool) -> Case:
    market = MarketData(random_walk_prices(N_DAYS, N_ASSETS, seed=seed + 3))
    grid: list[Params] = [{"fast": f, "slow": s, "direction": d} for f, s, d in SEARCH_GRID]
    search = build_trial_set(strategies.factory("trend_ma"), grid, market, COURT_CONFIG)
    params = search.params[search.selected]
    truth: Truth = "best_of_n_disclosed" if disclosed else "best_of_n_hidden"
    return _case(
        truth,
        seed,
        split,
        "trend_ma",
        params,
        market,
        trials=search if disclosed else None,
        oos_truth_sharpe=0.0,
    )


def zero_cost(seed: int, split: str) -> Case:
    market = MarketData(weak_signal_prices(N_DAYS, N_ASSETS, ar1=0.25, seed=seed + 4))
    return _case(
        "zero_cost", seed, split, "time_momentum", {"lookback": 1}, market, claim_config=ZERO_COST
    )


def survivorship(seed: int, split: str) -> Case:
    universe = survivorship_universe(N_DAYS, n_assets=50, seed=seed + 5)
    honest = MarketData(universe.prices)
    biased = MarketData(universe.survivor_prices)
    return _case(
        "survivorship",
        seed,
        split,
        "loser_reversal",
        {"lookback": 20, "quantile": 0.2},
        honest,
        claim_market=biased,
        oos_truth_sharpe=0.0,
        notes=(f"{len(universe.delisted)} of 50 names delisted; claim uses survivors only",),
    )


Builder = Callable[[int, str], Case]
BUILDERS: dict[str, Builder] = {
    "genuine_trend_ma": lambda s, sp: genuine(s, sp, "trend_ma"),
    "genuine_time_momentum": lambda s, sp: genuine(s, sp, "time_momentum"),
    "no_edge_trend_ma": lambda s, sp: no_edge(s, sp, "trend_ma"),
    "no_edge_time_momentum": lambda s, sp: no_edge(s, sp, "time_momentum"),
    "no_edge_loser_reversal": lambda s, sp: no_edge(s, sp, "loser_reversal"),
    "lookahead": lookahead,
    "scaler_leak": scaler_leak,
    "best_of_n_disclosed": lambda s, sp: best_of_n(s, sp, True),
    "best_of_n_hidden": lambda s, sp: best_of_n(s, sp, False),
    "zero_cost": zero_cost,
    "survivorship": survivorship,
}


def build_cases(seeds: Sequence[int], split: str, kinds: Sequence[str] | None = None) -> list[Case]:
    names = list(kinds) if kinds is not None else list(BUILDERS)
    unknown = set(names) - set(BUILDERS)
    if unknown:
        raise ValueError(f"unknown case kinds: {sorted(unknown)}")
    return [BUILDERS[k](seed, split) for seed in seeds for k in names]


def break_even_check(case: Case) -> float:
    """Ground-truth helper for zero_cost: annual turnover x 1 bp costs vs gross return."""
    strat = case.factory(case.params)
    res = run_backtest(strat.signals(case.market.close.copy()), case.market, ZERO_COST)
    turn = float(res.turnover.iloc[1:].mean())
    return float(res.returns.mean() / turn * 1e4) if turn > 0 else float("inf")


__all__ = [
    "BUILDERS",
    "COURT_CONFIG",
    "EXPECTED_TOOLS",
    "FLAWS",
    "TARGET",
    "Case",
    "Target",
    "Truth",
    "break_even_check",
    "build_cases",
]
