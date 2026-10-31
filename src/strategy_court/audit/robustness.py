"""Execution and parameter robustness.

Cost sensitivity (tool "cost_sensitivity")
* Cost sweep: re-run the backtest at increasing one-way linear costs (bps). The break-even cost
  is where the Sharpe ratio crosses zero; to first order it is also
  mean gross return per bar / mean turnover per bar (linear costs scale with turnover),
  reported as an analytic cross-check. Strategies whose break-even is below realistic costs
  for the traded universe (default 10 bps one-way; cost levels by anomaly in Novy-Marx &
  Velikov 2016) are not implementable.
* Execution-delay decay: shift the weights by d extra bars. A fast-decaying edge is extremely
  sensitive to execution timing; an exponential fit SR(d) = SR(0) exp(-k d) gives a half-life.

Parameter sensitivity (tool "parameter_sensitivity")
* Re-run on a neighbourhood of the chosen parameters (relative steps for numeric parameters).
  A genuine effect sits on a plateau; an overfit one is an isolated peak whose neighbours are
  much worse (cf. the overfitting discussion in Bailey et al. 2014).
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_court.audit._common import inconclusive, make_evidence
from strategy_court.audit.citations import BBLZ_2014, NOVY_MARX_VELIKOV_2016
from strategy_court.audit.trials import Params, StrategyFactory, build_trial_set
from strategy_court.backtest import BacktestConfig, MarketData, simulate, validate_weights
from strategy_court.schemas import Evidence, ParamValue, Severity

COST_TOOL = "cost_sensitivity"
PARAM_TOOL = "parameter_sensitivity"
COST_CITATIONS = (NOVY_MARX_VELIKOV_2016,)
PARAM_CITATIONS = (BBLZ_2014,)
DEFAULT_BPS = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0)
DEFAULT_DELAYS = (0, 1, 2, 3, 5, 10)


def _sharpe_rows(returns: np.ndarray, cfg: BacktestConfig) -> np.ndarray:
    x = returns[:, 1:] - cfg.rf_per_period
    sd = x.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 1e-12, x.mean(axis=1) / sd * math.sqrt(cfg.periods_per_year), 0.0)


def _with_linear_bps(cfg: BacktestConfig, bps: float) -> BacktestConfig:
    costs = cfg.costs.model_copy(update={"commission_bps": 0.0, "half_spread_bps": bps})
    return cfg.model_copy(update={"costs": costs})


def _crossing(x: Sequence[float], y: Sequence[float]) -> float:
    """First x where y crosses from > 0 to <= 0 (linear interpolation); inf if never; 0 if
    y[0] <= 0."""
    if y[0] <= 0:
        return 0.0
    for (x0, y0), (x1, y1) in itertools.pairwise(zip(x, y, strict=True)):
        if y1 <= 0 < y0:
            return x0 + (x1 - x0) * y0 / (y0 - y1)
    return float("inf")


@dataclass(frozen=True)
class CostSensitivityResult:
    bps: tuple[float, ...]
    sharpe: tuple[float, ...]
    annual_return: tuple[float, ...]
    break_even_bps: float
    break_even_bps_analytic: float
    annual_turnover: float
    delays: tuple[int, ...]
    delay_sharpe: tuple[float, ...]
    half_life_bars: float
    realistic_bps: float

    def to_evidence(self, strategy_id: str) -> Evidence:
        be = self.break_even_bps
        decays = (
            len(self.delay_sharpe) > 1
            and self.delay_sharpe[0] > 0
            and self.delay_sharpe[1] < 0.5 * self.delay_sharpe[0]
        )
        if be < self.realistic_bps:
            sev = Severity.HIGH
        elif be < 2 * self.realistic_bps or decays:
            sev = Severity.MEDIUM
        else:
            sev = Severity.INFO
        be_txt = "never (within the sweep)" if math.isinf(be) else f"{be:.1f} bps"
        delayed = self.delay_sharpe[1] if len(self.delay_sharpe) > 1 else float("nan")
        return make_evidence(
            COST_TOOL,
            strategy_id,
            flag=sev >= Severity.MEDIUM,
            severity=sev,
            statistic_name="break_even_bps",
            statistic=be if math.isfinite(be) else None,
            summary=(
                f"Edge disappears at one-way costs of {be_txt} (realistic: "
                f"{self.realistic_bps:.0f} bps; turnover {self.annual_turnover:.0f}x/yr); "
                f"Sharpe with one extra bar of delay: {delayed:.2f}."
            ),
            details={
                "break_even_bps": be,
                "break_even_bps_analytic": self.break_even_bps_analytic,
                "realistic_bps": self.realistic_bps,
                "annual_turnover": self.annual_turnover,
                "sharpe_at_bps": dict(zip([str(b) for b in self.bps], self.sharpe, strict=True)),
                "annual_return_at_bps": dict(
                    zip([str(b) for b in self.bps], self.annual_return, strict=True)
                ),
                "sharpe_at_delay": dict(
                    zip([str(d) for d in self.delays], self.delay_sharpe, strict=True)
                ),
                "delay_half_life_bars": self.half_life_bars,
            },
            plot_data={
                "bps": self.bps,
                "sharpe": self.sharpe,
                "delays": self.delays,
                "delay_sharpe": self.delay_sharpe,
            },
            citations=COST_CITATIONS,
        )


def _half_life(delays: Sequence[int], sharpe: Sequence[float]) -> float:
    pts = [(d, s) for d, s in zip(delays, sharpe, strict=True) if s > 0]
    if len(pts) < 2 or pts[0][0] != 0:
        return float("nan") if sharpe[0] <= 0 else 0.0
    d = np.array([p[0] for p in pts], dtype=float)
    y = np.log([p[1] for p in pts])
    slope = float(np.polyfit(d, y, 1)[0])
    return math.log(2) / -slope if slope < 0 else float("inf")


def cost_sensitivity(
    weights: pd.DataFrame,
    market: MarketData,
    config: BacktestConfig | None = None,
    bps: Sequence[float] = DEFAULT_BPS,
    delays: Sequence[int] = DEFAULT_DELAYS,
    realistic_bps: float = 10.0,
) -> CostSensitivityResult:
    cfg = config or BacktestConfig()
    w = validate_weights(weights, market)
    sharpe, ann_ret = [], []
    gross = None
    for level in bps:
        c = _with_linear_bps(cfg, level)
        res = simulate(w, market, c)
        sharpe.append(float(_sharpe_rows(res.returns, c)[0]))
        ann_ret.append(float(res.returns[0, 1:].mean() * cfg.periods_per_year))
        if level == 0:
            gross = res
    if gross is None:
        gross = simulate(w, market, _with_linear_bps(cfg, 0.0))
    mean_turn = float(gross.turnover[0, 1:].mean())
    analytic = (
        float(gross.returns[0, 1:].mean() / mean_turn * 1e4) if mean_turn > 0 else float("inf")
    )
    shifted = np.stack(
        [np.vstack([np.full((d, w.shape[1]), np.nan), w[: len(w) - d]]) for d in delays]
    )
    delay_sr = tuple(float(x) for x in _sharpe_rows(simulate(shifted, market, cfg).returns, cfg))
    return CostSensitivityResult(
        bps=tuple(float(b) for b in bps),
        sharpe=tuple(sharpe),
        annual_return=tuple(ann_ret),
        break_even_bps=_crossing(list(bps), sharpe),
        break_even_bps_analytic=max(analytic, 0.0),
        annual_turnover=mean_turn * cfg.periods_per_year,
        delays=tuple(int(d) for d in delays),
        delay_sharpe=delay_sr,
        half_life_bars=_half_life(delays, delay_sr),
        realistic_bps=realistic_bps,
    )


def cost_evidence(
    weights: pd.DataFrame, market: MarketData, strategy_id: str, **kw: object
) -> Evidence:
    w = validate_weights(weights, market)
    if not np.isfinite(w).any() or np.nansum(np.abs(w)) == 0:
        return inconclusive(
            COST_TOOL, strategy_id, "strategy never takes a position", COST_CITATIONS
        )
    return cost_sensitivity(weights, market, **kw).to_evidence(strategy_id)  # type: ignore[arg-type]


# ------------------------------------------------------------------ parameter neighbourhood


def _is_numeric(v: ParamValue) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def neighbourhood_grid(
    base: Params, rel_steps: Sequence[float] = (-0.3, -0.15, 0.0, 0.15, 0.3), max_points: int = 125
) -> list[dict[str, ParamValue]]:
    """Product grid over numeric params (ints rounded, duplicates removed); falls back to
    one-at-a-time sweeps when the product exceeds `max_points`. Base point is first."""

    def values(v: int | float) -> list[ParamValue]:
        raw = [v * (1 + s) for s in rel_steps]
        out: list[ParamValue] = [round(x) for x in raw] if isinstance(v, int) else list(raw)
        return list(dict.fromkeys(out))

    grids = {k: values(v) for k, v in base.items() if _is_numeric(v) and not isinstance(v, str)}
    points: list[dict[str, ParamValue]] = []
    if math.prod(len(g) for g in grids.values()) <= max_points:
        for combo in itertools.product(*grids.values()):
            points.append({**base, **dict(zip(grids, combo, strict=True))})
    else:
        for k, g in grids.items():
            points.extend({**base, k: v} for v in g)
    seen: dict[tuple[tuple[str, str], ...], dict[str, ParamValue]] = {}
    for p in [dict(base), *points]:
        seen.setdefault(tuple(sorted((k, repr(v)) for k, v in p.items())), p)
    return list(seen.values())


@dataclass(frozen=True)
class ParameterSensitivityResult:
    labels: tuple[str, ...]
    sharpe: tuple[float, ...]
    base_sharpe: float
    neighbour_median: float
    frac_ge_half: float
    frac_positive: float
    n_failed: int

    def to_evidence(self, strategy_id: str) -> Evidence:
        if self.base_sharpe <= 0:
            sev = Severity.MEDIUM
        elif self.frac_ge_half < 0.5:
            sev = Severity.HIGH
        elif self.frac_ge_half < 0.75:
            sev = Severity.MEDIUM
        else:
            sev = Severity.INFO
        shape = "plateau" if sev is Severity.INFO else "isolated peak"
        return make_evidence(
            PARAM_TOOL,
            strategy_id,
            flag=sev >= Severity.MEDIUM,
            severity=sev,
            statistic_name="fraction_neighbours_within_half",
            statistic=self.frac_ge_half,
            summary=(
                f"Chosen parameters: Sharpe {self.base_sharpe:.2f}; median of "
                f"{len(self.sharpe) - 1} neighbours {self.neighbour_median:.2f}; "
                f"{self.frac_ge_half:.0%} keep at least half the Sharpe ({shape})."
            ),
            details={
                "base_sharpe": self.base_sharpe,
                "neighbour_median_sharpe": self.neighbour_median,
                "fraction_neighbours_within_half": self.frac_ge_half,
                "fraction_neighbours_positive": self.frac_positive,
                "n_neighbours": len(self.sharpe) - 1,
                "n_failed_configurations": self.n_failed,
            },
            plot_data={"labels": self.labels, "sharpe": self.sharpe},
            citations=PARAM_CITATIONS,
        )


def parameter_sensitivity(
    factory: StrategyFactory,
    base_params: Params,
    market: MarketData,
    config: BacktestConfig | None = None,
    grid: Sequence[Params] | None = None,
) -> ParameterSensitivityResult:
    cfg = config or BacktestConfig()
    candidates: list[Params] = (
        list(grid) if grid is not None else list(neighbourhood_grid(base_params))
    )
    points: list[Params] = [base_params, *[p for p in candidates if dict(p) != dict(base_params)]]
    trials = build_trial_set(factory, points, market, cfg, selected=0)
    if trials.params[0] != base_params:
        raise ValueError("the base parameters themselves could not be evaluated")
    x = trials.returns - cfg.rf_per_period
    sd = x.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = np.where(sd > 1e-12, x.mean(axis=0) / sd * math.sqrt(cfg.periods_per_year), 0.0)
    base, others = float(sharpe[0]), sharpe[1:]
    return ParameterSensitivityResult(
        labels=trials.labels,
        sharpe=tuple(float(s) for s in sharpe),
        base_sharpe=base,
        neighbour_median=float(np.median(others)) if others.size else float("nan"),
        frac_ge_half=float((others >= 0.5 * base).mean()) if others.size and base > 0 else 0.0,
        frac_positive=float((others > 0).mean()) if others.size else float("nan"),
        n_failed=len(trials.failed),
    )


def parameter_evidence(
    factory: StrategyFactory,
    base_params: Params,
    market: MarketData,
    strategy_id: str,
    config: BacktestConfig | None = None,
) -> Evidence:
    if not any(
        isinstance(v, int | float) and not isinstance(v, bool) for v in base_params.values()
    ):
        return inconclusive(
            PARAM_TOOL, strategy_id, "no numeric parameters to perturb", PARAM_CITATIONS
        )
    res = parameter_sensitivity(factory, base_params, market, config)
    if len(res.sharpe) < 3:
        return inconclusive(PARAM_TOOL, strategy_id, "too few valid neighbours", PARAM_CITATIONS)
    return res.to_evidence(strategy_id)
