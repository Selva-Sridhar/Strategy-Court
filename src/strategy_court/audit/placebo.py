"""Placebo (permutation) null for timing skill.

H0: the strategy's positions carry no information about future returns. Placebos keep
*everything* about the positions — exposure, turnover, holding periods, cross-asset structure —
and only break their alignment with prices, so they pay the same costs:

* circular: shift the whole weight path in time by a random offset (wrapping). Exactly
  preserves the position process' distribution and autocorrelation; turnover differs only at
  the single wrap point.
* blocks: permute the order of holding runs (maximal spans of identical weight rows). Exact
  turnover for piecewise-constant strategies; for continuously rebalanced ones runs have length 1
  and this degrades to a row permutation, so `circular` is the default.

p-value = (1 + #{placebo Sharpe >= observed}) / (1 + B) — never zero (Phipson & Smyth 2010).
The data-snooping null is the bootstrap reality check (White 2000; Sullivan, Timmermann & White
1999), implemented in `snooping.py`.
"""

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from strategy_court.audit._common import inconclusive, make_evidence, severity_from_p
from strategy_court.audit.citations import PHIPSON_SMYTH_2010, STW_1999, WHITE_2000
from strategy_court.backtest import BacktestConfig, MarketData, simulate, validate_weights
from strategy_court.schemas import Evidence

TOOL = "placebo_null"
CITATIONS = (WHITE_2000, STW_1999, PHIPSON_SMYTH_2010)
Method = Literal["circular", "blocks"]


def circular_shift_placebos(
    weights: np.ndarray, n: int, rng: np.random.Generator, min_shift: int
) -> np.ndarray:
    t = weights.shape[0]
    if t <= 2 * min_shift:
        raise ValueError("series too short for the requested minimum shift")
    shifts = rng.integers(min_shift, t - min_shift + 1, n)
    return np.stack([np.roll(weights, int(s), axis=0) for s in shifts])


def _runs(weights: np.ndarray) -> list[tuple[int, int]]:
    same = np.all(
        (weights[1:] == weights[:-1]) | (np.isnan(weights[1:]) & np.isnan(weights[:-1])), axis=1
    )
    bounds = [0, *(np.flatnonzero(~same) + 1).tolist(), weights.shape[0]]
    return list(itertools.pairwise(bounds))


def run_block_placebos(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    runs = _runs(weights)
    out = np.empty((n, *weights.shape))
    for i in range(n):
        order = rng.permutation(len(runs))
        out[i] = np.concatenate([weights[runs[j][0] : runs[j][1]] for j in order])
    return out


def _annual_sharpes(returns: np.ndarray, rf: float, ppy: int) -> np.ndarray:
    x = returns[:, 1:] - rf
    sd = x.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 1e-12, x.mean(axis=1) / sd * np.sqrt(ppy), 0.0)


@dataclass(frozen=True)
class PlaceboResult:
    observed_sharpe: float
    null_sharpes: np.ndarray
    p_value: float
    method: Method
    turnover_ratio: float  # mean placebo turnover / observed turnover (should be ~1)

    @property
    def null_mean(self) -> float:
        return float(self.null_sharpes.mean())

    @property
    def null_std(self) -> float:
        return float(self.null_sharpes.std(ddof=1))

    @property
    def z(self) -> float:
        return (
            (self.observed_sharpe - self.null_mean) / self.null_std
            if self.null_std > 0
            else float("nan")
        )

    def to_evidence(self, strategy_id: str) -> Evidence:
        hist, edges = np.histogram(self.null_sharpes, bins=30)
        return make_evidence(
            TOOL,
            strategy_id,
            flag=self.p_value > 0.05,
            severity=severity_from_p(self.p_value),
            statistic_name="sharpe_vs_placebo_z",
            statistic=self.z,
            p_value=self.p_value,
            summary=(
                f"Observed Sharpe {self.observed_sharpe:.2f} vs {self.null_sharpes.size} "
                f"turnover-matched placebos (mean {self.null_mean:.2f}, sd {self.null_std:.2f}): "
                f"p = {self.p_value:.3f}."
            ),
            details={
                "observed_sharpe_annual": self.observed_sharpe,
                "placebo_mean": self.null_mean,
                "placebo_std": self.null_std,
                "placebo_q95": float(np.quantile(self.null_sharpes, 0.95)),
                "n_placebos": int(self.null_sharpes.size),
                "method": self.method,
                "turnover_ratio": self.turnover_ratio,
            },
            plot_data={"hist_counts": hist, "hist_edges": edges, "observed": self.observed_sharpe},
            citations=CITATIONS,
        )


def placebo_test(
    weights: pd.DataFrame,
    market: MarketData,
    config: BacktestConfig | None = None,
    n_placebos: int = 999,
    method: Method = "circular",
    min_shift: int | None = None,
    seed: int = 0,
    chunk: int = 100,
) -> PlaceboResult:
    cfg = config or BacktestConfig()
    w = validate_weights(weights, market)
    rng = np.random.default_rng(seed)
    base = simulate(w, market, cfg)
    observed = float(_annual_sharpes(base.returns, cfg.rf_per_period, cfg.periods_per_year)[0])
    obs_turn = float(base.turnover[0, 1:].mean())
    shift = min_shift if min_shift is not None else max(1, w.shape[0] // 10)
    null, turns = [], []
    for start in range(0, n_placebos, chunk):
        k = min(chunk, n_placebos - start)
        paths = (
            circular_shift_placebos(w, k, rng, shift)
            if method == "circular"
            else run_block_placebos(w, k, rng)
        )
        res = simulate(paths, market, cfg)
        null.append(_annual_sharpes(res.returns, cfg.rf_per_period, cfg.periods_per_year))
        turns.append(res.turnover[:, 1:].mean(axis=1))
    sims = np.concatenate(null)
    p = float((1 + (sims >= observed).sum()) / (1 + sims.size))
    ratio = float(np.concatenate(turns).mean() / obs_turn) if obs_turn > 0 else float("nan")
    return PlaceboResult(observed, sims, p, method, ratio)


def placebo_evidence(
    weights: pd.DataFrame, market: MarketData, strategy_id: str, **kw: object
) -> Evidence:
    w = validate_weights(weights, market)
    if np.nan_to_num(w).std(axis=0).max(initial=0.0) == 0:
        return inconclusive(
            TOOL, strategy_id, "positions never change, no timing to test", CITATIONS
        )
    if w.shape[0] < 50:
        return inconclusive(TOOL, strategy_id, "fewer than 50 bars", CITATIONS)
    return placebo_test(weights, market, **kw).to_evidence(strategy_id)  # type: ignore[arg-type]
