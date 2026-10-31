"""The trial set: every configuration that was tried while developing a strategy.

Multiple-testing tools (DSR, PBO, Haircut Sharpe, purged CV selection, Reality Check / SPA)
all consume the same object, so they agree on what was tried. A trial set is a matrix of
per-bar net returns, shape (n_bars, n_trials), plus the index of the selected trial.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy_court.audit._stats import sharpe_per_period
from strategy_court.backtest import BacktestConfig, MarketData, simulate, validate_weights
from strategy_court.backtest.strategy import Strategy
from strategy_court.schemas.strategy import ParamValue

Params = Mapping[str, ParamValue]
StrategyFactory = Callable[[Params], Strategy]


@dataclass(frozen=True)
class TrialSet:
    returns: np.ndarray  # (n_bars, n_trials) per-bar net simple returns
    labels: tuple[str, ...]
    selected: int
    params: tuple[Params, ...] = field(default_factory=tuple)
    failed: tuple[str, ...] = ()  # configurations that could not be evaluated (counted as trials)

    def __post_init__(self) -> None:
        if self.returns.ndim != 2 or self.returns.shape[1] != len(self.labels):
            raise ValueError("returns must be (n_bars, n_trials) with one label per trial")
        if not 0 <= self.selected < self.returns.shape[1]:
            raise ValueError("selected trial index out of range")

    @property
    def n_bars(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_trials(self) -> int:
        """Trials that were tried, including ones that failed to run."""
        return int(self.returns.shape[1]) + len(self.failed)

    @property
    def selected_returns(self) -> np.ndarray:
        return np.asarray(self.returns[:, self.selected])

    def sharpes(self) -> np.ndarray:
        return np.array(
            [sharpe_per_period(self.returns[:, j]) for j in range(self.returns.shape[1])]
        )

    @classmethod
    def from_returns(
        cls, returns: pd.DataFrame | np.ndarray, selected: int | str | None = None
    ) -> "TrialSet":
        if isinstance(returns, pd.DataFrame):
            labels = tuple(str(c) for c in returns.columns)
            matrix = returns.to_numpy(dtype=np.float64)
        else:
            matrix = np.asarray(returns, dtype=np.float64)
            labels = tuple(f"trial_{j}" for j in range(matrix.shape[1]))
        if isinstance(selected, str):
            selected = labels.index(selected)
        if selected is None:
            sr = [sharpe_per_period(matrix[:, j]) for j in range(matrix.shape[1])]
            selected = int(np.nanargmax(np.where(np.isfinite(sr), sr, -np.inf)))
        return cls(matrix, labels, int(selected))


def build_trial_set(
    factory: StrategyFactory,
    grid: Sequence[Params],
    market: MarketData,
    config: BacktestConfig | None = None,
    selected: int | None = None,
) -> TrialSet:
    """Evaluate every configuration with one batched engine call (trusted, in-process code).

    Configurations whose construction or signals fail are recorded in `failed` — they were
    still *tried*, so they still count towards the number of trials."""
    weights, labels, ok_params, failed = [], [], [], []
    for p in grid:
        label = ",".join(f"{k}={v}" for k, v in sorted(p.items()))
        try:
            weights.append(validate_weights(factory(p).signals(market.close.copy()), market))
        except Exception:
            failed.append(label)
            continue
        labels.append(label)
        ok_params.append(p)
    if not weights:
        raise ValueError("no configuration in the grid could be evaluated")
    res = simulate(np.stack(weights), market, config)
    returns = res.returns[:, 1:].T
    if selected is None:
        sr = np.array([sharpe_per_period(returns[:, j]) for j in range(returns.shape[1])])
        selected = int(np.nanargmax(np.where(np.isfinite(sr), sr, -np.inf)))
    return TrialSet(returns, tuple(labels), selected, tuple(ok_params), tuple(failed))
