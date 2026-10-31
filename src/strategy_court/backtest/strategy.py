"""The fixed strategy interface and trusted reference strategies.

Contract: `signals(prices)` receives close prices (index = bars, columns = assets) and returns
target weights of the same shape, decided at each bar's close. NaN = hold current position,
0 = flat. The engine executes at the next bar; strategies never control fills or costs.
`signals` must be pure, which is what makes the causality truncation test possible.
"""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from strategy_court.backtest.config import BacktestConfig
from strategy_court.backtest.engine import BacktestResult, run_backtest
from strategy_court.backtest.market import MarketData
from strategy_court.schemas.strategy import ParamValue


@runtime_checkable
class Strategy(Protocol):
    params: Mapping[str, ParamValue]

    def signals(self, prices: pd.DataFrame) -> pd.DataFrame: ...


def run_strategy(
    strategy: Strategy, market: MarketData, config: BacktestConfig | None = None
) -> BacktestResult:
    """Trusted, in-process execution. Untrusted (LLM-written) code goes through the sandbox."""
    return run_backtest(strategy.signals(market.close.copy()), market, config)


def _empty(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)


class BuyAndHold:
    """Equal-weight the assets trading on the first bar, then never rebalance."""

    def __init__(self) -> None:
        self.params: dict[str, ParamValue] = {}

    def signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = _empty(prices)
        live = prices.iloc[0].notna().to_numpy()
        w.iloc[0] = np.where(live, 1.0 / live.sum(), 0.0)
        return w


class EqualWeight:
    """Equal-weight all trading assets every `rebalance_every` bars; hold in between."""

    def __init__(self, rebalance_every: int = 21) -> None:
        if rebalance_every < 1:
            raise ValueError("rebalance_every must be >= 1")
        self.params: dict[str, ParamValue] = {"rebalance_every": rebalance_every}

    def signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        live = prices.notna()
        target = live.div(live.sum(axis=1), axis=0).fillna(0.0)
        w = _empty(prices)
        step = int(self.params["rebalance_every"])
        w.iloc[::step] = target.iloc[::step].to_numpy()
        return w


class MovingAverageCrossover:
    """Long 1/N in each asset whose fast SMA is above its slow SMA, flat otherwise."""

    def __init__(self, fast: int = 20, slow: int = 100) -> None:
        if not 1 <= fast < slow:
            raise ValueError("require 1 <= fast < slow")
        self.params: dict[str, ParamValue] = {"fast": fast, "slow": slow}

    def signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        fast = prices.rolling(int(self.params["fast"])).mean()
        slow = prices.rolling(int(self.params["slow"])).mean()
        long = (fast > slow).astype(float)
        return long / prices.shape[1]
