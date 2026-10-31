import time

import numpy as np
import pandas as pd
import pytest

from strategy_court.backtest import (
    BacktestConfig,
    BuyAndHold,
    CostModel,
    EqualWeight,
    MarketData,
    MovingAverageCrossover,
    Strategy,
    run_strategy,
    simulate,
    summarize,
)
from strategy_court.data import random_walk_prices, survivorship_universe


@pytest.mark.parametrize("strategy", [BuyAndHold(), EqualWeight(5), MovingAverageCrossover(5, 20)])
def test_reference_strategies_satisfy_contract_and_are_causal(strategy: Strategy) -> None:
    assert isinstance(strategy, Strategy)
    prices = random_walk_prices(n_days=200, n_assets=3, seed=4)
    full = strategy.signals(prices)
    assert full.shape == prices.shape
    for t in (30, 99, 150):
        partial = strategy.signals(prices.iloc[: t + 1])
        pd.testing.assert_series_equal(partial.iloc[-1], full.iloc[t], check_names=False)
    run_strategy(strategy, MarketData(prices))


def test_equal_weight_rebalances_on_schedule() -> None:
    prices = random_walk_prices(n_days=30, n_assets=4, seed=1)
    w = EqualWeight(10).signals(prices)
    assert w.notna().all(axis=1).sum() == 3
    np.testing.assert_allclose(w.iloc[0], 0.25)


def test_survivorship_bias_visible_through_engine() -> None:
    u = survivorship_universe(n_days=2520, n_assets=50, seed=3)
    cfg = BacktestConfig(costs=CostModel.zero())
    biased = summarize(run_strategy(EqualWeight(21), MarketData(u.survivor_prices), cfg))
    honest = summarize(run_strategy(EqualWeight(21), MarketData(u.prices), cfg))
    assert biased.cagr > honest.cagr


def test_costs_reduce_returns_monotonically() -> None:
    m = MarketData(random_walk_prices(n_days=500, n_assets=5, seed=2))
    strat = EqualWeight(1)
    results = [
        summarize(run_strategy(strat, m, BacktestConfig(costs=CostModel(half_spread_bps=bps))))
        for bps in (0, 5, 20)
    ]
    assert results[0].total_return > results[1].total_return > results[2].total_return


@pytest.mark.slow
def test_engine_throughput() -> None:
    m = MarketData(random_walk_prices(n_days=2520, n_assets=50, seed=0))
    w = np.random.default_rng(0).uniform(-0.04, 0.04, size=(200, 2520, 50))
    t0 = time.perf_counter()
    simulate(w[0], m)
    single = time.perf_counter() - t0
    t0 = time.perf_counter()
    simulate(w, m)
    batch = time.perf_counter() - t0
    print(f"\nsingle 2520x50: {single * 1e3:.1f} ms | batch 200x2520x50: {batch:.2f} s")
    assert single < 2.0 and batch < 30.0
