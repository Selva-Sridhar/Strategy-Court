import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from strategy_court.audit.robustness import (
    cost_evidence,
    cost_sensitivity,
    neighbourhood_grid,
    parameter_evidence,
    parameter_sensitivity,
)
from strategy_court.backtest import (
    BacktestConfig,
    BuyAndHold,
    CostModel,
    MarketData,
    MovingAverageCrossover,
)
from strategy_court.data import weak_signal_prices
from strategy_court.schemas import ParamValue, Severity

ZERO = BacktestConfig(costs=CostModel.zero(), timing="next_close")


def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    return np.sign(prices.pct_change(fill_method=None)).fillna(0.0) / prices.shape[1]


def test_high_turnover_edge_collapses_under_realistic_costs() -> None:
    prices = weak_signal_prices(n_days=2520, n_assets=2, ar1=0.25, seed=1)  # IC ~ 0.06
    m = MarketData(prices)
    res = cost_sensitivity(momentum(prices), m, ZERO)
    assert res.sharpe[0] > 0.8  # real but thin edge ...
    assert res.annual_turnover > 100
    assert res.break_even_bps < 10  # ... eaten by costs
    assert res.break_even_bps == pytest.approx(res.break_even_bps_analytic, rel=0.25)
    ev = res.to_evidence("s")
    assert ev.flag and ev.severity is Severity.HIGH


def test_buy_and_hold_is_cost_insensitive() -> None:
    prices = (
        weak_signal_prices(n_days=1000, n_assets=3, ar1=0.0, seed=2)
        * np.exp(np.linspace(0, 1, 1000))[:, None]
    )
    m = MarketData(prices)
    res = cost_sensitivity(BuyAndHold().signals(prices), m, ZERO)
    assert math.isinf(res.break_even_bps) or res.break_even_bps > 50
    assert not res.to_evidence("s").flag


def test_delay_decay_and_half_life() -> None:
    prices = weak_signal_prices(n_days=3000, n_assets=2, ar1=0.5, seed=3)
    res = cost_sensitivity(momentum(prices), MarketData(prices), ZERO)
    assert res.delay_sharpe[0] > 1.0
    assert res.delay_sharpe[1] < 0.75 * res.delay_sharpe[0]  # AR(1): edge decays ~ar1 per bar
    assert 0 < res.half_life_bars < 5


def test_cost_evidence_inconclusive_when_flat() -> None:
    prices = weak_signal_prices(n_days=100, seed=4)
    flat = pd.DataFrame(0.0, prices.index, prices.columns)
    assert cost_evidence(flat, MarketData(prices), "s").details["status"] == "inconclusive"


def test_neighbourhood_grid_rounds_ints_and_keeps_base_first() -> None:
    grid = neighbourhood_grid({"fast": 20, "slow": 100, "mode": "long"})
    assert grid[0] == {"fast": 20, "slow": 100, "mode": "long"}
    assert all(isinstance(p["fast"], int) and p["mode"] == "long" for p in grid)
    assert len(grid) == len({(p["fast"], p["slow"]) for p in grid}) == 25


def test_plateau_vs_isolated_peak() -> None:
    prices = (
        weak_signal_prices(n_days=1500, n_assets=1, ar1=0.0, seed=5)
        * np.exp(np.linspace(0, 1.2, 1500))[:, None]
    )
    m = MarketData(prices)

    class Lucky:
        """Long only at exactly k == 17, otherwise short: an isolated peak by construction."""

        def __init__(self, k: int) -> None:
            self.params: Mapping[str, ParamValue] = {"k": k}

        def signals(self, p: pd.DataFrame) -> pd.DataFrame:
            sign = 1.0 if self.params["k"] == 17 else -1.0
            return pd.DataFrame(sign, p.index, p.columns)

    def lucky(params: Mapping[str, ParamValue]) -> Lucky:
        k = params["k"]
        assert isinstance(k, int)
        return Lucky(k)

    spike = parameter_sensitivity(lucky, {"k": 17}, m, ZERO)
    assert spike.base_sharpe > 0 and spike.frac_ge_half == 0.0
    assert spike.to_evidence("s").severity is Severity.HIGH

    def mac(params: Mapping[str, ParamValue]) -> MovingAverageCrossover:
        fast, slow = params["fast"], params["slow"]
        assert isinstance(fast, int) and isinstance(slow, int)
        return MovingAverageCrossover(fast, slow)

    trend = parameter_sensitivity(mac, {"fast": 20, "slow": 100}, m, ZERO)
    assert trend.base_sharpe > 0 and trend.frac_ge_half > 0.75
    assert not trend.to_evidence("s").flag


def test_invalid_neighbours_are_counted_not_hidden() -> None:
    prices = weak_signal_prices(n_days=400, seed=6)
    m = MarketData(prices)

    def mac(params: Mapping[str, ParamValue]) -> MovingAverageCrossover:
        fast, slow = params["fast"], params["slow"]
        assert isinstance(fast, int) and isinstance(slow, int)
        return MovingAverageCrossover(fast, slow)  # raises when fast >= slow

    res = parameter_sensitivity(mac, {"fast": 40, "slow": 46}, m, ZERO)
    assert res.n_failed > 0
    assert parameter_evidence(mac, {"mode": "x"}, m, "s").details["status"] == "inconclusive"
