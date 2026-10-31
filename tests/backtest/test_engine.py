import numpy as np
import pandas as pd
import pytest

from strategy_court.backtest import (
    BacktestConfig,
    CostModel,
    ExecutionTiming,
    MarketData,
    WeightError,
    run_backtest,
    simulate,
    summarize,
)
from strategy_court.data import random_walk_prices

NAN = np.nan
ZERO = CostModel.zero()


def frame(rows: list[list[float]], cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or [f"A{i}" for i in range(len(rows[0]))]
    return pd.DataFrame(rows, pd.bdate_range("2021-01-04", periods=len(rows)), cols)


def cfg(**kw: object) -> BacktestConfig:
    base: dict[str, object] = {"initial_capital": 100.0, "costs": ZERO}
    return BacktestConfig.model_validate(base | kw)


def test_buy_and_hold_fills_at_next_close() -> None:
    m = MarketData(frame([[10], [11], [12.1], [9]]))
    w = frame([[1], [NAN], [NAN], [NAN]])
    res = run_backtest(w, m, cfg(timing=ExecutionTiming.NEXT_CLOSE))
    np.testing.assert_allclose(res.equity, [100, 100, 110, 100 * 9 / 11])
    assert res.turnover.iloc[1] == pytest.approx(1.0)
    assert (res.turnover.iloc[2:] == 0).all()


def test_buy_and_hold_fills_at_next_open() -> None:
    m = MarketData(frame([[10], [11], [12.1]]), open=frame([[10], [10.5], [11.5]]))
    res = run_backtest(frame([[1], [NAN], [NAN]]), m, cfg())
    np.testing.assert_allclose(res.equity, [100, 100 * 11 / 10.5, 100 * 12.1 / 10.5])


def test_missing_open_falls_back_to_close() -> None:
    m = MarketData(frame([[10], [11], [12.1]]), open=frame([[10], [NAN], [11.5]]))
    res = run_backtest(frame([[1], [NAN], [NAN]]), m, cfg())
    np.testing.assert_allclose(res.equity, [100, 100, 110])


def test_same_bar_information_cannot_be_monetised() -> None:
    """w_t = sign(r_t) uses only data known at close t. A naive engine that applies w_t to r_t
    reports a huge Sharpe; this engine (fill at t+1) reports noise on a random walk."""
    prices = random_walk_prices(n_days=10_000, seed=11)
    r = prices.pct_change(fill_method=None)
    w = np.sign(r).fillna(0.0)
    naive = (w * r).dropna().iloc[:, 0]
    assert naive.mean() / naive.std() * np.sqrt(252) > 10
    res = run_backtest(w, MarketData(prices), cfg(timing=ExecutionTiming.NEXT_CLOSE))
    assert abs(summarize(res).sharpe) < 0.5


def test_linear_costs_are_exact() -> None:
    m = MarketData(frame([[10], [10], [10]]))
    costs = CostModel(commission_bps=1, half_spread_bps=2)
    res = run_backtest(frame([[1], [NAN], [NAN]]), m, cfg(costs=costs))
    assert res.costs.iloc[1] == pytest.approx(100 * 3e-4)
    np.testing.assert_allclose(res.equity, [100, 99.97, 99.97])


def test_borrow_fee_accrues_on_shorts() -> None:
    m = MarketData(frame([[10], [10], [10]]))
    costs = CostModel.zero().model_copy(update={"borrow_bps_annual": 252.0})
    res = run_backtest(frame([[-1], [NAN], [NAN]]), m, cfg(costs=costs))
    np.testing.assert_allclose(res.equity, [100, 99.99, 99.98])


def test_cash_earns_risk_free_rate() -> None:
    m = MarketData(frame([[10]] * 253))
    res = run_backtest(frame([[0]] * 253), m, cfg(risk_free_rate=0.05))
    assert res.equity.iloc[-1] == pytest.approx(105.0)
    assert np.isnan(summarize(res).sharpe)  # zero excess-return variance -> NaN, not inf


def test_delisting_liquidates_at_last_close_with_delisting_return() -> None:
    m = MarketData(frame([[10], [10], [20], [NAN]]))
    res = run_backtest(frame([[1], [NAN], [NAN], [NAN]]), m, cfg(delisting_return=-0.3))
    np.testing.assert_allclose(res.equity, [100, 100, 200, 140])
    assert m.delisted_at == {"A0": pd.Timestamp("2021-01-06")}


def test_orders_on_untraded_assets_are_blocked_and_counted() -> None:
    m = MarketData(frame([[10, NAN], [10, NAN], [10, 5]]))
    res = run_backtest(frame([[0, 1], [0, NAN], [0, 0]]), m, cfg())
    assert res.blocked_orders == 1
    assert (res.weights["A1"] == 0).all()


def test_leverage_cap_scales_targets() -> None:
    m = MarketData(frame([[10], [10], [10]]))
    res = run_backtest(frame([[3], [3], [3]]), m, cfg(max_gross_leverage=2.0))
    assert res.gross_exposure.iloc[1] == pytest.approx(2.0)
    assert res.leverage_scaled_bars == 2


def test_ruin_freezes_equity_at_zero() -> None:
    m = MarketData(frame([[10], [10], [7], [20]]))
    res = run_backtest(frame([[5], [NAN], [NAN], [NAN]]), m, cfg())
    np.testing.assert_allclose(res.equity, [100, 100, 0, 0])
    assert res.ruined_at == m.index[2]
    assert summarize(res).ruined and summarize(res).cagr == -1.0


def test_impact_requires_volume() -> None:
    m = MarketData(frame([[10], [10]]))
    with pytest.raises(WeightError, match="volume"):
        run_backtest(frame([[1], [NAN]]), m, cfg(costs=CostModel(impact_coefficient=1.0)))


def test_square_root_impact_known_value() -> None:
    close = frame([[10.0], [11.0], [9.9], [10.0]])
    m = MarketData(close, volume=frame([[1000.0], [1000.0], [1000.0], [1000.0]]))
    costs = CostModel.zero().model_copy(update={"impact_coefficient": 0.5, "impact_window": 5})
    res = run_backtest(frame([[0], [0], [1], [NAN]]), m, cfg(costs=costs))
    sigma = np.std([0.1, -0.1], ddof=1)  # returns known at close of bar 2
    adv = np.mean([10_000, 11_000, 9_900])
    assert res.costs.iloc[3] == pytest.approx(0.5 * sigma * np.sqrt(100 / adv) * 100)
    assert res.impact_fallback_trades == 0


def test_impact_fallback_before_estimates_exist() -> None:
    m = MarketData(frame([[10.0], [10.0]]), volume=frame([[1e6], [1e6]]))
    costs = CostModel.zero().model_copy(
        update={"impact_coefficient": 1.0, "impact_fallback_bps": 50.0}
    )
    res = run_backtest(frame([[1], [NAN]]), m, cfg(costs=costs))
    assert res.costs.iloc[1] == pytest.approx(100 * 50e-4)
    assert res.impact_fallback_trades == 1


def test_weights_drift_between_rebalances() -> None:
    m = MarketData(frame([[10, 10], [10, 10], [20, 10]]))
    res = run_backtest(frame([[0.5, 0.5], [NAN, NAN], [NAN, NAN]]), m, cfg())
    assert res.equity.iloc[-1] == pytest.approx(150)
    drifted = frame([[0.5, 0.5], [NAN, NAN], [0.5, 0.5]])
    rebalanced = run_backtest(drifted, m, cfg(timing=ExecutionTiming.NEXT_CLOSE))
    assert rebalanced.turnover.iloc[2] == 0  # weights decided at bar 2 execute after the end


def test_batch_equals_individual_runs() -> None:
    prices = random_walk_prices(n_days=300, n_assets=3, seed=5)
    m = MarketData(prices)
    rng = np.random.default_rng(0)
    batch = rng.uniform(-1, 1, size=(4, 300, 3))
    c = BacktestConfig(costs=CostModel(borrow_bps_annual=50))
    together = simulate(batch, m, c).equity
    for p in range(4):
        np.testing.assert_allclose(together[p], simulate(batch[p], m, c).equity[0])


@pytest.mark.parametrize(
    "bad",
    [
        "shape",
        "index",
        "columns",
        "inf",
        "text",
    ],
)
def test_weight_contract_violations(bad: str) -> None:
    m = MarketData(frame([[10, 10], [11, 9], [12, 8]]))
    w = frame([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    if bad == "shape":
        w = w.iloc[:2]
    elif bad == "index":
        w.index = w.index + pd.Timedelta(days=1)
    elif bad == "columns":
        w.columns = ["X", "Y"]
    elif bad == "inf":
        w.iloc[0, 0] = np.inf
    else:
        w = w.astype(object)
        w.iloc[0, 0] = "buy"
    with pytest.raises(WeightError):
        run_backtest(w, m)
