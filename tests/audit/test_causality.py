import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strategy_court.audit.causality import (
    causal_rerun,
    causality_test,
    default_checkpoints,
    point_in_time_signals,
)
from strategy_court.backtest import BacktestConfig, CostModel, MarketData, MovingAverageCrossover
from strategy_court.data import random_walk_prices
from strategy_court.schemas import Severity

PRICES = random_walk_prices(n_days=300, n_assets=3, seed=21)


def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    return np.sign(prices.pct_change(fill_method=None)) / prices.shape[1]


def peek_one_bar(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices).shift(-1)


def centred_window(prices: pd.DataFrame) -> pd.DataFrame:
    trend = prices.rolling(21, center=True).mean()
    return (prices > trend).astype(float) / prices.shape[1]


def full_sample_zscore(prices: pd.DataFrame) -> pd.DataFrame:
    z = (prices - prices.mean()) / prices.std()
    return -z.clip(-1, 1) / prices.shape[1]


def future_max(prices: pd.DataFrame) -> pd.DataFrame:
    ahead = prices.iloc[::-1].cummax().iloc[::-1]
    return (ahead > prices * 1.05).astype(float) / prices.shape[1]


def length_dependent(prices: pd.DataFrame) -> pd.DataFrame:
    ramp = np.arange(len(prices)) / len(prices)
    return pd.DataFrame(np.tile(ramp[:, None], prices.shape[1]), prices.index, prices.columns)


def test_clean_strategies_pass() -> None:
    for fn in (momentum, MovingAverageCrossover(5, 30).signals):
        report = causality_test(fn, PRICES)
        assert report.status == "pass", report
        assert report.lookahead_depth is None and report.scope == "none"
        ev = report.to_evidence("s1")
        assert not ev.flag and ev.severity is Severity.INFO


def test_one_bar_peek_is_caught_with_depth_one() -> None:
    report = causality_test(peek_one_bar, PRICES)
    assert report.status == "fail"
    assert report.lookahead_depth == 1 and report.scope == "local"
    assert report.modes_failed() == ["mirror", "noise", "truncate"]
    assert report.leaking_columns == list(PRICES.columns)
    by_mode = {m: [c.ok for c in report.checks if c.mode == m] for m in ("truncate", "mirror")}
    assert not any(by_mode["truncate"]) and not any(by_mode["mirror"])  # deterministic power
    ev = report.to_evidence("s1")
    assert ev.flag and ev.severity is Severity.CRITICAL and ev.statistic > 0.9
    assert ev.citations[0].authors.startswith("Kaufman")


def test_magnitude_only_leak_needs_the_noise_perturbation() -> None:
    def future_volatility(prices: pd.DataFrame) -> pd.DataFrame:
        vol_ahead = np.log(prices).diff().abs().rolling(5).mean().shift(-5)
        return (vol_ahead < vol_ahead.median()).astype(float) / prices.shape[1]

    report = causality_test(future_volatility, PRICES)
    assert report.status == "fail"
    assert "noise" in report.modes_failed() and "mirror" not in report.modes_failed()


def test_centred_window_depth_is_half_window() -> None:
    report = causality_test(centred_window, PRICES)
    assert report.status == "fail" and report.scope == "local"
    assert report.lookahead_depth == 10


@pytest.mark.parametrize("fn", [full_sample_zscore, future_max])
def test_full_sample_statistics_are_global_leaks(fn: object) -> None:
    report = causality_test(fn, PRICES)  # type: ignore[arg-type]
    assert report.status == "fail" and report.scope == "global"


def test_truncation_and_perturbation_are_complementary() -> None:
    report = causality_test(length_dependent, PRICES)
    assert report.status == "fail"
    assert report.modes_failed() == ["truncate"]  # same length => perturbations can't see it


def test_nondeterministic_strategy_is_inconclusive_not_convicted() -> None:
    def noisy(prices: pd.DataFrame) -> pd.DataFrame:
        unseeded = np.random.default_rng()
        return pd.DataFrame(unseeded.random(prices.shape), prices.index, prices.columns)

    report = causality_test(noisy, PRICES)
    assert report.status == "inconclusive" and not report.deterministic
    ev = report.to_evidence("s1")
    assert not ev.flag and ev.severity is Severity.HIGH


def test_contract_errors_are_recorded_not_counted_as_leaks() -> None:
    def needs_history(prices: pd.DataFrame) -> pd.DataFrame:
        if len(prices) < 150:
            raise ValueError("warm-up")
        return momentum(prices)

    report = causality_test(needs_history, PRICES)
    assert report.status == "pass" and report.errors
    assert all("warm-up" in (c.error or "") for c in report.errors)

    def wrong_shape(prices: pd.DataFrame) -> pd.DataFrame:
        return momentum(prices).iloc[1:]

    bad = causality_test(wrong_shape, PRICES)
    assert bad.status == "inconclusive" and "index/columns" in bad.reason


def test_always_erroring_on_subsets_is_inconclusive() -> None:
    def only_full(prices: pd.DataFrame) -> pd.DataFrame:
        if len(prices) != len(PRICES) or not prices.equals(PRICES):
            raise RuntimeError("refuses other inputs")
        return momentum(prices)

    assert causality_test(only_full, PRICES).status == "inconclusive"


def test_checkpoints_cover_history() -> None:
    cps = default_checkpoints(300, 16, None)
    assert cps[0] == 30 and cps[-1] == 298 and len(cps) == 16


@settings(max_examples=60, deadline=None)
@given(lag=st.integers(-5, 5), window=st.integers(1, 30), n_assets=st.integers(1, 3))
def test_detects_every_negative_lag_and_never_flags_causal_ones(
    lag: int, window: int, n_assets: int
) -> None:
    prices = random_walk_prices(n_days=160, n_assets=n_assets, seed=window)

    def strategy(p: pd.DataFrame) -> pd.DataFrame:
        trend = p.pct_change(fill_method=None).rolling(window).mean()
        return np.sign(trend).shift(lag) / p.shape[1]

    status = causality_test(strategy, prices, n_checkpoints=8).status
    assert status == ("fail" if lag < 0 else "pass")


def test_point_in_time_signals_identity_for_causal_strategy() -> None:
    small = PRICES.iloc[:80]
    pit, errors = point_in_time_signals(momentum, small)
    pd.testing.assert_frame_equal(pit, momentum(small), check_freq=False)
    assert errors == 0


def test_honest_rerun_removes_the_fake_edge() -> None:
    """With t+1 execution a 1-bar peek earns nothing (the fill lag absorbs it, though it is
    still convicted above); a 2-bar peek is the smallest monetisable leak at next-close fills."""

    def peek_two_bars(prices: pd.DataFrame) -> pd.DataFrame:
        return momentum(prices).shift(-2)

    prices = random_walk_prices(n_days=400, n_assets=2, seed=8)
    cfg = BacktestConfig(costs=CostModel.zero())
    assert abs(causal_rerun(peek_one_bar, MarketData(prices), cfg).as_submitted.sharpe) < 1.5
    rerun = causal_rerun(peek_two_bars, MarketData(prices), cfg)
    assert rerun.as_submitted.sharpe > 5
    pit_sharpe = rerun.point_in_time.sharpe
    assert np.isnan(pit_sharpe) or abs(pit_sharpe) < 1.5
    assert set(rerun.as_dict()) >= {"sharpe_as_submitted", "sharpe_point_in_time"}
