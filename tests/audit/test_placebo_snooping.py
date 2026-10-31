import numpy as np
import pandas as pd
import pytest
from scipy import stats

from strategy_court.audit.placebo import (
    circular_shift_placebos,
    placebo_evidence,
    placebo_test,
    run_block_placebos,
)
from strategy_court.audit.snooping import reality_check, snooping_evidence
from strategy_court.audit.trials import TrialSet
from strategy_court.backtest import BacktestConfig, CostModel, MarketData
from strategy_court.data import random_walk_prices, weak_signal_prices

ZERO = BacktestConfig(costs=CostModel.zero())


def random_positions(index: pd.Index, rng: np.random.Generator, mean_run: int = 10) -> pd.DataFrame:
    """Piecewise-constant +/-1 positions, independent of prices."""
    n = len(index)
    w = np.empty(n)
    i = 0
    while i < n:
        length = int(rng.geometric(1 / mean_run))
        w[i : i + length] = rng.choice([-1.0, 1.0])
        i += length
    return pd.DataFrame({"SYN000": w}, index=index)


def test_placebos_preserve_position_distribution_and_runs() -> None:
    rng = np.random.default_rng(0)
    w = random_positions(pd.RangeIndex(500), rng).to_numpy()
    circ = circular_shift_placebos(w, 20, rng, 50)
    np.testing.assert_allclose(np.sort(circ, axis=1), np.repeat(np.sort(w, axis=0)[None], 20, 0))
    blocks = run_block_placebos(w, 20, rng)
    changes = (np.diff(w[:, 0]) != 0).sum()
    for path in blocks:
        assert (np.diff(path[:, 0]) != 0).sum() <= changes  # merging equal runs can only reduce
        np.testing.assert_allclose(np.sort(path, axis=0), np.sort(w, axis=0))


def test_placebo_p_values_are_uniform_under_the_null() -> None:
    rng = np.random.default_rng(1)
    pvals = []
    for i in range(80):
        prices = random_walk_prices(n_days=300, seed=100 + i)
        w = random_positions(prices.index, rng)
        pvals.append(placebo_test(w, MarketData(prices), ZERO, n_placebos=99, seed=i).p_value)
    assert stats.kstest(pvals, "uniform").pvalue > 0.01


def test_placebo_detects_genuine_timing_and_matches_turnover() -> None:
    # sign(r_t) earns r_{t+2} under t+1 fills; with AR(1) shocks the exploitable correlation
    # is ar1**2, so ar1 = 0.4 plants an information coefficient of ~0.16.
    prices = weak_signal_prices(n_days=2000, ar1=0.4, seed=2)
    w = np.sign(prices.pct_change(fill_method=None)).fillna(0.0)
    res = placebo_test(w, MarketData(prices), ZERO, n_placebos=199)
    assert res.p_value < 0.05 and res.observed_sharpe > 0
    assert res.turnover_ratio == pytest.approx(1.0, abs=0.01)
    ev = res.to_evidence("s")
    assert not ev.flag and ev.plot_data is not None


def test_placebo_inconclusive_for_static_positions() -> None:
    prices = random_walk_prices(n_days=200, seed=3)
    w = pd.DataFrame(1.0, prices.index, prices.columns)
    assert placebo_evidence(w, MarketData(prices), "s").details["status"] == "inconclusive"


def _noise_trials(rng: np.random.Generator, n: int = 750, m: int = 50) -> np.ndarray:
    return rng.normal(0.0, 0.01, (n, m))


def test_reality_check_size_and_spa_ordering() -> None:
    rng = np.random.default_rng(4)
    rejections = 0
    for i in range(40):
        res = reality_check(_noise_trials(rng), n_boot=300, seed=i)
        assert res.spa_p_lower <= res.spa_p_consistent <= res.spa_p_upper
        rejections += res.spa_p_consistent < 0.05
    assert rejections / 40 <= 0.125


def test_reality_check_finds_a_real_edge() -> None:
    rng = np.random.default_rng(5)
    r = _noise_trials(rng)
    r[:, 7] += 0.0025
    res = reality_check(r, n_boot=500)
    assert res.best_trial == 7 and res.rc_p < 0.05 and res.spa_p_consistent < 0.05


def test_spa_is_robust_to_adding_terrible_trials_where_rc_is_not() -> None:
    """Hansen's point: poor alternatives inflate the RC p-value but not SPA_c."""
    rng = np.random.default_rng(6)
    good = _noise_trials(rng, m=10)
    good[:, 0] += 0.0012
    bad = rng.normal(-0.004, 0.03, (750, 200))
    with_bad = reality_check(np.hstack([good, bad]), n_boot=500)
    without = reality_check(good, n_boot=500)
    assert with_bad.spa_p_consistent <= without.spa_p_consistent + 0.02
    assert with_bad.spa_p_consistent < with_bad.spa_p_upper


def test_snooping_evidence() -> None:
    rng = np.random.default_rng(7)
    ev = snooping_evidence(TrialSet.from_returns(_noise_trials(rng)), "s", n_boot=200)
    assert ev.tool == "data_snooping" and ev.p_value is not None
    one = TrialSet.from_returns(_noise_trials(rng, m=1))
    assert snooping_evidence(one, "s").details["status"] == "inconclusive"
