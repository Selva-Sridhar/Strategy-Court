import numpy as np
import pytest
from scipy import stats

from strategy_court.backtest.metrics import (
    annualised_volatility,
    cagr,
    kurtosis_raw,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    skewness,
    sortino_ratio,
)


def test_sharpe_matches_definition_and_rf() -> None:
    r = np.array([0.01, -0.005, 0.002, 0.007, -0.001])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected)
    shifted = (r - 0.001).mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r, rf_per_period=0.001) == pytest.approx(shifted)


def test_degenerate_inputs_give_nan_not_inf() -> None:
    flat = np.full(10, 0.001)
    assert np.isnan(sharpe_ratio(flat))
    assert np.isnan(sortino_ratio(flat))  # no downside
    assert np.isnan(skewness(flat)) and np.isnan(kurtosis_raw(flat))
    assert np.isnan(sharpe_ratio([0.01]))
    noisy_flat = (100 * 1.0002 ** np.arange(500))[1:] / (100 * 1.0002 ** np.arange(500))[:-1] - 1
    assert np.isnan(sharpe_ratio(noisy_flat))  # float noise is not variance


def test_sortino_uses_downside_over_all_periods() -> None:
    r = np.array([0.1, -0.1, 0.05, -0.05, 0.02])
    dd = np.sqrt((0.01 + 0.0025) / 5)
    assert sortino_ratio(r) == pytest.approx(r.mean() / dd * np.sqrt(252))


def test_cagr_and_volatility() -> None:
    eq = 100 * 1.001 ** np.arange(253)
    assert cagr(eq) == pytest.approx(1.001**252 - 1)
    assert cagr([100, 50, 0]) == -1.0
    r = np.array([0.01, -0.01] * 50)
    assert annualised_volatility(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252))


def test_drawdown_depth_and_duration() -> None:
    eq = [100, 120, 90, 95, 130, 117, 117]
    assert max_drawdown(eq) == pytest.approx(0.25)
    assert max_drawdown_duration(eq) == 2
    assert max_drawdown([100, 101, 102]) == 0.0


def test_moments_match_scipy_with_raw_kurtosis() -> None:
    r = np.random.default_rng(0).standard_t(5, 5000) * 0.01
    assert skewness(r) == pytest.approx(stats.skew(r))
    assert kurtosis_raw(r) == pytest.approx(stats.kurtosis(r) + 3)
    normal = np.random.default_rng(1).standard_normal(200_000)
    assert kurtosis_raw(normal) == pytest.approx(3.0, abs=0.05)
