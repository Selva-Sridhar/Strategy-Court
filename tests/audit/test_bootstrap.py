import math

import numpy as np
import pytest

from strategy_court.audit.bootstrap import (
    bootstrap_evidence,
    bootstrap_sharpe,
    optimal_block_length,
    sharpe_difference_hac,
    stationary_bootstrap_indices,
)


def ar1(n: int, phi: float, rng: np.random.Generator, mu: float = 0.0) -> np.ndarray:
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return mu + x * math.sqrt(1 - phi**2)  # unit unconditional variance


def test_block_length_grows_with_dependence() -> None:
    rng = np.random.default_rng(0)
    iid = optimal_block_length(rng.standard_normal(2000))
    mild = optimal_block_length(ar1(2000, 0.5, rng))
    strong = optimal_block_length(ar1(2000, 0.9, rng))
    assert iid < 3 < mild < strong
    assert strong <= math.ceil(min(3 * math.sqrt(2000), 2000 / 3))


def test_stationary_bootstrap_blocks_have_geometric_mean_length() -> None:
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap_indices(5000, 4, 8.0, rng)
    breaks = (np.diff(idx, axis=1) % 5000 != 1).sum()
    assert 5000 * 4 / (breaks + 4) == pytest.approx(8.0, rel=0.1)
    assert idx.min() >= 0 and idx.max() < 5000


@pytest.mark.slow
def test_ci_coverage_under_serial_dependence() -> None:
    """True per-period SR = 0.05 with AR(0.3) returns: the 90% percentile CI should cover
    the true Sharpe close to 90% of the time."""
    rng = np.random.default_rng(2)
    true_sr, hits, reps = 0.05, 0, 200
    for i in range(reps):
        r = ar1(600, 0.3, rng, mu=true_sr)
        res = bootstrap_sharpe(r, n_boot=500, alpha=0.10, periods_per_year=1, seed=i)
        hits += res.ci_percentile[0] <= true_sr <= res.ci_percentile[1]
    assert 0.83 <= hits / reps <= 0.96


def test_ci_quick_sanity_and_evidence() -> None:
    rng = np.random.default_rng(3)
    good = bootstrap_sharpe(rng.normal(0.002, 0.01, 2520), n_boot=500)
    lo, hi = good.ci_percentile
    assert 0 < lo < good.sharpe_annual < hi
    assert good.ci_asymptotic[0] < good.sharpe_annual < good.ci_asymptotic[1]
    ev = good.to_evidence("s")
    assert not ev.flag and ev.p_value is not None and ev.p_value < 0.01
    noise = bootstrap_evidence(rng.normal(0.0, 0.01, 500), "s", n_boot=300)
    assert noise.flag
    assert bootstrap_evidence([0.01] * 5, "s").details["status"] == "inconclusive"


def test_sharpe_difference_identical_series_is_zero_and_detects_gap() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(0.001, 0.01, 2000)
    same = sharpe_difference_hac(a, a)
    assert same.diff_annual == pytest.approx(0.0, abs=1e-12)
    better = sharpe_difference_hac(rng.normal(0.003, 0.01, 2000), rng.normal(0.0, 0.01, 2000))
    assert better.diff_annual > 0 and better.p_value < 0.01


@pytest.mark.slow
def test_sharpe_difference_has_correct_size() -> None:
    rng = np.random.default_rng(5)
    rejections = sum(
        sharpe_difference_hac(rng.normal(5e-4, 0.01, 750), rng.normal(5e-4, 0.01, 750)).p_value
        < 0.05
        for _ in range(600)
    )
    assert 0.025 < rejections / 600 < 0.08
