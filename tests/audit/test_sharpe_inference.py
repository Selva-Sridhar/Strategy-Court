"""Known-answer, identity and Monte-Carlo calibration tests for PSR/MinTRL, DSR and Haircut SR."""

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from strategy_court.audit.dsr import (
    deflated_sharpe_ratio,
    dsr_evidence,
    dsr_test,
    effective_trials,
    expected_max_sharpe,
)
from strategy_court.audit.haircut import (
    adjust_pvalues,
    haircut_evidence,
    haircut_sharpe,
    lo_annualisation_factor,
)
from strategy_court.audit.psr import (
    min_track_record_length,
    probabilistic_sharpe_ratio,
    psr_evidence,
    psr_test,
)
from strategy_court.audit.trials import TrialSet
from strategy_court.schemas import Severity

# ---------------------------------------------------------------- PSR / MinTRL


def test_variance_factor_reduces_to_lo_under_normality() -> None:
    sr, n = 0.1, 500
    expected = stats.norm.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + sr**2 / 2))
    assert probabilistic_sharpe_ratio(sr, n, 0.0, 3.0) == pytest.approx(expected)


@pytest.mark.parametrize(("sr", "skew", "kurt"), [(0.08, 0.0, 3.0), (0.12, -1.5, 9.0)])
def test_psr_at_min_trl_equals_confidence_level(sr: float, skew: float, kurt: float) -> None:
    """Algebraic identity: PSR evaluated at n = MinTRL is exactly 1 - alpha."""
    n = min_track_record_length(sr, skew, kurt, sr_benchmark=0.02, alpha=0.05)
    assert probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.02) == pytest.approx(0.95)


def test_negative_skew_and_fat_tails_lower_psr_and_lengthen_min_trl() -> None:
    assert probabilistic_sharpe_ratio(0.1, 300, -2.0, 10.0) < probabilistic_sharpe_ratio(
        0.1, 300, 0.0, 3.0
    )
    assert min_track_record_length(0.1, -2.0, 10.0) > min_track_record_length(0.1, 0.0, 3.0)
    assert math.isinf(min_track_record_length(0.0, 0.0, 3.0))


def test_psr_is_calibrated_under_the_null_with_fat_tails() -> None:
    """True SR = 0 with Student-t(5) returns: 1 - PSR(0) is a valid p-value (~ uniform)."""
    rng = np.random.default_rng(0)
    pvals = [1 - psr_test(rng.standard_t(5, 500) * 0.01).psr for _ in range(1500)]
    assert stats.kstest(pvals, "uniform").pvalue > 0.01


def _skewed_size(sigma: float, n: int = 250, reps: int = 3000) -> tuple[float, float]:
    """Rejection rates (corrected, normal-assumption) at a 5% level when true SR = SR*,
    for negatively skewed (negated log-normal) returns standardised to SR = 0.1."""
    rng = np.random.default_rng(1)
    raw = -rng.lognormal(0.0, sigma, size=(reps, n))
    mu = -math.exp(sigma**2 / 2)
    sd = math.sqrt((math.exp(sigma**2) - 1) * math.exp(sigma**2))
    true_sr = 0.1
    corrected = normal = 0
    for row in (raw - mu) / sd + true_sr:
        res = psr_test(row, benchmark_sharpe_annual=true_sr * math.sqrt(252))
        corrected += res.psr > 0.95
        normal += res.psr_normal > 0.95
    return corrected / reps, normal / reps


def test_moment_correction_controls_size_where_normal_formula_overrejects() -> None:
    corrected, normal = _skewed_size(sigma=0.5)  # skew ~ -1.75, realistic for strategies
    assert abs(corrected - 0.05) < abs(normal - 0.05)
    assert corrected < 0.07


def test_documented_finite_sample_limit_under_extreme_skew() -> None:
    corrected, normal = _skewed_size(sigma=1.0)  # skew ~ -6.2, kurtosis ~ 114
    assert corrected < normal  # still better than ignoring the moments ...
    assert corrected < 0.10  # ... but over-rejects, as stated in the psr module docstring


def test_psr_evidence() -> None:
    rng = np.random.default_rng(2)
    good = psr_evidence(rng.normal(0.002, 0.01, 2520), "s")
    noise = psr_evidence(rng.normal(0.0, 0.01, 250), "s")
    assert not good.flag and good.severity is Severity.INFO
    assert noise.flag and noise.severity >= Severity.MEDIUM
    assert psr_evidence([0.01] * 5, "s").details["status"] == "inconclusive"


# ---------------------------------------------------------------- DSR


def test_dsr_reproduces_bailey_lopez_de_prado_2014_numerical_example() -> None:
    """Paper's example: 100 trials, annual SR 2.5 over 5 years of daily data (1250 obs,
    250 days/yr), skew -3, kurtosis 10, variance of annual trial SRs 0.5
    => expected max SR ~ 0.1132 (daily) and DSR ~ 0.90."""
    sr = 2.5 / math.sqrt(250)
    dsr, sr0 = deflated_sharpe_ratio(sr, 1250, -3.0, 10.0, 100, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    assert dsr == pytest.approx(0.9004, abs=2e-3)


def test_expected_max_sharpe_matches_monte_carlo() -> None:
    rng = np.random.default_rng(3)
    for n in (10, 100, 1000):
        sims = rng.standard_normal((4000, n)).max(axis=1).mean()
        assert expected_max_sharpe(n, 1.0) == pytest.approx(sims, rel=0.03)
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_dsr_decreases_with_trials_and_effective_trials_bounds() -> None:
    values = [deflated_sharpe_ratio(0.1, 1000, 0, 3, n, 0.002)[0] for n in (2, 10, 100, 1000)]
    assert all(a > b for a, b in itertools.pairwise(values))
    rng = np.random.default_rng(4)
    base = rng.standard_normal((500, 1))
    assert effective_trials(np.repeat(base, 20, axis=1)) == pytest.approx(1.0)
    assert effective_trials(rng.standard_normal((5000, 20))) > 18


def test_dsr_flags_best_of_many_noise_strategies() -> None:
    rng = np.random.default_rng(5)
    noise = TrialSet.from_returns(rng.normal(0, 0.01, (1000, 200)))
    res = dsr_test(noise)
    assert res.sr > res.sr0 * 0.8  # the max looks good in isolation ...
    ev = dsr_evidence(noise, "s")
    assert ev.flag  # ... but is not distinguishable from luck
    real = rng.normal(0, 0.01, (1000, 10))
    real[:, 3] += 0.003
    assert not dsr_evidence(TrialSet.from_returns(real), "s").flag
    assert dsr_evidence(TrialSet.from_returns(real[:, :1]), "s").details["status"] == "inconclusive"


# ---------------------------------------------------------------- Haircut Sharpe


def test_p_value_adjustments_known_values() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(adjust_pvalues(p, "bonferroni"), [0.04, 0.16, 0.12, 0.02])
    np.testing.assert_allclose(adjust_pvalues(p, "holm"), [0.03, 0.06, 0.06, 0.02])
    bh = np.array([0.02, 0.04, 0.04, 0.02])
    c4 = 1 + 1 / 2 + 1 / 3 + 1 / 4
    np.testing.assert_allclose(adjust_pvalues(p, "bhy"), np.minimum(1, bh * c4))


def test_lo_factor_iid_and_ar1_closed_form() -> None:
    rng = np.random.default_rng(6)
    iid = rng.standard_normal(200_000)
    assert lo_annualisation_factor(iid, 12) == pytest.approx(math.sqrt(12), rel=0.01)
    phi, q = 0.3, 12
    x = np.zeros(400_000)
    e = rng.standard_normal(x.size)
    for t in range(1, x.size):
        x[t] = phi * x[t - 1] + e[t]
    k = np.arange(1, q)
    exact = q / math.sqrt(q + 2 * ((q - k) * phi**k).sum())
    assert lo_annualisation_factor(x, q, max_lag=q - 1) == pytest.approx(exact, rel=0.02)


def test_haircut_single_trial_is_zero_and_grows_with_trials() -> None:
    rng = np.random.default_rng(7)
    r = rng.normal(0.0015, 0.01, 2520)  # annual SR ~ 2.4 over 10 years
    one = haircut_sharpe(r, n_trials=1)
    assert one.haircut("bonferroni") == pytest.approx(0.0, abs=1e-9)
    cuts = [haircut_sharpe(r, n_trials=m).haircut("bonferroni") for m in (10, 100, 1000)]
    assert 0 < cuts[0] < cuts[1] < cuts[2] < 1
    weak = np.random.default_rng(7).normal(0.0008, 0.01, 2520)  # SR 0.69, p 0.014 alone
    assert haircut_sharpe(weak, n_trials=100).haircut("bonferroni") == 1.0
    res = haircut_sharpe(r, n_trials=50)
    assert res.adjusted_p["holm"] == pytest.approx(res.adjusted_p["bonferroni"])  # others null


def test_haircut_evidence_uses_actual_trials() -> None:
    rng = np.random.default_rng(8)
    noise = TrialSet.from_returns(rng.normal(0, 0.01, (1260, 100)))
    ev = haircut_evidence(noise, "s")
    assert ev.flag and ev.details["used_actual_trial_pvalues"] is True
    assert ev.details["n_trials"] == 100
