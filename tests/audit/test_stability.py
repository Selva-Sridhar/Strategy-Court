import numpy as np
import pandas as pd
import pytest
from scipy import stats

from strategy_court.audit.stability import (
    equal_means_wald,
    stability_evidence,
    stability_test,
    sup_wald,
    sup_wald_null,
    sup_wald_pvalue,
)
from strategy_court.schemas import Severity


def series(x: np.ndarray) -> pd.Series:
    return pd.Series(x, index=pd.bdate_range("2012-01-02", periods=x.size))


def test_simulated_null_matches_andrews_1993_critical_values() -> None:
    """Andrews (1993) Table 1, one restriction, 15% trimming: 10% ~ 7.17, 5% ~ 8.85, 1% ~ 12.35."""
    null = sup_wald_null()
    assert np.quantile(null, 0.90) == pytest.approx(7.17, abs=0.35)
    assert np.quantile(null, 0.95) == pytest.approx(8.85, abs=0.40)
    assert np.quantile(null, 0.99) == pytest.approx(12.35, abs=0.75)
    assert sup_wald_pvalue(8.85) == pytest.approx(0.05, abs=0.01)


def test_sup_wald_size_and_break_location() -> None:
    rng = np.random.default_rng(0)
    pvals = [sup_wald_pvalue(sup_wald(rng.normal(0, 1, 800))[0]) for _ in range(300)]
    assert 0.02 < np.mean(np.array(pvals) < 0.05) < 0.09
    x = np.concatenate([rng.normal(0.3, 1, 600), rng.normal(-0.1, 1, 400)])
    stat, tau = sup_wald(x)
    assert sup_wald_pvalue(stat) < 0.01
    assert abs(tau - 600) < 60


def test_equal_means_wald_is_chi2_under_the_null() -> None:
    rng = np.random.default_rng(1)
    pvals = [equal_means_wald(rng.normal(0, 1, 1000), 5)[1] for _ in range(400)]
    assert stats.kstest(pvals, "uniform").pvalue > 0.01


def test_regime_dependent_strategy_is_flagged() -> None:
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(0.0015, 0.01, 1260), rng.normal(-0.0003, 0.01, 1260)])
    res = stability_test(series(x))
    assert res.sup_wald_p < 0.05 and res.sharpe_after < 0.5 * res.sharpe_before
    assert res.break_date is not None and res.break_date.startswith(("2016", "2017"))
    ev = res.to_evidence("s")
    assert ev.flag and ev.severity is Severity.HIGH
    assert len(res.yearly_sharpes) == 10


def test_stable_strategy_passes() -> None:
    rng = np.random.default_rng(3)
    res = stability_test(series(rng.normal(0.001, 0.01, 2520)))
    assert res.frac_blocks_positive == 1.0 and res.sup_wald_p > 0.05
    assert not res.to_evidence("s").flag
    assert stability_evidence(series(np.zeros(100)), "s").details["status"] == "inconclusive"
