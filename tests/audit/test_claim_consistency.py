import math
from datetime import date, timedelta

import numpy as np
import pytest

from strategy_court.audit.claim_consistency import (
    check_claim,
    claim_evidence,
    drawdown_check,
    growth_bound_check,
    implied_vols,
    replication_check,
    simulate_mdd,
    track_record_check,
)
from strategy_court.schemas import ClaimedPerformance, CurvePoint, Hypothesis, Severity


def hyp(**claimed: object) -> Hypothesis:
    return Hypothesis(
        source_type="screenshot",
        universe=["SPY"],
        signal_description="claimed strategy",
        claimed=ClaimedPerformance.model_validate(claimed),
    )


def test_simulated_drawdown_matches_magdon_ismail_closed_form() -> None:
    """Driftless BM: E[MDD] = sqrt(pi/2) sigma sqrt(T) in log units (Magdon-Ismail et al. 2004).
    Sharpe = sigma/2 cancels the Ito term so the log-price is driftless."""
    sigma, years = 0.2, 5.0
    mdd = simulate_mdd(sharpe=sigma / 2, sigma=sigma, years=years, n_paths=6000)
    log_mdd = -np.log1p(-mdd)
    assert log_mdd.mean() == pytest.approx(
        math.sqrt(math.pi / 2) * sigma * math.sqrt(years), rel=0.04
    )


def test_growth_bound() -> None:
    assert growth_bound_check(1.0, 0.80).status == "implausible"
    assert growth_bound_check(2.0, 0.30).status == "consistent"
    assert growth_bound_check(3.5, 1.2).status == "consistent"  # loose at high Sharpe, by design
    assert implied_vols(1.0, 0.10) == pytest.approx(
        sorted([1 - math.sqrt(1 - 2 * math.log(1.1)), 1 + math.sqrt(1 - 2 * math.log(1.1))])
    )
    assert implied_vols(0.5, 0.80) == []


def test_drawdown_plausibility_both_tails() -> None:
    assert drawdown_check(3.0, 0.30, 0.45, 5.0).status == "implausible"  # far too deep
    assert drawdown_check(1.0, 0.15, 0.01, 10.0).status == "implausible"  # far too shallow
    assert drawdown_check(1.0, 0.12, 0.20, 10.0).status == "consistent"


def test_track_record_length() -> None:
    assert track_record_check(1.0, 1.0).status == "insufficient"
    assert track_record_check(2.0, 5.0).status == "consistent"


def _curve(
    sharpe: float, vol: float, years: int, seed: int, crash: float = 0.0
) -> list[CurvePoint]:
    rng = np.random.default_rng(seed)
    months = 12 * years
    mu = sharpe * vol / 12 - vol**2 / 24
    lr = mu + vol / math.sqrt(12) * rng.standard_normal(months)
    if crash:
        lr[months // 2] += math.log(1 - crash)
    v = 100 * np.exp(np.concatenate([[0.0], np.cumsum(lr)]))
    start = date(2015, 1, 1)
    return [
        CurvePoint(date=start + timedelta(days=round(30.44 * i)), value=x) for i, x in enumerate(v)
    ]


def test_curve_contradictions() -> None:
    curve = _curve(0.6, 0.15, 8, seed=1, crash=0.35)
    v = [p.value for p in curve]
    years = (curve[-1].date - curve[0].date).days / 365.25
    true_cagr = (v[-1] / v[0]) ** (1 / years) - 1
    claim = Hypothesis(
        source_type="screenshot",
        universe=["SPY"],
        signal_description="x",
        claimed=ClaimedPerformance(sharpe=3.0, cagr=true_cagr, max_drawdown=0.08),
        claimed_curve=curve,
    )
    by_check = {f.check: f.status for f in check_claim(claim).findings}
    assert by_check["curve_drawdown"] == "contradiction"
    assert by_check["curve_sharpe"] == "contradiction"
    assert by_check["curve_cagr"] == "consistent"


def test_replication_gap() -> None:
    assert replication_check(3.1, 4.0, [0.1, 0.3, -0.2]).status == "contradiction"
    assert replication_check(1.0, 4.0, [0.9, 0.4]).status == "consistent"


def test_evidence_for_viral_and_sober_claims() -> None:
    viral = hyp(  # "Sharpe 1.2, +150% a year, 3% max drawdown"
        sharpe=1.2,
        cagr=1.5,
        max_drawdown=0.03,
        period_start=date(2023, 1, 1),
        period_end=date(2024, 1, 1),
    )
    ev = claim_evidence(viral, "s", replicated_sharpes=[0.2, 0.4])
    assert ev.flag and ev.severity is Severity.HIGH
    statuses = {f["check"]: f["status"] for f in ev.details["findings"]}  # type: ignore[index, union-attr]
    assert statuses["growth_bound"] == "implausible"
    assert statuses["track_record"] == "insufficient"
    # one year is too short for replications at 0.2-0.4 to *contradict* a Sharpe of 1.2
    assert statuses["replication"] == "consistent"
    sober = hyp(
        sharpe=0.8,
        cagr=0.09,
        max_drawdown=0.22,
        period_start=date(2010, 1, 1),
        period_end=date(2024, 1, 1),
    )
    assert not claim_evidence(sober, "s").flag
    assert claim_evidence(hyp(), "s").details["findings"][0]["status"] == "not_checkable"  # type: ignore[index]
