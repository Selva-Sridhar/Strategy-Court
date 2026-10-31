"""Internal consistency of a performance claim, checked before (and independently of) any
replication. Each check returns a finding: consistent / implausible / contradiction /
insufficient / not_checkable.

1. Growth bound (Kelly 1956): for log-normal (GBM) wealth with excess Sharpe SR, log growth is
   rf + SR sigma - sigma^2 / 2, maximised at sigma = SR, so ln(1 + CAGR) <= rf + SR^2 / 2. A claim
   above the bound needs strongly positive skew or leverage the Sharpe ratio does not show.
2. Drawdown plausibility: with sigma implied by (SR, CAGR) — both roots of the growth equation
   are tried and the one most favourable to the claimant is kept, except roots above `max_vol`
   (default 100%/yr: the upper root lies beyond growth-optimal leverage and, at high Sharpe,
   implies absurd volatility that would excuse any drawdown) — simulate the maximum drawdown
   over the claimed horizon. Claimed MDD too *small* is judged under normal shocks (fat tails
   only make small drawdowns rarer); too *large* under Student-t(4) shocks (fat tails make large
   drawdowns likelier), so both judgements are conservative. Simulation is validated against
   the driftless closed form E[MDD] = sqrt(pi/2) sigma sqrt(T) (Magdon-Ismail et al. 2004).
3. Track-record length: the claimed period must exceed the Minimum Track Record Length of the
   claimed Sharpe (Bailey & López de Prado 2012).
4. Digitised curve (DePlot-style extraction, Liu et al. 2023): implied CAGR and Sharpe are
   compared with the claim; the curve's drawdown is a *lower bound* (sampling misses intra-point
   dips), so a claimed MDD smaller than it is a contradiction.
5. Replication gap: claimed Sharpe vs the best of K independent implementations of the claim,
   in units of the Sharpe standard error (Lo 2002): > 3 SE contradiction, 2-3 SE doubtful.
   The SE of the claimed estimate alone is used; claim and replication share data in most cases,
   which makes the true SE of their difference smaller, so the z-score is conservative.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from strategy_court.audit._common import make_evidence
from strategy_court.audit.citations import (
    BAILEY_LDP_2012,
    KELLY_1956,
    LIU_2023_DEPLOT,
    LO_2002,
    MAGDON_ISMAIL_2004,
)
from strategy_court.audit.psr import min_track_record_length
from strategy_court.schemas import CurvePoint, Evidence, Hypothesis, Severity

TOOL = "claim_consistency"
CITATIONS = (KELLY_1956, MAGDON_ISMAIL_2004, BAILEY_LDP_2012, LO_2002, LIU_2023_DEPLOT)
Status = Literal[
    "consistent", "doubtful", "implausible", "contradiction", "insufficient", "not_checkable"
]
PPY = 252


@dataclass(frozen=True)
class Finding:
    check: str
    status: Status
    message: str
    values: dict[str, float | None] = field(default_factory=dict)


def growth_bound_check(sharpe: float, cagr: float, rf_max: float = 0.05) -> Finding:
    g = math.log1p(cagr)
    bound = rf_max + sharpe**2 / 2 if sharpe > 0 else rf_max
    vals: dict[str, float | None] = {
        "log_growth": g,
        "max_log_growth": bound,
        "kelly_max_cagr": math.expm1(bound),
    }
    if g > bound:
        return Finding(
            "growth_bound",
            "implausible",
            f"CAGR {cagr:.0%} exceeds the maximum {math.expm1(bound):.0%} any log-normal "
            f"strategy with Sharpe {sharpe:.2f} can compound at (Kelly bound).",
            vals,
        )
    return Finding("growth_bound", "consistent", "CAGR is attainable at the claimed Sharpe.", vals)


def implied_vols(sharpe: float, cagr: float, rf: float = 0.0) -> list[float]:
    """Positive roots sigma of ln(1 + CAGR) - rf = SR sigma - sigma^2 / 2."""
    e = math.log1p(cagr) - rf
    disc = sharpe**2 - 2 * e
    if disc < 0:
        return []
    roots = [sharpe - math.sqrt(disc), sharpe + math.sqrt(disc)]
    return sorted({r for r in roots if r > 1e-6})


def simulate_mdd(
    sharpe: float,
    sigma: float,
    years: float,
    rf: float = 0.0,
    n_paths: int = 4000,
    shocks: Literal["normal", "t4"] = "normal",
    seed: int = 0,
) -> np.ndarray:
    """Maximum drawdowns (fractions) of GBM wealth with the given excess Sharpe and vol."""
    n = max(2, round(years * PPY))
    rng = np.random.default_rng(seed)
    drift = (rf + sharpe * sigma - sigma**2 / 2) / PPY
    out = np.empty(n_paths)
    for start in range(0, n_paths, 1000):
        k = min(1000, n_paths - start)
        if shocks == "normal":
            z = rng.standard_normal((k, n))
        else:
            z = rng.standard_t(4, (k, n)) / math.sqrt(2.0)
        x = np.cumsum(drift + sigma / math.sqrt(PPY) * z, axis=1)
        x = np.concatenate([np.zeros((k, 1)), x], axis=1)
        dd = (x - np.maximum.accumulate(x, axis=1)).min(axis=1)
        out[start : start + k] = -np.expm1(dd)
    return out


def drawdown_check(
    sharpe: float,
    cagr: float,
    mdd: float,
    years: float,
    rf: float = 0.0,
    seed: int = 0,
    max_vol: float = 1.0,
) -> Finding:
    vols = [s for s in implied_vols(sharpe, cagr, rf) if s <= max_vol]
    if not vols:
        return Finding(
            "drawdown",
            "not_checkable",
            f"no volatility up to {max_vol:.0%}/yr reconciles the claimed Sharpe and CAGR.",
        )
    p_small = max(
        float((simulate_mdd(sharpe, s, years, rf, seed=seed) <= mdd).mean()) for s in vols
    )
    p_large = max(
        float((simulate_mdd(sharpe, s, years, rf, shocks="t4", seed=seed) >= mdd).mean())
        for s in vols
    )
    vals: dict[str, float | None] = {
        "p_mdd_this_small": p_small,
        "p_mdd_this_large": p_large,
        "implied_vol_low": vols[0],
    }
    if p_small < 0.01:
        return Finding(
            "drawdown",
            "implausible",
            f"A {mdd:.0%} max drawdown over {years:.1f} years is too shallow for Sharpe "
            f"{sharpe:.2f} at {cagr:.0%} CAGR (P = {p_small:.3f}).",
            vals,
        )
    if p_large < 0.01:
        return Finding(
            "drawdown",
            "implausible",
            f"A {mdd:.0%} max drawdown is too deep for a genuine Sharpe of {sharpe:.2f} "
            f"(P = {p_large:.3f} even with fat tails).",
            vals,
        )
    return Finding("drawdown", "consistent", "Drawdown is plausible for the claimed Sharpe.", vals)


def track_record_check(sharpe: float, years: float, alpha: float = 0.05) -> Finding:
    sr = sharpe / math.sqrt(PPY)
    need = min_track_record_length(sr, 0.0, 3.0, 0.0, alpha) / PPY
    vals = {"min_track_record_years": need if math.isfinite(need) else None, "claimed_years": years}
    if years < need:
        return Finding(
            "track_record",
            "insufficient",
            f"{years:.1f} years cannot establish a Sharpe of {sharpe:.2f}; at least "
            f"{need:.1f} years are needed even with normal returns.",
            vals,
        )
    return Finding("track_record", "consistent", "Track record long enough for the claim.", vals)


def curve_checks(curve: Sequence[CurvePoint], hyp: Hypothesis) -> list[Finding]:
    if len(curve) < 12:
        return [Finding("curve", "not_checkable", "fewer than 12 digitised curve points.")]
    v = np.array([p.value for p in curve], dtype=np.float64)
    days = np.array([(p.date - curve[0].date).days for p in curve], dtype=np.float64)
    years = days[-1] / 365.25
    lr = np.diff(np.log(v))
    per_year = lr.size / years
    out = []
    c = hyp.claimed
    curve_cagr = (v[-1] / v[0]) ** (1 / years) - 1
    if c.cagr is not None:
        tol = max(0.02, 0.1 * abs(c.cagr))
        status: Status = "contradiction" if abs(curve_cagr - c.cagr) > tol else "consistent"
        out.append(
            Finding(
                "curve_cagr",
                status,
                f"Chart implies {curve_cagr:.1%} CAGR vs {c.cagr:.1%} claimed.",
                {"curve_cagr": curve_cagr, "claimed_cagr": c.cagr},
            )
        )
    curve_mdd = float(-np.expm1((np.log(v) - np.maximum.accumulate(np.log(v))).min()))
    if c.max_drawdown is not None:
        status = "contradiction" if c.max_drawdown < curve_mdd - 0.02 else "consistent"
        out.append(
            Finding(
                "curve_drawdown",
                status,
                f"Chart shows at least a {curve_mdd:.0%} drawdown vs {c.max_drawdown:.0%} claimed.",
                {"curve_mdd_lower_bound": curve_mdd, "claimed_mdd": c.max_drawdown},
            )
        )
    sd = lr.std(ddof=1)
    if c.sharpe is not None and sd > 0:
        sr_pp = lr.mean() / sd
        curve_sr = sr_pp * math.sqrt(per_year)
        se = math.sqrt((1 + sr_pp**2 / 2) / lr.size) * math.sqrt(per_year)
        z = (c.sharpe - curve_sr) / se
        status = "contradiction" if abs(z) > 3 else "consistent"
        out.append(
            Finding(
                "curve_sharpe",
                status,
                f"Chart implies Sharpe {curve_sr:.2f} ± {se:.2f} vs {c.sharpe:.2f} claimed "
                f"(z = {z:.1f}).",
                {"curve_sharpe": curve_sr, "se": se, "z": z},
            )
        )
    return out


def replication_check(sharpe: float, years: float, replicated: Sequence[float]) -> Finding:
    best = max(replicated)
    sr_pp = sharpe / math.sqrt(PPY)
    se = math.sqrt((1 + sr_pp**2 / 2) / (years * PPY)) * math.sqrt(PPY)
    z = (sharpe - best) / se
    vals: dict[str, float | None] = {
        "best_replicated_sharpe": best,
        "se": se,
        "z": z,
        "n_versions": float(len(replicated)),
    }
    if z > 3:
        return Finding(
            "replication",
            "contradiction",
            f"None of {len(replicated)} independent implementations gets near the claimed "
            f"Sharpe {sharpe:.2f} (best {best:.2f}, {z:.1f} standard errors short).",
            vals,
        )
    if z > 2:
        return Finding(
            "replication",
            "doubtful",
            f"The best of {len(replicated)} implementations reaches Sharpe {best:.2f} vs "
            f"{sharpe:.2f} claimed ({z:.1f} standard errors short).",
            vals,
        )
    return Finding(
        "replication", "consistent", "At least one implementation reproduces the claim.", vals
    )


def _years(hyp: Hypothesis) -> float | None:
    c = hyp.claimed
    if c.period_start and c.period_end:
        return (c.period_end - c.period_start).days / 365.25
    if hyp.claimed_curve and len(hyp.claimed_curve) >= 2:
        return (hyp.claimed_curve[-1].date - hyp.claimed_curve[0].date).days / 365.25
    return None


@dataclass(frozen=True)
class ClaimReport:
    findings: tuple[Finding, ...]

    def to_evidence(self, strategy_id: str) -> Evidence:
        # A too-short track record is missing evidence, not a flaw: reported (LOW, unflagged) so the
        # Judge can say so, but it cannot by itself count towards a conviction.
        bad = [f for f in self.findings if f.status in ("implausible", "contradiction")]
        doubtful = [f for f in self.findings if f.status == "doubtful"]
        short = [f for f in self.findings if f.status == "insufficient"]
        checked = [f for f in self.findings if f.status != "not_checkable"]
        sev = (
            Severity.HIGH
            if bad
            else Severity.MEDIUM
            if doubtful
            else Severity.LOW
            if short
            else Severity.INFO
        )
        if bad:
            summary = "Claim is internally inconsistent: " + " ".join(f.message for f in bad)
        elif doubtful or short:
            summary = " ".join(f.message for f in (*doubtful, *short))
        elif checked:
            summary = f"All {len(checked)} consistency checks passed."
        else:
            summary = "Claim states too little to check its consistency."
        return make_evidence(
            TOOL,
            strategy_id,
            flag=bool(bad or doubtful),
            severity=sev,
            statistic_name="n_inconsistencies",
            statistic=float(len(bad)),
            summary=summary,
            details={
                "findings": [
                    {"check": f.check, "status": f.status, "message": f.message, **f.values}
                    for f in self.findings
                ]
            },
            citations=CITATIONS,
        )


def check_claim(
    hyp: Hypothesis, replicated_sharpes: Sequence[float] | None = None, rf_max: float = 0.05
) -> ClaimReport:
    c = hyp.claimed
    years = _years(hyp)
    findings: list[Finding] = []
    if c.sharpe is not None and c.cagr is not None:
        findings.append(growth_bound_check(c.sharpe, c.cagr, rf_max))
        if c.max_drawdown is not None and years:
            findings.append(drawdown_check(c.sharpe, c.cagr, c.max_drawdown, years))
    if c.sharpe is not None and c.sharpe > 0 and years:
        findings.append(track_record_check(c.sharpe, years))
    if hyp.claimed_curve:
        findings.extend(curve_checks(hyp.claimed_curve, hyp))
    if c.sharpe is not None and years and replicated_sharpes:
        findings.append(replication_check(c.sharpe, years, replicated_sharpes))
    if not findings:
        findings.append(Finding("claim", "not_checkable", "no quantitative claim to check."))
    return ClaimReport(tuple(findings))


def claim_evidence(
    hyp: Hypothesis, strategy_id: str, replicated_sharpes: Sequence[float] | None = None
) -> Evidence:
    return check_claim(hyp, replicated_sharpes).to_evidence(strategy_id)
