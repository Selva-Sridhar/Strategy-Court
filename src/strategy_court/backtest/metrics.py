"""Performance metrics on simple per-period returns.

Conventions (stated, because audit tools depend on them)
--------------------------------------------------------
* Sharpe ratio: mean / standard deviation (ddof=1) of returns in excess of the per-period
  risk-free rate, annualised by sqrt(periods_per_year) (iid scaling; Lo 2002 shows the
  correction needed under serial correlation, applied in the inference tools, not here).
* Skewness and kurtosis are the plain moment estimators (bias=True); kurtosis is *raw*
  (normal = 3), the form used by PSR / DSR (Bailey & López de Prado 2012, 2014).
* Sortino ratio: excess mean over the target downside deviation computed over *all*
  periods, sqrt(mean(min(r - MAR, 0)^2)) (Sortino & van der Meer 1991).
* Max drawdown is a positive fraction of the running equity peak.
* Turnover is two-sided (sum of |traded value| / equity); holding period = gross exposure /
  one-way turnover.
* Degenerate inputs (zero variance, no downside, no drawdown) yield NaN, never inf.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from strategy_court.backtest.engine import BacktestResult

FloatArray = np.ndarray


def _arr(x: object) -> FloatArray:
    return np.asarray(x, dtype=np.float64).ravel()


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 and np.isfinite(den) else float("nan")


def _noise_floor(x: FloatArray) -> float:
    """Dispersion below this is floating-point noise (e.g. a constant series), i.e. zero.

    Returns computed as P1/P0 - 1 carry absolute error ~eps, hence the absolute 1e-12 floor
    (a per-period stdev of 1e-12 is economically zero)."""
    return float(max(64 * np.finfo(np.float64).eps * np.abs(x).max(initial=0.0), 1e-12))


def _dispersion(x: FloatArray, ddof: int = 1) -> float:
    sd = float(x.std(ddof=ddof))
    return sd if sd > _noise_floor(x) else 0.0


def sharpe_ratio(returns: object, rf_per_period: float = 0.0, periods_per_year: int = 252) -> float:
    r = _arr(returns) - rf_per_period
    if r.size < 2:
        return float("nan")
    return float(_safe_div(r.mean(), _dispersion(r)) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: object, rf_per_period: float = 0.0, periods_per_year: int = 252, mar: float = 0.0
) -> float:
    r = _arr(returns) - rf_per_period
    if r.size < 2:
        return float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(r - mar, 0.0) ** 2)))
    downside = downside if downside > _noise_floor(r) else 0.0
    return float(_safe_div(r.mean() - mar, downside) * np.sqrt(periods_per_year))


def annualised_volatility(returns: object, periods_per_year: int = 252) -> float:
    r = _arr(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if r.size >= 2 else float("nan")


def cagr(equity: object, periods_per_year: int = 252) -> float:
    e = _arr(equity)
    if e.size < 2 or e[0] <= 0:
        return float("nan")
    if e[-1] <= 0:
        return -1.0
    return float((e[-1] / e[0]) ** (periods_per_year / (e.size - 1)) - 1.0)


def drawdown_series(equity: object) -> FloatArray:
    e = _arr(equity)
    peak = np.maximum.accumulate(e)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(peak > 0, 1.0 - e / np.where(peak > 0, peak, 1.0), 0.0)


def max_drawdown(equity: object) -> float:
    dd = drawdown_series(equity)
    return float(dd.max()) if dd.size else float("nan")


def max_drawdown_duration(equity: object) -> int:
    """Longest run of consecutive bars strictly below the previous equity peak."""
    under = drawdown_series(equity) > 0
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def skewness(returns: object) -> float:
    r = _arr(returns)
    if r.size < 3 or _dispersion(r, ddof=0) == 0:
        return float("nan")
    return float(stats.skew(r, bias=True))


def kurtosis_raw(returns: object) -> float:
    r = _arr(returns)
    if r.size < 4 or _dispersion(r, ddof=0) == 0:
        return float("nan")
    return float(stats.kurtosis(r, fisher=False, bias=True))


@dataclass(frozen=True)
class PerformanceSummary:
    n_periods: int
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar: float
    skewness: float
    kurtosis: float
    hit_rate: float
    annual_turnover: float
    avg_gross_exposure: float
    avg_net_exposure: float
    avg_holding_period: float
    cost_drag_annual: float
    ruined: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def summarize(result: BacktestResult) -> PerformanceSummary:
    cfg = result.config
    ppy = cfg.periods_per_year
    r = result.returns.to_numpy(dtype=np.float64)
    eq = result.equity.to_numpy(dtype=np.float64)
    turn = result.turnover.iloc[1:].to_numpy(dtype=np.float64)
    gross = result.gross_exposure.iloc[1:].to_numpy(dtype=np.float64)
    active = r[r != 0]
    mdd = max_drawdown(eq)
    growth = cagr(eq, ppy)
    one_way = turn.mean() / 2 if turn.size else 0.0
    return PerformanceSummary(
        n_periods=int(r.size),
        total_return=float(eq[-1] / eq[0] - 1.0),
        cagr=growth,
        annual_volatility=annualised_volatility(r, ppy),
        sharpe=sharpe_ratio(r, cfg.rf_per_period, ppy),
        sortino=sortino_ratio(r, cfg.rf_per_period, ppy),
        max_drawdown=mdd,
        max_drawdown_duration=max_drawdown_duration(eq),
        calmar=_safe_div(growth, mdd),
        skewness=skewness(r),
        kurtosis=kurtosis_raw(r),
        hit_rate=float((active > 0).mean()) if active.size else float("nan"),
        annual_turnover=float(turn.mean() * ppy) if turn.size else 0.0,
        avg_gross_exposure=float(gross.mean()) if gross.size else 0.0,
        avg_net_exposure=float(result.net_exposure.iloc[1:].mean()),
        avg_holding_period=_safe_div(float(gross.mean()) if gross.size else 0.0, one_way),
        cost_drag_annual=float(result.cost_drag.mean() * ppy),
        ruined=result.ruined_at is not None,
    )
