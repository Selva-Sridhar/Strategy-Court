"""Batched, vectorised daily backtest engine.

Timing (no same-bar fills by construction)
------------------------------------------
Weights W[t] are decided at the close of bar t and executed at bar t+1 at the execution price
X[t+1] (next open, or next close). Bar s is simulated in two legs:

1. Delisting: positions in assets whose last close was bar s-1 are liquidated at that close
   times (1 + delisting_return).
2. Leg 1: holdings earn X[s] / C[s-1] - 1 (overnight leg, or the full bar under NEXT_CLOSE).
3. Trade at X[s] to target values W[s-1] * equity. NaN weight = hold, i.e. no trade.
   Costs (linear + market impact) are paid from cash.
4. Borrow fees on shorts and the risk-free rate on cash accrue over the bar.
5. Leg 2: the new holdings earn C[s] / X[s] - 1.

The state is held in currency (holding values and cash), so weight drift between rebalances
is exact, and turnover / costs are measured against drifted, not previous-target, weights.
Equity that reaches zero or below is ruin: the path is frozen at zero.

All paths of a (P, T, N) weight tensor are simulated together; the only Python loop is over
bars, every operation inside it is vectorised over paths and assets.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_court.backtest.config import BacktestConfig, ExecutionTiming
from strategy_court.backtest.market import MarketData


class WeightError(ValueError):
    """Strategy output does not satisfy the weight contract."""


@dataclass(frozen=True)
class Diagnostics:
    """Per-path counts of everything the engine had to override. Nothing is silent."""

    blocked_orders: np.ndarray  # non-zero weight on an asset not trading at that bar
    leverage_scaled_bars: np.ndarray  # bars where targets were scaled to max_gross_leverage
    impact_fallback_trades: np.ndarray  # trades charged the fallback because sigma/ADV unknown
    ruined_at: np.ndarray  # first bar with equity <= 0, or -1


@dataclass(frozen=True)
class BatchResult:
    equity: np.ndarray  # (P, T) currency, equity[:, 0] = initial capital
    returns: np.ndarray  # (P, T) net simple returns, returns[:, 0] = 0
    turnover: np.ndarray  # (P, T) sum |traded value| / pre-trade equity (two-sided)
    costs: np.ndarray  # (P, T) currency: linear + impact + borrow
    gross_exposure: np.ndarray  # (P, T) sum |post-trade holdings| / pre-trade equity
    net_exposure: np.ndarray  # (P, T)
    weights: np.ndarray | None  # (P, T, N) post-trade executed weights, if recorded
    diagnostics: Diagnostics


@dataclass(frozen=True)
class _Prepared:
    alive: np.ndarray
    newly_dead: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    sigma: np.ndarray | None
    adv: np.ndarray | None


def _prepare(market: MarketData, config: BacktestConfig) -> _Prepared:
    close = market.close.to_numpy(dtype=np.float64)
    alive = market.alive
    exec_px = close
    if config.timing is ExecutionTiming.NEXT_OPEN and market.open is not None:
        opn = market.open.to_numpy(dtype=np.float64)
        exec_px = np.where(np.isfinite(opn), opn, close)
    prev_close = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    held = np.vstack([np.zeros((1, close.shape[1]), dtype=bool), alive[:-1] & alive[1:]])
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = np.where(held, exec_px / prev_close - 1.0, 0.0)
        r2 = np.where(alive, close / exec_px - 1.0, 0.0)
    newly_dead = np.vstack([np.zeros((1, close.shape[1]), dtype=bool), alive[:-1] & ~alive[1:]])

    sigma = adv = None
    costs = config.costs
    if costs.impact_coefficient > 0:
        if market.volume is None:
            raise WeightError("market impact is enabled but no volume data was provided")
        w = costs.impact_window
        cc = market.close.pct_change(fill_method=None)
        sigma = cc.rolling(w, min_periods=2).std().shift(1).to_numpy(dtype=np.float64)
        dollar_volume = market.close * market.volume
        adv = dollar_volume.rolling(w, min_periods=1).mean().shift(1).to_numpy(dtype=np.float64)
    return _Prepared(alive, newly_dead, r1, r2, sigma, adv)


def validate_weights(weights: pd.DataFrame, market: MarketData) -> np.ndarray:
    """Check the strategy contract: same index/columns as close, finite or NaN (= hold)."""
    if not isinstance(weights, pd.DataFrame):
        raise WeightError(f"signals() must return a DataFrame, got {type(weights).__name__}")
    if not weights.index.equals(market.close.index):
        raise WeightError("weights index must equal the price index")
    if not weights.columns.equals(market.close.columns):
        raise WeightError("weights columns must equal the price columns")
    try:
        values = weights.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WeightError(f"weights must be numeric: {exc}") from exc
    if np.isinf(values).any():
        raise WeightError("weights contain +/-inf")
    return values


def simulate(
    weights: np.ndarray,
    market: MarketData,
    config: BacktestConfig | None = None,
    record_weights: bool = False,
) -> BatchResult:
    """Simulate P weight paths of shape (P, T, N) (or a single (T, N) path)."""
    cfg = config or BacktestConfig()
    w_all = np.asarray(weights, dtype=np.float64)
    if w_all.ndim == 2:
        w_all = w_all[None]
    n_bars, n_assets = market.shape
    if w_all.ndim != 3 or w_all.shape[1:] != (n_bars, n_assets):
        raise WeightError(f"weights must have shape (P, {n_bars}, {n_assets}), got {w_all.shape}")
    if np.isinf(w_all).any():
        raise WeightError("weights contain +/-inf")

    prep = _prepare(market, cfg)
    n_paths = w_all.shape[0]
    costs = cfg.costs
    linear = costs.linear_bps / 1e4
    borrow = costs.borrow_bps_annual / 1e4 / cfg.periods_per_year
    rf = cfg.rf_per_period
    keep = 1.0 + cfg.delisting_return

    h = np.zeros((n_paths, n_assets))
    cash = np.full(n_paths, cfg.initial_capital)
    equity = np.zeros((n_paths, n_bars))
    equity[:, 0] = cfg.initial_capital
    turnover = np.zeros((n_paths, n_bars))
    cost_out = np.zeros((n_paths, n_bars))
    gross = np.zeros((n_paths, n_bars))
    net = np.zeros((n_paths, n_bars))
    rec = np.zeros((n_paths, n_bars, n_assets)) if record_weights else None
    blocked = np.zeros(n_paths, dtype=np.int64)
    scaled = np.zeros(n_paths, dtype=np.int64)
    fallback = np.zeros(n_paths, dtype=np.int64)
    ruined_at = np.full(n_paths, -1, dtype=np.int64)

    for s in range(1, n_bars):
        w = w_all[:, s - 1, :]
        alive = prep.alive[s]

        dead = prep.newly_dead[s]
        if dead.any():
            cash += h[:, dead].sum(axis=1) * keep
            h[:, dead] = 0.0

        h *= 1.0 + prep.r1[s]
        e1 = cash + h.sum(axis=1)
        broke = e1 <= 0
        if broke.any():
            h[broke] = 0.0
            cash[broke] = 0.0
            e1[broke] = 0.0

        instructed = ~np.isnan(w)
        blocked += (instructed & (w != 0) & ~alive).sum(axis=1)
        target = np.where(instructed, w * e1[:, None], h)
        target[:, ~alive] = 0.0

        if cfg.max_gross_leverage is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                lev = np.where(e1 > 0, np.abs(target).sum(axis=1) / e1, 0.0)
            over = lev > cfg.max_gross_leverage
            if over.any():
                target[over] *= (cfg.max_gross_leverage / lev[over])[:, None]
                scaled += over

        trade = target - h
        traded = np.abs(trade)
        cost = traded.sum(axis=1) * linear
        if prep.sigma is not None and prep.adv is not None:
            sig, adv = prep.sigma[s], prep.adv[s]
            known = np.isfinite(sig) & np.isfinite(adv) & (adv > 0)
            # A near-zero ADV gives (correctly) unbounded impact; overflow to inf => ruin.
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                participation = np.where(known, traded / np.where(known, adv, 1.0), 0.0)
                impact = np.where(
                    known,
                    costs.impact_coefficient
                    * np.where(known, sig, 0.0)
                    * participation**costs.impact_exponent
                    * traded,
                    traded * costs.impact_fallback_bps / 1e4,
                )
            cost += impact.sum(axis=1)
            fallback += ((traded > 0) & ~known).sum(axis=1)

        h = target
        cash = cash - trade.sum(axis=1) - cost
        fee = np.clip(-h, 0.0, None).sum(axis=1) * borrow
        cash = (cash - fee) * (1.0 + rf)
        cost += fee

        with np.errstate(invalid="ignore", divide="ignore"):
            pos = e1 > 0
            turnover[:, s] = np.where(pos, traded.sum(axis=1) / np.where(pos, e1, 1.0), 0.0)
            gross[:, s] = np.where(pos, np.abs(h).sum(axis=1) / np.where(pos, e1, 1.0), 0.0)
            net[:, s] = np.where(pos, h.sum(axis=1) / np.where(pos, e1, 1.0), 0.0)
            if rec is not None:
                rec[:, s, :] = np.where(pos[:, None], h / np.where(pos, e1, 1.0)[:, None], 0.0)
        cost_out[:, s] = cost

        h *= 1.0 + prep.r2[s]
        e = cash + h.sum(axis=1)
        gone = e <= 0
        if gone.any():
            h[gone] = 0.0
            cash[gone] = 0.0
            e[gone] = 0.0
            ruined_at[(ruined_at < 0) & gone] = s
        equity[:, s] = e

    prev = equity[:, :-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        step = np.where(prev > 0, equity[:, 1:] / np.where(prev > 0, prev, 1.0) - 1.0, 0.0)
    returns = np.hstack([np.zeros((n_paths, 1)), step])
    return BatchResult(
        equity=equity,
        returns=returns,
        turnover=turnover,
        costs=cost_out,
        gross_exposure=gross,
        net_exposure=net,
        weights=rec,
        diagnostics=Diagnostics(blocked, scaled, fallback, ruined_at),
    )


@dataclass(frozen=True)
class BacktestResult:
    """Single-path result with labelled pandas objects."""

    equity: pd.Series
    returns: pd.Series  # from bar 1 (bar 0 has no return)
    turnover: pd.Series
    costs: pd.Series  # currency
    gross_exposure: pd.Series
    net_exposure: pd.Series
    weights: pd.DataFrame  # post-trade executed weights
    blocked_orders: int
    leverage_scaled_bars: int
    impact_fallback_trades: int
    ruined_at: pd.Timestamp | None
    config: BacktestConfig

    @property
    def cost_drag(self) -> pd.Series:
        """Costs as a fraction of the previous bar's equity."""
        prev = self.equity.shift(1).iloc[1:]
        return (self.costs.iloc[1:] / prev.where(prev > 0)).fillna(0.0)


def run_backtest(
    weights: pd.DataFrame, market: MarketData, config: BacktestConfig | None = None
) -> BacktestResult:
    cfg = config or BacktestConfig()
    res = simulate(validate_weights(weights, market), market, cfg, record_weights=True)
    idx = market.index
    assert res.weights is not None
    ruin = int(res.diagnostics.ruined_at[0])
    return BacktestResult(
        equity=pd.Series(res.equity[0], idx, name="equity"),
        returns=pd.Series(res.returns[0, 1:], idx[1:], name="returns"),
        turnover=pd.Series(res.turnover[0], idx, name="turnover"),
        costs=pd.Series(res.costs[0], idx, name="costs"),
        gross_exposure=pd.Series(res.gross_exposure[0], idx, name="gross_exposure"),
        net_exposure=pd.Series(res.net_exposure[0], idx, name="net_exposure"),
        weights=pd.DataFrame(res.weights[0], idx, market.close.columns),
        blocked_orders=int(res.diagnostics.blocked_orders[0]),
        leverage_scaled_bars=int(res.diagnostics.leverage_scaled_bars[0]),
        impact_fallback_trades=int(res.diagnostics.impact_fallback_trades[0]),
        ruined_at=idx[ruin] if ruin >= 0 else None,
        config=cfg,
    )
