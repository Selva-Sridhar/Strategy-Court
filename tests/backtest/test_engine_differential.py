"""Differential test: the vectorised, value-based engine must match an independent, share-based
ledger written with plain Python loops (no shared code), on hundreds of random markets."""

import math

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strategy_court.backtest import BacktestConfig, CostModel, ExecutionTiming, MarketData
from strategy_court.backtest.engine import simulate


def ledger(
    close: np.ndarray,
    opn: np.ndarray | None,
    volume: np.ndarray | None,
    weights: np.ndarray,
    cfg: BacktestConfig,
) -> list[float]:
    n_bars, n_assets = close.shape
    c = cfg.costs
    valid = [[not math.isnan(close[t][i]) for i in range(n_assets)] for t in range(n_bars)]
    first = [min(t for t in range(n_bars) if valid[t][i]) for i in range(n_assets)]
    last = [max(t for t in range(n_bars) if valid[t][i]) for i in range(n_assets)]

    def alive(t: int, i: int) -> bool:
        return first[i] <= t <= last[i]

    def sigma(s: int, i: int) -> float:
        rets = [
            close[j][i] / close[j - 1][i] - 1
            for j in range(max(1, s - c.impact_window), s)
            if valid[j][i] and valid[j - 1][i]
        ]
        if len(rets) < 2:
            return math.nan
        m = sum(rets) / len(rets)
        return math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))

    def adv(s: int, i: int) -> float:
        assert volume is not None
        dv = [
            close[j][i] * volume[j][i]
            for j in range(max(0, s - c.impact_window), s)
            if valid[j][i] and not math.isnan(volume[j][i])
        ]
        return sum(dv) / len(dv) if dv else math.nan

    rf = (1 + cfg.risk_free_rate) ** (1 / cfg.periods_per_year) - 1
    shares = [0.0] * n_assets
    cash = cfg.initial_capital
    equity = [cash]
    ruined = False
    for s in range(1, n_bars):
        if ruined:
            equity.append(0.0)
            continue
        for i in range(n_assets):
            if alive(s - 1, i) and not alive(s, i):
                cash += shares[i] * close[s - 1][i] * (1 + cfg.delisting_return)
                shares[i] = 0.0
        px = [0.0] * n_assets
        for i in range(n_assets):
            if alive(s, i):
                use_open = (
                    cfg.timing is ExecutionTiming.NEXT_OPEN
                    and opn is not None
                    and not math.isnan(opn[s][i])
                )
                px[i] = opn[s][i] if use_open and opn is not None else close[s][i]
        e1 = cash + sum(shares[i] * px[i] for i in range(n_assets) if alive(s, i))
        if e1 <= 0:
            ruined = True
            equity.append(0.0)
            continue
        target = [0.0] * n_assets
        for i in range(n_assets):
            w = weights[s - 1][i]
            if alive(s, i):
                target[i] = shares[i] * px[i] if math.isnan(w) else w * e1
        if cfg.max_gross_leverage is not None:
            lev = sum(abs(v) for v in target) / e1
            if lev > cfg.max_gross_leverage:
                target = [v * cfg.max_gross_leverage / lev for v in target]
        cost = 0.0
        spent = 0.0
        for i in range(n_assets):
            old = shares[i] * px[i] if alive(s, i) else 0.0
            dv = target[i] - old
            size = abs(dv)
            spent += dv
            cost += size * c.linear_bps / 1e4
            if c.impact_coefficient > 0:
                sg, ad = sigma(s, i), adv(s, i)
                if math.isfinite(sg) and math.isfinite(ad) and ad > 0:
                    cost += c.impact_coefficient * sg * (size / ad) ** c.impact_exponent * size
                else:
                    cost += size * c.impact_fallback_bps / 1e4
            shares[i] = target[i] / px[i] if alive(s, i) else 0.0
        fee = sum(max(-v, 0.0) for v in target) * c.borrow_bps_annual / 1e4 / cfg.periods_per_year
        cash = (cash - spent - cost - fee) * (1 + rf)
        e = cash + sum(shares[i] * close[s][i] for i in range(n_assets) if alive(s, i))
        if e <= 0:
            ruined = True
            e = 0.0
        equity.append(e)
    return equity


@st.composite
def scenario(draw: st.DrawFn) -> tuple[MarketData, np.ndarray, BacktestConfig]:
    n_bars = draw(st.integers(3, 25))
    n_assets = draw(st.integers(1, 4))
    steps = draw(
        st.lists(
            st.lists(st.floats(-0.25, 0.25), min_size=n_assets, max_size=n_assets),
            min_size=n_bars,
            max_size=n_bars,
        )
    )
    start = draw(st.lists(st.floats(5, 200), min_size=n_assets, max_size=n_assets))
    close = np.array(start) * np.exp(np.cumsum(np.array(steps), axis=0))
    for i in range(n_assets):
        lo = draw(st.integers(0, n_bars - 2))
        hi = draw(st.integers(lo + 1, n_bars - 1))
        close[:lo, i] = np.nan
        close[hi + 1 :, i] = np.nan

    opn = None
    if draw(st.booleans()):
        noise = draw(
            st.lists(
                st.lists(
                    st.one_of(st.floats(-0.05, 0.05), st.just(math.nan)),
                    min_size=n_assets,
                    max_size=n_assets,
                ),
                min_size=n_bars,
                max_size=n_bars,
            )
        )
        opn = close * np.exp(np.array(noise))
    volume = None
    if draw(st.booleans()):
        vol = draw(
            st.lists(
                st.lists(
                    st.one_of(st.just(0.0), st.floats(1, 1e6), st.just(math.nan)),
                    min_size=n_assets,
                    max_size=n_assets,
                ),
                min_size=n_bars,
                max_size=n_bars,
            )
        )
        volume = np.array(vol)

    weights = np.array(
        draw(
            st.lists(
                st.lists(
                    st.one_of(st.floats(-1.5, 1.5), st.just(math.nan)),
                    min_size=n_assets,
                    max_size=n_assets,
                ),
                min_size=n_bars,
                max_size=n_bars,
            )
        )
    )
    costs = CostModel(
        commission_bps=draw(st.floats(0, 10)),
        half_spread_bps=draw(st.floats(0, 10)),
        impact_coefficient=draw(st.floats(0, 1)) if volume is not None else 0.0,
        impact_exponent=draw(st.floats(0.3, 1.0)),
        impact_window=draw(st.integers(2, 5)),
        impact_fallback_bps=draw(st.floats(0, 100)),
        borrow_bps_annual=draw(st.floats(0, 500)),
    )
    cfg = BacktestConfig(
        initial_capital=1e6,
        timing=draw(st.sampled_from(list(ExecutionTiming))),
        costs=costs,
        risk_free_rate=draw(st.floats(0, 0.1)),
        delisting_return=draw(st.floats(-1, 0.5)),
        max_gross_leverage=draw(st.one_of(st.none(), st.floats(0.5, 3.0))),
    )
    idx = pd.bdate_range("2020-01-01", periods=n_bars)
    cols = [f"A{i}" for i in range(n_assets)]

    def frame(a: np.ndarray | None) -> pd.DataFrame | None:
        return None if a is None else pd.DataFrame(a, idx, cols)

    market = MarketData(pd.DataFrame(close, idx, cols), frame(opn), frame(volume))
    return market, weights, cfg


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenario())
def test_engine_matches_independent_share_ledger(
    case: tuple[MarketData, np.ndarray, BacktestConfig],
) -> None:
    market, weights, cfg = case
    expected = ledger(
        market.close.to_numpy(),
        None if market.open is None else market.open.to_numpy(),
        None if market.volume is None else market.volume.to_numpy(),
        weights,
        cfg,
    )
    got = simulate(weights, market, cfg).equity[0]
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-6)
