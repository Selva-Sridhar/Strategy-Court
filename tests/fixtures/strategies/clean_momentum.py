"""Clean: only past data, legitimate rolling/expanding/ewm and cross-sectional operations."""

import numpy as np
import pandas as pd

LOOKBACK = 20


def signals(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None)
    trend = prices.rolling(LOOKBACK).mean() / prices.rolling(5 * LOOKBACK).mean() - 1
    vol = rets.ewm(span=LOOKBACK).std()
    zscore = (rets - rets.expanding(20).mean()) / rets.expanding(20).std()
    score = (trend / vol).shift(1) + 0.0 * zscore
    ranks = score.rank(axis=1, pct=True)
    demeaned = ranks.sub(ranks.mean(axis=1), axis=0)
    noise = np.random.default_rng(42).normal(0, 1e-9, prices.shape)
    return demeaned / prices.shape[1] + noise
