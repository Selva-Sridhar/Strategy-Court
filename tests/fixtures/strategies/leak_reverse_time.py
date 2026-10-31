"""Leak: running maximum computed backwards in time (knows the future peak)."""

import pandas as pd


def signals(prices: pd.DataFrame) -> pd.DataFrame:
    future_peak = prices[::-1].cummax()[::-1]
    return (future_peak > prices * 1.05).astype(float) / prices.shape[1]
