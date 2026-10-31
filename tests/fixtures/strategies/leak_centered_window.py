"""Leak: centred moving average."""

import pandas as pd


def signals(prices: pd.DataFrame) -> pd.DataFrame:
    trend = prices.rolling(21, center=True).mean()
    return (prices > trend).astype(float) / prices.shape[1]
