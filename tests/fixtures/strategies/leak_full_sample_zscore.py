"""Leak: mean-reversion z-score normalised with full-sample moments."""

import pandas as pd


def signals(prices: pd.DataFrame) -> pd.DataFrame:
    z = (prices - prices.mean()) / prices.std()
    return -z.clip(-1, 1) / prices.shape[1]
