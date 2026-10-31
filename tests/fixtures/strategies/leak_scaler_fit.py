"""Leak: scaler fitted on the full sample (sklearn-style API, implemented inline)."""

import pandas as pd


class Scaler:
    def fit(self, x: pd.DataFrame) -> "Scaler":
        self.mu, self.sd = x.mean(), x.std()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return (x - self.mu) / self.sd


def signals(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None)
    scaled = Scaler().fit(rets).transform(rets)
    return (scaled.rolling(5).mean() > 0).astype(float) / prices.shape[1]
