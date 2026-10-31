"""Validated market data for the engine.

Conventions
-----------
* `close` is required; `open` and `volume` are optional and must share its index/columns.
* An asset is *alive* from its first to its last valid close. NaN before listing and after
  delisting is allowed; an interior gap is rejected (it would silently distort returns).
* An asset whose last valid close is before the final bar is treated as delisted there.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from strategy_court.data.store import DataError


def _check_aligned(name: str, frame: pd.DataFrame, ref: pd.DataFrame) -> None:
    if not frame.index.equals(ref.index) or not frame.columns.equals(ref.columns):
        raise DataError(f"'{name}' must have the same index and columns as 'close'")


@dataclass(frozen=True, eq=False)
class MarketData:
    close: pd.DataFrame
    open: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        c = self.close
        if not isinstance(c.index, pd.DatetimeIndex):
            raise DataError("close must have a DatetimeIndex")
        if not (c.index.is_monotonic_increasing and c.index.is_unique):
            raise DataError("close index must be strictly increasing")
        if len(c) < 2 or c.shape[1] < 1:
            raise DataError("need at least 2 bars and 1 asset")
        if not c.columns.is_unique:
            raise DataError("asset columns must be unique")
        values = c.to_numpy(dtype=np.float64)
        if np.isinf(values).any() or (values <= 0).any():
            raise DataError("close prices must be finite and strictly positive")
        valid = ~np.isnan(values)
        if (~valid.any(axis=0)).any():
            dead = list(c.columns[~valid.any(axis=0)])
            raise DataError(f"assets with no data: {dead}")
        gaps = self._inside(valid) & ~valid
        if gaps.any():
            bad = list(c.columns[gaps.any(axis=0)])
            raise DataError(f"interior gaps in close for {bad}; fill or drop them explicitly")
        if self.open is not None:
            _check_aligned("open", self.open, c)
            o = self.open.to_numpy(dtype=np.float64)
            if np.isinf(o).any() or (o <= 0).any():
                raise DataError("open prices must be finite and strictly positive")
        if self.volume is not None:
            _check_aligned("volume", self.volume, c)
            v = self.volume.to_numpy(dtype=np.float64)
            if np.isinf(v).any() or (v < 0).any():
                raise DataError("volume must be finite and non-negative")

    @staticmethod
    def _inside(valid: np.ndarray) -> np.ndarray:
        started = np.maximum.accumulate(valid, axis=0)
        ended = np.maximum.accumulate(valid[::-1], axis=0)[::-1]
        return np.asarray(started & ended)

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def assets(self) -> list[str]:
        return [str(c) for c in self.close.columns]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.close), self.close.shape[1])

    @cached_property
    def alive(self) -> np.ndarray:
        """(T, N) bool: asset is listed and has a close at this bar."""
        return self._inside(~np.isnan(self.close.to_numpy(dtype=np.float64)))

    @cached_property
    def delisted_at(self) -> dict[str, pd.Timestamp]:
        """Last bar with a valid close, for assets that stop trading before the final bar."""
        alive = self.alive
        last = alive.shape[0] - 1 - np.argmax(alive[::-1], axis=0)
        return {
            a: self.index[i]
            for a, i in zip(self.assets, last, strict=True)
            if i < alive.shape[0] - 1
        }
