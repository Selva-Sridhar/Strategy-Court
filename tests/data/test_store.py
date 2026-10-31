from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy_court.data import DataError, DataIntegrityError, DataStore, frame_sha256


def frame() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=5, name="date")
    return pd.DataFrame({"A": [1.0, 2, 3, np.nan, 5], "B": [5.0, 4, 3, 2, -0.0]}, index=idx)


def test_hash_is_content_sensitive_and_normalised() -> None:
    df = frame()
    assert frame_sha256(df) == frame_sha256(df.copy())
    changed = df.copy()
    changed.iloc[0, 0] = 1.0000001
    assert frame_sha256(changed) != frame_sha256(df)
    zero = df.copy()
    zero.iloc[4, 1] = 0.0
    assert frame_sha256(zero) == frame_sha256(df)  # -0.0 == 0.0


def test_save_load_roundtrip_preserves_hash(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    saved = store.save("x", frame())
    loaded = store.load("x")
    assert store.exists("x")
    assert loaded.sha256 == saved.sha256
    pd.testing.assert_frame_equal(loaded.frame, frame(), check_freq=False)


def test_tampering_is_detected(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.save("x", frame())
    tampered = frame()
    tampered.iloc[2, 1] = 99.0
    tampered.to_parquet(tmp_path / "x.parquet")
    with pytest.raises(DataIntegrityError):
        store.load("x")


def test_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        DataStore(tmp_path).load("nope")
