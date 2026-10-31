import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class DataError(Exception):
    """Market data could not be obtained or is malformed."""


class DataIntegrityError(DataError):
    """Cached data no longer matches its recorded hash (tampered or corrupted)."""


def frame_sha256(df: pd.DataFrame) -> str:
    """Canonical content hash of a numeric, date-indexed frame.

    Hashes column names, the index as int64 nanoseconds and the values as float64 with NaN and
    -0.0 normalised, so the hash is independent of Parquet/pyarrow metadata and library versions.
    """
    h = hashlib.sha256()
    h.update(json.dumps([str(c) for c in df.columns]).encode())
    ns = pd.DatetimeIndex(df.index).as_unit("ns").to_numpy(dtype="datetime64[ns]")
    h.update(ns.view(np.int64).tobytes())
    values = df.to_numpy(dtype=np.float64, na_value=np.nan) + 0.0
    values = np.where(np.isnan(values), np.nan, values)
    h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Dataset:
    name: str
    frame: pd.DataFrame
    sha256: str


class DataStore:
    """Parquet cache with a hash manifest; every load is integrity-checked."""

    MANIFEST = "manifest.json"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _manifest(self) -> dict[str, dict[str, str | int]]:
        path = self.root / self.MANIFEST
        if not path.exists():
            return {}
        data: dict[str, dict[str, str | int]] = json.loads(path.read_text("utf-8"))
        return data

    def exists(self, name: str) -> bool:
        return name in self._manifest() and (self.root / f"{name}.parquet").exists()

    def save(self, name: str, frame: pd.DataFrame) -> Dataset:
        digest = frame_sha256(frame)
        frame.to_parquet(self.root / f"{name}.parquet")
        manifest = self._manifest()
        index = pd.DatetimeIndex(frame.index)
        manifest[name] = {
            "sha256": digest,
            "rows": len(frame),
            "columns": ",".join(map(str, frame.columns)),
            "start": str(index.min().date()) if len(index) else "",
            "end": str(index.max().date()) if len(index) else "",
        }
        (self.root / self.MANIFEST).write_text(json.dumps(manifest, indent=2), "utf-8")
        return Dataset(name, frame, digest)

    def load(self, name: str) -> Dataset:
        entry = self._manifest().get(name)
        if entry is None:
            raise DataError(f"dataset '{name}' not in manifest")
        frame = pd.read_parquet(self.root / f"{name}.parquet")
        digest = frame_sha256(frame)
        if digest != entry["sha256"]:
            raise DataIntegrityError(f"dataset '{name}' hash mismatch")
        return Dataset(name, frame, digest)
