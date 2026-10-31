import json
from enum import StrEnum
from pathlib import Path

from strategy_court.llm.types import LLMRequest, LLMResponse


class CacheMode(StrEnum):
    AUTO = "auto"  # replay on hit, call provider and record on miss
    REPLAY = "replay"  # never call the provider; a miss is an error (tests, demos)
    RECORD = "record"  # always call the provider and overwrite the entry


class ReplayCache:
    """One JSON file per request hash, holding both request and response for inspection."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        return LLMResponse.model_validate(json.loads(path.read_text("utf-8"))["response"])

    def put(self, key: str, request: LLMRequest, response: LLMResponse) -> None:
        entry = {"request": request.model_dump(mode="json"), "response": response.model_dump()}
        self._path(key).write_text(json.dumps(entry, indent=2, ensure_ascii=False), "utf-8")
