import importlib

import strategy_court

SUBPACKAGES = [
    "schemas",
    "data",
    "backtest",
    "audit",
    "agents",
    "llm",
    "orchestration",
    "sandbox",
    "benchmark",
]


def test_version() -> None:
    assert strategy_court.__version__ == "0.1.0"


def test_subpackages_import() -> None:
    for name in SUBPACKAGES:
        importlib.import_module(f"strategy_court.{name}")
