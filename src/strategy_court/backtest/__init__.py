"""Vectorised daily backtest engine: t+1 fills, costs, metrics."""

from strategy_court.backtest.config import BacktestConfig, CostModel, ExecutionTiming
from strategy_court.backtest.engine import (
    BacktestResult,
    BatchResult,
    Diagnostics,
    WeightError,
    run_backtest,
    simulate,
    validate_weights,
)
from strategy_court.backtest.market import MarketData
from strategy_court.backtest.metrics import PerformanceSummary, summarize
from strategy_court.backtest.strategy import (
    BuyAndHold,
    EqualWeight,
    MovingAverageCrossover,
    Strategy,
    run_strategy,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BatchResult",
    "BuyAndHold",
    "CostModel",
    "Diagnostics",
    "EqualWeight",
    "ExecutionTiming",
    "MarketData",
    "MovingAverageCrossover",
    "PerformanceSummary",
    "Strategy",
    "WeightError",
    "run_backtest",
    "run_strategy",
    "simulate",
    "summarize",
    "validate_weights",
]
