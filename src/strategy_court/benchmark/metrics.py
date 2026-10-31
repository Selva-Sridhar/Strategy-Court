"""Benchmark metrics with honest uncertainty (Wilson score intervals for every proportion)."""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from strategy_court.schemas import VerdictLabel

LABELS = tuple(label.value for label in VerdictLabel)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson (1927) score interval for a binomial proportion; (0, 1) when n == 0."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.k, self.n)

    def as_dict(self) -> dict[str, object]:
        lo, hi = self.ci
        rate = self.value if self.n else None  # JSON-safe and comparable (no NaN)
        return {"k": self.k, "n": self.n, "rate": rate, "ci95": [lo, hi]}

    def fmt(self) -> str:
        lo, hi = self.ci
        return (
            f"{self.value:.0%} ({self.k}/{self.n}, 95% CI {lo:.0%}-{hi:.0%})" if self.n else "n/a"
        )


def meets_target(target: str, label: str) -> bool:
    return {
        "convicted": label == "convicted",
        "not_acquitted": label != "acquitted",
        "not_convicted": label != "convicted",
    }[target]


def confusion(truths: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    counts = Counter(zip(truths, labels, strict=True))
    return {t: {lab: counts.get((t, lab), 0) for lab in LABELS} for t in dict.fromkeys(truths)}
