"""Single source of literature references used by audit tools and surfaced on evidence cards.

Every entry must be verified (DOI / publisher page) before publication — tracked as T5.2.
"""

from strategy_court.schemas import Citation

KAUFMAN_2012 = Citation(
    authors="Kaufman, Rosset, Perlich & Stitelman",
    year=2012,
    title="Leakage in Data Mining: Formulation, Detection, and Avoidance",
    venue="ACM Transactions on Knowledge Discovery from Data 6(4)",
    doi="10.1145/2382577.2382579",
)

ARNOTT_HARVEY_MARKOWITZ_2019 = Citation(
    authors="Arnott, Harvey & Markowitz",
    year=2019,
    title="A Backtesting Protocol in the Era of Machine Learning",
    venue="Journal of Financial Data Science 1(1)",
)
