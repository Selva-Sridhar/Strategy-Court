"""Single source of literature references used by audit tools and surfaced on evidence cards.

Every entry must be verified (DOI / publisher page) before publication — tracked as T5.2.
"""

from strategy_court.schemas import Citation


def _c(authors: str, year: int, title: str, venue: str, doi: str | None = None) -> Citation:
    return Citation(authors=authors, year=year, title=title, venue=venue, doi=doi)


# Leakage / research protocol
KAUFMAN_2012 = _c(
    "Kaufman, Rosset, Perlich & Stitelman",
    2012,
    "Leakage in Data Mining: Formulation, Detection, and Avoidance",
    "ACM Transactions on Knowledge Discovery from Data 6(4)",
    "10.1145/2382577.2382579",
)
ARNOTT_HARVEY_MARKOWITZ_2019 = _c(
    "Arnott, Harvey & Markowitz",
    2019,
    "A Backtesting Protocol in the Era of Machine Learning",
    "Journal of Financial Data Science 1(1)",
)
LOPEZ_DE_PRADO_2018 = _c(
    "López de Prado", 2018, "Advances in Financial Machine Learning", "Wiley (ch. 7, 12)"
)

# Sharpe-ratio inference
LO_2002 = _c("Lo", 2002, "The Statistics of Sharpe Ratios", "Financial Analysts Journal 58(4)")
BAILEY_LDP_2012 = _c(
    "Bailey & López de Prado",
    2012,
    "The Sharpe Ratio Efficient Frontier",
    "Journal of Risk 15(2)",
)
BAILEY_LDP_2014 = _c(
    "Bailey & López de Prado",
    2014,
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, "
    "and Non-Normality",
    "Journal of Portfolio Management 40(5)",
)
LDP_2019 = _c(
    "López de Prado",
    2019,
    "A Data Science Solution to the Multiple-Testing Crisis in Financial Research",
    "Journal of Financial Data Science 1(1)",
)
LEDOIT_WOLF_2008 = _c(
    "Ledoit & Wolf",
    2008,
    "Robust Performance Hypothesis Testing with the Sharpe Ratio",
    "Journal of Empirical Finance 15(5)",
)
NEWEY_WEST_1994 = _c(
    "Newey & West",
    1994,
    "Automatic Lag Selection in Covariance Matrix Estimation",
    "Review of Economic Studies 61(4)",
)

# Overfitting / multiple testing
BBLZ_2014 = _c(
    "Bailey, Borwein, López de Prado & Zhu",
    2014,
    "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on "
    "Out-of-Sample Performance",
    "Notices of the American Mathematical Society 61(5)",
)
BBLZ_2017 = _c(
    "Bailey, Borwein, López de Prado & Zhu",
    2017,
    "The Probability of Backtest Overfitting",
    "Journal of Computational Finance 20(4)",
)
HARVEY_LIU_2015 = _c("Harvey & Liu", 2015, "Backtesting", "Journal of Portfolio Management 42(1)")
HARVEY_LIU_ZHU_2016 = _c(
    "Harvey, Liu & Zhu",
    2016,
    "... and the Cross-Section of Expected Returns",
    "Review of Financial Studies 29(1)",
)
HOLM_1979 = _c(
    "Holm",
    1979,
    "A Simple Sequentially Rejective Multiple Test Procedure",
    "Scandinavian Journal of Statistics 6(2)",
)
BENJAMINI_YEKUTIELI_2001 = _c(
    "Benjamini & Yekutieli",
    2001,
    "The Control of the False Discovery Rate in Multiple Testing under Dependency",
    "Annals of Statistics 29(4)",
)

# Data snooping / resampling
WHITE_2000 = _c("White", 2000, "A Reality Check for Data Snooping", "Econometrica 68(5)")
HANSEN_2005 = _c(
    "Hansen",
    2005,
    "A Test for Superior Predictive Ability",
    "Journal of Business & Economic Statistics 23(4)",
)
STW_1999 = _c(
    "Sullivan, Timmermann & White",
    1999,
    "Data-Snooping, Technical Trading Rule Performance, and the Bootstrap",
    "Journal of Finance 54(5)",
)
PHIPSON_SMYTH_2010 = _c(
    "Phipson & Smyth",
    2010,
    "Permutation P-values Should Never Be Zero",
    "Statistical Applications in Genetics and Molecular Biology 9(1)",
)
POLITIS_ROMANO_1994 = _c(
    "Politis & Romano",
    1994,
    "The Stationary Bootstrap",
    "Journal of the American Statistical Association 89(428)",
)
POLITIS_WHITE_2004 = _c(
    "Politis & White",
    2004,
    "Automatic Block-Length Selection for the Dependent Bootstrap",
    "Econometric Reviews 23(1)",
)
PATTON_POLITIS_WHITE_2009 = _c(
    "Patton, Politis & White",
    2009,
    "Correction to 'Automatic Block-Length Selection for the Dependent Bootstrap'",
    "Econometric Reviews 28(4)",
)

# Costs, stability, claims
NOVY_MARX_VELIKOV_2016 = _c(
    "Novy-Marx & Velikov",
    2016,
    "A Taxonomy of Anomalies and Their Trading Costs",
    "Review of Financial Studies 29(1)",
)
ANDREWS_1993 = _c(
    "Andrews",
    1993,
    "Tests for Parameter Instability and Structural Change with Unknown Change Point",
    "Econometrica 61(4)",
)
MCLEAN_PONTIFF_2016 = _c(
    "McLean & Pontiff",
    2016,
    "Does Academic Research Destroy Stock Return Predictability?",
    "Journal of Finance 71(1)",
)
MAGDON_ISMAIL_2004 = _c(
    "Magdon-Ismail, Atiya, Pratap & Abu-Mostafa",
    2004,
    "On the Maximum Drawdown of a Brownian Motion",
    "Journal of Applied Probability 41(1)",
)
KELLY_1956 = _c(
    "Kelly", 1956, "A New Interpretation of Information Rate", "Bell System Technical Journal 35(4)"
)
LIU_2023_DEPLOT = _c(
    "Liu et al.",
    2023,
    "DePlot: One-shot Visual Language Reasoning by Plot-to-Table Translation",
    "Findings of ACL 2023",
)
