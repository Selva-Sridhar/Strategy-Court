# Strategy Court — Project Plan & Knowledge Base

> Single source of truth for scope, design, and rationale. Status lives in [`TRACKER.md`](./TRACKER.md).
> Original brief: `Strategy Court Project Description.docx` (this folder).

---

## 1. Context

| Item | Detail |
|---|---|
| Event | Meta Global AI Developer Hackathon (Oct–Nov 2026) |
| Track | Indie developer — judged on creativity, technical ambition, and whether it works |
| Theme fit | "Build for Your World" (quant finance) / "Push the Frontier" (verification-first agents) |
| Internal deadline | **Tuesday, Nov 3, 2026** (moved from Sun Nov 1 to keep full depth; official deadline Nov 4) |
| Budget | $150 Meta Model API credit → all LLM calls cached (record/replay) |
| Reported judging weights (third-party source, unconfirmed) | Impact 30% · Creative use of Meta tools 25% · Demo/video 25% · Technical depth 20% |

### 1.1 One-line pitch
Strategy Court puts any trading strategy on trial. It audits claims with deterministic statistical tests and uses LLM agents only to propose, argue, and explain.

### 1.2 Problem
Backtests lie, and LLMs make it worse. A model can write a strategy with a stunning equity curve in seconds; the curve is often produced by look-ahead bias, data snooping, ignored costs, or leakage. Viral trading claims spread with no verification.

### 1.3 Solution
Input: a strategy as **code**, **text**, or a **screenshot** of a claim.
Output: a **verdict** with a credibility score, the evidence behind it, and an honest re-run with flaws fixed.

### 1.4 Core design principle
**LLMs propose and explain; deterministic code decides.**
The verdict is produced by a versioned, transparent rubric over test outputs — never by an LLM's opinion. Every number in the Judge's written opinion is machine-verified against the evidence.

### 1.5 Positioning (intentional)
- Quant-focused implementation and language throughout. Full technical depth is kept on purpose.
- No alpha claims, no investment advice. Public (yfinance) and synthetic data only.

---

## 2. Agreed decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **UI: React** (Vite + Tailwind + Recharts) served by **FastAPI + Server-Sent Events** | Best demo quality; live courtroom timeline |
| D2 | **Sandbox: Docker only** for untrusted (LLM-generated) code: `--network none`, CPU/mem/pids caps, read-only FS, timeout | Real isolation; subprocess on Windows can't block network or cap memory reliably |
| D2a | Repo-authored, trusted strategies (benchmark fixtures, tests) run in-process | Lets core + benchmark be developed/tested without Docker |
| D3 | **Real data: yfinance**, downloaded once, cached as Parquet with SHA-256 hash; read-only during trials | Credibility of real data + reproducibility |
| D4 | Synthetic data for ground truth (GARCH-t random walk, injected weak signal, survivorship universe) | Unarguable ground truth for the benchmark |
| D5 | Meta API details (endpoint, SDK, model names) **pending** — `llm/` sits behind a Protocol; replay cache + fake provider used until then | Phases 0–2 have zero LLM dependency |
| D6 | Research-backed methods; every tool and agent cites its source (docstring + `Evidence.citation` + UI card) | Depth and credibility |
| D7 | Rubric thresholds tuned on **dev seeds**, reported on **held-out seeds** | Avoid overfitting the court itself |
| D8 | Developer runs Docker/integration tests on their own PoC machine; code pushed to GitHub | No Docker on authoring machine |

---

## 3. System flow

```
 Input (code | text | screenshot)
        │
        ▼
 [Intake — Muse Spark vision] ──► Hypothesis (typed) + digitised claimed equity curve
        │                                   (code input skips Intake)
        ▼
 [Coder — Muse Code] ──► K interpretations of Strategy ──► static validation ──► Docker smoke test
        │                 (self-debug ≤ 3 attempts per version)
        ▼
 Arraignment (always-on, deterministic): AST scan · causality truncation · base backtest · claim consistency
        │
        ▼
 [Prosecutor — Muse Spark] ──► picks audit tools from registry (budget-limited, logged)
        │
        ▼
 [Defense — Muse Code] ──► patches only the flagged flaw (AST-diff guard, data hash check)
        │
        ▼
 Re-trial of patched strategy (same tools)
        │
        ▼
 Rubric (rubric.yaml, versioned) ──► score 0–100 + label (Acquitted / Not proven / Convicted)
        │
        ▼
 [Judge — Muse Spark] writes opinion ──► number-citation validator ──► (regenerate on mismatch)
        │
        ▼
 Verdict + evidence + honest re-run + JSON trace (replayable)
```

Purple = LLM step (Intake, Coder, Prosecutor, Defense, Judge-writer). Everything else is deterministic code.

---

## 4. Tech stack

| Layer | Choice |
|---|---|
| Core | Python 3.12, numpy, pandas, scipy, pydantic v2 |
| Data | yfinance (one-time pull) → Parquet (pyarrow); synthetic generators |
| API | FastAPI + SSE |
| UI | React + Vite + TypeScript + Tailwind + Recharts |
| Tests | pytest + hypothesis |
| Quality | ruff, mypy --strict, pre-commit, GitHub Actions CI |
| Sandbox | Docker (python:3.12-slim based image with numpy/pandas only) |
| LLM | Meta Model API (Muse Spark, Muse Code; optional SAM 3.1) via `llm/` Protocol + record/replay cache |

---

## 5. Key contracts

### 5.1 Strategy interface (what Muse Code must produce)
```python
class Strategy:
    params: dict
    def signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weights decided at close of t. The engine fills at t+1."""
```
- Engine owns fills, costs, sizing → a strategy cannot cheat on execution.
- `signals()` is pure → causality test: `signals(prices[:t]).iloc[-1] == signals(prices).loc[t]`.

### 5.2 Schemas (pydantic)
- **Hypothesis**: universe, signal description, holding period, rebalance frequency, claimed Sharpe / CAGR / max DD, claimed period, source type, optional digitised claimed curve.
- **StrategySpec**: source code, params, interpretation id (1..K), provenance.
- **Evidence**: `id`, `tool`, `statistic`, `p_value | flag`, `severity`, `details`, `plot_data`, `citation`.
- **Verdict**: score, label, evidence ids used, rubric version, opinion text, validator result.
- **TrialTrace**: ordered events, every LLM call (prompt hash, response, cost), every tool call, data hashes.

### 5.3 Hallucination guard
After the Judge writes its opinion, the validator extracts every number and checks it against values in the cited evidence (with tolerance). Any mismatch → regenerate (max N), else fall back to a templated opinion.

---

## 6. Audit tools (13 families, 14 registered tools + registry)

| # | Tool | Known-answer test | Reference |
|---|---|---|---|
| 1 | Causality truncation test (own contribution) | Planted `shift(-1)` fails; clean MA crossover passes | builds on Kaufman et al. (2012) |
| 2 | AST look-ahead scanner | Fixtures: negative shift, full-sample `fit()`, same-bar fill | Kaufman et al. (2012) |
| 3 | Purged + embargoed CV | Leak visible without purging, gone with it | López de Prado (2018) ch. 7 |
| 4 | PSR + Minimum Track Record Length | Reproduces paper's worked numbers | Bailey & López de Prado (2012) |
| 5 | Deflated Sharpe ratio | Reproduces paper's worked example | Bailey & López de Prado (2014) |
| 6 | PBO / CSCV | Pure-noise strategy set → PBO ≈ 0.5 | Bailey, Borwein, López de Prado & Zhu (2017) |
| 7 | Haircut Sharpe | Reproduces paper's example | Harvey & Liu (2015); Harvey, Liu & Zhu (2016) |
| 8 | Permutation / placebo null (turnover-matched) | p-values ~ uniform under null | White (2000); Sullivan, Timmermann & White (1999); Phipson & Smyth (2010) |
| 9 | Stationary bootstrap Sharpe CI | ~95% coverage on simulated data | Politis & Romano (1994); Politis & White (2004); Ledoit & Wolf (2008); Lo (2002) |
| 10 | Cost/slippage sweep + parameter-neighbourhood sensitivity | High-turnover strategy collapses at realistic costs | Novy-Marx & Velikov (2016) |
| 11 | Subperiod stability + sup-Wald break test | Planted regime-dependent strategy flagged | Andrews (1993); McLean & Pontiff (2016) |
| 12 | Claim consistency (Kelly growth bound, MDD plausibility, MinTRL, digitised curve, replication gap) | Impossible "Sharpe 3, 45% MDD" flagged | Kelly (1956); Magdon-Ismail et al. (2004); Bailey & López de Prado (2012); Liu et al. (2023, DePlot) |
| 13 | Data snooping across trials (White Reality Check, Hansen SPA) | Real edge found; SPA robust to adding poor trials | White (2000); Hansen (2005) |

Tool 10 is registered as two tools: `cost_sensitivity` (10a, incl. execution-delay decay) and `parameter_sensitivity` (10b). Static findings (tool 2) are capped at HIGH; only the dynamic causality test (tool 1) can produce CRITICAL evidence. Implementation status and verification notes per tool: `TRACKER.md`, Phase 1.
| — | Tool registry | Lists all tools + references | — |

Additional mechanism: **K-interpretation multiverse** — for text/screenshot claims Coder generates K plausible implementations; K is added to DSR's trial count and the Sharpe spread across K is reported (Simonsohn, Simmons & Nelson, 2020).

---

## 7. Agents

| Agent | Model | Job | Design reference |
|---|---|---|---|
| Intake | Muse Spark (vision), optional SAM 3.1 crop | Text/screenshot → `Hypothesis` + digitised equity curve | Liu et al. (2023, DePlot) |
| Coder | Muse Code | Hypothesis → K strategy versions behind fixed interface; self-debug ≤ 3 | Gao et al. (2023, PAL); Chen et al. (2024); Olausson et al. (2024) |
| Prosecutor | Muse Spark | Plans attack from strategy features; calls registry tools (budget, logged) | Yao et al. (2023, ReAct); Irving et al. (2018); Khan et al. (2024) |
| Defense | Muse Code | Patches only flagged flaw; AST-diff guard; data hash must match | Sharma et al. (2024, sycophancy) |
| Judge | Rubric + Muse Spark | Rubric decides; LLM writes explanation; validator checks numbers | Zheng et al. (2023); Gao et al. (2023, ALCE) |

Orchestrator: explicit **state machine**, not a free agent loop. Every transition emits an event (SSE to UI) and is persisted in the trace.

---

## 8. Rubric
- File: `rubric.yaml` (versioned; version recorded in each verdict).
- Evidence → weighted penalties → credibility score 0–100 → label: **Acquitted / Not proven / Convicted** (with counts).
- **Failed causality test ⇒ automatic conviction.**
- Rubric must not depend on Defense agreeableness.
- Thresholds tuned on dev seeds only (D7).

---

## 9. Benchmark & experiments

### 9.1 Planted-flaw benchmark
- Generators: GARCH-t random walk (true alpha = 0), weak injected signal with known IC, survivorship universe (simulated delistings).
- Flaws injected on clean base strategies: look-ahead, scaler leakage, best-of-N search, zero costs, survivorship.
- Metrics: recall per flaw, false-positive rate on clean + weak-signal strategies, verdict accuracy, confusion matrix, Wilson CIs, Benjamini–Hochberg where multiple tests are aggregated.
- Fixed seeds; `make benchmark` reproduces the table exactly. Dev/held-out seed split.

### 9.2 Adversarial overfitting stress test
Muse Code is given zero-alpha random-walk data and told to maximise in-sample Sharpe. Every output is a known false positive → report court's conviction rate (Perez et al., 2022, adapted).

### 9.3 Headline: 100-strategy survival
Muse Code generates ~100 strategy ideas; run all on cached yfinance data and on random-walk data; report survival counts. Cheap path + cache.

---

## 10. Repo layout

```
strategy-court/
├── pyproject.toml              # deps, ruff, mypy --strict, pytest
├── Makefile                    # test, lint, benchmark, headline, demo, sandbox-image
├── rubric.yaml
├── src/strategy_court/
│   ├── schemas/                # Hypothesis, StrategySpec, Evidence, Verdict, TrialTrace
│   ├── data/                   # yfinance loader, Parquet cache + hash, synthetic generators
│   ├── backtest/               # engine, costs, fills, metrics
│   ├── audit/                  # one module per tool + registry.py
│   ├── rubric.py
│   ├── agents/                 # intake, coder, prosecutor, defense, judge, prompts/
│   ├── llm/                    # Protocol, Meta client, fake provider, record/replay cache
│   ├── orchestration/          # state machine, events, tracing
│   ├── sandbox/                # Docker runner, image definition, AST pre-check
│   ├── validator.py            # Judge number-citation check
│   └── benchmark/              # flaw injectors, suite, adversarial, headline
├── api/main.py                 # FastAPI + SSE /trial
├── web/                        # React UI
├── tests/                      # mirrors src/, fixtures/
├── docs/methodology.md         # maths, assumptions, references (verified)
├── demo/                       # recorded LLM sessions, sample claims, screenshots
└── .github/workflows/ci.yml
```

---

## 11. Task breakdown
Full task list with IDs, owners, dependencies and status: **[`TRACKER.md`](./TRACKER.md)**.

Phases:
0. Foundations — scaffold, schemas, LLM layer, data, engine
1. Audit tools — 12 tools + registry
2. Verdict & benchmark — rubric, flaw injectors, harness
3. Agents & orchestration — sandbox, 5 agents, validator, state machine
4. Experiments — adversarial stress test, 100-strategy run, integration test
5. UI & delivery — React UI, methodology doc, README, video, submission

**Critical path:** engine → causality test → rubric → benchmark → orchestrator → UI.
Phases 0–2 have no LLM dependency.

---

## 12. Demo script (3 min)
1. **Hook (20s):** "Most AI-generated trading strategies are lies."
2. **Input (30s):** paste a viral-claim screenshot → extracted hypothesis + digitised curve.
3. **Trial (60s):** Prosecutor evidence appears live, Defense patches, Judge verdict — e.g. "Sharpe 3.1 → 0.2 after removing the leak", claimed vs honest curve overlay.
4. **Proof (40s):** benchmark table + adversarial stress-test result.
5. **Scale (20s):** 100-strategy survival number.
6. **Close (10s):** "Verification-first agents."

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Meta API specifics unknown | `llm/` Protocol + fake provider + replay cache; confirm on receipt |
| $150 credit | Cache everything; cheap path for bulk runs; cost meter in trace |
| No Docker on authoring machine | Trusted code in-process for dev; Docker tests marked and run on PoC machine |
| Scope vs Sunday deadline | Critical path first; no-LLM core first |
| Defense sycophancy | Rubric independent of Defense; AST-diff guard; data hash |
| Judge hallucination | Number-citation validator + templated fallback |
| Overfitting the court | Dev/held-out seed split |
| Data / IP rules | Check official rules when published; yfinance cached for research use only |
| Investment-advice perception | Explicit disclaimer; no alpha claims |
| Citation accuracy | Every reference verified (DOI/arXiv) before README |

---

## 14. References
> Listed from memory during planning — **each must be verified (DOI/arXiv) before publishing** (task T5.2).

**Quant / statistics**
- Andrews, D. W. K. (1993). Tests for Parameter Instability and Structural Change with Unknown Change Point. *Econometrica*, 61(4).
- Arnott, R., Harvey, C. R., & Markowitz, H. (2019). A Backtesting Protocol in the Era of Machine Learning. *Journal of Financial Data Science*, 1(1).
- Bailey, D. H., & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*, 15(2).
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5).
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-Mathematics and Financial Charlatanism. *Notices of the AMS*, 61(5).
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4).
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS-B*, 57(1).
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3).
- Bollerslev, T. (1987). A Conditionally Heteroskedastic Time Series Model for Speculative Prices and Rates of Return. *Review of Economics and Statistics*, 69(3).
- Brown, S. J., Goetzmann, W., Ibbotson, R. G., & Ross, S. A. (1992). Survivorship Bias in Performance Studies. *Review of Financial Studies*, 5(4).
- Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1).
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1).
- Hou, K., Xue, C., & Zhang, L. (2020). Replicating Anomalies. *Review of Financial Studies*, 33(5).
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining. *ACM TKDD*, 6(4).
- Ledoit, O., & Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio. *Journal of Empirical Finance*, 15(5).
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- McLean, R. D., & Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance*, 71(1).
- Novy-Marx, R., & Velikov, M. (2016). A Taxonomy of Anomalies and Their Trading Costs. *Review of Financial Studies*, 29(1).
- Phipson, B., & Smyth, G. K. (2010). Permutation P-values Should Never Be Zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1).
- Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *JASA*, 89(428).
- Politis, D. N., & White, H. (2004). Automatic Block-Length Selection for the Dependent Bootstrap. *Econometric Reviews*, 23(1).
- Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification Curve Analysis. *Nature Human Behaviour*, 4.
- Sullivan, R., Timmermann, A., & White, H. (1999). Data-Snooping, Technical Trading Rule Performance, and the Bootstrap. *Journal of Finance*, 54(5).
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5).

**Agents / LLMs**
- Chen, X., Lin, M., Schärli, N., & Zhou, D. (2024). Teaching Large Language Models to Self-Debug. *ICLR*.
- Gao, L., et al. (2023). PAL: Program-aided Language Models. *ICML*.
- Gao, T., Yen, H., Yu, J., & Chen, D. (2023). Enabling Large Language Models to Generate Text with Citations. *EMNLP*.
- Irving, G., Christiano, P., & Amodei, D. (2018). AI Safety via Debate. *arXiv:1805.00899*.
- Khan, A., et al. (2024). Debating with More Persuasive LLMs Leads to More Truthful Answers. *ICML*.
- Liu, F., et al. (2023). DePlot: One-shot Visual Language Reasoning by Plot-to-Table Translation. *Findings of ACL*.
- Olausson, T. X., et al. (2024). Is Self-Repair a Silver Bullet for Code Generation? *ICLR*.
- Perez, E., et al. (2022). Red Teaming Language Models with Language Models. *EMNLP*.
- Sharma, M., et al. (2024). Towards Understanding Sycophancy in Language Models. *ICLR*.
- Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR*.
- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS*.

---

## 15. Working conventions
- Every tool: own module, own test file, known-answer test, citation in docstring and `Evidence.citation`.
- Never let an LLM output a number that reaches the verdict.
- All LLM calls go through `llm/` and are recorded; tests use replay only.
- Data is hash-verified and read-only during trials.
- Tests needing Docker are marked `@pytest.mark.docker` and skipped when Docker is absent.
- Update `TRACKER.md` when starting/finishing a task; log decisions in its Decision Log.
