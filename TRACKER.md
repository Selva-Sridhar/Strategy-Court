# Strategy Court — Tracker

> Living status board. Design and rationale: [`PROJECT_PLAN.md`](./PROJECT_PLAN.md).
> Internal deadline: **Tuesday, Nov 3, 2026** (moved from Sun Nov 1 to keep full depth — D11). Official deadline: Nov 4.

**Status legend:** `TODO` · `IN PROGRESS` · `BLOCKED` · `REVIEW` · `DONE`
**Update rule:** change status when you start/finish a task; add notes for anything another teammate would need to know.

---

## Summary

| Phase | Tasks | Done | Target |
|---|---|---|---|
| 0 — Foundations | 5 | 5 | Wed Oct 28 |
| 1 — Audit tools | 13 | 13 | Fri Oct 30 |
| 2 — Verdict & benchmark | 3 | 3 | Fri Oct 30 |
| 3 — Agents & orchestration | 7 | 0 | Fri Oct 30 – Sat Oct 31 |
| 4 — Experiments | 3 | 0 | Sat Oct 31 |
| 5 — UI & delivery | 4 | 0 | Sat Oct 31 – Sun Nov 1 |
| **Total** | **35** | **21** | |

---

## Phase 0 — Foundations

| ID | Task | Owner | Depends on | Status | Done when | Notes |
|---|---|---|---|---|---|---|
| T0.1 | Repo scaffold: `src/strategy_court/`, pyproject (ruff, mypy --strict, pytest, hypothesis), pre-commit, CI, Makefile, `.gitignore` | Devin | — | DONE | `make test` / `make lint` pass | uv + `exclude-newer` guard; no `make` on Windows → commands in AGENTS.md |
| T0.2 | Pydantic schemas: Hypothesis, StrategySpec, Evidence (+citation), Verdict, TrialTrace | Devin | T0.1 | DONE | JSON round-trip tests pass | Frozen + `extra="forbid"` (bad LLM output fails loudly); `Evidence.numbers()` feeds the Judge validator; MDD stored as positive fraction |
| T0.3 | `llm/` Protocol, fake provider, record/replay cache, cost meter; Meta client stub | Devin | T0.2 | DONE | Replay works offline | AUTO/REPLAY/RECORD cache modes; strict pricing (unpriced model = error) + budget guard; `complete_structured` retries with validation feedback. **Meta provider still to add (B1)** |
| T0.4 | Data: yfinance loader + Parquet cache + SHA-256; GARCH-t, weak-signal, survivorship generators | Devin | T0.1 | DONE | Generator stats tests pass | Version-independent content hash; yfinance tested with mocks only — **user to run `STRATEGY_COURT_NETWORK=1 uv run pytest -m network` on PoC**. Weak signal = AR(1) in GARCH shocks (known lag-1 autocorr) so strategies stay price-only |
| T0.5 | Backtest engine: Strategy interface, t+1 fills, linear + sqrt-impact costs, metrics | Devin | T0.2, T0.4 | DONE | Buy-and-hold matches hand calc; same-bar fill impossible | Two-leg bar (next open, else close), value-based state (exact drift), linear + sqrt-impact (lagged sigma/ADV) + borrow + rf, delisting return, ruin, leverage cap, diagnostics. Verified by 400-case hypothesis differential test vs independent share ledger (mutation-checked). Speed: 150 ms single 2520x50; 200 paths in 0.48 s (2.4 ms/path) |

## Phase 1 — Audit tools

| ID | Task | Owner | Depends on | Status | Done when | Reference |
|---|---|---|---|---|---|---|
| T1.1 | Causality truncation test | Devin | T0.5 | DONE | `shift(-1)` fails; clean MA passes | Kaufman et al. 2012 (own contribution) — **critical path**. Determinism pre-check; truncate + mirror + noise checks (complementary, each proven by a test); look-ahead depth lower bound, local/global scope, leaking columns; point-in-time honest re-run. Hypothesis: 100% detection of negative lags, 0 false positives. 2520x50: test 0.6 s, honest re-run 14.6 s (1 call/bar). Sandbox (T3.1) must run the tool *inside* the container to avoid per-call container cost |
| T1.2 | AST look-ahead scanner | Devin | T0.1 | DONE | 3 fixtures flagged | Kaufman et al. 2012. 12 rules (look-ahead, same-bar P&L, unseeded RNG, hard-coded dates, security allowlist); constant resolution (`LAG=-2`); findings capped at HIGH (static = suspicion). 10 fixture strategies; **static and dynamic (T1.1) verdicts agree on all 7**. `security_findings()` = sandbox pre-check for T3.1 |
| T1.3 | Purged + embargoed CV | Devin | T0.5 | DONE | Leak appears w/o purge, gone with | López de Prado 2018. Purged K-fold + embargo + CPCV (splits, path assembly); audits the *selection procedure* over a trial set. 1-NN overlapping-label demo: OOS corr > 0.4 unpurged, ~0 purged |
| T1.4 | PSR + MinTRL | Devin | T0.5 | DONE | Paper numbers reproduced | Bailey & LdP 2012. Identity PSR(MinTRL)=1-alpha; Monte-Carlo size calibration; documented finite-sample limit at extreme skew (8% vs 5%) |
| T1.5 | Deflated Sharpe | Devin | T1.4 | DONE | Paper example reproduced | Bailey & LdP 2014. **Reproduces paper example: SR0 = 0.1132, DSR = 0.90**; E[max SR] vs Monte Carlo; effective trials (participation ratio) |
| T1.6 | PBO / CSCV | Devin | T0.5 | DONE | Noise set PBO ≈ 0.5 | Bailey et al. 2017. Vectorised via group sufficient statistics (12,870 splits in one matmul); equals brute force; noise ≈ 0.5, real edge ≈ 0 |
| T1.7 | Haircut Sharpe | Devin | T1.4 | DONE | Paper example reproduced | Harvey & Liu 2015. Bonferroni/Holm/BHY on *actual* trial p-values; Lo (2002) autocorrelation-corrected annualisation (AR(1) closed form). Known adjusted-p values tested (paper's own example not reproduced — not recalled reliably) |
| T1.8 | Permutation / placebo null | Devin | T0.5 | DONE | p ~ uniform under null | White 2000; STW 1999; Phipson & Smyth 2010. Circular-shift + run-block placebos (turnover ratio 1.00), p-values uniform (KS). **Plus tool 13 `data_snooping`: White RC + Hansen SPA (l/c/u)** |
| T1.9 | Stationary bootstrap Sharpe CI | Devin | T0.5 | DONE | ~95% coverage | Politis & Romano 1994; Politis & White 2004 / PPW 2009 block length; coverage test under AR(0.3); Ledoit-Wolf HAC Sharpe-difference test (size ~5%) |
| T1.10 | Cost sweep + parameter neighbourhood | Devin | T0.5 | DONE | High-turnover collapses | Novy-Marx & Velikov 2016. Break-even bps (sweep + analytic cross-check), execution-delay decay + half-life; parameter plateau vs isolated peak. Registered as two tools (10a, 10b) |
| T1.11 | Subperiod stability + sup-Wald | Devin | T0.5 | DONE | Regime strategy flagged | Andrews 1993. **Simulated null matches Andrews' critical values (7.17 / 8.85 / 12.35)**; HAC equal-means Wald ~ chi2; break date located |
| T1.12 | Claim consistency (implied vs claimed) | Devin | T1.4 | DONE | Impossible claim flagged | Kelly growth bound; MDD plausibility (normal for 'too small', t4 for 'too large'), simulation validated vs Magdon-Ismail closed form; MinTRL; digitised-curve CAGR/MDD/Sharpe; replication gap |
| T1.13 | Tool registry + Evidence emission + citations | Devin | T1.1–T1.12 | DONE | Registry lists 12 tools with refs | 14 registered tools (sections 1–13, 10a/10b), each cited; `TrialContext` with cached weights/backtest/trial set; missing inputs & tool failures -> inconclusive evidence (never raised, never silent); `catalogue()` for the Prosecutor; always-on arraignment = causality, ast_scanner, claim_consistency |

## Phase 2 — Verdict & benchmark

| ID | Task | Owner | Depends on | Status | Done when | Notes |
|---|---|---|---|---|---|---|
| T2.1 | `rubric.yaml` + `rubric.py` (auto-convict on causality fail) | Devin | T1.13 | DONE | Deterministic verdicts | v1.1.0 (`src/strategy_court/rubric.yaml`). Families with caps + diminishing returns; auto-conviction on proven look-ahead; acquittal needs conclusive look-ahead + significance; **conviction needs positive flaw evidence**. Property tests: order invariance, monotonicity, penalty identity, no conviction without a flaw |
| T2.2 | Flaw injectors: look-ahead, scaler leak, best-of-N, zero cost, survivorship | Devin | T0.5 | DONE | Each injector proven by test | 11 case kinds/seed (genuine x2, no-edge x3, 6 flaws incl. best-of-N disclosed vs hidden). Strategy *source* templates. New `trending_signal_prices` generator. Ground truth fixed ex ante (genuine = independent 20-yr OOS Sharpe >= 0.3 net of 10 bps) |
| T2.3 | Benchmark harness: recall, FPR, confusion, Wilson CI, dev/held-out seeds | Devin | T2.1, T2.2 | DONE | `make benchmark` reproducible | Wilson CIs, attribution, confusion; dev seeds 0-4 (calibration), held-out 100-109 run once. **Held-out (rubric 1.1.0): accuracy 79%; look-ahead / scaler / zero-cost 100% convicted; best-of-N disclosed 60% convicted, 0% acquitted; survivorship 50% convicted, 0% acquitted; best-of-N hidden 0% convicted (70% acquitted: known limit); genuine false conviction 5%; no-edge false acquittal 10%.** Results: `docs/benchmark/heldout/` |

## Phase 3 — Agents & orchestration

| ID | Task | Owner | Depends on | Status | Done when | Notes |
|---|---|---|---|---|---|---|
| T3.1 | Docker sandbox runner + image + AST pre-check | Devin | T0.5 | TODO | Malicious fixtures blocked | Docker tests run on user's PoC machine (D8) |
| T3.2 | Intake agent (vision → Hypothesis + digitised curve; optional SAM crop) | Devin | T0.3 | TODO | 5 sample claims parse | Needs B1 for live run |
| T3.3 | Coder agent (K interpretations, self-debug ≤ 3, K → DSR trials) | Devin | T3.1, T0.3 | TODO | K versions pass smoke test | Needs B1 for live run |
| T3.4 | Prosecutor agent (ReAct tool selection, budget, logged) | Devin | T1.13, T0.3 | TODO | Picks causality + cost on high-turnover | |
| T3.5 | Defense agent (constrained patch, AST-diff guard, data hash) | Devin | T3.1, T0.3 | TODO | Out-of-scope patch rejected | |
| T3.6 | Judge writer + number-citation validator | Devin | T2.1, T0.3 | TODO | Planted wrong number → regenerate | |
| T3.7 | Orchestrator state machine + events + JSON trace + replay | Devin | T3.2–T3.6 | TODO | Full trial runs offline from replay | **Critical path** |

## Phase 4 — Experiments

| ID | Task | Owner | Depends on | Status | Done when | Notes |
|---|---|---|---|---|---|---|
| T4.1 | Adversarial overfitting stress test | Devin | T3.3, T2.1 | TODO | Conviction rate + CI | Needs B1 |
| T4.2 | 100-strategy survival run (real + random walk) | Devin / User | T3.7 | TODO | Survival table | Needs B1; run on PoC machine |
| T4.3 | End-to-end integration test (replay) | Devin | T3.7 | TODO | Passes in CI | |

## Phase 5 — UI & delivery

| ID | Task | Owner | Depends on | Status | Done when | Notes |
|---|---|---|---|---|---|---|
| T5.1 | FastAPI SSE + React UI (timeline, evidence cards w/ citations, claimed vs honest curve, verdict, benchmark page) | Devin | T3.7 | TODO | Demo flow end-to-end | **Critical path** |
| T5.2 | `docs/methodology.md` + verify every reference | Devin | T1.13 | TODO | All refs verified | |
| T5.3 | README with results; demo recording + video | User / Devin | T4.*, T5.1 | TODO | 3-min video | |
| T5.4 | Submission dry run + submit | User | T5.3 | TODO | Submitted | |

---

## Blockers / pending inputs

| ID | Item | Owner | Needed by | Status |
|---|---|---|---|---|
| B1 | Meta API endpoint, SDK shape, model names, key (in env file — never committed) | User | Thu Oct 29 | OPEN |
| B2 | Personal GitHub repo: `https://github.com/Selva-Sridhar/Strategy-Court.git` (portfolio — never a work org). Commits use per-command identity (`git -c`), no git config changes; laptop signed in to personal account | User | Before first push | DONE |
| B3 | Docker installed on PoC machine for sandbox tests | User | Phase 3 | OPEN |
| B4 | Official hackathon rules (data / IP / open-source) | User | Before submission | OPEN |

---

## Decision log

| Date | ID | Decision | By |
|---|---|---|---|
| 2026-10-28 | D1 | React + FastAPI SSE for UI | User |
| 2026-10-28 | D2 | Docker-only sandbox for untrusted code; trusted fixtures in-process | User |
| 2026-10-28 | D3 | yfinance real data, cached Parquet + hash | User |
| 2026-10-28 | D5 | Proceed without Meta API; Protocol + replay until details arrive | User |
| 2026-10-28 | D6 | Research-backed methods with citations throughout | User |
| 2026-10-28 | — | Keep full technical depth; quant-focused framing (intentional) | User |
| 2026-10-30 | D14 | Dev calibration -> rubric 1.1.0 + tool fixes: PBO read with P(OOS loss); selection-bias tools count only the claimant's *disclosed* search (court probes never penalise); effective trials for max-selection via simulation (counts sign flips); tools don't flag consequences of having no edge (gross t >= 1 gates); short track record = LOW unflagged; conviction requires flaw evidence | Devin |
| 2026-10-30 | D15 | Held-out seeds 100-109 are now *used*. Known rubric issue seen there (not tuned): selection family alone (cap 45) can never reach conviction (<40). Fix candidates for v1.2 must be calibrated on dev and validated on **fresh held-out seeds 200-209** | Devin |
| 2026-10-30 | D12 | Added tool 13 `data_snooping` (White RC + Hansen SPA); T1.10 registered as two tools -> 14 registered tools | Devin |
| 2026-10-30 | D13 | Benchmark design note: with t+1 fills a lag-1 signal monetises only ar1^2, and a 1-bar peek earns ~nothing; plant monetisable signals/leaks (e.g. 2-bar peeks) in T2.2 | Devin |
| 2026-10-30 | D11 | Keep full technical depth everywhere; internal deadline moved from Sun Nov 1 to **Tue Nov 3** (official deadline Nov 4) | User |
| 2026-10-28 | D9 | Execution: next open if available, else next close (configurable) | User |
| 2026-10-28 | D10 | Engine is pure numpy (no C++, no numba); batched paths for throughput | User |
| 2026-10-28 | — | Every quant method to maximum technical depth (papers, known-answer + reference/property tests) | User |
| 2026-10-28 | — | No yfinance/network calls on authoring machine; test with synthetic data; user runs real-data tests on PoC | User |
| 2026-10-28 | — | Personal portfolio repo only; never a work GitHub org; never change git config; ask before any git write | User |
| 2026-10-28 | — | Added: K-interpretation multiverse, MinTRL + claim consistency, adversarial stress test, constrained-diff Defense, Haircut Sharpe | User |

---

## Change log

| Date | Change |
|---|---|
| 2026-10-28 | Tracker created |
| 2026-10-28 | T0.1 done: scaffold, uv env (pinned via uv.lock), ruff/mypy strict/pytest green, CI workflow |
| 2026-10-28 | T0.2 done: schemas (Hypothesis, StrategySpec, Evidence+Citation, Verdict, TrialTrace), 12 tests green |
| 2026-10-28 | T0.3 done: LLM layer (Protocol, FakeProvider, replay cache, cost/budget, structured output) |
| 2026-10-30 | T2.1–T2.3 done: rubric 1.1.0, benchmark cases + harness, held-out results committed; 204 tests green. **Phase 2 complete** |
| 2026-10-30 | T1.2–T1.13 done: all audit tools + registry; 181 tests green (incl. slow calibration). **Phase 1 complete** |
| 2026-10-30 | First push to GitHub (`4b9913a`, merged remote README); CI green on GitHub Actions. Auth: fine-grained PAT scoped to this repo (Contents + Workflows RW) |
| 2026-10-30 | T1.1 done: causality test + honest re-run; 96 tests green |
| 2026-10-28 | T0.5 done: backtest engine + metrics + Strategy protocol + reference strategies; 82 tests green. **Phase 0 complete** |
| 2026-10-28 | T0.4 done: DataStore (Parquet + hash manifest), yfinance loader (mocked tests + 1 opt-in network test), GARCH-t / weak-signal / survivorship generators; 41 tests green |
