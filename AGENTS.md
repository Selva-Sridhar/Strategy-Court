# AGENTS.md — Strategy Court

Plan and design: `PROJECT_PLAN.md`. Status: `TRACKER.md` (update it when starting/finishing a task).

## Environment
- Python 3.12 managed by **uv** (`uv sync` creates `.venv` from `uv.lock`).
- `[tool.uv] exclude-newer` in `pyproject.toml` blocks packages published in the last 7 days (supply-chain guard). Bump the date deliberately, never remove it.
- pandas 3.x (copy-on-write semantics) and numpy 2.x.

## Commands (Windows has no `make`; use these directly — Makefile mirrors them for Linux/CI)
- Install: `uv sync`
- Lint: `uv run ruff check src tests` and `uv run ruff format --check src tests`
- Format: `uv run ruff format src tests`
- Typecheck: `uv run mypy` (strict)
- Tests (fast): `uv run pytest -m "not slow and not live_llm and not network"`
- Tests (all): `uv run pytest`
- Benchmark (dev, for calibration): `uv run python -m strategy_court.benchmark.suite --split dev --out benchmark_results`
- Benchmark (held-out, headline; run only with a frozen rubric): `uv run python -m strategy_court.benchmark.suite --split heldout --out docs/benchmark/v<version>`

## Rubric protocol — IMPORTANT
- `src/strategy_court/rubric.yaml` is versioned; its fingerprint is stored in every verdict.
- Calibrate only on **dev** seeds (0-4). Run **held-out** seeds once per frozen rubric version and commit the results under `docs/benchmark/v<version>/`. Never tune on held-out.
- Held-out seeds are *spent* after one evaluation: 100-109 (v1.1.0), 200-209 (v1.2.0). A new rubric version needs a fresh set (record it in `USED_HELDOUT_SEEDS` in `benchmark/suite.py`).

## Test markers
- `docker` — needs a Docker daemon; auto-skipped when absent (authoring machine has no Docker; run on PoC machine / CI).
- `live_llm` — calls the real Meta Model API; runs only with `STRATEGY_COURT_LIVE_LLM=1`.
- `slow` — statistical calibration tests.
- `network` — hits real external services (yfinance); runs only with `STRATEGY_COURT_NETWORK=1`.

## Testing rule — IMPORTANT
- On the authoring machine, **never call the yfinance API** (or any market-data network service). Real-data code is tested with mocks only.
- All functional testing here uses **synthetic data**. The user runs real-data, Docker and live-LLM tests on their own PoC machine.

## Quant rigor standard — IMPORTANT
Every quant method must be implemented to maximum technical depth:
- Cite the source paper (docstring + `Evidence.citation`); state assumptions explicitly.
- Known-answer test against the paper's worked numbers where they exist.
- Independent reference implementation or property-based (hypothesis) test where feasible.
- Handle edge cases explicitly (NaN, zero variance, ruin, delisting) — never silently.
- Performance: numpy only (no numba, no C++). Vectorise; batch many paths in one call.

## Backtest engine semantics
- Weights decided at close t are executed at bar t+1: next **open** if available, else next close.
- NaN weight = hold (no instruction); 0 = flat. Engine works in value terms, so weight drift is exact.

## Conventions
- LLMs never produce a number that reaches the verdict; rubric decides.
- Every audit tool: own module, own test file, known-answer test, citation in docstring and `Evidence.citation`.
- Untrusted (LLM-generated) code runs only in the Docker sandbox. Repo-authored fixtures may run in-process.
- Never commit secrets; API keys live in an untracked `.env`.

## Git / GitHub — IMPORTANT
- This is a **personal portfolio** project. Only push to the user's personal GitHub repo, once they provide it.
- **Never access, clone, push to, or reference any work/employer GitHub org or repos** from this project. Remote: `https://github.com/Selva-Sridhar/Strategy-Court.git` only.
- Never change git config (global or local).
- **Always ask the user before any git commit, push, remote change, or other git write operation.**
