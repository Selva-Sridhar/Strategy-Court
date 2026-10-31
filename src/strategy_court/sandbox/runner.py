"""In-sandbox entry point: ``python -m strategy_court.sandbox.runner /work``.

Two processes inside the container:

* the *court* process (this module, trusted) reads the job, runs the audit tasks and writes the
  result;
* the *strategy worker* (spawned, untrusted) only executes the submitted strategy's
  ``signals()``. It never sees the job token, and everything it returns is decoded as a raw numeric
  array with ``allow_pickle=False`` — so it cannot execute code in, or tamper with, the court
  process that judges it.

Inside the worker a runtime import guard rejects any import made *by the strategy code* outside
the allowlist (library-internal imports are unaffected). The container adds the hard isolation
(no network, read-only filesystem, resource limits).
"""

import builtins
import io
import json
import multiprocessing as mp
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from strategy_court.audit.ast_scanner import ALLOWED_IMPORTS

JOB_FILE = "job.json"
RESULT_FILE = "result.json"
PRICES_FILE = "prices.parquet"
SOURCE_FILE = "strategy.py"
STRATEGY_MODULE = "strategy_under_test"
OK, ERR = b"O", b"E"


class StrategyCallError(RuntimeError):
    """The untrusted strategy failed, timed out or broke the output contract."""


# ----------------------------------------------------------------------------- worker side


def install_import_guard(allowed: frozenset[str] = ALLOWED_IMPORTS) -> None:
    """Reject imports *issued by strategy code* whose top-level package is not allowlisted."""
    real_import = builtins.__import__

    def guarded(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        caller = (globals or {}).get("__name__")
        if caller == STRATEGY_MODULE and (level > 0 or name.split(".")[0] not in allowed):
            raise ImportError(f"import of '{name}' is blocked inside the strategy sandbox")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded


def encode_array(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype=np.float64), allow_pickle=False)
    return buf.getvalue()


def decode_array(payload: bytes) -> np.ndarray:
    """Decode an untrusted payload: numeric .npy only, pickles rejected."""
    arr = np.load(io.BytesIO(payload), allow_pickle=False)
    if arr.dtype != np.float64 or arr.ndim != 2:
        raise StrategyCallError(f"strategy returned {arr.dtype} array of ndim {arr.ndim}")
    return np.asarray(arr)


def _worker_main(conn: Connection, source: str, class_name: str, guard: bool) -> None:
    if guard:
        install_import_guard()
    namespace: dict[str, object] = {"__name__": STRATEGY_MODULE, "np": np, "pd": pd}
    load_error = None
    try:
        exec(compile(source, "<strategy>", "exec"), namespace)
        cls = namespace[class_name]
        if not callable(cls):
            raise TypeError(f"'{class_name}' is not a class")
    except BaseException as exc:
        load_error = f"{type(exc).__name__}: {exc}"[:500]
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "stop":
            return
        _, params, prices = msg
        if load_error is not None:
            conn.send_bytes(ERR + f"could not load strategy: {load_error}".encode())
            continue
        try:
            out = cls(**params).signals(prices.copy())  # type: ignore[operator]
            if not isinstance(out, pd.DataFrame):
                raise TypeError(f"signals() returned {type(out).__name__}, expected DataFrame")
            if not out.index.equals(prices.index) or not out.columns.equals(prices.columns):
                raise ValueError("signals() index/columns do not match the input prices")
            conn.send_bytes(OK + encode_array(out.to_numpy(dtype=np.float64)))
        except BaseException as exc:
            conn.send_bytes(ERR + f"{type(exc).__name__}: {exc}"[:500].encode())


# ----------------------------------------------------------------------------- court side


class StrategyWorker:
    """Court-side handle to the untrusted strategy process."""

    def __init__(
        self, source: str, class_name: str, guard: bool = True, call_timeout: float = 30.0
    ):
        self.call_timeout = call_timeout
        self.calls = 0
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_worker_main, args=(child, source, class_name, guard), daemon=True
        )
        self._proc.start()
        child.close()

    def signals(self, params: Mapping[str, object], prices: pd.DataFrame) -> pd.DataFrame:
        if not self._proc.is_alive():
            raise StrategyCallError("strategy worker is not running")
        self.calls += 1
        self._conn.send(("signals", dict(params), prices))
        if not self._conn.poll(self.call_timeout):
            self.close(kill=True)
            raise StrategyCallError(f"strategy call timed out after {self.call_timeout:.0f}s")
        try:
            payload = self._conn.recv_bytes()
        except EOFError as exc:
            raise StrategyCallError("strategy worker died (resource limit or crash)") from exc
        if payload[:1] == ERR:
            raise StrategyCallError(payload[1:].decode(errors="replace"))
        values = decode_array(payload[1:])
        if values.shape != prices.shape:
            raise StrategyCallError(f"weights shape {values.shape} != prices shape {prices.shape}")
        if np.isinf(values).any():
            raise StrategyCallError("weights contain +/-inf")
        return pd.DataFrame(values, index=prices.index, columns=prices.columns)

    def bind(self, params: Mapping[str, object]) -> Callable[[pd.DataFrame], pd.DataFrame]:
        return lambda prices: self.signals(params, prices)

    def close(self, kill: bool = False) -> None:
        if self._proc.is_alive():
            if not kill:
                try:
                    self._conn.send(("stop",))
                    self._proc.join(2)
                except (BrokenPipeError, OSError):
                    pass
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(2)
        self._conn.close()


def _save(work: Path, name: str, values: np.ndarray) -> str:
    np.save(work / name, values, allow_pickle=False)
    return name


def run_task(
    task: Mapping[str, Any], prices: pd.DataFrame, worker: StrategyWorker, work: Path, i: int
) -> dict[str, object]:
    from strategy_court.audit.causality import causality_test, point_in_time_signals

    kind = task["kind"]
    params = task.get("params") or {}
    if kind == "smoke":
        head = prices.iloc[: min(len(prices), 250)]
        worker.signals(params, head)
        full = worker.signals(params, prices)
        return {"shape": list(full.shape), "nan_fraction": float(full.isna().to_numpy().mean())}
    if kind == "signals":
        files: list[str | None] = []
        errors: dict[str, str] = {}
        for j, p in enumerate(task["param_grid"]):
            try:
                w = worker.signals(p, prices)
                files.append(_save(work, f"weights_{i}_{j}.npy", w.to_numpy()))
            except StrategyCallError as exc:
                files.append(None)
                errors[str(j)] = str(exc)
        return {"files": files, "errors": errors}
    if kind == "causality":
        report = causality_test(worker.bind(params), prices, **task.get("options", {}))
        return {
            "evidence": report.to_evidence(task.get("strategy_id", "sandboxed")).model_dump(
                mode="json"
            )
        }
    if kind == "point_in_time":
        pit, n_err = point_in_time_signals(worker.bind(params), prices)
        return {"file": _save(work, f"pit_{i}.npy", pit.to_numpy()), "errors": n_err}
    raise ValueError(f"unknown task kind '{kind}'")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    work = Path(args[0] if args else "/work")
    job = json.loads((work / JOB_FILE).read_text("utf-8"))
    nonce = job.pop("nonce")
    (work / JOB_FILE).unlink()  # the token never coexists with untrusted code on disk
    prices = pd.read_parquet(work / PRICES_FILE)
    source = (work / SOURCE_FILE).read_text("utf-8")
    start = time.perf_counter()
    worker = StrategyWorker(
        source,
        job.get("class_name", "Strategy"),
        guard=not job.get("test_disable_inner_guards", False),
        call_timeout=float(job.get("call_timeout_s", 30.0)),
    )
    outputs: list[dict[str, object]] = []
    try:
        for i, task in enumerate(job["tasks"]):
            t0 = time.perf_counter()
            try:
                out = run_task(task, prices, worker, work, i)
                outputs.append({"kind": task["kind"], "status": "ok", "output": out})
            except Exception as exc:
                outputs.append(
                    {
                        "kind": task["kind"],
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                        "trace": traceback.format_exc(limit=3)[-1500:],
                    }
                )
            outputs[-1]["seconds"] = time.perf_counter() - t0
    finally:
        calls = worker.calls
        worker.close()
    result = {
        "nonce": nonce,
        "tasks": outputs,
        "strategy_calls": calls,
        "seconds": time.perf_counter() - start,
    }
    (work / RESULT_FILE).write_text(json.dumps(result), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
