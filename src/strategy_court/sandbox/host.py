"""Host side of the sandbox: prepare a job, run it, verify and decode the result.

`DockerSandbox` is the only backend allowed for untrusted (LLM-written) code. Isolation flags:
no network, read-only root filesystem, all Linux capabilities dropped, no privilege escalation,
non-root user, memory / swap / CPU / process-count caps, small noexec tmpfs, wall-clock timeout.

`TrustedLocalSandbox` runs the *same* runner as a local subprocess. It exists so the job protocol,
the in-container runner and its guards can be tested where Docker is unavailable, and it refuses to
run unless the caller explicitly declares the code repo-authored (trusted).
"""

import contextlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from strategy_court.audit.ast_scanner import Finding, security_findings
from strategy_court.sandbox import runner
from strategy_court.schemas import Evidence

TaskKind = Literal["smoke", "signals", "causality", "point_in_time"]
IMAGE = "strategy-court-sandbox:latest"
DOCKERFILE = "src/strategy_court/sandbox/Dockerfile"


class SandboxError(RuntimeError):
    """The sandbox could not produce a valid result."""


class SandboxRejectedError(SandboxError):
    """Static pre-check refused to run the code at all."""

    def __init__(self, findings: Sequence[Finding]) -> None:
        self.findings = list(findings)
        rules = ", ".join(sorted({f"{f.rule}@{f.line}" for f in findings}))
        super().__init__(f"code rejected before execution: {rules}")


class SandboxTimeoutError(SandboxError):
    """Wall-clock limit exceeded; the run was killed."""


@dataclass(frozen=True)
class SandboxLimits:
    cpus: float = 1.0
    memory_mb: int = 1024
    pids: int = 128
    tmpfs_mb: int = 64
    timeout_s: float = 900.0
    call_timeout_s: float = 30.0
    user: str = "65534:65534"  # nobody


@dataclass(frozen=True)
class Task:
    kind: TaskKind
    params: Mapping[str, object] = field(default_factory=dict)
    param_grid: Sequence[Mapping[str, object]] = ()
    options: Mapping[str, object] = field(default_factory=dict)
    strategy_id: str = "sandboxed"

    def to_json(self) -> dict[str, object]:
        out: dict[str, object] = {"kind": self.kind, "params": dict(self.params)}
        if self.kind == "signals":
            out["param_grid"] = [dict(p) for p in self.param_grid]
        if self.options:
            out["options"] = dict(self.options)
        out["strategy_id"] = self.strategy_id
        return out


@dataclass(frozen=True)
class TaskResult:
    kind: str
    ok: bool
    error: str | None = None
    seconds: float = 0.0
    weights: list[pd.DataFrame | None] = field(default_factory=list)
    evidence: Evidence | None = None
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxResult:
    tasks: list[TaskResult]
    strategy_calls: int
    seconds: float
    stderr: str = ""

    def task(self, kind: str) -> TaskResult:
        return next(t for t in self.tasks if t.kind == kind)


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill a process *and its descendants* (the strategy worker is a grandchild)."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    proc.kill()


def run_bounded(
    cmd: Sequence[str], timeout: float, on_timeout: Callable[[], None] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run with a hard wall-clock limit. Output goes to temp files, not pipes, so a surviving
    descendant can never block us; on timeout the whole process tree is killed."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        if sys.platform == "win32":
            proc = subprocess.Popen(
                list(cmd),
                stdout=out,
                stderr=err,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            proc = subprocess.Popen(list(cmd), stdout=out, stderr=err, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if on_timeout is not None:
                on_timeout()
            _kill_tree(proc)
            proc.wait(timeout=30)
            raise SandboxTimeoutError(f"sandbox exceeded {timeout:.0f}s and was killed") from exc
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(
            list(cmd),
            proc.returncode,
            out.read().decode(errors="replace"),
            err.read().decode(errors="replace"),
        )


def build_docker_command(work: Path, name: str, image: str, limits: SandboxLimits) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,size={limits.tmpfs_mb}m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(limits.pids),
        "--memory",
        f"{limits.memory_mb}m",
        "--memory-swap",
        f"{limits.memory_mb}m",
        "--cpus",
        str(limits.cpus),
        "--user",
        limits.user,
        "--volume",
        f"{work.resolve()}:/work:rw",
        image,
    ]


class _SandboxBase:
    def __init__(self, limits: SandboxLimits | None = None, precheck: bool = True) -> None:
        self.limits = limits or SandboxLimits()
        self.precheck = precheck

    def _execute(self, work: Path, name: str) -> subprocess.CompletedProcess[str]:
        raise NotImplementedError

    def run(
        self,
        source: str,
        prices: pd.DataFrame,
        tasks: Sequence[Task],
        class_name: str = "Strategy",
        test_disable_inner_guards: bool = False,
    ) -> SandboxResult:
        """`test_disable_inner_guards` turns off the pre-check and the runtime import guard so
        tests can verify that the *container* alone still contains hostile code."""
        if self.precheck and not test_disable_inner_guards:
            findings = security_findings(source)
            if findings:
                raise SandboxRejectedError(findings)
        nonce = secrets.token_hex(16)
        name = f"court-{secrets.token_hex(6)}"
        work = Path(tempfile.mkdtemp(prefix="court-job-"))
        try:
            work.chmod(0o777)  # container runs as an unprivileged uid
            (work / runner.SOURCE_FILE).write_text(source, "utf-8")
            prices.to_parquet(work / runner.PRICES_FILE)
            job = {
                "nonce": nonce,
                "class_name": class_name,
                "call_timeout_s": self.limits.call_timeout_s,
                "tasks": [t.to_json() for t in tasks],
                "test_disable_inner_guards": test_disable_inner_guards,
            }
            (work / runner.JOB_FILE).write_text(json.dumps(job), "utf-8")
            start = time.perf_counter()
            proc = self._execute(work, name)
            return self._collect(work, nonce, prices, tasks, proc, time.perf_counter() - start)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _collect(
        work: Path,
        nonce: str,
        prices: pd.DataFrame,
        tasks: Sequence[Task],
        proc: subprocess.CompletedProcess[str],
        seconds: float,
    ) -> SandboxResult:
        stderr = (proc.stderr or "")[-4000:]
        path = work / runner.RESULT_FILE
        if proc.returncode == 137:
            raise SandboxError(f"sandbox killed (memory or process limit). {stderr[-500:]}")
        if not path.exists():
            raise SandboxError(
                f"sandbox exited {proc.returncode} without a result. {stderr[-500:]}"
            )
        try:
            raw = json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise SandboxError("sandbox result is not valid JSON") from exc
        if not isinstance(raw, dict) or not secrets.compare_digest(str(raw.get("nonce")), nonce):
            raise SandboxError("sandbox result failed authentication (token mismatch)")
        outs = raw.get("tasks", [])
        if len(outs) != len(tasks):
            raise SandboxError("sandbox result has the wrong number of task outputs")
        results = []
        for task, out in zip(tasks, outs, strict=True):
            if out.get("kind") != task.kind:
                raise SandboxError("sandbox result tasks out of order")
            if out.get("status") != "ok":
                results.append(
                    TaskResult(task.kind, False, str(out.get("error")), out.get("seconds", 0.0))
                )
                continue
            payload = out.get("output", {})
            results.append(
                _decode_task(task, payload, work, prices, float(out.get("seconds", 0.0)))
            )
        return SandboxResult(results, int(raw.get("strategy_calls", 0)), seconds, stderr)


def _load(work: Path, name: str | None, prices: pd.DataFrame) -> pd.DataFrame | None:
    if name is None:
        return None
    path = (work / Path(name).name).resolve()
    if path.parent != work.resolve():
        raise SandboxError("sandbox output path escapes the job directory")
    values = np.load(path, allow_pickle=False)
    if values.shape != prices.shape or values.dtype != np.float64:
        raise SandboxError(f"sandbox weights have shape {values.shape}, expected {prices.shape}")
    return pd.DataFrame(values, index=prices.index, columns=prices.columns)


def _decode_task(
    task: Task, payload: Mapping[str, object], work: Path, prices: pd.DataFrame, seconds: float
) -> TaskResult:
    if task.kind == "signals":
        files = payload.get("files", [])
        assert isinstance(files, list)
        errors = payload.get("errors", {})
        assert isinstance(errors, dict)
        return TaskResult(
            task.kind,
            True,
            seconds=seconds,
            weights=[_load(work, f, prices) for f in files],
            info={"errors": errors},
        )
    if task.kind == "point_in_time":
        file = payload.get("file")
        assert isinstance(file, str)
        return TaskResult(
            task.kind,
            True,
            seconds=seconds,
            weights=[_load(work, file, prices)],
            info={"errors": payload.get("errors", 0)},
        )
    if task.kind == "causality":
        return TaskResult(
            task.kind, True, seconds=seconds, evidence=Evidence.model_validate(payload["evidence"])
        )
    return TaskResult(task.kind, True, seconds=seconds, info=dict(payload))


class DockerSandbox(_SandboxBase):
    """The only backend for untrusted code."""

    def __init__(
        self,
        image: str = IMAGE,
        limits: SandboxLimits | None = None,
        precheck: bool = True,
        docker: str = "docker",
    ) -> None:
        super().__init__(limits, precheck)
        self.image = image
        self.docker = docker

    def _execute(self, work: Path, name: str) -> subprocess.CompletedProcess[str]:
        cmd = build_docker_command(work, name, self.image, self.limits)
        cmd[0] = self.docker

        def kill_container() -> None:
            subprocess.run([self.docker, "kill", name], capture_output=True, check=False)

        return run_bounded(cmd, self.limits.timeout_s, on_timeout=kill_container)


class TrustedLocalSandbox(_SandboxBase):
    """Same runner and protocol without container isolation — repo-authored code only."""

    def __init__(self, *, trusted: bool, limits: SandboxLimits | None = None) -> None:
        if trusted is not True:
            raise PermissionError("TrustedLocalSandbox runs only code declared repo-authored")
        super().__init__(limits, precheck=True)

    def _execute(self, work: Path, name: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, "-m", "strategy_court.sandbox.runner", str(work)]
        return run_bounded(cmd, self.limits.timeout_s)


def docker_build_command(tag: str = IMAGE, context: Path = Path(".")) -> list[str]:
    return ["docker", "build", "-t", tag, "-f", str(context / DOCKERFILE), str(context)]
