"""Sandbox tests.

Local (run everywhere): job protocol, court/worker separation, import guard, numeric-only IPC,
result authentication, timeouts and error mapping — using the same runner as the container, on
repo-authored code only. Docker (marked, run in CI / PoC): the container contains hostile code
even with the inner guards switched off.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy_court.audit.causality import causality_test, point_in_time_signals
from strategy_court.benchmark import strategies
from strategy_court.data import random_walk_prices
from strategy_court.sandbox import (
    DockerSandbox,
    SandboxError,
    SandboxLimits,
    SandboxRejectedError,
    SandboxTimeoutError,
    Task,
    TrustedLocalSandbox,
    build_docker_command,
)
from strategy_court.sandbox.host import _SandboxBase, docker_build_command
from strategy_court.sandbox.runner import (
    StrategyCallError,
    StrategyWorker,
    decode_array,
    encode_array,
)
from strategy_court.schemas import Severity

PRICES = random_walk_prices(n_days=160, n_assets=3, seed=12)
CLEAN = strategies.source("trend_ma")
LEAKY = strategies.source("trend_ma_lookahead")
LOCAL = TrustedLocalSandbox(trusted=True, limits=SandboxLimits(timeout_s=120, call_timeout_s=20))


def strategy(body: str) -> str:
    return "import numpy as np\nimport pandas as pd\n\n" + textwrap.dedent(body)


# ------------------------------------------------------------------ host-side, pure


def test_docker_command_enforces_isolation() -> None:
    cmd = build_docker_command(Path("job"), "court-x", "img:1", SandboxLimits(memory_mb=512))
    joined = " ".join(cmd)
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--pids-limit 128",
        "--memory 512m",
        "--memory-swap 512m",
        "--cpus 1.0",
        "--user 65534:65534",
        "--rm",
    ):
        assert flag in joined, flag
    assert "noexec" in joined and cmd[-1] == "img:1"
    assert docker_build_command()[:2] == ["docker", "build"]


def test_precheck_rejects_unsafe_code_before_anything_runs() -> None:
    hostile = strategy(
        """
        import os
        class Strategy:
            def __init__(self):
                self.params = {}
            def signals(self, prices):
                os.system("echo pwned")
                return prices * 0
        """
    )
    with pytest.raises(SandboxRejectedError) as exc:
        DockerSandbox(docker="definitely-not-run").run(hostile, PRICES, [Task("smoke")])
    assert any(f.rule == "security.forbidden_import" for f in exc.value.findings)
    with pytest.raises(SandboxRejectedError):
        LOCAL.run("import strategy_court\n" + CLEAN, PRICES, [Task("smoke")])


def test_local_backend_requires_explicit_trust() -> None:
    with pytest.raises(PermissionError):
        TrustedLocalSandbox(trusted=False)


# ------------------------------------------------------------------ local end-to-end


def test_sandboxed_results_equal_in_process_results() -> None:
    params = {"fast": 5, "slow": 30, "direction": 1}
    grid = [params, {"fast": 10, "slow": 40, "direction": -1}]
    res = LOCAL.run(
        CLEAN,
        PRICES,
        [
            Task("smoke", params=params),
            Task("signals", param_grid=grid),
            Task("causality", params=params, options={"n_checkpoints": 5}, strategy_id="s1"),
            Task("point_in_time", params=params),
        ],
    )
    assert all(t.ok for t in res.tasks), [t.error for t in res.tasks]
    cls = strategies.compile_strategy("trend_ma")
    for p, got in zip(grid, res.task("signals").weights, strict=True):
        assert got is not None
        pd.testing.assert_frame_equal(got, cls(**p).signals(PRICES), check_freq=False)
    ev = res.task("causality").evidence
    assert ev is not None and ev.details["status"] == "pass" and ev.strategy_id == "s1"
    expected = causality_test(cls(**params).signals, PRICES, n_checkpoints=5).to_evidence("s1")
    assert ev.statistic == expected.statistic
    pit = res.task("point_in_time").weights[0]
    assert pit is not None
    pd.testing.assert_frame_equal(
        pit, point_in_time_signals(cls(**params).signals, PRICES)[0], check_freq=False
    )
    assert res.strategy_calls > len(PRICES)


def test_sandboxed_causality_convicts_a_leak() -> None:
    res = LOCAL.run(LEAKY, PRICES, [Task("causality", options={"n_checkpoints": 5})])
    ev = res.task("causality").evidence
    assert ev is not None and ev.severity is Severity.CRITICAL


def test_strategy_failures_are_reported_per_task() -> None:
    broken = strategy(
        """
        class Strategy:
            def __init__(self, k=1):
                self.params = {"k": k}
            def signals(self, prices):
                if self.params["k"] == 2:
                    raise ValueError("bad k")
                return prices.iloc[1:]
        """
    )
    res = LOCAL.run(
        broken, PRICES, [Task("smoke"), Task("signals", param_grid=[{"k": 1}, {"k": 2}])]
    )
    smoke, grid = res.tasks
    assert not smoke.ok and "index/columns" in (smoke.error or "")
    assert grid.ok and grid.weights == [None, None]
    assert "bad k" in str(grid.info["errors"])


def test_per_call_timeout_kills_a_hanging_strategy() -> None:
    hang = strategy(
        """
        class Strategy:
            def __init__(self):
                self.params = {}
            def signals(self, prices):
                while True:
                    pass
        """
    )
    quick = TrustedLocalSandbox(trusted=True, limits=SandboxLimits(timeout_s=120, call_timeout_s=3))
    res = quick.run(hang, PRICES, [Task("smoke"), Task("smoke")])
    assert "timed out" in (res.tasks[0].error or "")
    assert "not running" in (res.tasks[1].error or "")


def test_wall_clock_timeout_raises() -> None:
    hang = strategy(
        """
        class Strategy:
            def __init__(self):
                self.params = {}
            def signals(self, prices):
                while True:
                    pass
        """
    )
    tiny = TrustedLocalSandbox(trusted=True, limits=SandboxLimits(timeout_s=4, call_timeout_s=60))
    with pytest.raises(SandboxTimeoutError):
        tiny.run(hang, PRICES, [Task("smoke")])


# ------------------------------------------------------------------ worker isolation units


def test_runtime_import_guard_blocks_strategy_imports_only() -> None:
    sneaky = strategy(
        """
        class Strategy:
            def __init__(self):
                self.params = {}
            def signals(self, prices):
                mod = "soc" + "ket"
                __builtins__["__import__"](mod) if isinstance(__builtins__, dict) else None
                import socket
                return prices * 0
        """
    )
    worker = StrategyWorker(sneaky, "Strategy", guard=True, call_timeout=20)
    try:
        with pytest.raises(StrategyCallError, match="blocked inside the strategy sandbox"):
            worker.signals({}, PRICES)
        clean = StrategyWorker(CLEAN, "Strategy", guard=True, call_timeout=20)
        try:  # pandas' own lazy imports still work under the guard
            assert clean.signals({}, PRICES).shape == PRICES.shape
        finally:
            clean.close()
    finally:
        worker.close()


def test_untrusted_payloads_are_never_unpickled() -> None:
    import pickle

    with pytest.raises(ValueError):
        decode_array(pickle.dumps({"evil": object}))
    obj = np.empty((2, 2), dtype=object)
    with pytest.raises(ValueError):
        decode_array(encode_array(np.ones((2, 2)))[:10])  # truncated
    np.testing.assert_array_equal(decode_array(encode_array(np.eye(2))), np.eye(2))
    del obj


def _completed(code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["x"], code, "", "")


def test_result_authentication_and_integrity(tmp_path: Path) -> None:
    tasks = [Task("smoke")]

    def write(payload: object) -> None:
        (tmp_path / "result.json").write_text(json.dumps(payload), "utf-8")

    write({"nonce": "forged", "tasks": [{"kind": "smoke", "status": "ok", "output": {}}]})
    with pytest.raises(SandboxError, match="authentication"):
        _SandboxBase._collect(tmp_path, "real-token", PRICES, tasks, _completed(), 0.1)
    write({"nonce": "t", "tasks": []})
    with pytest.raises(SandboxError, match="number of task outputs"):
        _SandboxBase._collect(tmp_path, "t", PRICES, tasks, _completed(), 0.1)
    escape = [Task("point_in_time")]
    write(
        {
            "nonce": "t",
            "tasks": [
                {"kind": "point_in_time", "status": "ok", "output": {"file": "../../etc/passwd"}}
            ],
        }
    )
    with pytest.raises((SandboxError, FileNotFoundError)):
        _SandboxBase._collect(tmp_path, "t", PRICES, escape, _completed(), 0.1)
    (tmp_path / "result.json").unlink()
    with pytest.raises(SandboxError, match="killed"):
        _SandboxBase._collect(tmp_path, "t", PRICES, tasks, _completed(137), 0.1)
    with pytest.raises(SandboxError, match="without a result"):
        _SandboxBase._collect(tmp_path, "t", PRICES, tasks, _completed(1), 0.1)


# ------------------------------------------------------------------ Docker (CI / PoC)

IMAGE = "strategy-court-sandbox:test"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def docker_sandbox() -> DockerSandbox:
    subprocess.run(docker_build_command(IMAGE, REPO), check=True, cwd=REPO, timeout=1200)
    return DockerSandbox(
        image=IMAGE, limits=SandboxLimits(memory_mb=512, timeout_s=120, call_timeout_s=30)
    )


def hostile(body: str) -> str:
    """Hostile code that bypasses the pre-check/guard on purpose (test_disable_inner_guards)."""
    return strategy(
        f"""
        class Strategy:
            def __init__(self):
                self.params = {{}}
            def signals(self, prices):
{textwrap.indent(textwrap.dedent(body), " " * 16)}
                return prices * 0
        """
    )


@pytest.mark.docker
def test_docker_runs_clean_strategy(docker_sandbox: DockerSandbox) -> None:
    res = docker_sandbox.run(
        CLEAN, PRICES, [Task("smoke"), Task("causality", options={"n_checkpoints": 4})]
    )
    assert all(t.ok for t in res.tasks), [t.error for t in res.tasks]


@pytest.mark.docker
@pytest.mark.parametrize(
    ("body", "expect"),
    [
        (
            "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)",
            ("unreachable", "network", "errno"),
        ),
        ("open('/opt/court/pwned', 'w').write('x')", ("read-only", "permission")),
        ("open('/etc/passwd', 'a').write('x')", ("read-only", "permission")),
        ("import os\nos.setuid(0)", ("operation not permitted", "permission")),
    ],
)
def test_container_contains_hostile_code(
    docker_sandbox: DockerSandbox, body: str, expect: tuple[str, ...]
) -> None:
    res = docker_sandbox.run(hostile(body), PRICES, [Task("smoke")], test_disable_inner_guards=True)
    err = (res.tasks[0].error or "").lower()
    assert not res.tasks[0].ok and any(e in err for e in expect), err


@pytest.mark.docker
def test_container_memory_and_process_limits(docker_sandbox: DockerSandbox) -> None:
    mem = hostile("x = bytearray(2 * 1024**3)")
    try:
        res = docker_sandbox.run(mem, PRICES, [Task("smoke")], test_disable_inner_guards=True)
        assert not res.tasks[0].ok
    except SandboxError:
        pass  # whole container OOM-killed: also contained
    fork = hostile(
        "import subprocess\nprocs = [subprocess.Popen(['sleep', '30']) for _ in range(500)]"
    )
    res = docker_sandbox.run(fork, PRICES, [Task("smoke")], test_disable_inner_guards=True)
    assert not res.tasks[0].ok and "resource" in (res.tasks[0].error or "").lower()


@pytest.mark.docker
def test_container_wall_clock_timeout() -> None:
    sandbox = DockerSandbox(image=IMAGE, limits=SandboxLimits(timeout_s=15, call_timeout_s=600))
    with pytest.raises(SandboxTimeoutError):
        sandbox.run(
            hostile("while True:\n    pass"),
            PRICES,
            [Task("smoke")],
            test_disable_inner_guards=True,
        )
    assert shutil.which("docker")
