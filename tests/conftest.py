import os
import shutil
import subprocess

import pytest


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    docker_ok = _docker_available()
    live_llm_ok = os.environ.get("STRATEGY_COURT_LIVE_LLM") == "1"
    network_ok = os.environ.get("STRATEGY_COURT_NETWORK") == "1"
    skip_docker = pytest.mark.skip(reason="Docker daemon not available")
    skip_llm = pytest.mark.skip(reason="set STRATEGY_COURT_LIVE_LLM=1 to run live LLM tests")
    skip_net = pytest.mark.skip(reason="set STRATEGY_COURT_NETWORK=1 to run network tests")
    for item in items:
        if "docker" in item.keywords and not docker_ok:
            item.add_marker(skip_docker)
        if "live_llm" in item.keywords and not live_llm_ok:
            item.add_marker(skip_llm)
        if "network" in item.keywords and not network_ok:
            item.add_marker(skip_net)
