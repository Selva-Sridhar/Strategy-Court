# Sandbox image for untrusted strategy code. Build from the repo root:
#   docker build -t strategy-court-sandbox:latest -f src/strategy_court/sandbox/Dockerfile .
# Run only through strategy_court.sandbox.DockerSandbox (adds --network none, --read-only, caps...).
# TODO(PoC): pin the base image by digest once built on the PoC machine.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /opt/court
RUN pip install --no-cache-dir uv==0.11.19
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --locked --no-dev --no-cache \
    && chmod -R a+rX /opt/court

USER 65534:65534
WORKDIR /work
ENTRYPOINT ["/opt/court/.venv/bin/python", "-m", "strategy_court.sandbox.runner", "/work"]
