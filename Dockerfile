# pinned by digest, so a rebuild next month is the same Python 3.12.14; a move is a reviewed change
FROM python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# uv 0.12.6
COPY --from=ghcr.io/astral-sh/uv:0.12.6@sha256:88bc6eb1ccd4b82efd0e1b530caffabddf50dc2bf612e66c14ea25b8ee8a4d3d /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN apt-get update && apt-get install -y --no-install-recommends git \
      && rm -rf /var/lib/apt/lists/*
# `eval` carries ragas: without it the guest pass runs host side, beside the queue, not in it
RUN uv sync --frozen --no-install-project --group eval
COPY app ./app
RUN uv sync --frozen --group eval

CMD ["python", "--version"]
