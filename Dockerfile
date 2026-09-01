FROM python:3.12-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN apt-get update && apt-get install -y --no-install-recommends git \
      && rm -rf /var/lib/apt/lists/*
# dependencies are a function of the lock file, so a one-line edit does not reinstall them
RUN uv sync --frozen --no-install-project
COPY app ./app
RUN uv sync --frozen

CMD ["python", "--version"]
