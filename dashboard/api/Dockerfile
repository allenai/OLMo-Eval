# Image for both API services. Build context is dashboard/ (see skiff2.json):
#   docker build -f api/Dockerfile --build-arg API_MODE=ingest .
# API_MODE picks the routes (dashboard, ingest, or all for local development).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=api/uv.lock,target=uv.lock \
    --mount=type=bind,source=api/pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
COPY api/pyproject.toml api/uv.lock ./
COPY api/src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.12-slim-bookworm
RUN groupadd --system --gid 999 app \
 && useradd --system --gid 999 --uid 999 --no-create-home app
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ARG API_MODE=dashboard
ARG GIT_SHA=
ENV API_MODE=$API_MODE \
    GIT_SHA=$GIT_SHA \
    PORT=8000 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app
WORKDIR /app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn olmo_eval_api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*' --no-access-log"]
