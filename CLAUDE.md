# Development Commands

- Use `uv run` for Python commands
- Use `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/` for linting
- Use `uv run ty check src/` and `uv run pytest tests/ --ignore=tests/integration -v` for local verification

# Code Style

- Keep docstrings general; avoid implementation details that become stale
- Avoid comments that explain temporary or in-progress changes
- Design classes with stable interfaces; avoid coupling methods to specific fields

# Results Upload

- olmo-eval reaches the dashboard only through `src/olmo_eval/upload/` over HTTP; it must not import `olmo_eval_api` (the separate package in `dashboard/api`)
- Upload payloads follow `dashboard/contract/ingest-v1.schema.json`; change the contract there first

# Testing

- Adapt tests to match source code, not the reverse
- Always add unit tests for new code
