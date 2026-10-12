.PHONY: setup fix verify test lint type-check clean

setup:
	uv run --frozen pre-commit install

# Auto-fix formatting and lint issues
fix:
	./scripts/fix.sh

# Run all verification checks (lint, type-check, tests)
verify:
	./scripts/verify.sh

# Run unit tests only (no integration)
test:
	uv run pytest tests/ --ignore=tests/integration -v

# Run linter
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

# Run type checker
type-check:
	uv run ty check src/

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info htmlcov/ .coverage coverage.xml .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
