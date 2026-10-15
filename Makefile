.PHONY: setup fix verify test lint type-check file-sizes clean

setup:
	git config core.hooksPath .githooks

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

# Check that no tracked file exceeds the size limit
file-sizes:
	./scripts/check_file_sizes.sh

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info htmlcov/ .coverage coverage.xml .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
