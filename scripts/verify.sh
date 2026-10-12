#!/usr/bin/env bash
set -euo pipefail

# Parse arguments
SKIP_COVERAGE=false
NO_DOCKER=false
RUN_GPU=false
for arg in "$@"; do
    case $arg in
        --skip-coverage)
            SKIP_COVERAGE=true
            shift
            ;;
        --no-docker)
            NO_DOCKER=true
            shift
            ;;
        --gpu)
            RUN_GPU=true
            shift
            ;;
    esac
done

echo "Running verification checks..."

echo ""
echo "==> Syncing dependencies..."
uv sync --frozen --extra beaker --extra hf --extra s3 --extra sandbox

UV_RUN="uv run --frozen"

echo ""
echo "==> Checking formatting with ruff..."
$UV_RUN ruff format --check src/ tests/

echo ""
echo "==> Running ruff linter..."
$UV_RUN ruff check src/ tests/

echo ""
echo "==> Running ty type checker..."
$UV_RUN ty check src/

echo ""
echo "==> Running tests with coverage..."

# Build pytest arguments
PYTEST_ARGS="-v"
if [ "$SKIP_COVERAGE" = false ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src/olmo_eval --cov-report=term-missing --cov-report=html"
fi

if [ "$RUN_GPU" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --gpu"
fi

if [ "$NO_DOCKER" = true ]; then
    # Skip docker-based integration tests
    PYTEST_ARGS="$PYTEST_ARGS --no-docker"
fi

# Integration fixtures start and stop their own containers.
$UV_RUN pytest tests/ $PYTEST_ARGS

if [ "$SKIP_COVERAGE" = false ]; then
    echo ""
    echo "==> Coverage report generated in htmlcov/"
fi

echo ""
echo "All checks passed!"
