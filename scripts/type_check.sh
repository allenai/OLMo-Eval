#!/usr/bin/env bash
# Run ty against the same environment everywhere (CI, pre-commit hook, make).
#
# ty treats a missing optional package as Unknown, which hides type errors in
# code that uses it. The extras below are the ones whose types src/ depends on
# that the dev group does not already install; without them a check passes in
# one environment and fails in another.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

exec uv run --frozen --no-group vllm \
    --extra clients --extra litellm --extra agents \
    ty check src/ "$@"
