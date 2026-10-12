# Development

## Local Installation

```bash
# Clone the repository
git clone https://github.com/allenai/olmo-eval.git
cd olmo-eval

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12 if your machine does not already have it
uv python install 3.12

# Install dependencies and the package in editable mode from the checked-in
# lockfile so builds are reproducible. Run `uv lock` to update the lockfile.
uv sync --frozen

# Install git hooks
uv run pre-commit install

# Browse a few suites
uv run olmo-eval suite inspect mmlu
uv run olmo-eval suite inspect gpqa
uv run olmo-eval suite inspect olmobase:code

# Preview a run without loading a model
uv run olmo-eval run -m mock -t gsm8k --dry-run
```

## Common Commands

```bash
# Run lint, type checks, and unit tests directly
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/
uv run pytest tests/ --ignore=tests/integration -v

# Optional helper scripts
./scripts/fix.sh
./scripts/verify.sh
```

## Results Upload

Runs upload their results to the dashboard's ingest service by default (see
"Uploading Results and the Dashboard" in the README). For development:

```bash
# Skip uploads (tests set this automatically)
export OLMO_EVAL_UPLOAD=0

# Send uploads to a local API instead (see dashboard/README.md)
export OLMO_EVAL_API_URL=http://localhost:8000

# Build and validate every payload for a results directory without sending it
uv run olmo-eval results upload /tmp/results/ --dry-run
```

The client lives in `src/olmo_eval/upload/`. Its payloads must validate against
`dashboard/contract/ingest-v1.schema.json`, and `tests/upload/` checks them against it.
