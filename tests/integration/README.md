# Integration Tests

Tests in this directory need external services: Docker containers, a GPU, or downloaded
datasets. Unit tests (`pytest tests/ --ignore=tests/integration`) never run them.

Tests are selected with two root-level pytest options:

- `--gpu` runs tests marked `gpu`; without it they are skipped.
- `--no-docker` skips tests marked `integration`. For the vLLM tests it instead tells the
  fixtures to use an already running server rather than managing a container.

## Storage, metrics, and sandbox tests

`test_storage.py`, `test_storage_postgres_new.py`, `test_repository.py`, and
`test_metrics.py` use the Postgres and LocalStack services in `docker-compose.yml`.
`test_livecodebench_sandbox.py` needs Docker and the LiveCodeBench dataset.

```bash
docker compose -f tests/integration/docker-compose.yml up -d --wait
uv run pytest tests/integration/ -v
docker compose -f tests/integration/docker-compose.yml down -v
```

`./scripts/verify.sh` starts and stops these containers around the full test run. CI runs
`test_storage.py` against service containers.

## vLLM provider tests

`test_vllm_provider.py` is marked `gpu` and needs a CUDA GPU. vLLM is in the default
dependency group on Linux, so `uv sync --frozen` installs it.

```bash
# Starts a vLLM container with a small model, runs the tests, and stops it
uv run pytest tests/integration/test_vllm_provider.py -v --gpu

# Use a vLLM server you started yourself
uv run pytest tests/integration/test_vllm_provider.py -v --gpu --no-docker

# Use a different model
uv run pytest tests/integration/test_vllm_provider.py -v --gpu \
    --vllm-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

To manage the container manually:

```bash
docker compose -f tests/integration/docker-compose.vllm.yml up -d vllm
docker compose -f tests/integration/docker-compose.vllm.yml ps   # wait until healthy
uv run pytest tests/integration/test_vllm_provider.py -v --gpu --no-docker
docker compose -f tests/integration/docker-compose.vllm.yml down
```

A CPU-only profile exists for basic validation, but inference is very slow and may time
out:

```bash
docker compose -f tests/integration/docker-compose.vllm.yml --profile cpu up -d vllm-cpu
```

### Troubleshooting

- Container fails to start: check `docker logs olmo-eval-vllm-test`. Common causes are
  insufficient GPU memory (use a smaller model) and a missing nvidia-container-toolkit.
- Model loading times out: `VLLM_STARTUP_TIMEOUT` in `conftest.py` sets the limit.
