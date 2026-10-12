# Integration Tests

This directory contains integration tests that require external services like Docker, GPUs, or network access.

## Prerequisites

### For vLLM Provider Tests

1. **Docker with GPU support** (recommended):
   ```bash
   # Install nvidia-container-toolkit
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

   # Verify GPU access in Docker
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

2. **OR vLLM installed locally** with GPU:
   ```bash
   pip install vllm
   ```

## Running Integration Tests

### Quick Start (with Docker)

```bash
# Run all vLLM integration tests
pytest tests/integration/test_vllm_provider.py -v --integration

# The test harness will automatically:
# 1. Start a vLLM Docker container with a small model (Qwen2-0.5B)
# 2. Wait for the model to load (~2-5 minutes)
# 3. Run the tests
# 4. Stop the container
```

### Using a Pre-running vLLM Instance

If you already have vLLM running (locally or in Docker):

```bash
# Skip Docker management
pytest tests/integration/test_vllm_provider.py -v --integration --no-docker
```

### Using a Different Model

```bash
# Use a different model (must be compatible with vLLM)
pytest tests/integration/test_vllm_provider.py -v --integration \
    --vllm-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Manual Docker Setup

If you prefer to manage Docker manually:

```bash
# Start vLLM container
docker compose -f tests/integration/docker-compose.vllm.yml up -d vllm

# Wait for health check to pass
docker compose -f tests/integration/docker-compose.vllm.yml ps

# Run tests (skip Docker management)
pytest tests/integration/test_vllm_provider.py -v --integration --no-docker

# Stop container when done
docker compose -f tests/integration/docker-compose.vllm.yml down
```

### CPU-Only Testing (Experimental)

For environments without GPU:

```bash
# Start CPU-only vLLM (very slow, for basic validation only)
docker compose -f tests/integration/docker-compose.vllm.yml --profile cpu up -d vllm-cpu

# Note: CPU inference is extremely slow and may timeout
```

## Test Structure

```
tests/integration/
├── __init__.py              # Package marker
├── conftest.py              # Pytest fixtures and configuration
├── docker-compose.vllm.yml  # Docker Compose for vLLM
├── README.md                # This file
└── test_vllm_provider.py     # vLLM provider integration tests
```

## Test Categories

### `TestVLLMProviderGenerate`
Tests for text generation:
- Single/multiple prompts
- Sampling parameters (temperature, top_p, etc.)
- Stop sequences
- Multiple samples
- Logprobs during generation
- Deterministic generation

### `TestVLLMProviderLogprobs`
Tests for multiple-choice scoring via logprobs:
- Single/multiple requests
- Continuation scoring
- Correct answer detection

### `TestVLLMProviderEdgeCases`
Edge case handling:
- Empty prompts
- Long prompts
- Special characters
- Empty continuations

### `TestVLLMProviderWithTasks`
End-to-end task integration:
- ARC-style multiple choice
- Batch processing

## Troubleshooting

### Container fails to start
```bash
# Check logs
docker logs olmo-eval-vllm-test

# Common issues:
# - Not enough GPU memory: Try a smaller model
# - Missing CUDA drivers: Install nvidia-container-toolkit
```

### Tests timeout
```bash
# Increase timeout (default 5 minutes for model loading)
# Edit VLLM_STARTUP_TIMEOUT in conftest.py
```

### Out of GPU memory
```bash
# Use a smaller model
pytest tests/integration/ -v --integration --vllm-model "Qwen/Qwen2-0.5B"

# Or reduce GPU memory utilization in docker-compose.yml
```
