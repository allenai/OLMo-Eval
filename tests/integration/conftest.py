"""Pytest configuration and fixtures for integration tests."""

import subprocess
import time
from pathlib import Path

import pytest

DOCKER_COMPOSE_FILE = Path(__file__).parent / "docker-compose.vllm.yml"
VLLM_CONTAINER_NAME = "olmo-eval-vllm-test"
VLLM_STARTUP_TIMEOUT = 300  # 5 minutes for model loading


def pytest_addoption(parser):
    """Add custom command line options for vLLM tests."""
    parser.addoption(
        "--vllm-model",
        action="store",
        default="Qwen/Qwen2-0.5B",
        help="Model to use for vLLM integration tests",
    )


def _is_container_running(container_name: str) -> bool:
    """Check if a Docker container is running."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _is_container_healthy(container_name: str) -> bool:
    """Check if a Docker container is healthy."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Health.Status}}", container_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "healthy"


def _wait_for_vllm(timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
    """Wait for vLLM container to be healthy."""
    import httpx

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(5)
    return False


@pytest.fixture(scope="session")
def vllm_model(request) -> str:
    """Get the model name for vLLM tests."""
    return request.config.getoption("--vllm-model")


@pytest.fixture(scope="session")
def vllm_service(request):
    """Start vLLM Docker container for the test session.

    This fixture manages the lifecycle of the vLLM container:
    - Starts the container if not already running
    - Waits for it to be healthy
    - Yields control to tests
    - Stops the container after tests complete (unless --no-docker)
    """
    no_docker = request.config.getoption("--no-docker")

    if no_docker:
        # Assume vLLM is already running
        yield "http://localhost:8000"
        return

    # Check if container is already running
    already_running = _is_container_running(VLLM_CONTAINER_NAME)

    if not already_running:
        print("\nStarting vLLM container with docker-compose...")
        subprocess.run(
            ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "up", "-d", "vllm"],
            check=True,
        )

    # Wait for vLLM to be ready
    print("Waiting for vLLM to be ready (this may take a few minutes)...")
    if not _wait_for_vllm():
        # Get logs for debugging
        logs = subprocess.run(
            ["docker", "logs", VLLM_CONTAINER_NAME],
            capture_output=True,
            text=True,
        )
        print(f"vLLM logs:\n{logs.stdout}\n{logs.stderr}")
        pytest.fail("vLLM container failed to become healthy")

    print("vLLM is ready!")
    yield "http://localhost:8000"

    # Cleanup: stop container if we started it
    if not already_running and not no_docker:
        print("\nStopping vLLM container...")
        subprocess.run(
            ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "down"],
            check=False,
        )


@pytest.fixture(scope="session")
def vllm_provider(vllm_service, vllm_model):
    """Create a VLLMProvider instance connected to the test container.

    Note: This creates an in-process vLLM instance, not using the Docker
    container's API. For true integration testing with the container,
    use the OpenAI-compatible API directly.
    """
    # Skip if vLLM not installed
    pytest.importorskip("vllm")

    from olmo_eval.inference.vllm import VLLMProvider

    # Create provider with small memory footprint for testing
    provider = VLLMProvider(
        vllm_model,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        dtype="half",
    )

    yield provider


@pytest.fixture
def small_test_prompts() -> list[str]:
    """Provide a small set of test prompts."""
    return [
        "The capital of France is",
        "2 + 2 equals",
        "The color of the sky is",
    ]
