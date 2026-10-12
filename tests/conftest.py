"""Root pytest configuration."""

import pytest

# Register anyio plugin for async tests
pytest_plugins = ["anyio"]


@pytest.fixture(autouse=True)
def _no_results_upload(monkeypatch):
    """Keep tests from uploading results to the dashboard ingest service."""
    monkeypatch.setenv("OLMO_EVAL_UPLOAD", "0")
    monkeypatch.delenv("OLMO_EVAL_API_URL", raising=False)
    monkeypatch.delenv("OLMO_EVAL_LAUNCH_ID", raising=False)
    monkeypatch.setattr("olmo_eval.upload.auth._uploader_key", None)
    # Behave the same with or without Application Default Credentials on this machine.
    monkeypatch.setattr("olmo_eval.upload.auth.has_local_google_credentials", lambda: False)


@pytest.fixture
def anyio_backend():
    """Use only asyncio backend for async tests (trio not installed)."""
    return "asyncio"


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run GPU tests (requires GPU and vLLM)",
    )
    parser.addoption(
        "--no-docker",
        action="store_true",
        default=False,
        help="Skip Docker-based integration tests",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires Docker services)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on flags."""
    # Skip GPU tests unless --gpu flag is passed
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="GPU tests require --gpu flag")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip integration tests if --no-docker flag is passed
    if config.getoption("--no-docker"):
        skip_integration = pytest.mark.skip(reason="Integration tests skipped with --no-docker")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
