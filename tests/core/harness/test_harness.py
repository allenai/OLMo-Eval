"""Tests for Harness class."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import LMRequest, ProviderKind, RequestType
from olmo_eval.harness import clear_registry, register_tool
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.harness.harness import Harness, create_harness
from olmo_eval.harness.tools import tool


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the tool registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def mock_provider_config():
    """Create a mock provider config."""
    return ProviderConfig(kind=ProviderKind.MOCK, model="test-model")


@pytest.fixture
def sample_tool():
    """Create and register a sample tool."""

    @tool(name="test_search", description="Search for information")
    async def test_search(query: str) -> str:
        return f"Results for: {query}"

    register_tool(test_search)
    return test_search


class TestHarness:
    """Tests for the Harness class."""

    def test_harness_creation(self, mock_provider_config):
        """Test creating a Harness."""
        config = HarnessConfig(name="test", provider=mock_provider_config)
        harness = Harness(config)

        assert harness.config is config
        assert harness.model_name == "test-model"

    def test_harness_generate(self, mock_provider_config):
        """Test single-turn generate method."""
        config = HarnessConfig(name="test", provider=mock_provider_config)
        harness = Harness(config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Hello"},),
        )

        outputs = harness.generate([request])

        assert len(outputs) == 1
        assert len(outputs[0]) == 1
        # Mock provider returns "mock response" by default
        assert "mock" in outputs[0][0].text.lower() or outputs[0][0].text

    def test_harness_logprobs(self, mock_provider_config):
        """Test logprobs method."""
        config = HarnessConfig(name="test", provider=mock_provider_config)
        harness = Harness(config)

        request = LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Test prompt",
            continuations=("continuation",),
        )

        outputs = harness.logprobs([request])
        assert len(outputs) == 1

    def test_harness_apply_config_with_tools(self, mock_provider_config, sample_tool):
        """Test that _apply_config injects tool schemas."""
        config = HarnessConfig(
            name="with_tools",
            provider=mock_provider_config,
            tools=("test_search",),
        )
        harness = Harness(config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Search something"},),
        )

        transformed = harness._apply_config(request)

        assert transformed.tools is not None
        assert len(transformed.tools) == 1
        assert transformed.tools[0].name == "test_search"

    def test_harness_apply_config_with_system_prompt(self, mock_provider_config):
        """Test that _apply_config injects system prompt."""
        config = HarnessConfig(
            name="with_prompt",
            provider=mock_provider_config,
            system_prompt="You are a helpful assistant.",
        )
        harness = Harness(config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Hello"},),
        )

        transformed = harness._apply_config(request)

        assert transformed.system_prompt == "You are a helpful assistant."

    def test_harness_apply_config_preserves_images(self, mock_provider_config):
        """_apply_config must carry LMRequest.images through to the provider."""
        config = HarnessConfig(name="with_images", provider=mock_provider_config)
        harness = Harness(config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Describe this image."},),
            images=("fake-image",),
        )

        transformed = harness._apply_config(request)

        assert transformed.images == ("fake-image",)

    def test_harness_apply_config_images_default_none(self, mock_provider_config):
        """Text requests (no images) stay images=None through _apply_config."""
        config = HarnessConfig(name="no_images", provider=mock_provider_config)
        harness = Harness(config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Hello"},),
        )

        transformed = harness._apply_config(request)

        assert transformed.images is None

    def test_harness_inject_system_prompt(self, mock_provider_config):
        """Test system prompt injection into messages."""
        config = HarnessConfig(
            name="inject_test",
            provider=mock_provider_config,
            system_prompt="System message",
        )
        harness = Harness(config)

        messages = ({"role": "user", "content": "Hello"},)
        injected = harness._inject_system_prompt(messages)

        assert len(injected) == 2
        assert injected[0]["role"] == "system"
        assert injected[0]["content"] == "System message"
        assert injected[1]["role"] == "user"

    def test_harness_no_inject_if_system_exists(self, mock_provider_config):
        """Test that system prompt isn't injected if one already exists."""
        config = HarnessConfig(
            name="no_inject",
            provider=mock_provider_config,
            system_prompt="New system",
        )
        harness = Harness(config)

        messages = (
            {"role": "system", "content": "Existing system"},
            {"role": "user", "content": "Hello"},
        )
        injected = harness._inject_system_prompt(messages)

        assert len(injected) == 2  # No new system message added
        assert injected[0]["content"] == "Existing system"

    def test_harness_no_inject_if_no_prompt(self, mock_provider_config):
        """Test that nothing is injected if no system prompt configured."""
        config = HarnessConfig(name="no_prompt", provider=mock_provider_config)
        harness = Harness(config)

        messages = ({"role": "user", "content": "Hello"},)
        injected = harness._inject_system_prompt(messages)

        assert injected == messages


class TestCreateHarness:
    """Tests for create_harness factory function."""

    def test_create_harness_with_mock_provider(self):
        """Test create_harness with mock provider type."""
        config = HarnessConfig(
            name="default",
            provider=ProviderConfig(kind=ProviderKind.MOCK, model="test-model"),
        )
        harness = create_harness(config)

        assert harness.config.name == "default"
        assert harness.provider.model_name == "test-model"

    def test_create_harness_with_config(self):
        """Test create_harness with explicit config."""
        config = HarnessConfig(
            name="explicit",
            provider=ProviderConfig(kind=ProviderKind.MOCK, model="test-model"),
        )
        harness = create_harness(config)

        assert harness.config.name == "explicit"

    def test_create_harness_with_dict(self):
        """Test create_harness with dict config."""
        config_dict = {
            "name": "from_dict",
            "provider": {"kind": "mock", "model_name": "test-model"},
            "system_prompt": "Test prompt",
            "max_turns": 5,
        }
        harness = create_harness(config_dict)

        assert harness.config.name == "from_dict"
        assert harness.config.system_prompt == "Test prompt"
        assert harness.config.max_turns == 5


class TestHarnessRun:
    """Tests for Harness.run() multi-turn execution."""

    @pytest.mark.anyio
    async def test_harness_run_no_scaffold_raises(self, mock_provider_config):
        """Test run raises error when no scaffold is configured."""
        config = HarnessConfig(
            name="no_scaffold",
            provider=mock_provider_config,
        )
        harness = Harness(config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Hello"},),
        )

        with pytest.raises(RuntimeError, match="No scaffold configured"):
            await harness.run(request)


class TestImageRequestRejection:
    """A provider that cannot render images must reject image requests loudly."""

    def _harness_with(self, provider):
        harness = Harness(HarnessConfig(name="test"))
        harness._provider = provider
        return harness

    def _image_request(self):
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "How many cats?"},),
            images=("/tmp/cat.png",),
        )

    def test_text_provider_rejects_image_requests(self):
        class TextOnly:
            supports_images = False

            def generate(self, requests, sampling_params=None):  # pragma: no cover
                raise AssertionError("must not be reached")

        harness = self._harness_with(TextOnly())
        with pytest.raises(ValueError, match="does not support image requests"):
            harness.generate([self._image_request()])

    def test_multimodal_provider_passes(self):
        class Vision:
            supports_images = True

            def generate(self, requests, sampling_params=None):
                return [[]]

        harness = self._harness_with(Vision())
        assert harness.generate([self._image_request()]) == [[]]

    def test_text_requests_unaffected(self):
        class TextOnly:
            supports_images = False

            def generate(self, requests, sampling_params=None):
                return [[]]

        harness = self._harness_with(TextOnly())
        request = LMRequest(request_type=RequestType.CHAT, prompt="2+2?")
        assert harness.generate([request]) == [[]]
