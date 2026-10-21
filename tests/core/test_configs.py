"""Tests for olmo_eval.core.configs module."""

# Import to ensure suites are registered
import olmo_eval.evals  # noqa: F401
from olmo_eval.common.configs import expand_tasks, get_provider_config
from olmo_eval.harness.config import ProviderConfig


class TestExpandTasks:
    """Tests for expand_tasks function."""

    def test_expand_single_task(self):
        """Test expanding a single task (no expansion needed)."""
        result = expand_tasks(["arc_challenge"])

        assert result == ["arc_challenge"]

    def test_expand_multiple_tasks(self):
        """Test expanding multiple tasks (no expansion needed)."""
        result = expand_tasks(["arc_challenge", "arc_easy"])

        assert result == ["arc_challenge", "arc_easy"]

    def test_expand_suite(self):
        """Test expanding a suite to its tasks."""
        result = expand_tasks(["mt_mbpp_v2fix"])

        # mt_mbpp_v2fix should expand to multiple tasks (17 languages)
        assert len(result) > 1
        assert all(isinstance(t, str) for t in result)

    def test_expand_mixed_tasks_and_suites(self):
        """Test expanding mix of tasks and suites."""
        result = expand_tasks(["humaneval", "mt_mbpp_v2fix"])

        # Should have humaneval plus all mt_mbpp_v2fix tasks
        assert "humaneval" in result
        assert len(result) > 2

    def test_expand_empty_list(self):
        """Test expanding empty list."""
        result = expand_tasks([])

        assert result == []

    def test_expand_preserves_task_order(self):
        """Test that task order is preserved."""
        result = expand_tasks(["arc_easy", "arc_challenge"])

        assert result[0] == "arc_easy"
        assert result[1] == "arc_challenge"

    def test_expand_suite_with_priority(self):
        """Test expanding a suite with priority suffix."""
        result = expand_tasks(["mt_mbpp_v2fix@high"])

        # All expanded tasks should have the priority suffix
        assert len(result) > 1
        assert all(t.endswith("@high") for t in result)


class TestGetProviderConfig:
    """Tests for get_provider_config function."""

    def test_get_preset_model(self):
        """Test getting a preset model config."""
        config = get_provider_config("llama3.1-8b")

        assert isinstance(config, ProviderConfig)
        assert config.model == "meta-llama/Meta-Llama-3.1-8B"
        assert config.get_provider_name() == "vllm"

    def test_get_unknown_model_as_hf_path(self):
        """Test that unknown model name is treated as HF path."""
        config = get_provider_config("some-org/custom-model")

        assert config.model == "some-org/custom-model"
        assert config.get_provider_name() == "vllm_server"  # Default

    def test_get_model_with_override(self):
        """Test getting model with field override."""
        config = get_provider_config("llama3.1-8b", kind="hf")

        assert config.model == "meta-llama/Meta-Llama-3.1-8B"
        assert config.get_provider_name() == "hf"

    def test_get_model_with_multiple_overrides(self):
        """Test getting model with multiple overrides."""
        config = get_provider_config(
            "llama3.1-8b",
            kind="hf",
            dtype="float16",
            revision="main",
            force_download=True,
        )

        assert config.get_provider_name() == "hf"
        assert config.dtype == "float16"
        assert config.revision == "main"
        assert config.force_download is True

    def test_get_unknown_model_with_overrides(self):
        """Test unknown model with overrides."""
        config = get_provider_config(
            "custom/model",
            kind="hf",
        )

        assert config.model == "custom/model"
        assert config.get_provider_name() == "hf"

    def test_get_model_extra_args_merged(self):
        """Test that kwargs are merged for presets."""
        # Override with additional kwargs
        config = get_provider_config(
            "llama3.1-8b",
            kwargs={"custom_arg": "value"},
        )

        assert "custom_arg" in config.kwargs
        assert config.kwargs["custom_arg"] == "value"

    def test_preset_not_mutated(self):
        """Test that getting with overrides doesn't mutate preset."""
        original = get_provider_config("llama3.1-8b")
        _ = get_provider_config("llama3.1-8b", kind="hf")
        after = get_provider_config("llama3.1-8b")

        assert original.get_provider_name() == after.get_provider_name() == "vllm"

    def test_tokenizer_override_preset(self):
        """Test tokenizer override on a preset model."""
        config = get_provider_config("llama3.1-8b", tokenizer="allenai/dolma2-tokenizer")

        assert config.model == "meta-llama/Meta-Llama-3.1-8B"
        assert config.tokenizer == "allenai/dolma2-tokenizer"

    def test_tokenizer_override_custom_model(self):
        """Test tokenizer override on a custom (non-preset) model."""
        config = get_provider_config(
            "custom/my-model",
            tokenizer="custom/my-tokenizer",
        )

        assert config.model == "custom/my-model"
        assert config.tokenizer == "custom/my-tokenizer"

    def test_tokenizer_default_is_none(self):
        """Test that tokenizer defaults to None for models without preset tokenizer."""
        config = get_provider_config("llama3.1-8b")

        # llama3.1 doesn't have a preset tokenizer - defaults to None
        assert config.tokenizer is None


class TestProviderConfigKwargsPassthrough:
    """Unknown provider keys become constructor kwargs for every kind (so a typo
    surfaces in the provider instead of being dropped silently)."""

    def test_unknown_keys_reach_kwargs(self):
        from olmo_eval.inference.providers.config import ProviderConfig

        config = ProviderConfig.from_dict(
            {"kind": "hf", "model": "m", "multimodal": True, "max_crops": 24}
        )
        assert config.kwargs["multimodal"] is True
        assert config.kwargs["max_crops"] == 24

    def test_known_fields_stay_fields(self):
        from olmo_eval.inference.providers.config import ProviderConfig

        config = ProviderConfig.from_dict({"kind": "hf", "model": "m", "dtype": "float32"})
        assert config.dtype == "float32"
        assert "dtype" not in config.kwargs
