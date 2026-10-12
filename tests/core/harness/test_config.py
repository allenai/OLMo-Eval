"""Tests for HarnessConfig."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from olmo_eval.common.execution import ProcessScoringPoolConfig
from olmo_eval.harness import clear_registry, register_tool
from olmo_eval.harness.config import HarnessConfig, harness_config
from olmo_eval.harness.sandbox import SandboxConfig, SandboxMode
from olmo_eval.harness.sandbox.config import DEFAULT_DEPLOYMENT_TIMEOUT
from olmo_eval.harness.tools import tool


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the tool registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def sample_tool():
    """Create a sample tool for testing."""

    @tool(name="sample_tool", description="A sample tool")
    async def sample(query: str) -> str:
        return query

    register_tool(sample)
    return sample


class TestHarnessConfig:
    """Tests for HarnessConfig dataclass."""

    def test_default_config(self):
        """Test creating a default HarnessConfig."""
        config = HarnessConfig(name="default")

        assert config.name == "default"
        assert config.tool_names == ()
        assert config.system_prompt is None
        assert config.tool_choice == "auto"
        assert config.max_turns is None
        assert config.max_concurrency is None
        assert config.scaffold is None

    def test_config_with_tools(self, sample_tool):
        """Test HarnessConfig with tool names."""
        config = HarnessConfig(
            name="with_tools",
            tools=("sample_tool",),
        )

        assert config.has_tools is True
        tools = config.resolved_tools
        assert len(tools) == 1
        assert tools[0].name == "sample_tool"

    def test_config_tool_schemas(self, sample_tool):
        """Test getting tool schemas from config."""
        config = HarnessConfig(
            name="schemas",
            tools=("sample_tool",),
        )

        schemas = config.tool_schemas
        assert len(schemas) == 1
        assert schemas[0].name == "sample_tool"

    def test_config_has_tools_false(self):
        """Test has_tools is False when no tools configured."""
        config = HarnessConfig(name="no_tools")
        assert config.has_tools is False

    def test_config_serialization(self):
        """Test to_dict / from_dict round-trip."""
        config = HarnessConfig(
            name="serialize",
            tools=("tool_a", "tool_b"),
            system_prompt="Test prompt",
            tool_choice="required",
            max_turns=5,
            max_concurrency=4,
            scaffold="openai_agents",
            scaffold_kwargs={"enable_compaction": False},
            required_secrets=("API_KEY",),
            scoring_process_pools={
                "cpu": ProcessScoringPoolConfig(workers=6),
            },
            sandbox_pool_instances=32,
            sandbox_pool_min_instances=12,
            sandboxes=(
                SandboxConfig(
                    image="python:3.12",
                    mode=SandboxMode.DOCKER,
                ),
            ),
        )

        d = config.to_dict()
        restored = HarnessConfig.from_dict(d)

        assert "scaffold" in d
        assert "backend" not in d
        assert "scaffold_kwargs" in d
        assert "backend_kwargs" not in d

        assert restored.name == config.name
        assert restored.tool_names == config.tool_names
        assert restored.system_prompt == config.system_prompt
        assert restored.tool_choice == config.tool_choice
        assert restored.max_turns == config.max_turns
        assert restored.max_concurrency == config.max_concurrency
        assert restored.scaffold == config.scaffold
        assert restored.scaffold_kwargs == config.scaffold_kwargs
        assert restored.required_secrets == config.required_secrets
        assert restored.scoring_process_pools == config.scoring_process_pools
        assert restored.sandbox_pool_instances == 32
        assert restored.sandbox_pool_min_instances == 12
        assert len(restored.sandboxes) == 1
        assert restored.sandboxes[0].instances is None

    def test_from_dict_accepts_legacy_backend_keys(self):
        """Legacy backend keys should deserialize to the scaffold API."""
        restored = HarnessConfig.from_dict(
            {
                "name": "legacy",
                "backend": "openai_agents",
                "backend_kwargs": {"enable_compaction": False},
                "scoring_process_pools": {
                    "cpu": {"workers": 5, "start_method": "spawn", "max_tasks_per_child": 128}
                },
                "sandbox_pool_instances": 8,
                "sandbox_pool_min_instances": 3,
            }
        )

        serialized = restored.to_dict()

        assert restored.scaffold == "openai_agents"
        assert restored.scaffold_kwargs == {"enable_compaction": False}
        assert restored.scoring_process_pools == {
            "cpu": ProcessScoringPoolConfig(
                workers=5,
                start_method="spawn",
                max_tasks_per_child=128,
            )
        }
        assert restored.sandbox_pool_instances == 8
        assert restored.sandbox_pool_min_instances == 3
        assert "backend" not in serialized
        assert "backend_kwargs" not in serialized
        assert serialized["scaffold"] == "openai_agents"
        assert serialized["scaffold_kwargs"] == {"enable_compaction": False}

    def test_sandbox_config_resolved_min_instances_clamps(self):
        """Configured startup minimums should never exceed resolved executor count."""
        config = SandboxConfig(
            image="python:3.12",
            mode=SandboxMode.DOCKER,
            instances=1,
            min_instances=24,
        )

        assert config.resolved_min_instances == 1

    def test_sandbox_config_deployment_timeout_round_trip(self):
        """Modal deployment lifetime is configurable independently of request timeouts."""
        config = SandboxConfig(
            image="python:3.12",
            mode=SandboxMode.MODAL,
            runtime_timeout=120.0,
            deployment_timeout=14_400.0,
        )

        restored = SandboxConfig.from_dict(config.to_dict())

        assert restored.runtime_timeout == 120.0
        assert restored.deployment_timeout == 14_400.0

    def test_sandbox_config_default_deployment_timeout_exceeds_one_hour(self):
        config = SandboxConfig(image="python:3.12", mode=SandboxMode.MODAL)

        assert config.deployment_timeout == DEFAULT_DEPLOYMENT_TIMEOUT
        assert config.deployment_timeout > 3600.0

    def test_config_immutable(self):
        """Test that HarnessConfig is frozen (immutable)."""
        from dataclasses import FrozenInstanceError

        config = HarnessConfig(name="frozen")

        with pytest.raises(FrozenInstanceError):
            config.name = "changed"

    def test_validate_secrets_all_present(self):
        """Test validate_secrets when all secrets are present."""
        with patch.dict(os.environ, {"TEST_KEY": "value", "OTHER_KEY": "value2"}):
            config = HarnessConfig(
                name="test",
                required_secrets=("TEST_KEY", "OTHER_KEY"),
            )
            missing = config.validate_secrets()
            assert missing == []

    def test_validate_secrets_missing(self):
        """Test validate_secrets when some secrets are missing."""
        with patch.dict(os.environ, {"PRESENT_KEY": "value"}, clear=True):
            config = HarnessConfig(
                name="test",
                required_secrets=("PRESENT_KEY", "MISSING_KEY"),
            )
            missing = config.validate_secrets()
            assert "MISSING_KEY" in missing
            assert "PRESENT_KEY" not in missing

    def test_with_tools_method(self, sample_tool):
        """Test with_tools creates a new config with additional tools."""
        config = HarnessConfig(name="original")
        new_config = config.with_tools("sample_tool")

        assert config.tool_names == ()  # Original unchanged
        assert new_config.tool_names == ("sample_tool",)

    def test_with_system_prompt_method(self):
        """Test with_system_prompt creates a new config."""
        config = HarnessConfig(name="original", system_prompt="old")
        new_config = config.with_system_prompt("new prompt")

        assert config.system_prompt == "old"  # Original unchanged
        assert new_config.system_prompt == "new prompt"


class TestHarnessConfigFactory:
    """Tests for harness_config factory function."""

    def test_harness_config_with_tool_objects(self):
        """Test harness_config accepts Tool objects and registers them."""

        @tool(name="factory_tool")
        async def factory_func(x: str) -> str:
            return x

        register_tool(factory_func)

        config = harness_config(
            name="factory_test",
            tools=[factory_func],  # factory_func is now a Tool object
        )

        assert "factory_tool" in config.tool_names
        # Tool should be registered
        from olmo_eval.harness import get_tool

        assert get_tool("factory_tool") is factory_func

    def test_harness_config_with_tool_names(self, sample_tool):
        """Test harness_config accepts tool names."""
        config = harness_config(
            name="names_test",
            tools=["sample_tool"],
        )

        assert config.tool_names == ("sample_tool",)

    def test_harness_config_mixed_tools(self, sample_tool):
        """Test harness_config with mix of Tool objects and names."""

        @tool(name="another_tool")
        async def another_func(x: str) -> str:
            return x

        config = harness_config(
            name="mixed_test",
            tools=["sample_tool", another_func],  # another_func is now a Tool object
        )

        assert "sample_tool" in config.tool_names
        assert "another_tool" in config.tool_names

    def test_harness_config_all_params(self):
        """Test harness_config with all parameters."""
        config = harness_config(
            name="full_test",
            tools=[],
            system_prompt="Test prompt",
            tool_choice="none",
            max_turns=15,
            max_concurrency=16,
            scoring_process_pools={"cpu": ProcessScoringPoolConfig(workers=4)},
            scaffold="openai_agents",
            required_secrets=["SECRET"],
        )

        assert config.name == "full_test"
        assert config.system_prompt == "Test prompt"
        assert config.tool_choice == "none"
        assert config.max_turns == 15
        assert config.max_concurrency == 16
        assert config.scoring_process_pools == {"cpu": ProcessScoringPoolConfig(workers=4)}
        assert config.scaffold == "openai_agents"
        assert config.required_secrets == ("SECRET",)


class TestScaffoldKwargsIsolation:
    """to_dict must not hand out the config's own scaffold_kwargs dict.

    The override path writes into the dict to_dict returns, and presets are
    cached for the life of the process, so an alias let one launch's overrides
    survive into every later config built from the same preset.
    """

    def _config(self) -> HarnessConfig:
        """A config carrying nested scaffold_kwargs, which is what aliases."""
        return HarnessConfig(
            name="test",
            scaffold="openai_agents",
            scaffold_kwargs={"model_settings": {"temperature": 0.0}},
        )

    def test_mutating_the_returned_dict_leaves_the_config_alone(self):
        """to_dict returns a copy, not the config's own nested dict."""
        config = self._config()

        serialized = config.to_dict()
        serialized["scaffold_kwargs"]["model_settings"]["temperature"] = 1.0

        assert config.scaffold_kwargs["model_settings"]["temperature"] == 0.0

    def test_an_override_does_not_mutate_the_config_it_was_applied_to(self):
        """Applying -o overrides leaves the config they were applied to intact."""
        from olmo_eval.cli.run.config import _apply_harness_overrides

        config = self._config()

        applied = _apply_harness_overrides(
            config, ["scaffold_kwargs.model_settings.reasoning_effort=none"]
        )

        assert applied.scaffold_kwargs["model_settings"]["reasoning_effort"] == "none"
        assert "reasoning_effort" not in config.scaffold_kwargs["model_settings"]

    def test_two_overrides_of_one_config_do_not_accumulate(self):
        """A second launch must not inherit the first launch's overrides."""
        from olmo_eval.cli.run.config import _apply_harness_overrides

        config = self._config()

        _apply_harness_overrides(config, ["scaffold_kwargs.model_settings.reasoning_effort=none"])
        second = _apply_harness_overrides(config, ["scaffold_kwargs.model_settings.top_p=0.9"])

        assert "reasoning_effort" not in second.scaffold_kwargs["model_settings"]
