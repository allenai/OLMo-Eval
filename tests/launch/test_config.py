"""Tests for olmo_eval.launch.config module."""

import tempfile

from olmo_eval.launch.config import (
    EvalConfig,
    get_model_short_name,
    get_tasks_short_name,
)


class TestGetModelShortName:
    """Tests for get_model_short_name function."""

    def test_simple_model_name(self):
        """Test simple model name returns as-is (lowercased)."""
        assert get_model_short_name("llama3.1-8b") == "llama3.1-8b"

    def test_huggingface_path(self):
        """Test HuggingFace path returns last component."""
        assert get_model_short_name("meta-llama/Llama-3.1-8B") == "llama-3.1-8b"

    def test_local_path(self):
        """Test local checkpoint path returns everything after checkpoints/."""
        path = "/weka/checkpoints/lucas/model/step1000-hf"
        assert get_model_short_name(path) == "lucas/model/step1000-hf"

    def test_local_path_with_trailing_slash(self):
        """Test local checkpoint path with trailing slash."""
        path = "/weka/checkpoints/lucas/model/step1000-hf/"
        assert get_model_short_name(path) == "lucas/model/step1000-hf"

    def test_alias_overrides_name(self):
        """Test alias is used when provided."""
        path = "/weka/checkpoints/lucas/model/step1000-hf/"
        assert get_model_short_name(path, alias="my-model-1k") == "my-model-1k"

    def test_s3_checkpoint_path(self):
        """Test S3 checkpoint path returns everything after checkpoints/."""
        path = "s3://ai2-llm/checkpoints/lucas/olmo3_1b_v2/step61007-hf/"
        assert get_model_short_name(path) == "lucas/olmo3_1b_v2/step61007-hf"

    def test_alias_is_lowercased(self):
        """Test alias is lowercased."""
        assert get_model_short_name("some-model", alias="My-Model-Name") == "my-model-name"

    def test_long_path_uses_last_16_chars(self):
        """Test very long last component uses last 16 chars of full path."""
        long_component = "a" * 40
        result = get_model_short_name(f"/weka/models/{long_component}")
        assert len(result) == 16
        assert result == "a" * 16

    def test_non_checkpoint_path(self):
        """Test non-checkpoint path returns last component."""
        assert get_model_short_name("/weka/models/my-model-name") == "my-model-name"


class TestGetTasksShortName:
    """Tests for get_tasks_short_name function."""

    def test_single_task(self):
        """Test single task returns task name."""
        assert get_tasks_short_name(["mmlu"]) == "mmlu"

    def test_single_task_with_priority(self):
        """Test single task strips @priority suffix."""
        assert get_tasks_short_name(["mmlu@high"]) == "mmlu"

    def test_single_task_with_variant(self):
        """Test single task strips :variant suffix."""
        assert get_tasks_short_name(["arc:mc"]) == "arc"

    def test_single_task_with_stacked_variant(self):
        """Test single task strips stacked variant suffixes."""
        assert get_tasks_short_name(["arc_easy:mc:full"]) == "arc"

    def test_two_tasks(self):
        """Test two tasks joined with underscore."""
        assert get_tasks_short_name(["gsm8k", "arc_challenge"]) == "gsm8k_arc"

    def test_three_tasks(self):
        """Test three tasks joined with underscore."""
        assert get_tasks_short_name(["mmlu", "gsm8k", "hellaswag"]) == "mmlu_gsm8k_hellaswa"

    def test_four_or_more_tasks(self):
        """Test 4+ tasks uses first task and count."""
        result = get_tasks_short_name(["mmlu", "gsm8k", "hellaswag", "arc_challenge"])
        assert result == "mmlu_3more"

    def test_many_tasks(self):
        """Test many tasks uses first task and count."""
        tasks = ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "winogrande", "truthfulqa"]
        result = get_tasks_short_name(tasks)
        assert result == "mmlu_5more"

    def test_empty_list(self):
        """Test empty task list returns placeholder."""
        assert get_tasks_short_name([]) == "notasks"

    def test_strips_challenge_suffix(self):
        """Test _challenge suffix is removed."""
        assert get_tasks_short_name(["arc_challenge"]) == "arc"

    def test_long_task_name_truncated(self):
        """Test long task names are truncated."""
        result = get_tasks_short_name(["verylongtasknamethatshouldbetruncated"])
        assert len(result) <= 24

    def test_mixed_priorities_and_variants(self):
        """Test tasks with mixed priorities and variants."""
        tasks = ["mmlu@high", "gsm8k", "arc_easy:mc@low"]
        result = get_tasks_short_name(tasks)
        assert result == "mmlu_gsm8k_arc"


class TestEvalConfig:
    """Tests for EvalConfig dataclass."""

    def test_from_yaml_simple_models(self):
        """Test loading YAML with simple string models."""
        yaml_content = """
name: test-eval
models:
  - llama3.1-8b
  - olmo-2-7b
tasks:
  - mmlu
  - gsm8k
cluster: h100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name)

            assert config.name == "test-eval"
            assert len(config.models) == 2
            assert config.models[0] == "llama3.1-8b"
            assert config.models[1] == "olmo-2-7b"
            assert config.cluster == "h100"

    def test_from_yaml_with_cli_overrides(self):
        """Test YAML loading with CLI-style overrides."""
        yaml_content = """
name: test-eval
models:
  - llama3.1-8b
tasks:
  - mmlu
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = EvalConfig.from_yaml(f.name, overrides=["priority=high"])

            assert config.priority == "high"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config = EvalConfig.from_dict(
            {
                "name": "test-eval",
                "models": ["llama3.1-8b"],
                "tasks": ["mmlu"],
                "cluster": "a100",
            }
        )

        assert config.name == "test-eval"
        assert config.models == ["llama3.1-8b"]
        assert config.tasks == ["mmlu"]
        assert config.cluster == "a100"

    def test_to_yaml(self):
        """Test exporting config to YAML."""
        config = EvalConfig.from_dict(
            {
                "name": "test-eval",
                "models": ["llama3.1-8b"],
                "tasks": ["mmlu"],
            }
        )

        yaml_str = config.to_yaml()
        assert "name: test-eval" in yaml_str
        assert "llama3.1-8b" in yaml_str

    def test_default_values(self):
        """Test default values are set correctly."""
        config = EvalConfig.from_dict(
            {
                "name": "test",
                "models": ["model"],
                "tasks": ["task"],
            }
        )

        assert config.priority == "normal"
        assert config.preemptible is None
        assert config.min_runtime is None
        assert config.timeout == "24h"
        assert config.gpus == 1
