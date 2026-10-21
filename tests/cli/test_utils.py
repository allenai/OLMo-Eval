"""Tests for CLI utility functions."""

import click
import pytest

from olmo_eval.cli.utils import (
    FlaggedArg,
    _format_transformers_runtime_rows,
    extract_priority_from_overrides,
    process_ordered_args,
    reconstruct_ordered_args,
)


class TestFlaggedArg:
    """Tests for FlaggedArg dataclass."""

    def test_creation(self):
        """Test creating FlaggedArg."""
        arg = FlaggedArg(flag="m", value="llama3.1-8b")
        assert arg.flag == "m"
        assert arg.value == "llama3.1-8b"


class TestReconstructOrderedArgs:
    """Tests for reconstruct_ordered_args function."""

    def test_empty_args(self):
        """Test with empty arguments."""
        result = reconstruct_ordered_args([])
        assert result == []

    def test_single_task(self):
        """Test single task argument."""
        result = reconstruct_ordered_args(["-t", "mmlu"])
        assert len(result) == 1
        assert result[0].flag == "t"
        assert result[0].value == "mmlu"

    def test_single_task_long_form(self):
        """Test single task with --task."""
        result = reconstruct_ordered_args(["--task", "mmlu"])
        assert len(result) == 1
        assert result[0].flag == "t"
        assert result[0].value == "mmlu"

    def test_task_with_equals(self):
        """Test task with = syntax."""
        result = reconstruct_ordered_args(["-t=mmlu"])
        assert len(result) == 1
        assert result[0].flag == "t"
        assert result[0].value == "mmlu"

    def test_multiple_tasks(self):
        """Test multiple tasks."""
        result = reconstruct_ordered_args(
            [
                "-t",
                "mmlu",
                "-t",
                "gsm8k",
            ]
        )
        assert len(result) == 2
        assert result[0] == FlaggedArg("t", "mmlu")
        assert result[1] == FlaggedArg("t", "gsm8k")

    def test_tasks_with_overrides(self):
        """Test tasks with overrides in order."""
        result = reconstruct_ordered_args(
            [
                "-t",
                "mmlu",
                "-o",
                "limit=100",
                "-o",
                "batch_size=8",
                "-t",
                "gsm8k",
                "-o",
                "limit=50",
            ]
        )
        assert len(result) == 5
        assert result[0] == FlaggedArg("t", "mmlu")
        assert result[1] == FlaggedArg("o", "limit=100")
        assert result[2] == FlaggedArg("o", "batch_size=8")
        assert result[3] == FlaggedArg("t", "gsm8k")
        assert result[4] == FlaggedArg("o", "limit=50")

    def test_ignores_other_flags(self):
        """Test that non-relevant flags are ignored."""
        result = reconstruct_ordered_args(
            [
                "-m",
                "model1",
                "--cluster",
                "h100",
                "-t",
                "task1",
                "--gpus",
                "4",
            ]
        )
        # Only -t is tracked, -m is not handled by reconstruct_ordered_args
        assert len(result) == 1
        assert result[0] == FlaggedArg("t", "task1")


class TestProcessOrderedArgs:
    """Tests for process_ordered_args function."""

    def test_empty_list(self):
        """Test with empty list."""
        task_overrides, harness_overrides = process_ordered_args([])
        assert task_overrides == {}
        assert harness_overrides == []

    def test_single_task_no_overrides(self):
        """Test single task without overrides."""
        ordered = [FlaggedArg("t", "mmlu")]
        task_overrides, harness_overrides = process_ordered_args(ordered)
        assert task_overrides == {"mmlu": []}
        assert harness_overrides == []

    def test_single_task_with_overrides(self):
        """Test single task with overrides."""
        ordered = [
            FlaggedArg("t", "mmlu"),
            FlaggedArg("o", "limit=100"),
        ]
        task_overrides, harness_overrides = process_ordered_args(ordered)
        assert task_overrides == {"mmlu": ["limit=100"]}
        assert harness_overrides == []

    def test_task_accepts_sandbox_allocation_weight_override(self):
        """Task overrides should allow sandbox allocation weight hints."""
        ordered = [
            FlaggedArg("t", "bigcodebench:olmo3base"),
            FlaggedArg("o", "sandbox_allocation_weight=6.0"),
        ]

        task_overrides, harness_overrides = process_ordered_args(ordered)

        assert task_overrides == {"bigcodebench:olmo3base": ["sandbox_allocation_weight=6.0"]}
        assert harness_overrides == []

    def test_multiple_tasks_with_overrides(self):
        """Test multiple tasks with overrides."""
        ordered = [
            FlaggedArg("t", "mmlu"),
            FlaggedArg("o", "limit=100"),
            FlaggedArg("t", "gsm8k"),  # No overrides
        ]
        task_overrides, harness_overrides = process_ordered_args(ordered)
        assert task_overrides == {
            "mmlu": ["limit=100"],
            "gsm8k": [],
        }
        assert harness_overrides == []

    def test_harness_with_overrides(self):
        """Test harness with overrides."""
        ordered = [
            FlaggedArg("h", "default"),
            FlaggedArg("o", "max_turns=10"),
        ]
        task_overrides, harness_overrides = process_ordered_args(ordered)
        assert task_overrides == {}
        assert harness_overrides == ["max_turns=10"]

    def test_harness_accepts_scoring_process_pool_override(self):
        """Harness overrides should allow scoring_process_pools dotlist paths."""
        ordered = [
            FlaggedArg("h", "default"),
            FlaggedArg("o", "scoring_process_pools.cpu.workers=8"),
        ]

        task_overrides, harness_overrides = process_ordered_args(ordered)

        assert task_overrides == {}
        assert harness_overrides == ["scoring_process_pools.cpu.workers=8"]

    def test_override_without_preceding_flag_raises(self):
        """Test that -o without preceding -t or --harness raises error."""
        ordered = [FlaggedArg("o", "gpus=4")]
        with pytest.raises(click.UsageError, match="-o/--override must follow"):
            process_ordered_args(ordered)


class TestExtractPriorityFromOverrides:
    """Tests for extract_priority_from_overrides function."""

    def test_empty_overrides(self):
        """Test with empty overrides dict."""
        priority, filtered = extract_priority_from_overrides({})
        assert priority is None
        assert filtered == {}

    def test_no_priority_override(self):
        """Test with no priority in overrides."""
        task_overrides = {"mmlu": ["limit=100", "batch_size=8"]}
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority is None
        assert filtered == {"mmlu": ["limit=100", "batch_size=8"]}

    def test_priority_extracted(self):
        """Test priority is extracted from overrides."""
        task_overrides = {"mmlu": ["priority=urgent", "limit=100"]}
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority == "urgent"
        assert filtered == {"mmlu": ["limit=100"]}

    def test_priority_only_override(self):
        """Test when only priority is in overrides - task not in filtered result."""
        task_overrides = {"mmlu": ["priority=high"]}
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority == "high"
        assert filtered == {}  # No other overrides left

    def test_multiple_tasks_with_priority(self):
        """Test priority from multiple tasks - last one wins."""
        task_overrides = {
            "mmlu": ["priority=normal", "limit=100"],
            "gsm8k": ["priority=urgent"],
        }
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority == "urgent"  # Last one wins
        assert filtered == {"mmlu": ["limit=100"]}

    def test_preserves_non_priority_overrides(self):
        """Test that non-priority overrides are preserved."""
        task_overrides = {
            "mmlu": ["limit=100", "priority=high", "batch_size=8"],
            "gsm8k": ["limit=50"],
        }
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority == "high"
        assert filtered == {
            "mmlu": ["limit=100", "batch_size=8"],
            "gsm8k": ["limit=50"],
        }


class TestEndToEndOrdering:
    """End-to-end tests for CLI arg ordering."""

    def test_full_command_line(self):
        """Test reconstructing and processing a full command line."""
        args = [
            "run",
            "-t",
            "mmlu",
            "-o",
            "limit=100",
            "-t",
            "gsm8k",
            "--output-dir",
            "/tmp/results",
        ]
        ordered = reconstruct_ordered_args(args)
        task_overrides, harness_overrides = process_ordered_args(ordered)

        assert task_overrides == {
            "mmlu": ["limit=100"],
            "gsm8k": [],
        }
        assert harness_overrides == []

    def test_worker_batching_launch_shape_resolves_harness_overrides(self):
        """The async worker launch command should preserve worker and batch overrides."""
        from olmo_eval.cli.run.config import RunConfigBuilder

        args = [
            "-H",
            "default",
            "-o",
            "provider.kind=vllm_server",
            "-o",
            "provider.kwargs.reasoning_parser=qwen3",
            "-o",
            "provider.trust_remote_code=true",
            "-o",
            "provider.num_instances=8",
            "-o",
            "batching.chunk_size=2",
            "-m",
            "Qwen/Qwen3-4B-Thinking-2507",
            "-t",
            "hmmt_nov_2025:pass_at_32@urgent",
        ]
        ordered = reconstruct_ordered_args(args)
        task_overrides, harness_overrides = process_ordered_args(ordered)

        config = RunConfigBuilder(
            model="Qwen/Qwen3-4B-Thinking-2507",
            task=("hmmt_nov_2025:pass_at_32",),
            output_dir="/tmp/results",
            harness_preset="default",
            cli_task_overrides=task_overrides,
            cli_harness_overrides=harness_overrides,
        ).build()

        assert str(config.harness_config.provider.kind) == "vllm_server"
        assert config.harness_config.provider.kwargs["reasoning_parser"] == "qwen3"
        assert config.harness_config.provider.trust_remote_code is True
        assert config.harness_config.provider.num_instances == 8
        assert config.harness_config.batching is not None
        assert config.harness_config.batching.chunk_size == 2

    def test_run_config_builder_wires_force_download_model_flag(self):
        """The CLI flag should become provider.force_download."""
        from olmo_eval.cli.run.config import RunConfigBuilder

        config = RunConfigBuilder(
            model="Qwen/Qwen3-4B-Thinking-2507",
            task=("hmmt_nov_2025:pass_at_32",),
            output_dir="/tmp/results",
            force_download_model=True,
        ).build()

        assert config.harness_config.provider.force_download is True


class TestFormatTransformersRuntimeRows:
    """Tests for runtime summary formatting of transformers versions."""

    def test_non_isolated_runtime_uses_single_transformers_row(self):
        """Non-isolated runs should keep the compact single-row summary."""
        rows = _format_transformers_runtime_rows("5.8.0", None, None)

        assert rows == [("Transformers", "5.8.0")]

    def test_isolated_runtime_shows_main_and_vllm_rows(self):
        """Isolated vLLM runs should distinguish main and server environments."""
        rows = _format_transformers_runtime_rows(
            "5.8.0",
            "5.0.0.dev0",
            "/opt/vllm-venv/bin/python",
        )

        assert rows == [
            ("Transformers (main)", "5.8.0"),
            ("Transformers (vLLM)", "5.0.0.dev0"),
        ]

    def test_isolated_runtime_surfaces_missing_main_transformers(self):
        """The summary should stay explicit even if only the isolated env has transformers."""
        rows = _format_transformers_runtime_rows(
            None,
            "5.0.0.dev0",
            "/opt/vllm-venv/bin/python",
        )

        assert rows == [
            ("Transformers (main)", "[dim]NOT INSTALLED[/dim]"),
            ("Transformers (vLLM)", "5.0.0.dev0"),
        ]


def test_task_override_accepts_every_representable_task_config_field() -> None:
    """The accepted-override list is derived from TaskConfig, so it cannot go stale."""
    import dataclasses

    import click
    import pytest

    from olmo_eval.cli.utils import _NON_CLI_TASK_FIELDS, FlaggedArg, process_ordered_args
    from olmo_eval.evals.tasks.common.base import TaskConfig

    for field in dataclasses.fields(TaskConfig):
        ordered = [FlaggedArg("t", "mmlu"), FlaggedArg("o", f"{field.name}=x")]
        if field.name in _NON_CLI_TASK_FIELDS:
            with pytest.raises(click.UsageError):
                process_ordered_args(ordered)
            continue
        task_overrides, _ = process_ordered_args(ordered)
        assert task_overrides["mmlu"] == [f"{field.name}=x"], field.name
    assert {"answer_extractor", "sandbox_env"} == _NON_CLI_TASK_FIELDS
