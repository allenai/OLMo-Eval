"""Tests for olmo_eval.launch.beaker module."""

import pytest

from olmo_eval.cli.beaker.config_loader import LaunchConfigLoader
from olmo_eval.common.constants.infrastructure import cluster_has_weka
from olmo_eval.launch.beaker import (
    BeakerEnvSecret,
    BeakerJobConfig,
    BeakerWekaBucket,
    _parse_timeout,
    build_install_command,
    calculate_experiment_splits,
    normalize_provider_package,
    parse_install_spec,
    parse_task_with_priority,
    resolve_clusters,
    validate_min_runtime,
    validate_priority_configuration,
)


class TestResolveClustors:
    """Tests for cluster resolution."""

    def test_resolve_h100_alias(self):
        """Test resolving h100 alias."""
        clusters = resolve_clusters("h100")
        assert "ai2/jupiter" in clusters
        assert "ai2/ceres" in clusters

    def test_resolve_a100_alias(self):
        """Test resolving a100 alias."""
        clusters = resolve_clusters("a100")
        assert clusters == ["ai2/saturn"]

    def test_resolve_aus_alias(self):
        """Test resolving aus alias."""
        clusters = resolve_clusters("aus")
        assert "ai2/jupiter" in clusters
        assert "ai2/neptune" in clusters
        assert "ai2/saturn" in clusters
        assert "ai2/ceres" in clusters

    def test_resolve_full_name(self):
        """Test that full cluster names pass through."""
        clusters = resolve_clusters("ai2/jupiter")
        assert clusters == ["ai2/jupiter"]

    def test_resolve_list_of_clusters(self):
        """Test resolving a list of clusters."""
        clusters = resolve_clusters(["ai2/jupiter", "ai2/saturn"])
        assert "ai2/jupiter" in clusters
        assert "ai2/saturn" in clusters

    def test_resolve_mixed_aliases_and_names(self):
        """Test resolving mixed aliases and full names."""
        clusters = resolve_clusters(["h100", "ai2/saturn"])
        assert "ai2/jupiter" in clusters
        assert "ai2/ceres" in clusters
        assert "ai2/saturn" in clusters

    def test_resolve_legacy_cluster_name(self):
        """Test resolving legacy cluster names."""
        clusters = resolve_clusters("ai2/jupiter-cirrascale-2")
        assert clusters == ["ai2/jupiter"]

    def test_resolve_deduplicates(self):
        """Test that duplicate clusters are removed."""
        clusters = resolve_clusters(["h100", "ai2/jupiter"])
        assert clusters.count("ai2/jupiter") == 1


class TestClusterWekaSupport:
    """Tests for cluster Weka availability checks."""

    @pytest.mark.parametrize(
        "cluster",
        [
            "ai2/titan",
            "ai2/titan-batch-b200-aus-ib",
            "ai2/titan-cirrascale",
        ],
    )
    def test_titan_clusters_have_weka(self, cluster):
        """Titan clusters should receive Weka-backed cache environment variables."""
        assert cluster_has_weka(cluster) is True


class TestParseTimeout:
    """Tests for timeout parsing."""

    def test_parse_hours(self):
        """Test parsing hours."""
        ns = _parse_timeout("24h")
        assert ns == 24 * 3600_000_000_000

    def test_parse_minutes(self):
        """Test parsing minutes."""
        ns = _parse_timeout("30m")
        assert ns == 30 * 60_000_000_000

    def test_parse_seconds(self):
        """Test parsing seconds."""
        ns = _parse_timeout("90s")
        assert ns == 90 * 1_000_000_000

    def test_parse_combined(self):
        """Test parsing combined time units."""
        ns = _parse_timeout("1h30m")
        expected = 1 * 3600_000_000_000 + 30 * 60_000_000_000
        assert ns == expected

    def test_parse_invalid_returns_default(self):
        """Test that invalid timeout returns 24h default."""
        ns = _parse_timeout("invalid")
        assert ns == 86400_000_000_000  # 24h in ns


class TestBeakerEnvSecret:
    """Tests for BeakerEnvSecret."""

    def test_creation(self):
        """Test creating a secret."""
        secret = BeakerEnvSecret(name="HF_TOKEN", secret="my_hf_token")
        assert secret.name == "HF_TOKEN"
        assert secret.secret == "my_hf_token"


class TestBeakerWekaBucket:
    """Tests for BeakerWekaBucket."""

    def test_default_mount(self):
        """Test that mount path is auto-generated."""
        bucket = BeakerWekaBucket(bucket="oe-eval-default")
        assert bucket.bucket == "oe-eval-default"
        assert bucket.mount == "/weka/oe-eval-default"

    def test_custom_mount(self):
        """Test custom mount path."""
        bucket = BeakerWekaBucket(bucket="oe-eval-default", mount="/custom/path")
        assert bucket.mount == "/custom/path"


class TestBeakerJobConfig:
    """Tests for BeakerJobConfig."""

    def test_minimal_config(self):
        """Test creating minimal config with required fields."""
        config = BeakerJobConfig(
            name="test-job",
            command=["echo", "hello"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
        )
        assert config.name == "test-job"
        assert config.command == ["echo", "hello"]
        assert config.num_gpus == 0
        assert config.cluster == "h100"
        assert config.workspace == "ai2/oe-data"
        assert config.budget == "ai2/oe-other"
        assert config.priority == "normal"
        assert config.preemptible is True
        assert config.timeout == "24h"

    def test_full_config(self):
        """Test creating full config with all options."""
        config = BeakerJobConfig(
            name="test-job",
            command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
            num_gpus=4,
            shared_memory="20GiB",
            cluster=["ai2/jupiter", "ai2/saturn"],
            priority="high",
            preemptible=False,
            timeout="48h",
            retries=2,
            workspace="ai2/custom-workspace",
            budget="ai2/custom-budget",
            beaker_image="custom-image",
            description="Test description",
            weka_buckets=[BeakerWekaBucket("custom-bucket")],
            nfs=True,
            env_vars={"CUSTOM_VAR": "value"},
            env_secrets=[BeakerEnvSecret("CUSTOM_SECRET", "secret_name")],
        )
        assert config.num_gpus == 4
        assert config.cluster == ["ai2/jupiter", "ai2/saturn"]
        assert config.priority == "high"
        assert config.preemptible is False
        assert config.retries == 2
        assert config.nfs is True
        assert len(config.weka_buckets) == 1

    def test_default_weka_buckets(self):
        """Test default Weka buckets are set and properly configured."""
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
        )
        # Just verify defaults exist and have valid mount paths
        assert len(config.weka_buckets) >= 1
        for bucket in config.weka_buckets:
            assert bucket.bucket  # Non-empty bucket name
            assert bucket.mount and bucket.mount.startswith("/weka/")

    def test_default_secrets(self):
        """Test default env_secrets is empty (secrets are added during launch)."""
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
        )
        # env_secrets defaults to empty list; secrets are injected during launch
        assert len(config.env_secrets) == 0


class TestValidateMinRuntime:
    """Tests for min runtime validation."""

    @pytest.mark.parametrize("value", ["2h", "30m", "90s", "1h30m", "0h5m", " 4h "])
    def test_valid_durations(self, value):
        assert validate_min_runtime(value) == value.strip()

    @pytest.mark.parametrize("value", ["", "2", "1d", "-1h", "2 hours", "30m1h"])
    def test_invalid_durations(self, value):
        with pytest.raises(ValueError, match="invalid min runtime"):
            validate_min_runtime(value)

    @pytest.mark.parametrize("value", ["0s", "0h", "0h0m0s"])
    def test_zero_duration_rejected(self, value):
        with pytest.raises(ValueError, match="must be greater than zero"):
            validate_min_runtime(value)


class TestBeakerJobConfigMinRuntime:
    """Tests for min runtime handling in BeakerJobConfig."""

    def _config(self, **kwargs):
        return BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
            **kwargs,
        )

    def test_min_runtime_replaces_preemptible(self):
        config = self._config(min_runtime="2h")
        assert config.min_runtime == "2h"
        assert config.preemptible is None

    def test_preemptible_false_with_min_runtime_rejected(self):
        with pytest.raises(ValueError, match="cannot be combined with min_runtime"):
            self._config(preemptible=False, min_runtime="2h")

    def test_invalid_min_runtime_rejected(self):
        with pytest.raises(ValueError, match="invalid min runtime"):
            self._config(min_runtime="2")

    def test_preemptible_none_without_min_runtime_defaults_true(self):
        assert self._config(preemptible=None).preemptible is True

    @pytest.mark.parametrize(
        ("kwargs", "expected_preemptible", "expected_min_runtime"),
        [
            ({}, True, None),
            ({"preemptible": False}, False, None),
            ({"min_runtime": "2h"}, None, "2h"),
        ],
    )
    def test_launch_passes_preemption_to_gantry(
        self, kwargs, expected_preemptible, expected_min_runtime
    ):
        from unittest.mock import MagicMock, patch

        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher(workspace="ai2/oe-data")
        launcher._beaker = MagicMock(user_name="test-user")
        config = self._config(weka_buckets=[], **kwargs)

        with patch("gantry.api.launch_experiment") as mock_launch:
            launcher.launch(config)

        call_kwargs = mock_launch.call_args.kwargs
        assert call_kwargs["preemptible"] is expected_preemptible
        assert call_kwargs["min_runtime"] == expected_min_runtime

    @pytest.mark.parametrize(
        ("kwargs", "expected_context"),
        [
            ({}, {"preemptible": True}),
            ({"preemptible": False}, {"preemptible": False}),
            ({"min_runtime": "1h30m"}, {"minRuntime": 90 * 60 * 1_000_000_000}),
        ],
    )
    def test_task_spec_context(self, kwargs, expected_context):
        """The Beaker spec sends minRuntime without the deprecated preemptible field."""
        from beaker.types import BeakerTaskSpec

        config = self._config(**kwargs)
        spec = BeakerTaskSpec.new(
            "test",
            beaker_image="test-image",
            preemptible=config.preemptible,
            min_runtime=config.min_runtime,
        )

        context = spec.to_json()["context"]
        context.pop("priority", None)
        assert context == expected_context

    @pytest.mark.parametrize(
        ("kwargs", "expected_line"),
        [
            ({}, "Preemptible: yes"),
            ({"preemptible": False}, "Preemptible: no"),
            ({"min_runtime": "1h30m"}, "Min runtime: 1h30m"),
        ],
    )
    def test_dry_run_header(self, kwargs, expected_line, capsys):
        from olmo_eval.launch import BeakerLauncher

        BeakerLauncher()._print_dry_run_config(self._config(**kwargs), ["ai2/jupiter"])

        assert expected_line in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("context", "expected_line"),
        [
            ({"preemptible": True}, "Preemptible: yes"),
            ({"minRuntime": 90 * 60 * 1_000_000_000}, "Min runtime: 1h30m"),
        ],
    )
    def test_print_experiment_config_header(self, context, expected_line, capsys):
        from olmo_eval.launch import print_experiment_config

        print_experiment_config({"tasks": [{"context": context}]}, name="test")

        assert expected_line in capsys.readouterr().out


class TestGoogleCredentialInjection:
    """Google credentials reach gantry as google_credentials_secret."""

    def _launch(self, dry_run: bool):
        from unittest.mock import MagicMock, patch

        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher(workspace="ai2/oe-data")
        launcher._beaker = MagicMock(user_name="test-user")
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
            weka_buckets=[],
            inject_gcs_credentials=True,
        )
        with (
            patch("gantry.api.launch_experiment") as mock_launch,
            patch(
                "olmo_eval.launch.beaker.gcs.ensure_gcs_secrets",
                return_value="test-user_GOOGLE_CREDENTIALS",
            ) as mock_ensure,
        ):
            launcher.launch(config, dry_run=dry_run)
        return mock_launch.call_args.kwargs, mock_ensure

    def test_launch_writes_and_injects_the_secret(self):
        kwargs, mock_ensure = self._launch(dry_run=False)
        mock_ensure.assert_called_once_with("ai2/oe-data")
        assert kwargs["google_credentials_secret"] == "test-user_GOOGLE_CREDENTIALS"

    def test_dry_run_injects_without_writing_the_secret(self):
        kwargs, mock_ensure = self._launch(dry_run=True)
        mock_ensure.assert_not_called()
        assert kwargs["google_credentials_secret"] == "test-user_GOOGLE_CREDENTIALS"


class TestUploaderKeyInjection:
    """The uploader key reaches the job as OLMO_EVAL_UPLOAD_CREDENTIALS."""

    def _launch(self, dry_run: bool):
        from unittest.mock import MagicMock, patch

        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher(workspace="ai2/oe-data")
        launcher._beaker = MagicMock(user_name="test-user")
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
            weka_buckets=[],
            inject_upload_credentials=True,
        )
        with (
            patch("gantry.api.launch_experiment") as mock_launch,
            patch(
                "olmo_eval.launch.beaker.uploader.ensure_uploader_secret",
                return_value="olmo_eval_uploader_key",
            ) as mock_ensure,
        ):
            launcher.launch(config, dry_run=dry_run)
        return mock_launch.call_args.kwargs, mock_ensure

    def test_launch_writes_and_injects_the_key(self):
        kwargs, mock_ensure = self._launch(dry_run=False)
        mock_ensure.assert_called_once_with("ai2/oe-data")
        assert ("OLMO_EVAL_UPLOAD_CREDENTIALS", "olmo_eval_uploader_key") in kwargs["env_secrets"]
        assert kwargs["google_credentials_secret"] is None

    def test_dry_run_injects_without_reading_the_key(self):
        kwargs, mock_ensure = self._launch(dry_run=True)
        mock_ensure.assert_not_called()
        assert ("OLMO_EVAL_UPLOAD_CREDENTIALS", "olmo_eval_uploader_key") in kwargs["env_secrets"]


class TestBeakerLauncherImport:
    """Tests for BeakerLauncher import behavior."""

    def test_launcher_imports_without_beaker(self):
        """Test that BeakerLauncher can be imported without beaker-py installed.

        The actual beaker import should be lazy (only when using the launcher).
        """
        from olmo_eval.launch import BeakerLauncher

        # Should be able to instantiate without error
        launcher = BeakerLauncher()
        assert launcher._beaker is None

    def test_config_imports_work(self):
        """Test that config classes can be imported."""
        from olmo_eval.launch import (
            BeakerEnvSecret,
            BeakerJobConfig,
            BeakerWekaBucket,
        )

        # All should be importable
        assert BeakerEnvSecret is not None
        assert BeakerJobConfig is not None
        assert BeakerWekaBucket is not None


class TestParseTaskWithPriority:
    """Tests for task priority parsing."""

    def test_task_only_uses_default(self):
        """Test task without priority uses default."""
        task, priority = parse_task_with_priority("mmlu")
        assert task == "mmlu"
        assert priority == "normal"

    def test_task_with_priority(self):
        """Test task with @priority suffix."""
        task, priority = parse_task_with_priority("mmlu@high")
        assert task == "mmlu"
        assert priority == "high"

    def test_task_with_variant_and_priority(self):
        """Test task with a variant and priority."""
        task, priority = parse_task_with_priority("arc_easy:mc@high")
        assert task == "arc_easy:mc"
        assert priority == "high"

    def test_custom_default_priority(self):
        """Test using custom default priority."""
        task, priority = parse_task_with_priority("mmlu", default_priority="high")
        assert task == "mmlu"
        assert priority == "high"

    def test_explicit_priority_overrides_default(self):
        """Test that explicit @priority overrides default."""
        task, priority = parse_task_with_priority("mmlu@low", default_priority="high")
        assert task == "mmlu"
        assert priority == "low"

    def test_all_valid_priorities(self):
        """Test all valid priority values."""
        for p in ("low", "normal", "high", "urgent"):
            task, priority = parse_task_with_priority(f"mmlu@{p}")
            assert priority == p

    def test_invalid_priority_raises(self):
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="Invalid priority"):
            parse_task_with_priority("mmlu@invalid")

    def test_invalid_priority_error_message(self):
        """Test error message includes valid options."""
        with pytest.raises(ValueError, match="low, normal, high, urgent"):
            parse_task_with_priority("mmlu@bad")


class TestValidatePriorityConfiguration:
    """Tests for priority configuration validation."""

    def test_tasks_without_priority_use_default(self):
        """Test tasks without @priority suffix use default."""
        result = validate_priority_configuration(["mmlu", "gsm8k"])
        assert result == {"normal": ["mmlu", "gsm8k"]}

    def test_tasks_without_priority_custom_default(self):
        """Test tasks without @priority suffix use custom default."""
        result = validate_priority_configuration(["mmlu", "gsm8k"], "high")
        assert result == {"high": ["mmlu", "gsm8k"]}

    def test_tasks_with_priority_suffixes(self):
        """Test tasks with @priority suffixes are grouped correctly."""
        result = validate_priority_configuration(["mmlu@high", "gsm8k@normal", "arc@high"])
        assert result == {"high": ["mmlu", "arc"], "normal": ["gsm8k"]}

    def test_mixed_tasks_with_and_without_priority(self):
        """Test mixed tasks (some with, some without @priority)."""
        result = validate_priority_configuration(["mmlu@high", "gsm8k", "arc@low"])
        # gsm8k should use default "normal"
        assert result == {"high": ["mmlu"], "normal": ["gsm8k"], "low": ["arc"]}

    def test_all_priority_levels(self):
        """Test all valid priority levels work."""
        result = validate_priority_configuration(["a@low", "b@normal", "c@high", "d@urgent"])
        assert result == {"low": ["a"], "normal": ["b"], "high": ["c"], "urgent": ["d"]}

    def test_empty_tasks_list(self):
        """Test empty tasks list returns empty dict."""
        result = validate_priority_configuration([])
        assert result == {}

    def test_tuple_input(self):
        """Test that tuple input works (from CLI)."""
        result = validate_priority_configuration(("mmlu", "gsm8k"))
        assert result == {"normal": ["mmlu", "gsm8k"]}

    def test_task_with_variant_and_priority(self):
        """Test task with variant (:) and @priority suffix."""
        result = validate_priority_configuration(["arc_easy:mc@high"])
        assert result == {"high": ["arc_easy:mc"]}


class TestCalculateExperimentSplits:
    """Tests for calculate_experiment_splits function."""

    def test_single_node_no_split(self):
        """Test case where total GPUs fit on single node."""
        # 4 instances × 2 GPUs = 8 GPUs (fits on 8-GPU node)
        result = calculate_experiment_splits(
            tasks=["a", "b", "c", "d"],
            gpus_per_model=2,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["tasks"] == ["a", "b", "c", "d"]
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 4

    def test_split_into_two_experiments(self):
        """Test case requiring split into 2 experiments."""
        # 4 instances × 4 GPUs = 16 GPUs, max 8 per node = 2 experiments
        result = calculate_experiment_splits(
            tasks=["a", "b", "c", "d"],
            gpus_per_model=4,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 2
        # Each experiment gets 2 instances (8 GPUs) and 2 tasks
        assert result[0]["tasks"] == ["a", "b"]
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 2
        assert result[1]["tasks"] == ["c", "d"]
        assert result[1]["num_gpus"] == 8
        assert result[1]["parallelism"] == 2

    def test_split_into_multiple_experiments(self):
        """Test case requiring split into many experiments."""
        # 8 instances × 4 GPUs = 32 GPUs, max 8 per node = 4 experiments
        result = calculate_experiment_splits(
            tasks=["a", "b", "c", "d", "e", "f", "g", "h"],
            gpus_per_model=4,
            parallelism=8,
            max_gpus_per_node=8,
        )
        assert len(result) == 4
        for split in result:
            assert split["num_gpus"] == 8
            assert split["parallelism"] == 2

    def test_single_task_no_split(self):
        """Test with single task, no split needed."""
        result = calculate_experiment_splits(
            tasks=["mmlu"],
            gpus_per_model=2,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["tasks"] == ["mmlu"]
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 4

    def test_single_task_with_split(self):
        """Test single task distributed across splits."""
        # With 1 task and 2 required experiments, task goes to first experiment
        result = calculate_experiment_splits(
            tasks=["mmlu"],
            gpus_per_model=4,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["tasks"] == ["mmlu"]
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 2

    def test_parallelism_one_no_split(self):
        """Test with parallelism=1, should never split."""
        result = calculate_experiment_splits(
            tasks=["a", "b", "c"],
            gpus_per_model=4,
            parallelism=1,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["tasks"] == ["a", "b", "c"]
        assert result[0]["num_gpus"] == 4
        assert result[0]["parallelism"] == 1

    def test_exactly_fits_node(self):
        """Test when total GPUs exactly equal max per node."""
        # 2 instances × 4 GPUs = 8 GPUs = max
        result = calculate_experiment_splits(
            tasks=["a", "b"],
            gpus_per_model=4,
            parallelism=2,
            max_gpus_per_node=8,
        )
        assert len(result) == 1
        assert result[0]["num_gpus"] == 8
        assert result[0]["parallelism"] == 2

    def test_uneven_task_distribution(self):
        """Test with odd number of tasks split unevenly."""
        # 3 tasks split across 2 experiments
        result = calculate_experiment_splits(
            tasks=["a", "b", "c"],
            gpus_per_model=4,
            parallelism=4,
            max_gpus_per_node=8,
        )
        assert len(result) == 2
        assert result[0]["tasks"] == ["a", "b"]
        assert result[1]["tasks"] == ["c"]

    def test_more_experiments_than_tasks(self):
        """Test when splitting would create more experiments than tasks."""
        # 4 experiments needed, but only 2 tasks
        result = calculate_experiment_splits(
            tasks=["a", "b"],
            gpus_per_model=4,
            parallelism=8,
            max_gpus_per_node=8,
        )
        # Should create experiments for all tasks, even if some splits are empty
        assert len(result) == 2
        assert result[0]["tasks"] == ["a"]
        assert result[1]["tasks"] == ["b"]

    def test_gpu_calculation(self):
        """Test that GPU calculations are correct."""
        # 3 instances × 2 GPUs = 6 GPUs (fits on 8-GPU node)
        result = calculate_experiment_splits(
            tasks=["a"],
            gpus_per_model=2,
            parallelism=3,
            max_gpus_per_node=8,
        )
        assert result[0]["num_gpus"] == 6
        assert result[0]["parallelism"] == 3

    def test_large_model_exceeds_node(self):
        """Test edge case where single model instance exceeds node GPUs."""
        # Model needs 16 GPUs but max is 8 - falls back to 1 instance
        result = calculate_experiment_splits(
            tasks=["a", "b"],
            gpus_per_model=16,
            parallelism=2,
            max_gpus_per_node=8,
        )
        # Should still work, using 1 instance per experiment
        assert len(result) == 2
        assert result[0]["num_gpus"] == 16
        assert result[0]["parallelism"] == 1


class TestBeakerJobConfigTaskPackages:
    """Tests for BeakerJobConfig task_packages field."""

    def test_task_packages_default_none(self):
        """Test that task_packages defaults to None."""
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
        )
        assert config.task_packages is None

    def test_task_packages_can_be_set(self):
        """Test that task_packages can be set."""
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
            task_packages=["special-lib==1.0", "git+https://github.com/user/repo"],
        )
        assert config.task_packages == ["special-lib==1.0", "git+https://github.com/user/repo"]


class TestBuildCommandWithTaskPackages:
    """Tests for command building with task packages."""

    def test_install_cmd_includes_task_packages(self):
        """Test that task packages are installed in the generated install command."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        install_cmd = launcher._build_install_cmd(
            extras=[],
            env_exports=None,
            task_packages=["special-lib==1.0", "another-pkg"],
        )

        # Check that task packages are being installed
        assert "uv pip install 'special-lib==1.0'" in install_cmd
        assert "uv pip install 'another-pkg'" in install_cmd

    def test_provider_packages_installed_before_task_packages(self):
        """Test that provider packages are installed before task packages."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        install_cmd = launcher._build_install_cmd(
            extras=[],
            env_exports=None,
            provider_packages=["vllm==0.14.0"],
            task_packages=["task-dep==1.0"],
        )

        # Provider packages should be installed before task packages
        provider_pos = install_cmd.find("uv pip install 'vllm==0.14.0'")
        task_pos = install_cmd.find("uv pip install 'task-dep==1.0'")
        assert provider_pos < task_pos

    def test_cuda_constraints_exclude_torch_companions_and_cutlass(self):
        """Only Torch itself and the CUDA packages are pinned to the base image.

        Provider releases pin torchvision/torchaudio alongside their Torch
        generation, and vLLM pins CUTLASS DSL independently of Torch's CUDA
        package set, so neither may be frozen to the image's build.
        """
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        install_cmd = launcher._build_install_cmd(
            extras=[],
            env_exports=None,
            provider_packages=["vllm==0.14.0"],
        )

        assert (
            "uv pip freeze -q | grep -E '^(torch(==| @ )|nvidia-)' "
            "| grep -vE '^nvidia-cutlass-dsl' > /tmp/cuda-constraints.txt"
        ) in install_cmd
        assert "grep -E '^(torch|nvidia-)' > /tmp/cuda-constraints.txt" not in install_cmd

    def test_no_task_packages_if_none(self):
        """Test that no extra install steps if task_packages is None."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        install_cmd = launcher._build_install_cmd(
            extras=[],
            env_exports=None,
            task_packages=None,
        )

        # Should only have the base install, not any extra pip install for task packages
        # Count occurrences of 'uv pip install' - should only be the base install
        install_count = install_cmd.count("uv pip install")
        assert install_count == 1  # Only the base olmo-eval install

    def test_task_packages_git_url_normalized(self):
        """Test that git URLs in task packages are normalized."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        install_cmd = launcher._build_install_cmd(
            extras=[],
            env_exports=None,
            task_packages=["https://github.com/user/repo@v1.0"],
        )

        # GitHub URL should get git+ prefix
        assert "uv pip install 'git+https://github.com/user/repo@v1.0'" in install_cmd

    def test_provider_packages_use_isolated_vllm_venv(self):
        """Provider packages should install into the isolated vLLM venv."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        install_cmd = launcher._build_install_cmd(
            extras=["vllm", "clients"],
            env_exports=None,
            provider_packages=["git+https://github.com/user/repo@v1.0"],
            vllm_isolated_venv=True,
        )

        assert (
            "cd /gantry-runtime && uv export --extra vllm "
            "--no-default-groups --no-hashes --no-emit-project "
            "--no-header --no-annotate --frozen 2>/dev/null "
            "| grep -vE '^(torch|nvidia-)' | grep -vE ' @ ' "
            "> /tmp/vllm-lock-constraints.txt"
        ) in install_cmd
        assert "-c /tmp/vllm-lock-constraints.txt -e '.[vllm]'" in install_cmd
        assert (
            "uv --no-config --no-cache pip install --python /opt/vllm-venv/bin/python "
            "--refresh --refresh-package repo --reinstall-package repo "
            "'repo @ git+https://github.com/user/repo@v1.0' -c /tmp/cuda-constraints.txt"
        ) in install_cmd
        assert "[isolated-vllm-check]" not in install_cmd

    def test_task_packages_stay_in_main_env_with_isolated_vllm(self):
        """Task packages should continue installing in the main app environment."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        install_cmd = launcher._build_install_cmd(
            extras=["vllm", "clients"],
            env_exports=None,
            provider_packages=["provider-dep==1.0"],
            task_packages=["task-dep==1.0"],
            vllm_isolated_venv=True,
        )

        assert (
            "uv pip install --python /opt/vllm-venv/bin/python "
            "--refresh-package provider-dep --reinstall-package provider-dep "
            "'provider-dep==1.0' -c /tmp/cuda-constraints.txt"
        ) in install_cmd
        assert "uv pip install 'task-dep==1.0' -c /tmp/cuda-constraints.txt" in install_cmd
        assert (
            "uv pip install --python /opt/vllm-venv/bin/python "
            "'task-dep==1.0' -c /tmp/cuda-constraints.txt"
        ) not in install_cmd
        assert "[isolated-vllm-check]" not in install_cmd

    def test_provider_packages_can_enable_isolated_vllm_venv_without_vllm_extra(self):
        """A custom provider package should still use the isolated vLLM venv."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        install_cmd = launcher._build_install_cmd(
            extras=["clients"],
            env_exports=None,
            provider_packages=["https://github.com/user/vllm@custom"],
            vllm_isolated_venv=True,
        )

        assert "uv venv /opt/vllm-venv" in install_cmd
        assert "export VLLM_PYTHON=/opt/vllm-venv/bin/python" in install_cmd
        assert (
            "cd /gantry-runtime && uv pip install "
            "--python /opt/vllm-venv/bin/python "
            "--cache-dir \"$UV_CACHE_DIR\" -e '.[vllm]'"
        ) not in install_cmd
        assert (
            "uv --no-config --no-cache pip install --python /opt/vllm-venv/bin/python "
            "--refresh --refresh-package vllm --reinstall-package vllm "
            "'vllm @ git+https://github.com/user/vllm@custom' -c /tmp/cuda-constraints.txt"
        ) in install_cmd
        assert "[isolated-vllm-check]" not in install_cmd

    def test_olmo_core_extra_does_not_try_to_build_flash_attention(self):
        """Default OLMo-core jobs must not compile flash-attn at startup."""
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher()
        install_cmd = launcher._build_install_cmd(
            extras=["olmo_core"],
            env_exports=None,
        )

        assert "uv pip install -e '.[olmo_core,beaker]' -c /tmp/cuda-constraints.txt" in install_cmd
        assert "flash-attn" not in install_cmd
        assert "--no-build-isolation-package" not in install_cmd


class TestNormalizeProviderPackage:
    """Tests for normalize_provider_package function."""

    def test_github_url_gets_git_prefix(self):
        """Test GitHub URL gets git+ prefix."""
        result = normalize_provider_package("https://github.com/user/vllm")
        assert result == "git+https://github.com/user/vllm"

    def test_github_url_with_branch_gets_git_prefix(self):
        """Test GitHub URL with branch gets git+ prefix."""
        result = normalize_provider_package("https://github.com/user/vllm@my-branch")
        assert result == "git+https://github.com/user/vllm@my-branch"

    def test_gitlab_url_gets_git_prefix(self):
        """Test GitLab URL gets git+ prefix."""
        result = normalize_provider_package("https://gitlab.com/user/package")
        assert result == "git+https://gitlab.com/user/package"

    def test_git_plus_url_unchanged(self):
        """Test git+ URL passes through unchanged."""
        result = normalize_provider_package("git+https://github.com/user/vllm@v0.14.0")
        assert result == "git+https://github.com/user/vllm@v0.14.0"

    def test_named_direct_reference_preserved(self):
        """Named direct references should keep the package name and normalize the source."""
        result = normalize_provider_package(
            "transformers @ https://github.com/user/transformers@main"
        )
        assert result == "transformers @ git+https://github.com/user/transformers@main"

    def test_pypi_version_unchanged(self):
        """Test PyPI version spec passes through unchanged."""
        result = normalize_provider_package("vllm==0.14.0")
        assert result == "vllm==0.14.0"

    def test_pypi_version_range_unchanged(self):
        """Test PyPI version range passes through unchanged."""
        result = normalize_provider_package("vllm>=0.13.0,<0.15.0")
        assert result == "vllm>=0.13.0,<0.15.0"

    def test_pypi_with_extras_unchanged(self):
        """Test PyPI with extras passes through unchanged."""
        result = normalize_provider_package("vllm[runai]==0.14.0")
        assert result == "vllm[runai]==0.14.0"

    def test_local_path_unchanged(self):
        """Test local path passes through unchanged."""
        result = normalize_provider_package("/path/to/local/vllm")
        assert result == "/path/to/local/vllm"

    def test_package_name_only_unchanged(self):
        """Test package name only passes through unchanged."""
        result = normalize_provider_package("vllm")
        assert result == "vllm"

    def test_strips_install_flags(self):
        """Test that install flags are stripped from package spec."""
        result = normalize_provider_package(
            "git+https://github.com/user/repo@v1.0 --no-build-isolation"
        )
        assert result == "git+https://github.com/user/repo@v1.0"


class TestParseInstallSpec:
    """Tests for parse_install_spec function."""

    def test_no_flags(self):
        """Test package without flags."""
        pkg, flags = parse_install_spec("vllm==0.14.0")
        assert pkg == "vllm==0.14.0"
        assert flags == []

    def test_single_flag(self):
        """Test package with single flag."""
        pkg, flags = parse_install_spec(
            "git+https://github.com/user/repo@v1.0 --no-build-isolation"
        )
        assert pkg == "git+https://github.com/user/repo@v1.0"
        assert flags == ["--no-build-isolation"]

    def test_multiple_flags(self):
        """Test package with multiple flags."""
        pkg, flags = parse_install_spec("vllm==0.14.0 --no-deps --force-reinstall")
        assert pkg == "vllm==0.14.0"
        assert flags == ["--no-deps", "--force-reinstall"]

    def test_flag_with_value(self):
        """Test flag with value."""
        pkg, flags = parse_install_spec("pkg==1.0 --config-settings build=release")
        assert pkg == "pkg==1.0"
        assert flags == ["--config-settings", "build=release"]


class TestBuildInstallCommand:
    """Tests for build_install_command function."""

    def test_simple_package(self):
        """Test simple package without flags."""
        cmd = build_install_command("vllm==0.14.0", "/tmp/constraints.txt")
        assert cmd == "uv pip install 'vllm==0.14.0' -c /tmp/constraints.txt"

    def test_package_with_flag(self):
        """Test package with install flag."""
        cmd = build_install_command(
            "git+https://github.com/user/repo@v1.0 --no-build-isolation",
            "/tmp/constraints.txt",
        )
        expected = (
            "uv pip install --no-build-isolation "
            "'git+https://github.com/user/repo@v1.0' -c /tmp/constraints.txt"
        )
        assert cmd == expected

    def test_github_url_normalized(self):
        """Test GitHub URL gets git+ prefix."""
        cmd = build_install_command("https://github.com/user/repo@v1.0", "/tmp/constraints.txt")
        assert (
            cmd == "uv pip install 'git+https://github.com/user/repo@v1.0' -c /tmp/constraints.txt"
        )

    def test_no_constraints(self):
        """Test without constraints file."""
        cmd = build_install_command("vllm==0.14.0", None)
        assert cmd == "uv pip install 'vllm==0.14.0'"

    def test_virtualenv_target(self):
        """Test targeting a specific virtualenv via --python."""
        cmd = build_install_command("vllm==0.14.0", "/tmp/constraints.txt", "/opt/vllm-venv")
        assert (
            cmd == "uv pip install --python /opt/vllm-venv/bin/python 'vllm==0.14.0' "
            "-c /tmp/constraints.txt"
        )

    def test_force_reinstall_targets_named_package(self):
        """Forced installs should target the overridden package by name."""
        cmd = build_install_command(
            "transformers==5.8.0",
            "/tmp/constraints.txt",
            "/opt/vllm-venv",
            force_reinstall=True,
        )
        assert (
            cmd == "uv pip install --python /opt/vllm-venv/bin/python "
            "--refresh-package transformers --reinstall-package transformers "
            "'transformers==5.8.0' -c /tmp/constraints.txt"
        )

    def test_force_reinstall_targets_git_repo_name(self):
        """Forced git installs should infer the distribution name from the repo."""
        cmd = build_install_command(
            "git+https://github.com/user/transformers.git@custom-branch",
            "/tmp/constraints.txt",
            "/opt/vllm-venv",
            force_reinstall=True,
        )
        assert (
            cmd == "uv --no-config --no-cache pip install --python /opt/vllm-venv/bin/python "
            "--refresh --refresh-package transformers --reinstall-package transformers "
            "'transformers @ git+https://github.com/user/transformers.git@custom-branch' "
            "-c /tmp/constraints.txt"
        )

    def test_force_reinstall_strips_extras_from_named_direct_ref_target(self):
        """Reinstall flags should use the base distribution name, not extras."""
        cmd = build_install_command(
            "ai2-olmo-core[torchao,transformers] @ "
            "git+https://github.com/allenai/OLMo-core.git@feature-branch",
            "/tmp/constraints.txt",
            force_reinstall=True,
        )
        assert (
            cmd == "uv --no-config --no-cache pip install --refresh "
            "--refresh-package ai2-olmo-core --reinstall-package ai2-olmo-core "
            "'ai2-olmo-core[torchao,transformers] @ "
            "git+https://github.com/allenai/OLMo-core.git@feature-branch' "
            "-c /tmp/constraints.txt"
        )

    def test_force_reinstall_normalizes_named_direct_ref_target(self):
        """Named direct-ref reinstall targets should be PEP 503-normalized."""
        cmd = build_install_command(
            "flash_attn.interface[dev] @ "
            "git+https://github.com/user/flash_attn.interface.git@feature-branch",
            "/tmp/constraints.txt",
            force_reinstall=True,
        )
        assert (
            cmd == "uv --no-config --no-cache pip install --refresh "
            "--refresh-package flash-attn-interface --reinstall-package flash-attn-interface "
            "'flash_attn.interface[dev] @ "
            "git+https://github.com/user/flash_attn.interface.git@feature-branch' "
            "-c /tmp/constraints.txt"
        )

    def test_force_reinstall_normalizes_git_repo_target(self):
        """Forced git installs should normalize inferred repo names."""
        cmd = build_install_command(
            "git+https://github.com/user/flash_attn.interface.git@feature-branch",
            "/tmp/constraints.txt",
            force_reinstall=True,
        )
        assert (
            cmd == "uv --no-config --no-cache pip install --refresh "
            "--refresh-package flash-attn-interface --reinstall-package flash-attn-interface "
            "'flash-attn-interface @ "
            "git+https://github.com/user/flash_attn.interface.git@feature-branch' "
            "-c /tmp/constraints.txt"
        )

    def test_force_reinstall_normalizes_egg_fragment_target(self):
        """Explicit egg fragments should normalize to uv's package target form."""
        cmd = build_install_command(
            "git+https://github.com/user/repo.git@feature-branch#egg=flash_attn.interface",
            "/tmp/constraints.txt",
            force_reinstall=True,
        )
        assert (
            cmd == "uv --no-config --no-cache pip install --refresh "
            "--refresh-package flash-attn-interface --reinstall-package flash-attn-interface "
            "'flash-attn-interface @ "
            "git+https://github.com/user/repo.git@feature-branch#egg=flash_attn.interface' "
            "-c /tmp/constraints.txt"
        )

    def test_force_reinstall_normalizes_local_path_target(self):
        """Forced local path installs should normalize inferred path names."""
        cmd = build_install_command(
            "/mnt/packages/flash_attn.interface",
            "/tmp/constraints.txt",
            force_reinstall=True,
        )
        assert (
            cmd == "uv --no-config --no-cache pip install --refresh "
            "--refresh-package flash-attn-interface --reinstall-package flash-attn-interface "
            "'flash-attn-interface @ /mnt/packages/flash_attn.interface' "
            "-c /tmp/constraints.txt"
        )

    def test_force_reinstall_normalizes_plain_package_target(self):
        """Plain package specs should also target canonical distribution names."""
        cmd = build_install_command(
            "flash_attn.interface==1.0.0",
            "/tmp/constraints.txt",
            force_reinstall=True,
        )
        assert (
            cmd == "uv pip install --refresh-package flash-attn-interface "
            "--reinstall-package flash-attn-interface "
            "'flash_attn.interface==1.0.0' -c /tmp/constraints.txt"
        )

    def test_force_reinstall_targets_wheel_distribution_name(self):
        """Forced wheel installs should target the distribution, not the wheel filename."""
        cmd = build_install_command(
            "https://example.org/wheels/flash_attn_interface-3.0.0-cp312-cp312-linux_x86_64.whl",
            "/tmp/constraints.txt",
            force_reinstall=True,
        )
        assert (
            cmd == "uv --no-config --no-cache pip install --refresh "
            "--refresh-package flash-attn-interface "
            "--reinstall-package flash-attn-interface "
            "'flash-attn-interface @ https://example.org/wheels/"
            "flash_attn_interface-3.0.0-cp312-cp312-linux_x86_64.whl' "
            "-c /tmp/constraints.txt"
        )

    def test_force_reinstall_respects_existing_reinstall_flags(self):
        """Explicit reinstall flags should not be duplicated."""
        cmd = build_install_command(
            "transformers==5.8.0 --reinstall-package transformers",
            "/tmp/cuda-constraints.txt",
            "/opt/vllm-venv",
            force_reinstall=True,
        )
        assert (
            cmd == "uv pip install --python /opt/vllm-venv/bin/python "
            "--reinstall-package transformers "
            "'transformers==5.8.0' -c /tmp/cuda-constraints.txt"
        )


class TestDetectGpuRequirement:
    """Tests for GPU requirement detection in LaunchConfigLoader."""

    def test_provider_num_instances_counted_for_gpus(self):
        """provider.num_instances controls GPU allocation."""
        loader = LaunchConfigLoader(
            config_path=None,
            cli_args={
                "harness": "dr_tulu",
                "harness_overrides": [
                    "provider.num_instances=4",
                    "provider.kwargs.tensor_parallel_size=2",
                ],
            },
        )

        gpus = loader._detect_gpu_requirement(
            model_spec="Qwen/Qwen3-8B",
            harness_name="dr_tulu",
            harness_overrides=[
                "provider.num_instances=4",
                "provider.kwargs.tensor_parallel_size=2",
            ],
        )

        # 4 instances × 2 TP = 8 GPUs for main provider
        assert gpus == 8

    def test_main_and_auxiliary_gpu_sum(self):
        """Total GPUs should include both main and auxiliary providers."""
        loader = LaunchConfigLoader(
            config_path=None,
            cli_args={
                "harness": "dr_tulu",
                "harness_overrides": [
                    "provider.num_instances=4",
                    "auxiliary_providers.judge.kind=vllm_server",
                    "auxiliary_providers.judge.model=Qwen/Qwen3-8B",
                    "auxiliary_providers.judge.num_instances=1",
                ],
            },
        )

        gpus = loader._detect_gpu_requirement(
            model_spec="Qwen/Qwen3-8B",
            harness_name="dr_tulu",
            harness_overrides=[
                "provider.num_instances=4",
                "auxiliary_providers.judge.kind=vllm_server",
                "auxiliary_providers.judge.model=Qwen/Qwen3-8B",
                "auxiliary_providers.judge.num_instances=1",
            ],
        )

        # 4 instances × 1 TP (main) + 1 instance × 1 TP (aux) = 5 GPUs
        assert gpus == 5

    def test_api_backed_auxiliary_no_gpus(self):
        """API-backed auxiliary providers should not consume GPUs."""
        loader = LaunchConfigLoader(
            config_path=None,
            cli_args={
                "harness": "dr_tulu",
                "harness_overrides": [
                    "provider.num_instances=2",
                    "auxiliary_providers.judge.kind=litellm",
                    "auxiliary_providers.judge.model=gpt-4o",
                ],
            },
        )

        gpus = loader._detect_gpu_requirement(
            model_spec="Qwen/Qwen3-8B",
            harness_name="dr_tulu",
            harness_overrides=[
                "provider.num_instances=2",
                "auxiliary_providers.judge.kind=litellm",
                "auxiliary_providers.judge.model=gpt-4o",
            ],
        )

        # 2 instances × 1 TP (main) + 0 (litellm is API-backed) = 2 GPUs
        assert gpus == 2

    def test_external_auxiliary_server_no_gpus(self):
        """Auxiliary providers with base_url should not consume GPUs."""
        loader = LaunchConfigLoader(
            config_path=None,
            cli_args={
                "harness": "dr_tulu",
                "harness_overrides": [
                    "provider.num_instances=2",
                    "auxiliary_providers.judge.kind=vllm_server",
                    "auxiliary_providers.judge.model=some-model",
                    "auxiliary_providers.judge.base_url=http://external:8000/v1",
                ],
            },
        )

        gpus = loader._detect_gpu_requirement(
            model_spec="Qwen/Qwen3-8B",
            harness_name="dr_tulu",
            harness_overrides=[
                "provider.num_instances=2",
                "auxiliary_providers.judge.kind=vllm_server",
                "auxiliary_providers.judge.model=some-model",
                "auxiliary_providers.judge.base_url=http://external:8000/v1",
            ],
        )

        # 2 instances × 1 TP (main) + 0 (external server) = 2 GPUs
        assert gpus == 2

    def test_local_auxiliary_counted_with_api_backed_main_provider(self):
        """Local auxiliary instances require GPUs even when the main provider does not."""
        loader = LaunchConfigLoader(config_path=None, cli_args={})

        gpus = loader._detect_gpu_requirement(
            model_spec="gpt-4o",
            harness_name="dr_tulu",
            harness_overrides=[
                "auxiliary_providers.judge.kind=vllm_server",
                "auxiliary_providers.judge.model=Qwen/Qwen3-8B",
                "auxiliary_providers.judge.num_instances=3",
            ],
        )

        assert gpus == 3

    def test_rejects_gpu_count_below_main_and_auxiliary_requirements(self):
        """An explicit allocation must fit every local provider instance."""
        loader = LaunchConfigLoader(
            config_path=None,
            cli_args={
                "model": ("Qwen/Qwen3-8B",),
                "task": ("olmobase:code",),
                "cluster": "h100",
                "workspace": "ai2/test-workspace",
                "budget": "ai2/test-budget",
                "gpus": 4,
                "harness": "dr_tulu",
                "harness_overrides": [
                    "provider.num_instances=2",
                    "auxiliary_providers.judge.kind=vllm_server",
                    "auxiliary_providers.judge.model=Qwen/Qwen3-8B",
                    "auxiliary_providers.judge.num_instances=3",
                ],
            },
        )

        with pytest.raises(SystemExit) as exc_info:
            loader.load()

        assert exc_info.value.code == 1

    def test_auto_sets_gpu_count_to_main_and_auxiliary_requirements(self):
        """The default allocation grows to fit all local provider instances."""
        loader = LaunchConfigLoader(
            config_path=None,
            cli_args={
                "model": ("Qwen/Qwen3-8B",),
                "task": ("olmobase:code",),
                "cluster": "h100",
                "workspace": "ai2/test-workspace",
                "budget": "ai2/test-budget",
                "gpus": None,
                "harness": "dr_tulu",
                "harness_overrides": [
                    "provider.num_instances=2",
                    "auxiliary_providers.judge.kind=vllm_server",
                    "auxiliary_providers.judge.model=Qwen/Qwen3-8B",
                    "auxiliary_providers.judge.num_instances=3",
                ],
            },
        )

        config = loader.load()

        assert config.gpus == 5


class TestLaunchConfigLoaderExperimentNames:
    """Tests for auto-generated Beaker experiment names."""

    def test_generated_name_strips_task_priority_suffixes(self):
        loader = LaunchConfigLoader(
            config_path=None,
            cli_args={
                "model": ("XiaomiMiMo/MiMo-7B-Base", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"),
                "task": ("olmobase:code@urgent", "olmobase:code_fim@urgent"),
                "cluster": "h100",
                "workspace": "ai2/test-workspace",
                "budget": "ai2/test-budget",
                "gpus": 1,
            },
        )

        config = loader.load()

        assert config.name == "olmobase_code-olmobase_code_fim"

    def test_generated_name_strips_priority_for_many_tasks(self):
        loader = LaunchConfigLoader(
            config_path=None,
            cli_args={
                "model": ("XiaomiMiMo/MiMo-7B-Base",),
                "task": ("olmobase:code@urgent", "olmobase:code_fim@urgent", "olmobase:math@high"),
                "cluster": "h100",
                "workspace": "ai2/test-workspace",
                "budget": "ai2/test-budget",
                "gpus": 1,
            },
        )

        config = loader.load()

        assert config.name == "mimo-7b-base-olmobase_code-and-2-more"


class TestLaunchConfigLoaderPreemption:
    """Tests for resolving preemptible and min_runtime across the CLI and config file."""

    def _load(self, cli_args=None, yaml_body=None, tmp_path=None):
        base_cli_args = {
            "model": ("XiaomiMiMo/MiMo-7B-Base",),
            "task": ("mmlu",),
            "cluster": "h100",
            "workspace": "ai2/test-workspace",
            "budget": "ai2/test-budget",
            "gpus": 1,
        }
        config_path = None
        if yaml_body is not None:
            assert tmp_path is not None
            path = tmp_path / "config.yaml"
            path.write_text("name: test\nmodels: [m]\ntasks: [mmlu]\n" + yaml_body)
            config_path = str(path)
        loader = LaunchConfigLoader(
            config_path=config_path, cli_args={**base_cli_args, **(cli_args or {})}
        )
        return loader.load()

    def test_default_is_preemptible(self):
        config = self._load()
        assert config.preemptible is True
        assert config.min_runtime is None

    def test_cli_min_runtime(self):
        config = self._load({"min_runtime": "2h"})
        assert config.preemptible is None
        assert config.min_runtime == "2h"

    @pytest.mark.parametrize("preemptible", [True, False])
    def test_cli_min_runtime_with_preemptible_rejected(self, preemptible):
        with pytest.raises(SystemExit):
            self._load({"min_runtime": "2h", "preemptible": preemptible})

    def test_cli_invalid_min_runtime_rejected(self):
        with pytest.raises(SystemExit):
            self._load({"min_runtime": "two hours"})

    def test_config_file_min_runtime(self, tmp_path):
        config = self._load(yaml_body="min_runtime: 4h\n", tmp_path=tmp_path)
        assert config.preemptible is None
        assert config.min_runtime == "4h"

    @pytest.mark.parametrize("preemptible", ["true", "false"])
    def test_config_file_min_runtime_with_preemptible_rejected(self, preemptible, tmp_path):
        with pytest.raises(SystemExit):
            self._load(
                yaml_body=f"preemptible: {preemptible}\nmin_runtime: 4h\n", tmp_path=tmp_path
            )

    def test_config_file_preemptible_false(self, tmp_path):
        config = self._load(yaml_body="preemptible: false\n", tmp_path=tmp_path)
        assert config.preemptible is False
        assert config.min_runtime is None

    def test_cli_min_runtime_replaces_config_file_preemptible(self, tmp_path):
        config = self._load(
            {"min_runtime": "2h"}, yaml_body="preemptible: false\n", tmp_path=tmp_path
        )
        assert config.preemptible is None
        assert config.min_runtime == "2h"

    def test_cli_preemptible_replaces_config_file_min_runtime(self, tmp_path):
        config = self._load(
            {"preemptible": False}, yaml_body="min_runtime: 4h\n", tmp_path=tmp_path
        )
        assert config.preemptible is False
        assert config.min_runtime is None
