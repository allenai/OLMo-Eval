"""Tests for beaker launch functionality."""

from __future__ import annotations

import pytest

from olmo_eval.common.constants.infrastructure import BEAKER_UV_CACHE_DIR
from olmo_eval.inference.providers.config import ProviderConfig


class TestProviderConfigDependenciesValidation:
    """Tests for ProviderConfig.from_dict dependencies validation."""

    def test_dependencies_as_list_accepted(self):
        """Test that dependencies as a list is accepted."""
        config = ProviderConfig.from_dict(
            {"dependencies": ["https://github.com/user/repo@v1.0", "some-package==1.0"]}
        )
        assert config.dependencies == (
            "https://github.com/user/repo@v1.0",
            "some-package==1.0",
        )

    def test_dependencies_as_empty_list_accepted(self):
        """Test that empty dependencies list is accepted."""
        config = ProviderConfig.from_dict({"dependencies": []})
        assert config.dependencies == ()

    def test_dependencies_default_to_empty(self):
        """Test that dependencies default to empty tuple."""
        config = ProviderConfig.from_dict({})
        assert config.dependencies == ()

    def test_dependencies_as_string_rejected(self):
        """Test that dependencies as a string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ProviderConfig.from_dict({"dependencies": "https://github.com/user/repo@v1.0"})

        assert "must be a list, not a string" in str(exc_info.value)
        assert "https://github.com/user/repo@v1.0" in str(exc_info.value)
        assert "provider.dependencies=[url1,url2]" in str(exc_info.value)

    def test_dependencies_string_does_not_iterate_chars(self):
        """Test that string dependencies don't silently iterate to characters.

        This is the bug we're preventing - tuple("abc") gives ('a', 'b', 'c').
        """
        with pytest.raises(ValueError):
            ProviderConfig.from_dict({"dependencies": "abc"})


class TestHarnessOverridesProviderDependencies:
    """Tests for harness overrides with provider.dependencies."""

    def test_apply_harness_overrides_with_list_dependencies(self):
        """Test that harness overrides with list dependencies work."""
        from olmo_eval.cli.run.config import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset

        preset = get_harness_preset("default")

        # Apply override with list syntax (as OmegaConf parses it)
        overrides = ["provider.dependencies=[https://github.com/user/repo@v1.0]"]
        result = _apply_harness_overrides(preset, overrides)

        assert result.provider.dependencies == ("https://github.com/user/repo@v1.0",)

    def test_apply_harness_overrides_with_string_dependencies_raises(self):
        """Test that harness overrides with string dependencies raises error."""
        from olmo_eval.cli.run.config import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset

        preset = get_harness_preset("default")

        # This is what happens when user forgets brackets
        overrides = ["provider.dependencies=https://github.com/user/repo@v1.0"]

        with pytest.raises(ValueError, match="must be a list, not a string"):
            _apply_harness_overrides(preset, overrides)

    def test_harness_provider_dependencies_in_job_config(self):
        """Test that harness provider.dependencies end up in BeakerJobConfig."""
        from unittest.mock import patch

        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        # Create a minimal launch config with harness overrides
        launch_config = LaunchConfig(
            name="test",
            model_specs=["test-model"],
            task_specs=["humaneval"],  # Use real task
            cluster="h100",
            workspace="ai2/test",
            budget="ai2/test",
            harness="default",
            harness_overrides=["provider.dependencies=[https://github.com/user/repo@v1.0]"],
        )

        # Create a minimal experiment plan
        exp = ExperimentPlan(
            name="test",
            model_spec="test-model",
            priority="normal",
            tasks=["humaneval"],  # Use real task
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )

        # Create job assembler with minimal required params
        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        # Mock cluster_has_weka to avoid cluster lookups
        with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
            job_config = assembler.assemble(exp)

        # The harness override should have been extracted
        assert job_config.provider_packages is not None
        assert "https://github.com/user/repo@v1.0" in job_config.provider_packages

    def test_provider_package_overrides_vllm_extra(self):
        """Test that provider.package overrides the default vllm extra."""
        from olmo_eval.cli.run.config import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset

        preset = get_harness_preset("default")

        # Apply provider.package override
        overrides = ["provider.package=https://github.com/user/vllm@custom"]
        result = _apply_harness_overrides(preset, overrides)

        assert result.provider.package == "https://github.com/user/vllm@custom"

    def test_provider_package_skips_vllm_extra_in_job_config(self):
        """Test that provider.package causes vllm extra to be skipped."""
        from unittest.mock import patch

        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        # Create config with provider.package override
        launch_config = LaunchConfig(
            name="test",
            model_specs=["test-model"],
            task_specs=["humaneval"],
            cluster="h100",
            workspace="ai2/test",
            budget="ai2/test",
            harness="default",
            harness_overrides=["provider.package=https://github.com/user/vllm@custom"],
        )

        exp = ExperimentPlan(
            name="test",
            model_spec="test-model",
            priority="normal",
            tasks=["humaneval"],
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )

        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
            job_config = assembler.assemble(exp)

        # vllm should NOT be in extras (provider.package overrides it)
        assert "vllm" not in job_config.extras
        # But provider.package should be in provider_packages
        assert job_config.provider_packages is not None
        assert "https://github.com/user/vllm@custom" in job_config.provider_packages

    def test_olmo_core_provider_package_replaces_olmo_core_extra(self):
        """OLMo-core package overrides should replace the bundled olmo_core extra."""
        from unittest.mock import patch

        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        package = "ai2-olmo-core[torchao,transformers]==2.3.0"
        launch_config = LaunchConfig(
            name="test",
            model_specs=["/weka/checkpoints/model/step100"],
            task_specs=["humaneval"],
            cluster="h100",
            workspace="ai2/test",
            budget="ai2/test",
            harness="default",
            harness_overrides=[
                "provider.kind=olmo_core",
                f"provider.package={package}",
            ],
        )
        exp = ExperimentPlan(
            name="test",
            model_spec="/weka/checkpoints/model/step100",
            priority="normal",
            tasks=["humaneval"],
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )
        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
            job_config = assembler.assemble(exp)

        assert "olmo_core" not in job_config.extras
        assert "vllm" not in job_config.extras
        assert job_config.provider_packages == [package]

    def test_olmo_core_source_package_override_binds_distribution_name(self):
        """Bare OLMo-core source URLs should reinstall the ai2-olmo-core distribution."""
        from unittest.mock import patch

        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        source = "https://github.com/allenai/OLMo-core.git@feature-branch"
        launch_config = LaunchConfig(
            name="test",
            model_specs=["/weka/checkpoints/model/step100"],
            task_specs=["humaneval"],
            cluster="h100",
            workspace="ai2/test",
            budget="ai2/test",
            harness="default",
            harness_overrides=[
                "provider.kind=olmo_core",
                f"provider.package={source}",
            ],
        )
        exp = ExperimentPlan(
            name="test",
            model_spec="/weka/checkpoints/model/step100",
            priority="normal",
            tasks=["humaneval"],
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )
        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
            job_config = assembler.assemble(exp)

        assert "olmo_core" not in job_config.extras
        assert job_config.provider_packages == [
            "ai2-olmo-core[torchao,transformers] @ "
            "git+https://github.com/allenai/OLMo-core.git@feature-branch"
        ]

    @pytest.mark.parametrize("provider_kind", ["olmo_core", "olmo_core_vlm"])
    def test_olmo_core_provider_package_accepts_version_shorthand(self, provider_kind):
        """Version-only OLMo-core package overrides should target ai2-olmo-core."""
        from olmo_eval.cli.beaker.job_assembler import normalize_provider_package_for_kind

        assert (
            normalize_provider_package_for_kind(provider_kind, "2.3.0")
            == "ai2-olmo-core[torchao,transformers]==2.3.0"
        )
        assert (
            normalize_provider_package_for_kind(provider_kind, ">=2.3,<2.4")
            == "ai2-olmo-core[torchao,transformers]>=2.3,<2.4"
        )
        assert (
            normalize_provider_package_for_kind(
                provider_kind, "https://github.com/allenai/OLMo-core.git@feature-branch"
            )
            == "ai2-olmo-core[torchao,transformers] @ "
            "git+https://github.com/allenai/OLMo-core.git@feature-branch"
        )

    def test_apply_harness_overrides_with_global_sandbox_override(self):
        """Test that sandboxes={...} sets the shared pool and common sandbox fields."""
        from olmo_eval.cli.run.config import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset
        from olmo_eval.harness.sandbox import SandboxMode

        preset = get_harness_preset("codex_universal")

        result = _apply_harness_overrides(
            preset,
            [
                'sandboxes={"mode":"modal","instances":64,"min_instances":24,'
                '"registry_auth":{"provider":"gcp"}}'
            ],
        )

        assert result.sandbox_pool_instances == 64
        assert result.sandbox_pool_min_instances == 24
        assert all(sandbox.mode == SandboxMode.MODAL for sandbox in result.sandboxes)
        assert all(sandbox.registry_auth is not None for sandbox in result.sandboxes)
        assert all(sandbox.registry_auth.provider == "gcp" for sandbox in result.sandboxes)

    def test_get_task_configs_applies_sandbox_allocation_weight_override(self):
        """Beaker task config preview should carry scheduler-only weight overrides."""
        from olmo_eval.cli.beaker.launch import _get_task_configs

        task_configs = _get_task_configs(
            ["bigcodebench:olmo3base"],
            {"bigcodebench:olmo3base": ["sandbox_allocation_weight=6.0"]},
        )

        assert task_configs["bigcodebench:olmo3base"].sandbox_allocation_weight == 6.0


class TestJobConfigAssemblerEnvironment:
    """Tests for Beaker job environment assembly."""

    def test_titan_cluster_gets_uv_cache_dir(self):
        """Titan clusters should get Weka-backed UV cache env vars."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        launch_config = LaunchConfig(
            name="test",
            model_specs=["test-model"],
            task_specs=["humaneval"],
            cluster="ai2/titan-cirrascale",
            workspace="ai2/test",
            budget="ai2/test",
            uv_cache_dir=BEAKER_UV_CACHE_DIR,
        )
        exp = ExperimentPlan(
            name="test",
            model_spec="test-model",
            priority="normal",
            tasks=["humaneval"],
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )
        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        job_config = assembler.assemble(exp)

        assert job_config.env_vars["UV_CACHE_DIR"] == BEAKER_UV_CACHE_DIR
        assert job_config.env_vars["UV_LINK_MODE"] == "copy"

    def test_force_download_model_is_forwarded_to_run_command(self):
        """Beaker task jobs should pass the cache-refresh flag into olmo-eval run."""
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        launch_config = LaunchConfig(
            name="test",
            model_specs=["test-model"],
            task_specs=["humaneval"],
            cluster="h100",
            workspace="ai2/test",
            budget="ai2/test",
            force_download_model=True,
        )
        exp = ExperimentPlan(
            name="test",
            model_spec="test-model",
            priority="normal",
            tasks=["humaneval"],
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )
        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        command = assembler._build_command(exp)

        assert "--force-download-model" in command


class TestPreemptionForwarding:
    """Tests that preemption settings reach the Beaker job config."""

    @pytest.mark.parametrize(
        ("preemptible", "min_runtime"),
        [(True, None), (False, None), (None, "2h")],
    )
    def test_task_job_forwards_preemption(self, preemptible, min_runtime):
        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        launch_config = LaunchConfig(
            name="test",
            model_specs=["test-model"],
            task_specs=["humaneval"],
            cluster="h100",
            workspace="ai2/test",
            budget="ai2/test",
            preemptible=preemptible,
            min_runtime=min_runtime,
        )
        exp = ExperimentPlan(
            name="test",
            model_spec="test-model",
            priority="normal",
            tasks=["humaneval"],
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )
        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )

        job_config = assembler.assemble(exp)

        assert job_config.preemptible is preemptible
        assert job_config.min_runtime == min_runtime

    def test_external_eval_job_forwards_min_runtime(self):
        from olmo_eval.cli.beaker.job_assembler import assemble_external_eval_job

        job_config = assemble_external_eval_job(
            name="test",
            model="test-model",
            external_evals=["tau2_bench"],
            cluster="h100",
            num_gpus=1,
            workspace="ai2/test",
            beaker_image="test-image",
            preemptible=None,
            min_runtime="2h",
        )

        assert job_config.preemptible is None
        assert job_config.min_runtime == "2h"

    def test_resolve_preemption(self):
        from olmo_eval.cli.beaker.config_loader import resolve_preemption

        assert resolve_preemption(None, None) == (True, None)
        assert resolve_preemption(False, None) == (False, None)
        assert resolve_preemption(None, "2h") == (None, "2h")
        with pytest.raises(ValueError, match="cannot both be set"):
            resolve_preemption(True, "2h")


class TestTaskExpansionInExperimentSummary:
    """Tests for task expansion in _build_experiment_summary."""

    def test_expanded_tasks_match_task_configs(self):
        """Test that expanded tasks are used to lookup task configs."""
        from olmo_eval.common.configs import expand_tasks

        # Verify that minerva_math:olmo3base expands to multiple tasks
        expanded = expand_tasks(["minerva_math:olmo3base"])
        assert len(expanded) > 1

        # The expanded tasks should have different names than the input
        assert "minerva_math:olmo3base" not in expanded
        # But should contain minerva_math variants
        assert any("minerva_math" in t for t in expanded)

    def test_build_experiment_summary_uses_expanded_tasks(self):
        """Test that _build_experiment_summary expands tasks before lookup."""
        from unittest.mock import MagicMock, patch

        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.launch import _build_experiment_summary
        from olmo_eval.common.configs import expand_tasks

        # Create experiment with unexpanded task spec
        exp = ExperimentPlan(
            name="test",
            model_spec="allenai/test-model",
            priority="normal",
            tasks=["minerva_math:olmo3base"],  # Unexpanded suite name
            original_task_specs=["minerva_math:olmo3base"],
            total_expanded_tasks=7,
            num_gpus=1,
        )

        # Create mock job config
        mock_job_config = MagicMock()

        # Get the actual expanded task names
        expanded = expand_tasks(["minerva_math:olmo3base"])

        # Create task_configs_by_spec keyed by EXPANDED task names
        # (This simulates how _get_task_configs works)
        task_configs_by_spec = {}
        for task_name in expanded:
            mock_config = MagicMock()
            mock_config.name = task_name
            task_configs_by_spec[task_name] = mock_config

        # Patch the harness and provider lookups that happen inside the function
        with patch("olmo_eval.harness.get_harness_preset") as mock_harness:
            mock_harness_config = MagicMock()
            mock_harness_config.scaffold = None
            mock_harness_config.sandboxes = ()
            mock_harness_config.merge_provider = MagicMock(return_value=mock_harness_config)
            mock_harness.return_value = mock_harness_config

            with patch("olmo_eval.common.configs.get_provider_config") as mock_provider:
                mock_provider.return_value = MagicMock()

                summary = _build_experiment_summary(exp, mock_job_config, task_configs_by_spec)

        # The summary should have found all expanded tasks
        # because _build_experiment_summary now expands the task specs
        assert len(summary.tasks) == len(expanded)

    def test_unexpanded_task_spec_would_miss_lookup(self):
        """Test that without expansion, task lookup would fail.

        This documents the bug we fixed - unexpanded task specs
        don't match keys in task_configs_by_spec.
        """
        from olmo_eval.common.configs import expand_tasks

        # The suite name
        suite_spec = "minerva_math:olmo3base"

        # The expanded tasks have different names
        expanded = expand_tasks([suite_spec])

        # Create dict keyed by expanded names
        task_configs_by_spec = {task: f"config_for_{task}" for task in expanded}

        # The unexpanded suite name is NOT a key
        assert suite_spec not in task_configs_by_spec

        # But expanded tasks ARE keys
        for task in expanded:
            assert task in task_configs_by_spec


class TestProviderKindConsistency:
    """The Beaker install step and the in-container run must agree on the provider kind.

    Regression tests for a job that installed vLLM into the isolated server venv (because
    the harness preset says vllm_server) and then ran the model preset's in-process vllm
    provider, which could not import vLLM from the main venv.
    """

    CASES = [
        # (model, harness, overrides, expected kind)
        ("olmo-3-1025-7b", "default", [], "vllm"),
        ("olmo-3-1025-7b", "default", ["provider.kind=vllm_server"], "vllm_server"),
        ("olmo-3-1025-7b", None, ["provider.kind=vllm_server"], "vllm_server"),
        ("olmo-3-1025-7b", None, [], "vllm"),
        ("gpt-4o", "default", [], "litellm"),
        ("Qwen/Qwen3-4B", "default", [], "vllm_server"),
        ("Qwen/Qwen3-4B", "default", ["provider.kind=vllm"], "vllm"),
        ("Qwen/Qwen3-4B", "default", ['provider={"kind":"vllm"}'], "vllm"),
        # Scaffolds need an OpenAI client, so in-process vllm runs as a server
        ("llama3.1-8b-instruct", "simple_agent", [], "vllm_server"),
        ("gpt-4o", "simple_agent", [], "litellm"),
    ]

    @staticmethod
    def _assemble(model: str, harness: str | None, overrides: list[str]):
        from unittest.mock import patch

        from olmo_eval.cli.beaker.config_loader import LaunchConfig
        from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
        from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler

        launch_config = LaunchConfig(
            name="test",
            model_specs=[model],
            task_specs=["humaneval"],
            cluster="h100",
            workspace="ai2/test",
            budget="ai2/test",
            harness=harness,
            harness_overrides=overrides,
        )
        exp = ExperimentPlan(
            name="test",
            model_spec=model,
            priority="normal",
            tasks=["humaneval"],
            original_task_specs=["humaneval"],
            total_expanded_tasks=1,
            num_gpus=1,
        )
        assembler = JobConfigAssembler(
            config=launch_config,
            effective_image="test-image",
            effective_groups=[],
            beaker_username="test-user",
            common_secrets=[],
            store_secrets=[],
            task_secrets=[],
            inject_aws_credentials=False,
            inject_gcs_credentials=False,
        )
        with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
            return assembler.assemble(exp)

    @pytest.mark.parametrize(("model", "harness", "overrides", "expected"), CASES)
    def test_run_resolves_expected_kind(self, model, harness, overrides, expected):
        from olmo_eval.cli.run.config import RunConfigBuilder

        config = RunConfigBuilder(
            model=model,
            task=("humaneval",),
            output_dir="/tmp/results",
            harness_preset=harness,
            cli_harness_overrides=overrides,
        ).build()

        assert str(config.harness_config.provider.kind) == expected

    @pytest.mark.parametrize(("model", "harness", "overrides", "expected"), CASES)
    def test_launcher_resolves_expected_kind(self, model, harness, overrides, expected):
        from olmo_eval.cli.beaker.job_assembler import resolve_provider_kind
        from olmo_eval.cli.run.config import _apply_harness_overrides
        from olmo_eval.harness import get_harness_preset

        preset = None
        if harness:
            preset = _apply_harness_overrides(get_harness_preset(harness), overrides)

        assert resolve_provider_kind(model, preset, overrides) == expected

    @pytest.mark.parametrize(("model", "harness", "overrides", "expected"), CASES)
    def test_isolated_vllm_venv_only_for_vllm_server(self, model, harness, overrides, expected):
        job_config = self._assemble(model, harness, overrides)

        assert job_config.vllm_isolated_venv is (expected == "vllm_server")
        assert ("vllm" in job_config.extras) is (expected in ("vllm", "vllm_server"))
        assert ("litellm" in job_config.extras) is (expected == "litellm")


class TestProviderKindOverride:
    """Tests for extracting an explicit provider.kind from CLI overrides."""

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ([], None),
            (["provider.max_model_len=4096"], None),
            (["provider.kind=vllm_server"], "vllm_server"),
            (['provider={"kind":"litellm"}'], "litellm"),
            (["provider.kind=vllm", "provider.kind=vllm_server"], "vllm_server"),
            (["provider.kwargs.kind=vllm"], None),
            (["providers.kind=vllm"], None),
            (["batching.chunk_size=2"], None),
            (["provider.dependencies.0=example-package==2"], None),
            (
                ["provider.dependencies.0=example-package==2", "provider.kind=vllm_server"],
                "vllm_server",
            ),
        ],
    )
    def test_provider_kind_override(self, overrides, expected):
        from olmo_eval.cli.run.config import provider_kind_override

        assert provider_kind_override(overrides) == expected

    def test_indexed_provider_override_with_harness_config(self):
        from olmo_eval.cli.run.config import _apply_harness_overrides, merge_model_provider
        from olmo_eval.common.configs import get_provider_config
        from olmo_eval.harness import get_harness_preset

        overrides = ["provider.dependencies.0=example-package==2"]
        preset = get_harness_preset("default").with_provider_overrides(
            dependencies=["example-package==1"]
        )
        preset = _apply_harness_overrides(preset, overrides)
        merged = merge_model_provider(preset, get_provider_config("olmo-3-1025-7b"), overrides)

        assert list(merged.provider.dependencies) == ["example-package==2"]
