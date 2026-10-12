"""Tests for olmo_eval.runner module."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

# Import to ensure tasks and suites are registered
import olmo_eval.evals  # noqa: F401
import olmo_eval.evals.tasks  # noqa: F401
from olmo_eval.common.execution import ProcessScoringPoolConfig, default_process_pool_workers
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import ProcessScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.evals.tasks.common import Task, TaskConfig, get_task
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.harness.sandbox import SandboxConfig, SandboxMode
from olmo_eval.runners import AsyncEvalRunner, ValidationError
from olmo_eval.runners.asynq.runner import (
    _DEFAULT_SANDBOX_ENV,
    _determine_scoring_limits,
    _plan_process_scoring_pools,
    _plan_sandbox_configs,
    _shutdown_process_scoring_pools,
)
from olmo_eval.runners.asynq.types import TaskTracker
from olmo_eval.runners.processing.utils import compute_task_hash


def make_harness_config(model_name: str = "llama3.1-8b") -> HarnessConfig:
    """Create a HarnessConfig with the given model name."""
    return HarnessConfig(
        name="test",
        provider=ProviderConfig(model=model_name),
    )


@dataclass(frozen=True, slots=True)
class DefaultProcessScorer(ProcessScorer):
    name: str = "default_process"

    def process_score(self, instance: Instance, output: LMOutput) -> float:
        return 1.0


@dataclass(frozen=True, slots=True)
class NamedProcessScorer(ProcessScorer):
    name: str = "named_process"
    process_pool_name: ClassVar[str] = "math"

    def process_score(self, instance: Instance, output: LMOutput) -> float:
        return 1.0


class ProcessTask(Task):
    """Minimal task used for process-pool planning tests."""

    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="q", gold_answer="a")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    def extract_answer(self, output: LMOutput) -> str | None:
        return output.text


def _make_process_tracker(
    spec: str,
    scorer_cls: type[ProcessScorer],
    *,
    total_instances: int,
) -> TaskTracker:
    metric = AccuracyMetric(scorer=scorer_cls)
    task = ProcessTask(
        TaskConfig(
            name=spec.replace(":", "_"),
            data_source="test/dataset",
            metrics=(metric,),
        )
    )
    return TaskTracker(
        model_name="test-model",
        spec=spec,
        task=task,
        total_instances=total_instances,
    )


class TestAsyncEvalRunnerValidation:
    """Tests for AsyncEvalRunner.validate method."""

    def test_validate_valid_task(self):
        """Test validation passes for valid task with metrics."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=["humaneval:bpb"],
        )
        # Should not raise
        runner.validate()

    def test_validate_valid_suite(self):
        """Test validation passes for valid suite with metrics."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=["mt_mbpp_v2fix:bpb"],
        )
        # Should not raise
        runner.validate()

    def test_validate_multiple_valid_specs(self):
        """Test validation passes for multiple valid specs with metrics."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=["humaneval:bpb", "mbpp:bpb"],
        )
        # Should not raise
        runner.validate()

    def test_validate_task_without_metrics_fails(self):
        """Test validation fails for task without metrics configured."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=["humaneval"],
        )
        with pytest.raises(ValidationError, match="no metrics configured"):
            runner.validate()

    def test_validate_invalid_task_raises(self):
        """Test validation fails for unknown task."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=["nonexistent_task"],
        )
        with pytest.raises(ValidationError, match="Unknown task or suite"):
            runner.validate()

    def test_validate_invalid_suite_raises(self):
        """Test validation fails for unknown suite."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=["nonexistent:suite"],
        )
        with pytest.raises(ValidationError, match="Unknown task or suite"):
            runner.validate()

    def test_validate_invalid_variant_raises(self):
        """Test validation fails for an unknown variant."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=["humaneval:nonexistent_variant"],
        )
        with pytest.raises(ValidationError, match="Unknown variant"):
            runner.validate()

    def test_validate_collects_multiple_errors(self):
        """Test validation collects all errors."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=["bad_task1", "bad_task2", "humaneval:bpb"],
        )
        with pytest.raises(ValidationError) as exc_info:
            runner.validate()

        error_msg = str(exc_info.value)
        assert "bad_task1" in error_msg
        assert "bad_task2" in error_msg

    def test_validate_mixed_valid_and_invalid(self):
        """Test validation fails if any spec is invalid."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=["humaneval:bpb", "nonexistent", "mbpp:bpb"],
        )
        with pytest.raises(ValidationError, match="nonexistent"):
            runner.validate()

    def test_validate_empty_task_specs(self):
        """Test validation fails with empty task specs."""
        runner = AsyncEvalRunner(
            harness_config=make_harness_config(),
            task_specs=[],
        )
        with pytest.raises(ValidationError):
            runner.validate()


class TestNamedSandboxPlanning:
    """Tests for named sandbox allocation in AsyncEvalRunner."""

    @staticmethod
    def make_codex_universal_like_sandboxes(
        *,
        default_instances: int | None = None,
        default_min_instances: int | None = None,
        bigcodebench_instances: int | None = None,
        bigcodebench_min_instances: int | None = None,
    ) -> tuple[SandboxConfig, SandboxConfig]:
        return (
            SandboxConfig(
                instances=default_instances,
                min_instances=default_min_instances,
                image="default-sandbox:latest",
                mode=SandboxMode.DOCKER,
                inject_swerex=True,
            ),
            SandboxConfig(
                instances=bigcodebench_instances,
                min_instances=bigcodebench_min_instances,
                image="bigcodebench-sandbox:latest",
                mode=SandboxMode.DOCKER,
                capabilities=frozenset({"sandbox:bigcodebench"}),
                inject_swerex=True,
            ),
        )

    def test_named_sandbox_uses_matching_preset_capacity(self):
        """Named preset sandboxes should contribute their own executor budget."""
        trackers = {
            "bigcodebench:olmo3base": TaskTracker(
                model_name="test-model",
                spec="bigcodebench:olmo3base",
                task=get_task("bigcodebench:olmo3base"),
                total_instances=2,
            )
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(bigcodebench_instances=64),
            ["bigcodebench:olmo3base"],
            trackers,
            sandbox_pool_instances=None,
        )

        assert plan is not None
        assert plan.budget == 64
        assert plan.allocated == {"bigcodebench": 64}
        assert len(plan.sandboxes) == 1
        assert plan.sandboxes[0].capabilities == frozenset({"sandbox:bigcodebench"})
        assert plan.sandboxes[0].instances == 64

    def test_dynamic_named_sandbox_uses_global_pool(self):
        """Dynamically declared sandbox envs should draw from the shared sandbox pool."""
        trackers = {
            "ds1000:olmo3base": TaskTracker(
                model_name="test-model",
                spec="ds1000:olmo3base",
                task=get_task("ds1000:olmo3base"),
                total_instances=1,
            )
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(),
            ["ds1000:olmo3base"],
            trackers,
            sandbox_pool_instances=8,
        )

        assert plan is not None
        assert plan.budget == 8
        assert plan.allocated == {"ds1000": 8}
        assert len(plan.sandboxes) == 1
        assert plan.sandboxes[0].capabilities == frozenset({"sandbox:ds1000"})
        assert plan.sandboxes[0].instances == 8

    def test_default_and_named_envs_share_global_pool_when_auto_allocated(self):
        """Auto-managed sandboxes should share the global pool proportionally."""
        trackers = {
            "humaneval:pass_at_1": TaskTracker(
                model_name="test-model",
                spec="humaneval:pass_at_1",
                task=get_task("humaneval:pass_at_1"),
                total_instances=3,
            ),
            "bigcodebench:olmo3base": TaskTracker(
                model_name="test-model",
                spec="bigcodebench:olmo3base",
                task=get_task("bigcodebench:olmo3base"),
                total_instances=2,
            ),
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(),
            ["humaneval:pass_at_1", "bigcodebench:olmo3base"],
            trackers,
            sandbox_pool_instances=64,
        )

        assert plan is not None
        assert plan.budget == 64
        assert plan.allocated[_DEFAULT_SANDBOX_ENV] + plan.allocated["bigcodebench"] == 64
        assert plan.allocated[_DEFAULT_SANDBOX_ENV] == 4
        assert plan.allocated["bigcodebench"] == 60

    def test_pool_minimum_is_distributed_with_auto_allocated_envs(self):
        """Shared startup minimums should follow the weighted execution split."""
        trackers = {
            "humaneval:pass_at_1": TaskTracker(
                model_name="test-model",
                spec="humaneval:pass_at_1",
                task=get_task("humaneval:pass_at_1"),
                total_instances=18,
            ),
            "bigcodebench:olmo3base": TaskTracker(
                model_name="test-model",
                spec="bigcodebench:olmo3base",
                task=get_task("bigcodebench:olmo3base"),
                total_instances=1,
            ),
            "ds1000:olmo3base": TaskTracker(
                model_name="test-model",
                spec="ds1000:olmo3base",
                task=get_task("ds1000:olmo3base"),
                total_instances=1,
            ),
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(),
            ["humaneval:pass_at_1", "bigcodebench:olmo3base", "ds1000:olmo3base"],
            trackers,
            sandbox_pool_instances=64,
            sandbox_pool_min_instances=24,
        )

        assert plan is not None
        assert plan.min_required == {_DEFAULT_SANDBOX_ENV: 6, "bigcodebench": 9, "ds1000": 9}
        sandboxes_by_cap = {cfg.capabilities: cfg for cfg in plan.sandboxes}
        assert sandboxes_by_cap[frozenset({"bash"})].instances == 15
        assert sandboxes_by_cap[frozenset({"bash"})].min_instances == 6
        assert sandboxes_by_cap[frozenset({"sandbox:bigcodebench"})].instances == 24
        assert sandboxes_by_cap[frozenset({"sandbox:bigcodebench"})].min_instances == 9
        assert sandboxes_by_cap[frozenset({"sandbox:ds1000"})].instances == 25
        assert sandboxes_by_cap[frozenset({"sandbox:ds1000"})].min_instances == 9

    def test_default_and_named_envs_add_explicit_instances_on_top_of_pool(self):
        """Explicit sandbox counts should be preserved alongside the shared pool."""
        trackers = {
            "humaneval:pass_at_1": TaskTracker(
                model_name="test-model",
                spec="humaneval:pass_at_1",
                task=get_task("humaneval:pass_at_1"),
                total_instances=3,
            ),
            "bigcodebench:olmo3base": TaskTracker(
                model_name="test-model",
                spec="bigcodebench:olmo3base",
                task=get_task("bigcodebench:olmo3base"),
                total_instances=2,
            ),
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(
                default_instances=5,
                bigcodebench_instances=None,
            ),
            ["humaneval:pass_at_1", "bigcodebench:olmo3base"],
            trackers,
            sandbox_pool_instances=64,
        )

        assert plan is not None
        assert plan.budget == 69
        assert plan.allocated[_DEFAULT_SANDBOX_ENV] == 5
        assert plan.allocated["bigcodebench"] == 64
        assert {cfg.capabilities for cfg in plan.sandboxes} == {
            frozenset({"bash"}),
            frozenset({"sandbox:bigcodebench"}),
        }

    def test_auto_allocated_minimums_are_clamped_to_distributed_capacity(self):
        """Auto-managed configs should not keep impossible per-config startup minimums."""
        trackers = {
            "humaneval:pass_at_1": TaskTracker(
                model_name="test-model",
                spec="humaneval:pass_at_1",
                task=get_task("humaneval:pass_at_1"),
                total_instances=3,
            ),
            "bigcodebench:olmo3base": TaskTracker(
                model_name="test-model",
                spec="bigcodebench:olmo3base",
                task=get_task("bigcodebench:olmo3base"),
                total_instances=2,
            ),
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(
                default_min_instances=24,
                bigcodebench_min_instances=24,
            ),
            ["humaneval:pass_at_1", "bigcodebench:olmo3base"],
            trackers,
            sandbox_pool_instances=64,
        )

        assert plan is not None
        sandboxes_by_cap = {cfg.capabilities: cfg for cfg in plan.sandboxes}
        assert sandboxes_by_cap[frozenset({"bash"})].instances == 4
        assert sandboxes_by_cap[frozenset({"bash"})].min_instances == 4
        assert sandboxes_by_cap[frozenset({"sandbox:bigcodebench"})].instances == 60
        assert sandboxes_by_cap[frozenset({"sandbox:bigcodebench"})].min_instances == 24

    def test_warns_when_default_explicit_instances_bypass_shared_pool(self, caplog):
        """Explicit default capacity with no pool should warn when named envs stay at one each."""
        trackers = {
            "humaneval:pass_at_1": TaskTracker(
                model_name="test-model",
                spec="humaneval:pass_at_1",
                task=get_task("humaneval:pass_at_1"),
                total_instances=15,
            ),
            "bigcodebench:olmo3base": TaskTracker(
                model_name="test-model",
                spec="bigcodebench:olmo3base",
                task=get_task("bigcodebench:olmo3base"),
                total_instances=228,
            ),
            "ds1000:olmo3base": TaskTracker(
                model_name="test-model",
                spec="ds1000:olmo3base",
                task=get_task("ds1000:olmo3base"),
                total_instances=1000,
            ),
        }

        with caplog.at_level("WARNING", logger="olmo_eval.runners.asynq.runner"):
            plan = _plan_sandbox_configs(
                self.make_codex_universal_like_sandboxes(default_instances=32),
                ["humaneval:pass_at_1", "bigcodebench:olmo3base", "ds1000:olmo3base"],
                trackers,
                sandbox_pool_instances=None,
            )

        assert plan is not None
        assert plan.allocated == {
            _DEFAULT_SANDBOX_ENV: 32,
            "bigcodebench": 1,
            "ds1000": 1,
        }
        assert "auto-managed with no shared sandbox pool" in caplog.text
        assert "bigcodebench, ds1000" in caplog.text

    def test_default_sandbox_defaults_to_one_executor_when_unset(self):
        """Default-only execution should materialize one executor when no pool is set."""
        trackers = {
            "humaneval:pass_at_1": TaskTracker(
                model_name="test-model",
                spec="humaneval:pass_at_1",
                task=get_task("humaneval:pass_at_1"),
                total_instances=1,
            ),
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(),
            ["humaneval:pass_at_1"],
            trackers,
            sandbox_pool_instances=None,
        )

        assert plan is not None
        assert plan.budget == 1
        assert plan.allocated == {_DEFAULT_SANDBOX_ENV: 1}
        assert len(plan.sandboxes) == 1
        assert plan.sandboxes[0].capabilities == frozenset({"bash"})
        assert plan.sandboxes[0].instances == 1

    def test_non_execution_tasks_do_not_plan_default_sandbox(self):
        """Non-execution tasks should not consume default sandbox capacity."""
        trackers = {
            "humaneval:bpb": TaskTracker(
                model_name="test-model",
                spec="humaneval:bpb",
                task=get_task("humaneval:bpb"),
                total_instances=3,
            ),
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(),
            ["humaneval:bpb"],
            trackers,
            sandbox_pool_instances=64,
        )

        assert plan is None

    def test_named_non_execution_variants_do_not_plan_sandboxes(self):
        """Named sandbox envs should be ignored when tasks do not execute code."""
        trackers = {
            "bigcodebench:bpb": TaskTracker(
                model_name="test-model",
                spec="bigcodebench:bpb",
                task=get_task("bigcodebench:bpb"),
                total_instances=10,
            ),
            "ds1000:bpb": TaskTracker(
                model_name="test-model",
                spec="ds1000:bpb",
                task=get_task("ds1000:bpb"),
                total_instances=10,
            ),
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(),
            ["bigcodebench:bpb", "ds1000:bpb"],
            trackers,
            sandbox_pool_instances=64,
        )

        assert plan is None

    def test_weighted_allocation_favors_named_execution_sandboxes(self):
        """Named execution sandboxes should get weighted capacity beyond raw item counts."""
        trackers = {
            "humaneval:pass_at_1": TaskTracker(
                model_name="test-model",
                spec="humaneval:pass_at_1",
                task=get_task("humaneval:pass_at_1"),
                total_instances=18,
            ),
            "bigcodebench:olmo3base": TaskTracker(
                model_name="test-model",
                spec="bigcodebench:olmo3base",
                task=get_task("bigcodebench:olmo3base"),
                total_instances=1,
            ),
            "ds1000:olmo3base": TaskTracker(
                model_name="test-model",
                spec="ds1000:olmo3base",
                task=get_task("ds1000:olmo3base"),
                total_instances=1,
            ),
        }

        plan = _plan_sandbox_configs(
            self.make_codex_universal_like_sandboxes(),
            ["humaneval:pass_at_1", "bigcodebench:olmo3base", "ds1000:olmo3base"],
            trackers,
            sandbox_pool_instances=64,
        )

        assert plan is not None
        assert plan.env_sandbox_items == {_DEFAULT_SANDBOX_ENV: 18, "bigcodebench": 5, "ds1000": 5}
        assert plan.env_work_units == {
            _DEFAULT_SANDBOX_ENV: 18.0,
            "bigcodebench": 30.0,
            "ds1000": 33.0,
        }
        assert plan.allocated == {_DEFAULT_SANDBOX_ENV: 15, "bigcodebench": 24, "ds1000": 25}


class TestProcessPoolPlanning:
    """Tests for process-backed scoring pool planning."""

    def test_auto_creates_default_cpu_pool(self):
        trackers = {
            "process:test": _make_process_tracker(
                "process:test",
                DefaultProcessScorer,
                total_instances=7,
            )
        }

        plan = _plan_process_scoring_pools(
            {},
            ["process:test"],
            trackers,
            scoring_concurrency=6,
        )

        assert plan is not None
        assert plan.auto_created == frozenset({"cpu"})
        assert plan.pools == {"cpu": ProcessScoringPoolConfig(workers=6)}
        assert plan.demand["cpu"].task_count == 1
        assert plan.demand["cpu"].scorer_count == 1
        assert plan.demand["cpu"].response_jobs == 7

    def test_auto_created_cpu_pool_uses_default_worker_count_when_unset(self):
        trackers = {
            "process:test": _make_process_tracker(
                "process:test",
                DefaultProcessScorer,
                total_instances=2,
            )
        }

        plan = _plan_process_scoring_pools(
            {},
            ["process:test"],
            trackers,
            scoring_concurrency=None,
        )

        assert plan is not None
        assert plan.pools["cpu"] == ProcessScoringPoolConfig(workers=default_process_pool_workers())

    def test_uses_explicit_named_process_pool(self):
        trackers = {
            "process:named": _make_process_tracker(
                "process:named",
                NamedProcessScorer,
                total_instances=4,
            )
        }
        configured = {
            "math": ProcessScoringPoolConfig(
                workers=3,
                start_method="spawn",
                max_tasks_per_child=128,
            )
        }

        plan = _plan_process_scoring_pools(
            configured,
            ["process:named"],
            trackers,
            scoring_concurrency=8,
        )

        assert plan is not None
        assert plan.auto_created == frozenset()
        assert plan.pools == configured
        assert plan.demand["math"].response_jobs == 4

    def test_rejects_missing_non_default_process_pool(self):
        trackers = {
            "process:named": _make_process_tracker(
                "process:named",
                NamedProcessScorer,
                total_instances=3,
            )
        }

        with pytest.raises(ValueError, match="math"):
            _plan_process_scoring_pools(
                {},
                ["process:named"],
                trackers,
                scoring_concurrency=8,
            )

    def test_rejects_missing_default_pool_when_explicit_pools_exist(self):
        trackers = {
            "process:test": _make_process_tracker(
                "process:test",
                DefaultProcessScorer,
                total_instances=3,
            )
        }

        with pytest.raises(ValueError, match="cpu"):
            _plan_process_scoring_pools(
                {"math": ProcessScoringPoolConfig(workers=2)},
                ["process:test"],
                trackers,
                scoring_concurrency=8,
            )

    def test_rejects_non_reconstructible_process_scorer_during_planning(self):
        @dataclass(frozen=True, slots=True)
        class LocalProcessScorer(ProcessScorer):
            name: str = "local_process"

            def process_score(self, instance: Instance, output: LMOutput) -> float:
                return 1.0

        task = ProcessTask(
            TaskConfig(
                name="process_bad",
                data_source="test/dataset",
                metrics=(AccuracyMetric(scorer=LocalProcessScorer),),
            )
        )
        trackers = {
            "process:bad": TaskTracker(
                model_name="test-model",
                spec="process:bad",
                task=task,
                total_instances=1,
            )
        }

        with pytest.raises(ValueError, match="module scope"):
            _plan_process_scoring_pools(
                {"cpu": ProcessScoringPoolConfig(workers=2)},
                ["process:bad"],
                trackers,
                scoring_concurrency=8,
            )

    def test_default_response_limit_uses_largest_available_resource(self):
        trackers = {
            "process:test": _make_process_tracker(
                "process:test",
                DefaultProcessScorer,
                total_instances=3,
            )
        }
        plan = _plan_process_scoring_pools(
            {"cpu": ProcessScoringPoolConfig(workers=5)},
            ["process:test"],
            trackers,
            scoring_concurrency=None,
        )

        response_limit, context_limit = _determine_scoring_limits(
            HarnessConfig(name="test", provider=ProviderConfig(model="model")),
            [{"instances": 2}],
            plan,
        )

        assert (response_limit, context_limit) == (5, 5)

    def test_explicit_context_limit_does_not_cap_other_resources(self):
        trackers = {
            "process:test": _make_process_tracker(
                "process:test",
                DefaultProcessScorer,
                total_instances=3,
            )
        }
        plan = _plan_process_scoring_pools(
            {"cpu": ProcessScoringPoolConfig(workers=6)},
            ["process:test"],
            trackers,
            scoring_concurrency=4,
        )

        response_limit, context_limit = _determine_scoring_limits(
            HarnessConfig(
                name="test",
                provider=ProviderConfig(model="model"),
                scoring_concurrency=4,
            ),
            [{"instances": 9}, {"instances": 2}],
            plan,
        )

        assert response_limit == 11
        assert context_limit == 4

    def test_shutdown_logs_confirmation(self, monkeypatch):
        class Manager:
            def __init__(self) -> None:
                self.shutdown_called = False

            def shutdown(self) -> None:
                self.shutdown_called = True

        import olmo_eval.runners.asynq.runner as runner_module

        manager = Manager()
        fake_logger = MagicMock()
        monkeypatch.setattr(runner_module, "runner_logger", fake_logger)

        _shutdown_process_scoring_pools(manager)

        assert manager.shutdown_called
        fake_logger.info.assert_any_call("Stopping process scoring pools...")
        messages = [call.args[0] for call in fake_logger.info.call_args_list]
        assert any(message.startswith("Stopped process scoring pools in ") for message in messages)
        fake_logger.warning.assert_not_called()

    def test_shutdown_logs_warning_on_failure(self, monkeypatch):
        class Manager:
            def shutdown(self) -> None:
                raise RuntimeError("boom")

        import olmo_eval.runners.asynq.runner as runner_module

        fake_logger = MagicMock()
        monkeypatch.setattr(runner_module, "runner_logger", fake_logger)

        _shutdown_process_scoring_pools(Manager())

        fake_logger.info.assert_called_once_with("Stopping process scoring pools...")
        fake_logger.warning.assert_called_once_with(
            "Failed to stop process scoring pools",
            exc_info=True,
        )


class TestTaskConfigSandboxAllocationWeight:
    """Tests for scheduler-only sandbox allocation weights."""

    def test_rejects_non_positive_or_non_finite_weights(self):
        """Sandbox allocation weight must be positive and finite."""
        for bad_weight in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="sandbox_allocation_weight"):
                TaskConfig(name="bad", sandbox_allocation_weight=bad_weight)

    def test_sandbox_allocation_weight_does_not_affect_task_hash(self):
        """Scheduler-only weights should stay out of task serialization and hashing."""
        config = get_task("bigcodebench:olmo3base").config
        unweighted = replace(config, sandbox_allocation_weight=1.0)

        assert config.to_dict() == unweighted.to_dict()
        assert compute_task_hash(config.to_dict()) == compute_task_hash(unweighted.to_dict())


class TestSuiteAggregations:
    """Tests for compute_suite_aggregations function."""

    def test_suite_aggregation_basic(self):
        """Test basic suite aggregation without overrides."""
        # Create mock task results that match expanded mt_mbpp_v2fix tasks
        # Get actual tasks in mt_mbpp_v2fix suite
        from olmo_eval.evals.suites import get_suite
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

        suite = get_suite("mt_mbpp_v2fix")
        expanded_tasks = suite.expand()

        # Create mock results for each task (nested format)
        task_results = {}
        for task in expanded_tasks:
            task_results[task] = {"metrics": {"accuracy": {"exact_match": 0.75}}}

        result = compute_suite_aggregations(["mt_mbpp_v2fix"], task_results)

        assert "mt_mbpp_v2fix" in result
        assert result["mt_mbpp_v2fix"]["metrics"]["accuracy"]["exact_match"] == 0.75
        assert result["mt_mbpp_v2fix"]["num_tasks"] == len(expanded_tasks)

    def test_suite_aggregation_with_priority(self):
        """Test suite aggregation with priority suffix."""
        from olmo_eval.evals.suites import get_suite
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

        suite = get_suite("mt_mbpp_v2fix")
        expanded_tasks = suite.expand()

        # Create mock results with priority suffix
        task_results = {}
        for task in expanded_tasks:
            task_results[f"{task}@high"] = {"metrics": {"accuracy": {"exact_match": 0.85}}}

        result = compute_suite_aggregations(["mt_mbpp_v2fix@high"], task_results)

        assert "mt_mbpp_v2fix@high" in result
        assert result["mt_mbpp_v2fix@high"]["metrics"]["accuracy"]["exact_match"] == pytest.approx(
            0.85
        )

    def test_suite_aggregation_non_suite_ignored(self):
        """Test that non-suite specs are ignored."""
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

        task_results = {"humaneval": {"metrics": {"accuracy": {"exact_match": 0.75}}}}

        result = compute_suite_aggregations(["humaneval"], task_results)

        # humaneval is a task, not a suite
        assert result == {}

    def test_suite_aggregation_average_of_averages(self):
        """Test AVERAGE_OF_AVERAGES aggregation weights children equally."""
        from olmo_eval.evals.suites.registry import (
            _REGISTRY,
            AggregationStrategy,
            Suite,
        )
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

        # Create a nested suite for testing
        nested_suite = Suite(
            name="_test_nested",
            tasks=("task_a", "task_b", "task_c"),  # 3 tasks
            aggregation=AggregationStrategy.AVERAGE,
        )

        # Create an average-of-averages suite
        aoa_suite = Suite(
            name="_test_aoa",
            tasks=(
                "task_single",  # Individual task
                nested_suite,  # Nested suite with 3 tasks
            ),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        )

        # Register temporarily
        _REGISTRY["_test_aoa"] = aoa_suite

        try:
            # Create task results (nested format):
            # - task_single: 1.0
            # - task_a, task_b, task_c: 0.4, 0.5, 0.6 (average = 0.5)
            # Expected AVERAGE_OF_AVERAGES: (1.0 + 0.5) / 2 = 0.75
            # (NOT simple average: (1.0 + 0.4 + 0.5 + 0.6) / 4 = 0.625)
            task_results = {
                "task_single": {"metrics": {"bits_per_byte": {"bpb_scorer": 1.0}}},
                "task_a": {"metrics": {"bits_per_byte": {"bpb_scorer": 0.4}}},
                "task_b": {"metrics": {"bits_per_byte": {"bpb_scorer": 0.5}}},
                "task_c": {"metrics": {"bits_per_byte": {"bpb_scorer": 0.6}}},
            }

            result = compute_suite_aggregations(["_test_aoa"], task_results)

            # Check top-level suite aggregation
            assert "_test_aoa" in result
            # Average of averages: (1.0 + 0.5) / 2 = 0.75
            assert result["_test_aoa"]["metrics"]["bits_per_byte"]["bpb_scorer"] == pytest.approx(
                0.75
            )
            assert result["_test_aoa"]["num_tasks"] == 4  # All tasks included
            assert result["_test_aoa"]["num_children"] == 2  # 2 children
            assert result["_test_aoa"]["aggregation"] == "average_of_averages"
            assert result["_test_aoa"]["nested_suites"] == ["_test_nested"]

            # Check nested suite aggregation is also reported
            assert "_test_nested" in result
            # Nested suite average: (0.4 + 0.5 + 0.6) / 3 = 0.5
            assert result["_test_nested"]["metrics"]["bits_per_byte"][
                "bpb_scorer"
            ] == pytest.approx(0.5)
            assert result["_test_nested"]["num_tasks"] == 3
            assert result["_test_nested"]["aggregation"] == "average"
            assert result["_test_nested"]["parent_suite"] == "_test_aoa"
        finally:
            # Clean up
            del _REGISTRY["_test_aoa"]

    def test_suite_aggregation_weighted_average(self):
        """Test WEIGHTED_AVERAGE weights each task by its instance count."""
        from olmo_eval.evals.suites.registry import (
            _REGISTRY,
            AggregationStrategy,
            Suite,
        )
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

        weighted_suite = Suite(
            name="_test_weighted",
            tasks=("task_small", "task_large"),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        )
        _REGISTRY["_test_weighted"] = weighted_suite

        try:
            task_results = {
                "task_small": {
                    "metrics": {"accuracy": {"exact_match": 0.9}},
                    "primary_metric": "accuracy:exact_match",
                    "num_instances": 100,
                },
                "task_large": {
                    "metrics": {"accuracy": {"exact_match": 0.5}},
                    "primary_metric": "accuracy:exact_match",
                    "num_instances": 300,
                },
            }

            result = compute_suite_aggregations(["_test_weighted"], task_results)

            # (0.9 * 100 + 0.5 * 300) / 400 = 0.6, versus 0.7 unweighted.
            assert result["_test_weighted"]["metrics"]["accuracy"]["exact_match"] == pytest.approx(
                0.6
            )
            assert result["_test_weighted"]["metrics"]["primary_score"]["average"] == pytest.approx(
                0.6
            )
            assert result["_test_weighted"]["aggregation"] == "weighted_average"
        finally:
            del _REGISTRY["_test_weighted"]

    def test_suite_aggregation_weighted_average_without_counts(self):
        """Test WEIGHTED_AVERAGE omits the aggregate when a count is missing."""
        from olmo_eval.evals.suites.registry import (
            _REGISTRY,
            AggregationStrategy,
            Suite,
        )
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

        weighted_suite = Suite(
            name="_test_weighted_no_counts",
            tasks=("task_small", "task_large"),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        )
        _REGISTRY["_test_weighted_no_counts"] = weighted_suite

        try:
            task_results = {
                "task_small": {
                    "metrics": {"accuracy": {"exact_match": 0.9}},
                    "num_instances": 100,
                },
                "task_large": {"metrics": {"accuracy": {"exact_match": 0.5}}},
            }

            result = compute_suite_aggregations(["_test_weighted_no_counts"], task_results)

            # An unweighted mean under a weighted suite's name would be a
            # different statistic wearing the same label.
            assert "_test_weighted_no_counts" not in result
        finally:
            del _REGISTRY["_test_weighted_no_counts"]

    def test_weighted_child_of_average_of_averages_is_weighted(self):
        """Test a WEIGHTED_AVERAGE child is collapsed with its own weighting."""
        from olmo_eval.evals.suites.registry import (
            _REGISTRY,
            AggregationStrategy,
            Suite,
        )
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

        child = Suite(
            name="_test_weighted_child_runner",
            tasks=("task_small", "task_large"),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        )
        parent = Suite(
            name="_test_weighted_child_runner_parent",
            tasks=("task_standalone", child),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        )
        _REGISTRY["_test_weighted_child_runner_parent"] = parent

        try:
            task_results = {
                "task_standalone": {
                    "metrics": {"accuracy": {"exact_match": 1.0}},
                    "num_instances": 50,
                },
                "task_small": {
                    "metrics": {"accuracy": {"exact_match": 0.9}},
                    "num_instances": 100,
                },
                "task_large": {
                    "metrics": {"accuracy": {"exact_match": 0.5}},
                    "num_instances": 300,
                },
            }

            result = compute_suite_aggregations(
                ["_test_weighted_child_runner_parent"], task_results
            )

            # The child collapses to its weighted mean of 0.6, then the parent
            # weights its two children equally: (1.0 + 0.6) / 2.
            child_metrics = result["_test_weighted_child_runner"]["metrics"]
            assert child_metrics["accuracy"]["exact_match"] == pytest.approx(0.6)
            parent_metrics = result["_test_weighted_child_runner_parent"]["metrics"]
            assert parent_metrics["accuracy"]["exact_match"] == pytest.approx(0.8)
        finally:
            del _REGISTRY["_test_weighted_child_runner_parent"]

    def test_weighted_child_without_counts_is_dropped_from_parent(self):
        """Test a WEIGHTED_AVERAGE child missing a count is left out of its parent."""
        from olmo_eval.evals.suites.registry import (
            _REGISTRY,
            AggregationStrategy,
            Suite,
        )
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

        child = Suite(
            name="_test_weighted_child_no_counts",
            tasks=("task_small", "task_large"),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        )
        parent = Suite(
            name="_test_weighted_child_no_counts_parent",
            tasks=("task_standalone", child),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        )
        _REGISTRY["_test_weighted_child_no_counts_parent"] = parent

        try:
            task_results = {
                "task_standalone": {
                    "metrics": {"accuracy": {"exact_match": 1.0}},
                    "num_instances": 50,
                },
                "task_small": {
                    "metrics": {"accuracy": {"exact_match": 0.9}},
                    "num_instances": 100,
                },
                "task_large": {"metrics": {"accuracy": {"exact_match": 0.5}}},
            }

            result = compute_suite_aggregations(
                ["_test_weighted_child_no_counts_parent"], task_results
            )

            # The child has no weighted mean, so the parent averages only the
            # standalone task.
            assert "_test_weighted_child_no_counts" not in result
            parent_result = result["_test_weighted_child_no_counts_parent"]
            assert parent_result["metrics"]["accuracy"]["exact_match"] == pytest.approx(1.0)
            assert parent_result["num_children"] == 1
            assert parent_result["nested_suites"] == []
        finally:
            del _REGISTRY["_test_weighted_child_no_counts_parent"]

    def test_weighted_child_suite_tasks_are_collapsed_in_log_summary(self, capsys):
        """Test a WEIGHTED_AVERAGE child collapses its tasks in log_summary."""
        from olmo_eval.evals.suites.registry import (
            _REGISTRY,
            AggregationStrategy,
            Suite,
        )
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations
        from olmo_eval.runners.processing.metrics import log_summary

        child = Suite(
            name="_test_weighted_child_display",
            tasks=("task_small", "task_large"),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        )
        parent = Suite(
            name="_test_weighted_child_display_parent",
            tasks=("task_standalone", child),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        )
        _REGISTRY["_test_weighted_child_display_parent"] = parent

        try:
            task_results = {
                "task_standalone": {
                    "metrics": {"accuracy": {"exact_match": 1.0}},
                    "num_instances": 50,
                },
                "task_small": {
                    "metrics": {"accuracy": {"exact_match": 0.9}},
                    "num_instances": 100,
                },
                "task_large": {
                    "metrics": {"accuracy": {"exact_match": 0.5}},
                    "num_instances": 300,
                },
            }

            suite_aggs = compute_suite_aggregations(
                ["_test_weighted_child_display_parent"], task_results
            )
            log_summary({"tasks": task_results, "suites": suite_aggs})
            captured = capsys.readouterr().out

            assert "task_standalone" in captured
            assert "_test_weighted_child_display" in captured
            assert "task_small" not in captured
            assert "task_large" not in captured
        finally:
            del _REGISTRY["_test_weighted_child_display_parent"]

    def test_collapsed_tasks_in_log_summary(self, capsys):
        """Test that child suite tasks are collapsed in log_summary display.

        When a parent suite uses AVERAGE_OF_AVERAGES and a child suite uses
        AVERAGE, the child's individual tasks should not appear as separate
        rows — only the sub-suite average row should be shown.
        """
        from olmo_eval.evals.suites.registry import (
            _REGISTRY,
            AggregationStrategy,
            Suite,
        )
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations
        from olmo_eval.runners.processing.metrics import log_summary

        nested_suite = Suite(
            name="_test_nested_display",
            tasks=("task_a", "task_b", "task_c"),
            aggregation=AggregationStrategy.AVERAGE,
        )
        aoa_suite = Suite(
            name="_test_aoa_display",
            tasks=("task_standalone", nested_suite),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        )
        _REGISTRY["_test_aoa_display"] = aoa_suite

        try:
            task_results = {
                "task_standalone": {"metrics": {"pass_at_1": {"code_exec": 0.8}}},
                "task_a": {"metrics": {"pass_at_1": {"code_exec": 0.4}}},
                "task_b": {"metrics": {"pass_at_1": {"code_exec": 0.5}}},
                "task_c": {"metrics": {"pass_at_1": {"code_exec": 0.6}}},
            }

            suite_aggs = compute_suite_aggregations(["_test_aoa_display"], task_results)
            results = {"tasks": task_results, "suites": suite_aggs}

            log_summary(results)
            captured = capsys.readouterr().out

            # Standalone task should appear
            assert "task_standalone" in captured
            # Nested suite average should appear
            assert "_test_nested_display" in captured
            # Parent suite should appear
            assert "_test_aoa_display" in captured
            # Individual nested tasks should NOT appear (collapsed into sub-suite)
            assert "task_a" not in captured
            assert "task_b" not in captured
            assert "task_c" not in captured
        finally:
            del _REGISTRY["_test_aoa_display"]

    def test_non_averaged_child_suite_not_collapsed(self):
        """Test that DISPLAY_ONLY child suites do not collapse their tasks."""
        from olmo_eval.evals.suites.registry import (
            _REGISTRY,
            AggregationStrategy,
            Suite,
        )
        from olmo_eval.runners.processing.aggregation import compute_suite_aggregations

        nested_suite = Suite(
            name="_test_nested_display_only",
            tasks=("task_x", "task_y"),
            aggregation=AggregationStrategy.DISPLAY_ONLY,
        )
        aoa_suite = Suite(
            name="_test_aoa_no_collapse",
            tasks=("task_z", nested_suite),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        )
        _REGISTRY["_test_aoa_no_collapse"] = aoa_suite

        try:
            task_results = {
                "task_z": {"metrics": {"acc": {"em": 0.9}}},
                "task_x": {"metrics": {"acc": {"em": 0.3}}},
                "task_y": {"metrics": {"acc": {"em": 0.7}}},
            }

            suite_aggs = compute_suite_aggregations(["_test_aoa_no_collapse"], task_results)

            # DISPLAY_ONLY child should have parent_suite but aggregation != "average"
            nested_data = suite_aggs.get("_test_nested_display_only", {})
            assert nested_data.get("parent_suite") == "_test_aoa_no_collapse"
            assert nested_data.get("aggregation") == "display_only"

            # The collapse logic only applies to aggregation=="average" children,
            # so DISPLAY_ONLY tasks should NOT be collapsed
            collapsed: set[str] = set()
            for suite_data in suite_aggs.values():
                if suite_data.get("parent_suite") and suite_data.get("aggregation") == "average":
                    collapsed.update(suite_data.get("tasks", []))
            assert collapsed == set()
        finally:
            del _REGISTRY["_test_aoa_no_collapse"]


class TestGetPrimaryMetric:
    """Tests for get_primary_metric function.

    Note: Metrics are now nested: {metric_name: {scorer_name: value}}.
    The preferred parameter uses "metric:scorer" format, and the result
    is ("metric:scorer", value).
    """

    def test_preferred_metric_used_when_present(self):
        """Test that preferred metric is used when specified and present."""
        from olmo_eval.runners.processing.utils import get_primary_metric

        metrics = {
            "accuracy": {"exact_match": 0.75},
            "bits_per_byte": {"bpb_scorer": 0.5},
            "f1": {"f1_scorer": 0.8},
        }
        result = get_primary_metric(metrics, preferred="bits_per_byte:bpb_scorer")

        assert result == ("bits_per_byte:bpb_scorer", 0.5)

    def test_preferred_metric_ignored_when_not_present(self):
        """Test fallback when preferred metric is not in metrics dict."""
        from olmo_eval.runners.processing.utils import get_primary_metric

        metrics = {"accuracy": {"exact_match": 0.75}, "f1": {"f1_scorer": 0.8}}
        result = get_primary_metric(metrics, preferred="bits_per_byte:bpb_scorer")

        # Falls back to accuracy (first scorer alphabetically)
        assert result == ("accuracy:exact_match", 0.75)

    def test_accuracy_fallback_when_no_preferred(self):
        """Test that accuracy is used when no preferred metric specified."""
        from olmo_eval.runners.processing.utils import get_primary_metric

        metrics = {"accuracy": {"exact_match": 0.75}, "bits_per_byte": {"bpb_scorer": 0.5}}
        result = get_primary_metric(metrics)

        assert result == ("accuracy:exact_match", 0.75)

    def test_alphabetical_fallback_when_no_accuracy(self):
        """Test alphabetical fallback when no accuracy and no preferred."""
        from olmo_eval.runners.processing.utils import get_primary_metric

        metrics = {"f1": {"f1_scorer": 0.8}, "bits_per_byte": {"bpb_scorer": 0.5}}
        result = get_primary_metric(metrics)

        # bits_per_byte comes before f1 alphabetically, bpb_scorer is first (only) scorer
        assert result == ("bits_per_byte:bpb_scorer", 0.5)

    def test_empty_metrics_returns_none(self):
        """Test that empty metrics returns None."""
        from olmo_eval.runners.processing.utils import get_primary_metric

        result = get_primary_metric({})
        assert result is None

    def test_preferred_none_same_as_not_specified(self):
        """Test that preferred=None behaves same as not specifying."""
        from olmo_eval.runners.processing.utils import get_primary_metric

        metrics = {"accuracy": {"exact_match": 0.75}, "bits_per_byte": {"bpb_scorer": 0.5}}

        result_none = get_primary_metric(metrics, preferred=None)
        result_default = get_primary_metric(metrics)

        assert result_none == result_default
