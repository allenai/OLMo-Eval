"""Tests for olmo_eval.tasks.registry module."""

from collections.abc import Iterator

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import (
    Task,
    clear_registry,
    get_base_task_name,
    get_task,
    get_task_dependencies,
    list_tasks,
    list_variants,
    parse_overrides,
    register,
    register_variant,
)
from olmo_eval.evals.tasks.common.registry import _configs, _tasks, _variants


class DummyTask(Task):
    """A minimal task implementation for testing."""

    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="What is 2+2?", gold_answer="4")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    def extract_answer(self, output: LMOutput) -> str:
        return output.text.strip()


@pytest.fixture(autouse=True)
def clean_registry():
    """Provide an isolated registry for testing.

    Saves the current registry state, clears it for the test,
    then restores it afterward.
    """
    # Save original state
    original_tasks = _tasks.copy()
    original_configs = _configs.copy()
    original_variants = {k: v.copy() for k, v in _variants.items()}

    clear_registry()
    yield

    # Restore original state
    clear_registry()
    _tasks.update(original_tasks)
    _configs.update(original_configs)
    _variants.update(original_variants)


class TestRegister:
    """Tests for the @register decorator."""

    def test_register_task(self):
        """Test basic task registration."""

        @register("test_task")
        class TestTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        assert "test_task" in list_tasks()

    def test_register_duplicate_raises(self):
        """Test that registering duplicate task names raises an error."""

        @register("duplicate")
        class FirstTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        with pytest.raises(ValueError, match="already registered"):

            @register("duplicate")
            class SecondTask(DummyTask):
                data_source = DataSource(path="test/dataset")

    def test_register_preserves_class(self):
        """Test that @register returns the original class."""

        @register("preserved")
        class PreservedTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        assert PreservedTask.__name__ == "PreservedTask"
        assert issubclass(PreservedTask, Task)


class TestRegisterVariant:
    """Tests for register_variant function."""

    def test_register_variant(self):
        """Test registering a variant for an existing task."""

        @register("base_task")
        class BaseTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        register_variant("base_task", "custom", num_fewshot=5, limit=100)

        variants = list_variants("base_task")
        assert "custom" in variants["base_task"]

    def test_register_variant_unknown_task_raises(self):
        """Test that registering a variant for an unknown task raises error."""
        with pytest.raises(ValueError, match="unknown task"):
            register_variant("nonexistent", "variant", num_fewshot=5)

    def test_register_multiple_variants(self):
        """Test registering multiple variants for one task."""

        @register("multi_variant")
        class MultiVariantTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        register_variant("multi_variant", "fast", limit=10)
        register_variant("multi_variant", "full", limit=None)
        register_variant("multi_variant", "fewshot", num_fewshot=5)

        variants = list_variants("multi_variant")
        assert set(variants["multi_variant"]) == {"fast", "full", "fewshot"}


class TestGetTask:
    """Tests for get_task function."""

    def test_get_task_by_name(self):
        """Test getting a task by simple name."""

        @register("simple_task")
        class SimpleTask(DummyTask):
            data_source = DataSource(path="test/dataset")
            num_fewshot = 0

        task = get_task("simple_task")
        assert isinstance(task, SimpleTask)
        assert task.config.name == "simple_task"
        assert task.config.num_fewshot == 0

    def test_get_task_with_variant(self):
        """Test getting a task with variant overrides."""

        @register("variant_task")
        class VariantTask(DummyTask):
            data_source = DataSource(path="test/dataset")
            num_fewshot = 0

        register_variant("variant_task", "fewshot", num_fewshot=5)

        # Without variant
        task_base = get_task("variant_task")
        assert task_base.config.num_fewshot == 0

        # With variant
        task_variant = get_task("variant_task:fewshot")
        assert task_variant.config.num_fewshot == 5

    def test_get_task_unknown_raises(self):
        """Test that getting unknown task raises KeyError."""
        with pytest.raises(KeyError, match="Unknown task"):
            get_task("nonexistent_task")

    def test_get_task_with_unknown_variant_raises(self):
        """Test that an unknown variant raises KeyError."""

        @register("fallback_task")
        class FallbackTask(DummyTask):
            data_source = DataSource(path="test/dataset")
            num_fewshot = 3

        with pytest.raises(KeyError, match="Unknown variant 'unknown_variant'"):
            get_task("fallback_task:unknown_variant")


class TestListTasks:
    """Tests for list_tasks function."""

    def test_list_tasks_empty(self):
        """Test list_tasks with empty registry."""
        assert list_tasks() == []

    def test_list_tasks_returns_sorted(self):
        """Test that list_tasks returns sorted names."""

        @register("zebra")
        class ZebraTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        @register("alpha")
        class AlphaTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        @register("middle")
        class MiddleTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        tasks = list_tasks()
        assert tasks == ["alpha", "middle", "zebra"]


class TestListVariants:
    """Tests for list_variants function."""

    def test_list_variants_all(self):
        """Test listing all variants."""

        @register("task_a")
        class TaskA(DummyTask):
            data_source = DataSource(path="test/dataset")

        @register("task_b")
        class TaskB(DummyTask):
            data_source = DataSource(path="test/dataset")

        register_variant("task_a", "variant1")
        register_variant("task_a", "variant2")
        register_variant("task_b", "variant3")

        all_variants = list_variants()
        assert "task_a" in all_variants
        assert "task_b" in all_variants
        assert set(all_variants["task_a"]) == {"variant1", "variant2"}
        assert all_variants["task_b"] == ["variant3"]

    def test_list_variants_filtered(self):
        """Test listing variants for a specific task."""

        @register("filtered")
        class FilteredTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        register_variant("filtered", "fast")
        register_variant("filtered", "slow")

        variants = list_variants("filtered")
        assert "filtered" in variants
        assert len(variants) == 1
        assert set(variants["filtered"]) == {"fast", "slow"}

    def test_list_variants_no_variants(self):
        """Test listing variants for a task with none registered."""

        @register("no_variants")
        class NoVariantsTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        variants = list_variants("no_variants")
        assert variants == {"no_variants": []}


class TestClearRegistry:
    """Tests for clear_registry function."""

    def test_clear_registry(self):
        """Test that clear_registry removes all entries."""

        @register("to_clear")
        class ToClearTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        register_variant("to_clear", "variant")

        assert "to_clear" in list_tasks()
        assert list_variants("to_clear")["to_clear"] == ["variant"]

        clear_registry()

        assert list_tasks() == []
        assert list_variants() == {}


class TestGetBaseTaskName:
    """Tests for get_base_task_name utility function."""

    def test_simple_task_name(self):
        """Test with a simple task name (no modifiers)."""
        assert get_base_task_name("arc_easy") == "arc_easy"

    def test_task_with_variant(self):
        """Test that variants are preserved."""
        assert get_base_task_name("arc_easy:mc") == "arc_easy:mc"
        assert get_base_task_name("humaneval:3shot:bpb") == "humaneval:3shot:bpb"

    def test_task_with_priority(self):
        """Test that priority suffix is stripped."""
        assert get_base_task_name("arc_easy@high") == "arc_easy"
        assert get_base_task_name("arc_easy@low") == "arc_easy"
        assert get_base_task_name("arc_easy:mc@high") == "arc_easy:mc"


class TestGetTaskDependencies:
    """Tests for get_task_dependencies function."""

    def test_empty_dependencies_returns_empty_list(self):
        """Test that tasks without dependencies return empty list."""

        @register("no_deps_task")
        class NoDepTask(DummyTask):
            data_source = DataSource(path="test/dataset")

        result = get_task_dependencies(["no_deps_task"])
        assert result == []

    def test_single_task_with_dependencies(self):
        """Test extracting dependencies from a single task."""

        @register("deps_task")
        class DepsTask(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = ["pkg1==1.0", "pkg2>=2.0"]

        result = get_task_dependencies(["deps_task"])
        assert result == ["pkg1==1.0", "pkg2>=2.0"]

    def test_multiple_tasks_merge_dependencies(self):
        """Test that dependencies from multiple tasks are merged."""

        @register("task_a")
        class TaskA(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = ["pkg1==1.0"]

        @register("task_b")
        class TaskB(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = ["pkg2==2.0"]

        result = get_task_dependencies(["task_a", "task_b"])
        assert result == ["pkg1==1.0", "pkg2==2.0"]

    def test_dependencies_deduplicated(self):
        """Test that duplicate dependencies are removed."""

        @register("dup_task_a")
        class DupTaskA(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = ["pkg1==1.0", "pkg2==2.0"]

        @register("dup_task_b")
        class DupTaskB(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = ["pkg1==1.0", "pkg3==3.0"]  # pkg1 is duplicate

        result = get_task_dependencies(["dup_task_a", "dup_task_b"])
        # pkg1==1.0 appears in first task, should only appear once
        assert result == ["pkg1==1.0", "pkg2==2.0", "pkg3==3.0"]

    def test_preserves_order(self):
        """Test that order is preserved (first occurrence wins)."""

        @register("order_task_a")
        class OrderTaskA(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = ["c-pkg", "a-pkg"]

        @register("order_task_b")
        class OrderTaskB(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = ["b-pkg"]

        result = get_task_dependencies(["order_task_a", "order_task_b"])
        # Order should be preserved: c-pkg, a-pkg from task_a, then b-pkg from task_b
        assert result == ["c-pkg", "a-pkg", "b-pkg"]

    def test_mixed_tasks_with_and_without_deps(self):
        """Test mixing tasks with and without dependencies."""

        @register("with_deps")
        class WithDeps(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = ["special-lib"]

        @register("without_deps")
        class WithoutDeps(DummyTask):
            data_source = DataSource(path="test/dataset")

        result = get_task_dependencies(["without_deps", "with_deps"])
        assert result == ["special-lib"]

    def test_empty_task_list(self):
        """Test with empty task list."""
        result = get_task_dependencies([])
        assert result == []

    def test_task_with_variant_inherits_base_dependencies(self):
        """Test that variants can override dependencies."""

        @register("base_deps")
        class BaseDepsTask(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = ["base-pkg"]

        # Register variant that adds to dependencies
        register_variant("base_deps", "extra", dependencies=["extra-pkg"])

        # Base task should have base dependencies
        base_result = get_task_dependencies(["base_deps"])
        assert base_result == ["base-pkg"]

        # Variant should have overridden dependencies (replace, not merge)
        variant_result = get_task_dependencies(["base_deps:extra"])
        assert variant_result == ["extra-pkg"]

    def test_git_url_dependencies(self):
        """Test tasks with git URL dependencies."""

        @register("git_deps_task")
        class GitDepsTask(DummyTask):
            data_source = DataSource(path="test/dataset")
            dependencies = [
                "git+https://github.com/user/repo@v1.0",
                "https://github.com/user/another-repo",
            ]

        result = get_task_dependencies(["git_deps_task"])
        assert result == [
            "git+https://github.com/user/repo@v1.0",
            "https://github.com/user/another-repo",
        ]


class TestParseOverridesDependencies:
    """Tests for parse_overrides handling of dependencies field."""

    def test_parse_dependencies_json_list(self):
        """Test parsing dependencies as JSON list."""
        result = parse_overrides('dependencies=["pkg1==1.0", "pkg2>=2.0"]')
        assert result == {"dependencies": ["pkg1==1.0", "pkg2>=2.0"]}

    def test_parse_dependencies_single_value(self):
        """Test parsing single dependency value (not JSON) becomes list."""
        result = parse_overrides("dependencies=special-lib")
        assert result == {"dependencies": ["special-lib"]}

    def test_parse_dependencies_with_other_overrides(self):
        """Test parsing dependencies alongside other overrides."""
        result = parse_overrides('num_fewshot=5,dependencies=["pkg1"]')
        assert result == {"num_fewshot": 5, "dependencies": ["pkg1"]}

    def test_parse_dependencies_empty_list(self):
        """Test parsing empty dependencies list."""
        result = parse_overrides("dependencies=[]")
        assert result == {"dependencies": []}

    def test_parse_dependencies_git_urls(self):
        """Test parsing dependencies with git URLs."""
        result = parse_overrides('dependencies=["git+https://github.com/user/repo@v1.0"]')
        assert result == {"dependencies": ["git+https://github.com/user/repo@v1.0"]}


class TestParseOverridesSamplingParams:
    """Tests for parse_overrides coercion of SamplingParams fields."""

    def test_parse_truncate_prompt_tokens_is_int(self):
        """truncate_prompt_tokens reaches the provider as an int, not a string."""
        result = parse_overrides("truncate_prompt_tokens=32768")
        assert result == {"truncate_prompt_tokens": 32768}

    def test_parse_truncate_prompt_tokens_sentinel(self):
        """The -1 sentinel meaning 'the model's max input length' survives parsing."""
        assert parse_overrides("truncate_prompt_tokens=-1") == {"truncate_prompt_tokens": -1}
