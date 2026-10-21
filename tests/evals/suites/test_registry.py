"""Tests for olmo_eval.evals.suites.registry module."""

import re

import pytest

from olmo_eval.evals.suites.registry import (
    _REGISTRY,
    AggregationStrategy,
    Suite,
    format_tasks,
    get_suite,
    list_suites,
    make_suite,
    register,
    search_suites,
    suite_exists,
)


@pytest.fixture
def isolated_registry():
    """Provide an isolated registry for testing.

    Saves the current registry state, clears it for the test,
    then restores it afterward.
    """
    original = _REGISTRY.copy()
    _REGISTRY.clear()
    yield _REGISTRY
    _REGISTRY.clear()
    _REGISTRY.update(original)


class TestAggregationStrategy:
    """Tests for AggregationStrategy enum."""

    def test_enum_values(self):
        """Test that all expected values exist."""
        assert AggregationStrategy.NONE.value == "none"
        assert AggregationStrategy.AVERAGE.value == "average"
        assert AggregationStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert AggregationStrategy.AVERAGE_OF_AVERAGES.value == "average_of_averages"
        assert AggregationStrategy.DISPLAY_ONLY.value == "display_only"

    def test_enum_is_string(self):
        """Test that enum values are strings."""
        assert isinstance(AggregationStrategy.AVERAGE, str)
        assert AggregationStrategy.AVERAGE == "average"


class TestSuite:
    """Tests for Suite dataclass."""

    def test_suite_creation_minimal(self):
        """Test creating a suite with minimal arguments."""
        suite = Suite(name="test", tasks=("task1", "task2"))

        assert suite.name == "test"
        assert suite.tasks == ("task1", "task2")
        assert suite.aggregation == AggregationStrategy.AVERAGE  # default
        assert suite.description == ""  # default

    def test_suite_creation_full(self):
        """Test creating a suite with all arguments."""
        suite = Suite(
            name="full_suite",
            tasks=("task1", "task2", "task3"),
            aggregation=AggregationStrategy.DISPLAY_ONLY,
            description="A test suite",
        )

        assert suite.name == "full_suite"
        assert suite.tasks == ("task1", "task2", "task3")
        assert suite.aggregation == AggregationStrategy.DISPLAY_ONLY
        assert suite.description == "A test suite"

    def test_suite_is_frozen(self):
        """Test that suite is immutable."""
        suite = Suite(name="test", tasks=("task1",))

        with pytest.raises(AttributeError):
            suite.name = "changed"

    def test_expanded_tasks_simple(self):
        """Test expanding a suite with only string tasks."""
        suite = Suite(name="test", tasks=("task1", "task2", "task3"))

        assert suite.expanded_tasks == ("task1", "task2", "task3")

    def test_expanded_tasks_nested(self):
        """Test expanding a suite with nested suites."""
        inner = Suite(name="inner", tasks=("task1", "task2"))
        outer = Suite(name="outer", tasks=(inner, "task3"))

        assert outer.expanded_tasks == ("task1", "task2", "task3")

    def test_expanded_tasks_deeply_nested(self):
        """Test expanding deeply nested suites."""
        level1 = Suite(name="level1", tasks=("a", "b"))
        level2 = Suite(name="level2", tasks=(level1, "c"))
        level3 = Suite(name="level3", tasks=(level2, "d"))

        assert level3.expanded_tasks == ("a", "b", "c", "d")

    def test_expand_alias(self):
        """Test that expand() is an alias for expanded_tasks."""
        suite = Suite(name="test", tasks=("task1", "task2"))

        assert suite.expand() == suite.expanded_tasks

    def test_repr_simple(self):
        """Test string representation of simple suite."""
        suite = Suite(name="test", tasks=("t1", "t2", "t3"))

        assert repr(suite) == "Suite('test', 3 tasks)"

    def test_repr_nested(self):
        """Test string representation of nested suite."""
        inner = Suite(name="inner", tasks=("t1", "t2"))
        outer = Suite(name="outer", tasks=(inner, "t3"))

        # outer has 2 items but expands to 3 tasks
        assert repr(outer) == "Suite('outer', 2 items -> 3 tasks)"


class TestRegister:
    """Tests for register function."""

    def test_register_suite(self, isolated_registry):
        """Test basic suite registration."""
        suite = Suite(name="new_suite", tasks=("task1",))
        result = register(suite)

        assert result is suite  # Returns same suite
        assert "new_suite" in isolated_registry
        assert isolated_registry["new_suite"] is suite

    def test_register_duplicate_raises(self, isolated_registry):
        """Test that registering duplicate name raises error."""
        suite1 = Suite(name="duplicate", tasks=("task1",))
        suite2 = Suite(name="duplicate", tasks=("task2",))

        register(suite1)

        with pytest.raises(ValueError, match="already registered"):
            register(suite2)

    def test_register_chaining(self, isolated_registry):
        """Test that register returns suite for chaining."""
        suite = register(Suite(name="chained", tasks=("task1",)))

        assert suite.name == "chained"
        assert suite_exists("chained")


class TestGetSuite:
    """Tests for get_suite function."""

    def test_get_existing_suite(self, isolated_registry):
        """Test getting a registered suite."""
        suite = Suite(name="existing", tasks=("task1",))
        register(suite)

        result = get_suite("existing")

        assert result is suite

    def test_get_nonexistent_raises(self, isolated_registry):
        """Test that getting unknown suite raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_suite("nonexistent")


class TestListSuites:
    """Tests for list_suites function."""

    def test_list_suites_empty(self, isolated_registry):
        """Test listing with empty registry."""
        result = list_suites()

        assert result == ()

    def test_list_suites_returns_sorted(self, isolated_registry):
        """Test that list_suites returns sorted names."""
        register(Suite(name="zebra", tasks=("t1",)))
        register(Suite(name="alpha", tasks=("t2",)))
        register(Suite(name="middle", tasks=("t3",)))

        result = list_suites()

        assert result == ("alpha", "middle", "zebra")


class TestSearchSuites:
    """Tests for search_suites function."""

    def test_search_exact_string(self, isolated_registry):
        """Test searching with exact string match."""
        register(Suite(name="mmlu:mc", tasks=("t1",)))
        register(Suite(name="mmlu:rc", tasks=("t2",)))
        register(Suite(name="arc:mc", tasks=("t3",)))

        result = search_suites("mmlu")

        assert set(result) == {"mmlu:mc", "mmlu:rc"}

    def test_search_regex_pattern(self, isolated_registry):
        """Test searching with regex pattern."""
        register(Suite(name="core:mc", tasks=("t1",)))
        register(Suite(name="core:rc", tasks=("t2",)))
        register(Suite(name="basic:mc", tasks=("t3",)))

        pattern = re.compile(r":mc$")
        result = search_suites(pattern)

        assert set(result) == {"core:mc", "basic:mc"}

    def test_search_no_matches(self, isolated_registry):
        """Test searching with no matches."""
        register(Suite(name="test", tasks=("t1",)))

        result = search_suites("nonexistent")

        assert result == ()


class TestSuiteExists:
    """Tests for suite_exists function."""

    def test_exists_true(self, isolated_registry):
        """Test suite_exists returns True for registered suite."""
        register(Suite(name="exists", tasks=("t1",)))

        assert suite_exists("exists") is True

    def test_exists_false(self, isolated_registry):
        """Test suite_exists returns False for unknown suite."""
        assert suite_exists("unknown") is False


class TestMakeSuite:
    """Tests for make_suite helper function."""

    def test_make_suite_creates_and_registers(self, isolated_registry):
        """Test that make_suite creates and registers a suite."""
        suite = make_suite(
            name="made_suite",
            tasks=("task1", "task2"),
            aggregation=AggregationStrategy.DISPLAY_ONLY,
            description="Test description",
        )

        assert suite.name == "made_suite"
        assert suite.tasks == ("task1", "task2")
        assert suite.aggregation == AggregationStrategy.DISPLAY_ONLY
        assert suite.description == "Test description"
        assert suite_exists("made_suite")

    def test_make_suite_default_aggregation(self, isolated_registry):
        """Test make_suite uses default aggregation."""
        suite = make_suite(name="default_agg", tasks=("task1",))

        assert suite.aggregation == AggregationStrategy.AVERAGE


class TestFormatTasks:
    """Tests for format_tasks helper function."""

    def test_format_tasks_basic(self):
        """Test basic task formatting."""
        categories = ("cat1", "cat2", "cat3")
        template = "{}:mc:full"

        result = format_tasks(categories, template)

        assert result == ("cat1:mc:full", "cat2:mc:full", "cat3:mc:full")

    def test_format_tasks_complex_template(self):
        """Test formatting with more complex template."""
        categories = ("arc", "hellaswag")
        template = "eval_{}:rc:full"

        result = format_tasks(categories, template)

        assert result == ("eval_arc:rc:full", "eval_hellaswag:rc:full")

    def test_format_tasks_empty_categories(self):
        """Test formatting with empty categories."""
        result = format_tasks((), "{}")

        assert result == ()

    def test_format_tasks_no_placeholder(self):
        """Test formatting when template has no placeholder."""
        categories = ("a", "b")
        template = "static"

        result = format_tasks(categories, template)

        assert result == ("static", "static")


class TestIntegration:
    """Integration tests with actual registered suites."""

    def test_actual_suites_registered(self):
        """Test that suites are actually registered after import."""
        # Import triggers registration
        import olmo_eval.evals.suites  # noqa: F401

        suites = list_suites()
        assert len(suites) > 0

    def test_get_actual_suite(self):
        """Test getting an actual registered suite."""
        import olmo_eval.evals.suites  # noqa: F401

        # mt_mbpp_v2fix should be registered by code.py
        if suite_exists("mt_mbpp_v2fix"):
            suite = get_suite("mt_mbpp_v2fix")
            assert isinstance(suite, Suite)
            assert suite.name == "mt_mbpp_v2fix"

    def test_search_actual_suites(self):
        """Test searching actual registered suites."""
        import olmo_eval.evals.suites  # noqa: F401

        # Search for mbpp suites
        mbpp_suites = search_suites("mbpp")
        assert len(mbpp_suites) > 0
        assert all("mbpp" in name for name in mbpp_suites)

    def test_nested_suite_expansion(self):
        """Test that nested suites expand correctly."""
        import olmo_eval.evals.suites  # noqa: F401

        # Find a suite that likely has nested suites
        for name in list_suites():
            suite = get_suite(name)
            expanded = suite.expand()
            # All expanded tasks should be strings
            assert all(isinstance(t, str) for t in expanded)
            # Expanded count should be >= direct task count
            assert len(expanded) >= len(suite.tasks)
