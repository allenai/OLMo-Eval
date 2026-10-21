"""Tests for the BFCL tasks, their prompts, and the checker they score with."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from olmo_eval.common.scorers.bfcl import (
    BFCLScorer,
    DecodeError,
    ast_checker,
    decode_text,
    decode_text_lenient,
    decode_tool_calls,
    is_empty_output,
    is_function_calling_format,
)
from olmo_eval.common.scorers.bfcl.constants import Language
from olmo_eval.common.types import Instance, LMOutput, RequestType, compute_task_hash
from olmo_eval.common.types.tools import ToolCall
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks.bfcl import (
    BASE_INSTRUCTION,
    EXEMPLARS,
    PYTHON_EXEMPLARS,
    SYSTEM_PROMPT,
    TASK_CATEGORIES,
    BFCLTask,
    Exemplar,
    build_tool_schemas,
    category_from_id,
    fewshot_source_for,
    prepare_function_docs,
)
from olmo_eval.evals.tasks.common import get_task


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


needs_java = pytest.mark.skipif(
    not _has("tree_sitter_java"), reason="tree-sitter-java is not installed"
)
needs_javascript = pytest.mark.skipif(
    not _has("tree_sitter_javascript"), reason="tree-sitter-javascript is not installed"
)


def function_doc(
    name: str, properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Documentation for {name}.",
        "parameters": {
            "type": "dict",
            "properties": properties,
            "required": required if required is not None else list(properties),
        },
    }


AREA_DOC = function_doc(
    "geometry.triangle_area",
    {
        "base": {"type": "integer", "description": "Base of the triangle."},
        "height": {"type": "integer", "description": "Height of the triangle."},
        "unit": {"type": "string", "description": "Unit of measure."},
    },
    required=["base", "height"],
)

AREA_ANSWER = [{"geometry.triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]

SIMPLE_ENTRY = {
    "id": "simple_0",
    "question": [[{"role": "user", "content": "Area of a triangle with base 10 and height 5?"}]],
    "function": [AREA_DOC],
}


def simple_task(spec: str = "bfcl_simple:base") -> BFCLTask:
    """A task whose possible answers are supplied rather than downloaded."""
    task = get_task(spec)
    assert isinstance(task, BFCLTask)
    task._answers = {"simple_0": AREA_ANSWER}
    return task


def simple_instance(spec: str = "bfcl_simple:base") -> Instance:
    instance = simple_task(spec).process_doc(SIMPLE_ENTRY)
    assert instance is not None
    return instance


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def test_decodes_a_bare_call_without_the_brackets() -> None:
    assert decode_text("geometry.triangle_area(base=10, height=5)") == [
        {"geometry.triangle_area": {"base": 10, "height": 5}}
    ]


def test_decodes_several_calls_and_strips_code_fences() -> None:
    text = "```\n[f(a=1), g(b='x')]\n```"

    assert decode_text(text) == [{"f": {"a": 1}}, {"g": {"b": "x"}}]


def test_decodes_nested_and_negative_argument_values() -> None:
    decoded = decode_text("[f(items=[1, 2], mapping={'k': 'v'}, offset=-3, flag=True)]")

    assert decoded == [{"f": {"items": [1, 2], "mapping": {"k": "v"}, "offset": -3, "flag": True}}]


def test_folds_arithmetic_arguments_to_a_number() -> None:
    assert decode_text("[f(total=2 + 3 * 4)]") == [{"f": {"total": 14}}]


def test_does_not_execute_a_call_written_inside_an_argument() -> None:
    # The reference implementation evaluates such an expression; running text a
    # model wrote is not worth the fidelity, so it is kept as source instead.
    decoded = decode_text("[f(total=__import__('os').getpid() + 1)]")

    assert decoded == [{"f": {"total": "__import__('os').getpid() + 1"}}]


def test_a_bare_name_argument_decodes_to_its_own_text() -> None:
    assert decode_text("[f(handle=sessionToken)]") == [{"f": {"handle": "sessionToken"}}]


def test_prose_does_not_decode_as_calls() -> None:
    with pytest.raises(DecodeError):
        decode_text("None of the provided functions can answer that.")


def test_lenient_decoding_accepts_a_json_tool_call_list() -> None:
    text = '[{"name": "f", "arguments": {"a": 1}}]'

    assert decode_text_lenient(text) == [{"f": {"a": 1}}]
    with pytest.raises(DecodeError):
        decode_text(text)


def test_lenient_decoding_accepts_a_tool_call_tag() -> None:
    text = 'Sure.\n<tool_call>{"name": "f", "arguments": "{\\"a\\": 1}"}</tool_call>'

    assert decode_text_lenient(text) == [{"f": {"a": 1}}]


def test_lenient_decoding_still_reports_prose_as_a_failure() -> None:
    with pytest.raises(DecodeError):
        decode_text_lenient("I cannot help with that.")


def test_native_tool_calls_are_read_back_under_the_dataset_name() -> None:
    calls = [ToolCall.create("call_1", "geometry_triangle_area", {"base": 10, "height": 5})]

    decoded = decode_tool_calls(calls, {"geometry_triangle_area": "geometry.triangle_area"})

    assert decoded == [{"geometry.triangle_area": {"base": 10, "height": 5}}]


def test_native_tool_calls_with_unparsable_arguments_fail_to_decode() -> None:
    calls = [ToolCall.create("call_1", "f", "{not json")]

    with pytest.raises(DecodeError):
        decode_tool_calls(calls, {})


def test_format_and_emptiness_checks() -> None:
    assert is_function_calling_format([{"f": {"a": 1}}])
    assert is_function_calling_format([])
    assert not is_function_calling_format([{"f": "a"}])
    assert not is_function_calling_format("f(a=1)")

    assert is_empty_output([])
    assert is_empty_output([{}])
    assert is_empty_output("not calls")
    assert not is_empty_output([{"f": {}}])


@needs_java
def test_java_call_text_decodes_to_a_call() -> None:
    decoded = decode_text(
        '[DatabaseConnector.connect(url="jdbc:mysql://localhost/test", timeoutSeconds=30)]',
        Language.JAVA,
    )

    assert decoded == [
        {
            "DatabaseConnector.connect": {
                "url": "jdbc:mysql://localhost/test",
                "timeoutSeconds": "30",
            }
        }
    ]


@needs_javascript
def test_javascript_call_text_decodes_to_a_call() -> None:
    decoded = decode_text("[setElementText(elementId='title', text='Hello')]", Language.JAVASCRIPT)

    assert decoded == [{"setElementText": {"elementId": "title", "text": "Hello"}}]


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


def check_simple(calls: list[dict[str, Any]]) -> bool:
    return ast_checker([AREA_DOC], calls, AREA_ANSWER, Language.PYTHON, "simple")["valid"]


def test_a_correct_call_passes_with_the_optional_parameter_left_out() -> None:
    assert check_simple([{"geometry.triangle_area": {"base": 10, "height": 5}}])


def test_a_correct_call_passes_with_the_optional_parameter_supplied() -> None:
    assert check_simple([{"geometry.triangle_area": {"base": 10, "height": 5, "unit": "units"}}])


def test_a_wrong_value_fails() -> None:
    assert not check_simple([{"geometry.triangle_area": {"base": 11, "height": 5}}])


def test_a_missing_required_parameter_fails() -> None:
    assert not check_simple([{"geometry.triangle_area": {"base": 10}}])


def test_an_undeclared_parameter_fails() -> None:
    assert not check_simple([{"geometry.triangle_area": {"base": 10, "height": 5, "color": "red"}}])


def test_the_wrong_function_fails() -> None:
    assert not check_simple([{"geometry.square_area": {"base": 10, "height": 5}}])


def test_a_simple_category_rejects_more_than_one_call() -> None:
    call = {"geometry.triangle_area": {"base": 10, "height": 5}}

    assert not check_simple([call, call])


def test_strings_compare_without_case_spacing_or_punctuation() -> None:
    doc = function_doc("book", {"when": {"type": "string", "description": "Date to book."}})
    answer = [{"book": {"when": ["April 1, 2024"]}}]

    result = ast_checker(
        [doc], [{"book": {"when": "april 1 2024"}}], answer, Language.PYTHON, "simple"
    )

    assert result["valid"]


def test_an_integer_is_accepted_where_a_float_is_declared() -> None:
    doc = function_doc("weigh", {"kg": {"type": "float", "description": "Weight."}})
    answer = [{"weigh": {"kg": [70.0]}}]

    assert ast_checker([doc], [{"weigh": {"kg": 70}}], answer, Language.PYTHON, "simple")["valid"]


def test_a_list_argument_is_compared_against_the_accepted_lists() -> None:
    doc = function_doc(
        "tag",
        {"names": {"type": "array", "items": {"type": "string"}, "description": "Tags."}},
    )
    answer = [{"tag": {"names": [["red", "blue"]]}}]

    assert ast_checker(
        [doc], [{"tag": {"names": ["Red", "Blue"]}}], answer, Language.PYTHON, "simple"
    )["valid"]
    assert not ast_checker([doc], [{"tag": {"names": ["red"]}}], answer, Language.PYTHON, "simple")[
        "valid"
    ]


def test_a_dictionary_argument_compares_key_by_key() -> None:
    doc = function_doc("configure", {"options": {"type": "dict", "description": "Options."}})
    answer = [{"configure": {"options": [{"mode": ["fast"], "retries": [3, ""]}]}}]

    assert ast_checker(
        [doc], [{"configure": {"options": {"mode": "fast"}}}], answer, Language.PYTHON, "simple"
    )["valid"]
    assert not ast_checker(
        [doc],
        [{"configure": {"options": {"mode": "slow"}}}],
        answer,
        Language.PYTHON,
        "simple",
    )["valid"]


def test_a_list_of_dictionaries_compares_in_order() -> None:
    doc = function_doc(
        "query",
        {"filters": {"type": "array", "items": {"type": "dict"}, "description": "Filters."}},
    )
    answer = [{"query": {"filters": [[{"field": ["age"]}, {"field": ["job"]}]]}}]

    assert ast_checker(
        [doc],
        [{"query": {"filters": [{"field": "age"}, {"field": "job"}]}}],
        answer,
        Language.PYTHON,
        "simple",
    )["valid"]
    assert not ast_checker(
        [doc],
        [{"query": {"filters": [{"field": "job"}, {"field": "age"}]}}],
        answer,
        Language.PYTHON,
        "simple",
    )["valid"]


def test_parallel_calls_may_arrive_in_any_order() -> None:
    doc = function_doc(
        "forecast",
        {
            "city": {"type": "string", "description": "City."},
            "days": {"type": "integer", "description": "Days."},
        },
    )
    answer = [
        {"forecast": {"city": ["Boston"], "days": [3]}},
        {"forecast": {"city": ["Denver"], "days": [3]}},
    ]
    calls = [
        {"forecast": {"city": "Denver", "days": 3}},
        {"forecast": {"city": "Boston", "days": 3}},
    ]

    assert ast_checker([doc], calls, answer, Language.PYTHON, "parallel")["valid"]
    assert not ast_checker([doc], calls[:1], answer, Language.PYTHON, "parallel")["valid"]


def test_the_multiple_category_checks_the_one_chosen_function() -> None:
    hotel = function_doc("hotel.search", {"city": {"type": "string", "description": "City."}})
    flight = function_doc("flight.search", {"origin": {"type": "string", "description": "From."}})
    answer = [{"hotel.search": {"city": ["Kyoto"]}}]

    assert ast_checker(
        [flight, hotel], [{"hotel.search": {"city": "Kyoto"}}], answer, Language.PYTHON, "multiple"
    )["valid"]
    assert not ast_checker(
        [flight, hotel],
        [{"flight.search": {"origin": "Kyoto"}}],
        answer,
        Language.PYTHON,
        "multiple",
    )["valid"]


# ---------------------------------------------------------------------------
# Java and JavaScript value conversion, as the reference implementation checks it
# ---------------------------------------------------------------------------


@needs_java
def test_java_values_convert_to_their_python_equivalents() -> None:
    from olmo_eval.common.scorers.bfcl.java_js import java_type_converter as convert

    assert convert("true", "boolean") is True
    assert convert("false", "boolean") is False
    assert convert("123", "integer") == 123
    assert convert("-123", "integer") == -123
    assert convert("3.14f", "float") == pytest.approx(3.14)
    assert convert("3e-3F", "float") == pytest.approx(3e-3)
    assert convert("3.14", "double") == pytest.approx(3.14)
    assert convert("123L", "long") == 123
    assert convert("abc", "String") == "abc"
    assert convert("123", "any") == "123"
    assert convert("new int[]{1, 2, 3}", "Array") == [1, 2, 3]
    assert convert("new int[] { }", "Array") == []
    assert convert('new Object[]{1, "abc", true}', "Array") == [1, "abc", True]
    assert convert('new ArrayList<>(Arrays.asList("a", "b"))', "ArrayList") == ["a", "b"]
    assert convert("new ArrayList<>()", "ArrayList") == []
    assert convert('new HashMap<String, String>() {{ put("key", "value"); }}', "HashMap") == {
        "key": "value"
    }
    assert convert("new HashMap<>()", "HashMap") == {}
    # Text that does not match the declared type stays a string, so it fails as
    # a wrong value rather than as a decoding error.
    assert convert("3.14", "float") == "3.14"
    assert convert("invalid", "boolean") == "invalid"


@needs_java
def test_java_collection_elements_honor_their_declared_type() -> None:
    from olmo_eval.common.scorers.bfcl.java_js import _parse_java_array, _parse_java_arraylist

    assert _parse_java_array("new long[]{1L, 2L, 3L}", nested_type="long") == [1, 2, 3]
    assert _parse_java_array("new long[]{1L, 2, 3L}", nested_type="long") == [1, "2", 3]
    assert _parse_java_arraylist(
        "new ArrayList<Integer>(Arrays.asList(1, 2, 3))", nested_type="integer"
    ) == [1, 2, 3]
    assert _parse_java_arraylist(
        "new ArrayList<Character>() {{ add('a'); add('b'); }}", nested_type="char"
    ) == ["a", "b"]


@needs_javascript
def test_javascript_values_convert_to_their_python_equivalents() -> None:
    from olmo_eval.common.scorers.bfcl.java_js import js_type_converter as convert

    assert convert("true", "Boolean") is True
    assert convert("false", "Boolean") is False
    assert convert("123", "integer") == 123
    assert convert("3.14", "float") == pytest.approx(3.14)
    assert convert("123n", "Bigint") == 123
    assert convert("abc", "String") == "abc"
    assert convert("'abc'", "String") == "abc"
    assert convert("[1, 2, 3]", "array") == [1, 2, 3]
    assert convert("new Array(1, 2, 3)", "array") == [1, 2, 3]
    assert convert("[]", "array") == []
    assert convert("{'key': 'value'}", "dict") == {"key": "value"}
    assert convert("{'key': 123}", "dict") == {"key": 123}
    assert convert("{}", "dict") == {}


# ---------------------------------------------------------------------------
# Instances and prompts
# ---------------------------------------------------------------------------


def test_an_instance_carries_the_raw_documents_and_the_prepared_ones() -> None:
    instance = simple_instance()

    assert instance.question == "Area of a triangle with base 10 and height 5?"
    assert instance.metadata["id"] == "simple_0"
    assert instance.metadata["test_category"] == "simple"
    assert instance.metadata["ground_truth"] == AREA_ANSWER
    # The checker compares against the documents as the dataset ships them.
    assert instance.metadata["functions"][0]["description"] == AREA_DOC["description"]
    assert instance.metadata["prepared_functions"][0]["description"].endswith(
        "Note that the provided function is in Python 3 syntax."
    )


def test_a_tool_name_is_made_valid_and_mapped_back() -> None:
    instance = simple_instance()

    assert instance.tools is not None
    assert instance.tools[0].name == "geometry_triangle_area"
    assert instance.metadata["name_map"] == {"geometry_triangle_area": "geometry.triangle_area"}


def test_tool_schemas_use_json_schema_types() -> None:
    schemas, _ = build_tool_schemas([AREA_DOC])

    assert schemas[0].parameters["type"] == "object"
    assert schemas[0].parameters["properties"]["base"]["type"] == "integer"
    assert schemas[0].parameters["required"] == ["base", "height"]


def test_tool_names_that_would_collide_are_kept_distinct() -> None:
    schemas, name_map = build_tool_schemas([function_doc("a.b", {}), function_doc("a_b", {})])

    assert len({schema.name for schema in schemas}) == 2
    assert set(name_map.values()) == {"a.b", "a_b"}


def test_java_documents_describe_arguments_as_source_text() -> None:
    doc = function_doc(
        "Files.read",
        {"path": {"type": "String", "description": "Path to read."}},
    )

    prepared = prepare_function_docs([doc], Language.JAVA)

    assert prepared[0]["parameters"]["properties"]["path"]["type"] == "string"
    assert (
        "Java String type parameter"
        in prepared[0]["parameters"]["properties"]["path"]["description"]
    )
    # The original document is untouched, so the checker still sees the real type.
    assert doc["parameters"]["properties"]["path"]["type"] == "String"


def test_the_default_regime_sends_tool_schemas_and_no_system_prompt() -> None:
    task = simple_task("bfcl_simple")
    request = task.format_request(simple_instance("bfcl_simple"))

    assert request.request_type == RequestType.CHAT
    assert request.system_prompt is None
    assert request.tools is not None and request.tools[0].name == "geometry_triangle_area"
    assert [message["role"] for message in request.messages] == ["user"]


def test_the_prompt_regime_writes_the_functions_into_a_system_message() -> None:
    task = simple_task("bfcl_simple:prompt")
    request = task.format_request(simple_instance("bfcl_simple:prompt"))

    assert request.tools is None
    system = request.messages[0]
    assert system["role"] == "system"
    assert system["content"].startswith(SYSTEM_PROMPT.split("{functions}")[0])
    assert "geometry.triangle_area" in system["content"]
    assert "[func_name1(params_name1=params_value1" in system["content"]


def test_the_prompt_regime_keeps_a_question_s_own_system_message() -> None:
    entry = {
        **SIMPLE_ENTRY,
        "question": [
            [
                {"role": "system", "content": "Answer as a travel agent."},
                {"role": "user", "content": "Area of a triangle with base 10 and height 5?"},
            ]
        ],
    }
    task = simple_task("bfcl_simple:prompt")
    instance = task.process_doc(entry)
    assert instance is not None

    system = task.format_request(instance).messages[0]["content"]

    assert system.endswith("Answer as a travel agent.")
    assert "[func_name1(params_name1=params_value1" in system


def test_the_prompt_is_a_completion_ending_at_the_answer_header() -> None:
    task = simple_task()
    request = task.format_request(simple_instance())

    assert request.request_type == RequestType.COMPLETION
    assert request.messages == ()
    assert request.prompt.startswith(BASE_INSTRUCTION)
    assert request.prompt.endswith(
        "### Query:\nArea of a triangle with base 10 and height 5?\n\n### Answer:\n"
    )
    assert request.prompt.count("### Answer:") == 6


def test_the_prompt_shows_the_requested_number_of_exemplars() -> None:
    assert len(simple_task("bfcl_simple:base:0shot").get_fewshot()) == 0
    assert len(simple_task("bfcl_simple:base:2shot").get_fewshot()) == 2
    assert len(simple_task().get_fewshot()) == 5


def test_every_language_supplies_exemplars_for_every_shot_count() -> None:
    for spec in ("bfcl_java:base", "bfcl_javascript:base", "bfcl_simple:base"):
        assert len(get_task(spec).get_fewshot()) == 5


def test_exemplars_demonstrate_declining_as_well_as_calling() -> None:
    answers = [example.gold_answer or "" for example in simple_task().get_fewshot()]

    assert any(answer.startswith("[") for answer in answers)
    assert any(not answer.startswith("[") for answer in answers)


# ---------------------------------------------------------------------------
# Answer extraction and scoring
# ---------------------------------------------------------------------------


def score_one(task: BFCLTask, instance: Instance, output: LMOutput) -> float:
    from olmo_eval.common.types import Response

    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])
    task._extract_answers([response])
    return BFCLScorer().score(instance, output)


def test_a_correct_native_tool_call_scores_one() -> None:
    task = simple_task("bfcl_simple")
    output = LMOutput(
        text="",
        tool_calls=[ToolCall.create("1", "geometry_triangle_area", {"base": 10, "height": 5})],
    )

    assert score_one(task, simple_instance("bfcl_simple"), output) == 1.0


def test_text_instead_of_a_tool_call_scores_zero_and_says_so() -> None:
    task = simple_task("bfcl_simple")
    output = LMOutput(text="[geometry.triangle_area(base=10, height=5)]", tool_calls=None)

    assert score_one(task, simple_instance("bfcl_simple"), output) == 0.0
    assert output.metadata["bfcl_no_tool_calls_with_text"] is True


def test_a_correct_call_scores_one() -> None:
    task = simple_task()
    output = LMOutput(text="[geometry.triangle_area(base=10, height=5)]")

    assert score_one(task, simple_instance(), output) == 1.0


def test_a_reply_in_a_json_tool_call_format_still_scores() -> None:
    # A base model reaches for the format it saw in pretraining; the answer is
    # judged on the call it chose, not on the shape it wrote it in.
    task = simple_task()
    output = LMOutput(
        text='[{"name": "geometry.triangle_area", "arguments": {"base": 10, "height": 5}}]'
    )

    assert score_one(task, simple_instance(), output) == 1.0


def test_an_undecodable_reply_scores_zero_and_records_why() -> None:
    task = simple_task()
    output = LMOutput(text="I am not able to help with that.")

    assert score_one(task, simple_instance(), output) == 0.0
    assert "bfcl_decode_error" in output.metadata


def test_irrelevance_rewards_declining_and_punishes_calling() -> None:
    task = get_task("bfcl_irrelevance:base")
    assert isinstance(task, BFCLTask)
    task._answers = {}
    entry = {
        "id": "irrelevance_0",
        "question": [[{"role": "user", "content": "Write me a haiku."}]],
        "function": [AREA_DOC],
    }
    instance = task.process_doc(entry)
    assert instance is not None

    declined = LMOutput(text="None of the provided functions can write a haiku.")
    called = LMOutput(text="[geometry.triangle_area(base=10, height=5)]")

    assert score_one(task, instance, declined) == 1.0
    assert score_one(task, instance, called) == 0.0


def test_relevance_rewards_calling_and_punishes_declining() -> None:
    task = get_task("bfcl_live_relevance:base")
    assert isinstance(task, BFCLTask)
    task._answers = {}
    entry = {
        "id": "live_relevance_0-0-0",
        "question": [[{"role": "user", "content": "Area of a triangle with base 10?"}]],
        "function": [AREA_DOC],
    }
    instance = task.process_doc(entry)
    assert instance is not None

    assert score_one(task, instance, LMOutput(text="[geometry.triangle_area(base=10)]")) == 1.0
    assert score_one(task, instance, LMOutput(text="I need the height as well.")) == 0.0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_exemplar_set_is_named_in_the_task_configuration() -> None:
    # Nothing about hand-written exemplars reaches TaskConfig on its own, so
    # the name is what puts them into a run's stored configuration.
    assert get_task("bfcl_simple:base").config.fewshot_source == fewshot_source_for(Language.PYTHON)
    assert get_task("bfcl_live:base").config.fewshot_source == fewshot_source_for(Language.PYTHON)
    assert get_task("bfcl_java:base").config.fewshot_source == fewshot_source_for(Language.JAVA)
    assert get_task("bfcl_javascript:base").config.fewshot_source == fewshot_source_for(
        Language.JAVASCRIPT
    )


def test_each_language_s_exemplars_are_named_distinctly() -> None:
    names = {fewshot_source_for(language) for language in Language}

    assert len(names) == 3
    assert all(name.startswith("bfcl_fixed_") for name in names)


def test_the_exemplar_name_is_stable_and_reaches_the_task_hash() -> None:
    config = get_task("bfcl_simple:base").config

    assert fewshot_source_for(Language.PYTHON) == fewshot_source_for(Language.PYTHON)
    assert config.to_dict()["fewshot_source"] == config.fewshot_source
    assert compute_task_hash(config.to_dict()) != compute_task_hash(
        {**config.to_dict(), "fewshot_source": "something else"}
    )


def test_editing_an_exemplar_changes_the_name_it_is_recorded_under(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Otherwise two runs prompted differently would share a task hash.
    before = fewshot_source_for(Language.PYTHON)
    edited = (
        Exemplar(question="Reworded.", answer="[f(a=1)]", functions=PYTHON_EXEMPLARS[0].functions),
        *PYTHON_EXEMPLARS[1:],
    )
    monkeypatch.setitem(EXEMPLARS, Language.PYTHON, edited)

    assert fewshot_source_for(Language.PYTHON) != before


def test_the_category_of_an_entry_comes_from_its_id() -> None:
    assert category_from_id("simple_0") == "simple"
    assert category_from_id("parallel_multiple_7") == "parallel_multiple"
    assert category_from_id("live_parallel_multiple_3-0-0") == "live_parallel_multiple"


def test_every_single_turn_category_is_registered() -> None:
    for task_name in TASK_CATEGORIES:
        for variant in ("", ":prompt", ":base", ":base:2shot"):
            assert get_task(f"{task_name}{variant}") is not None


def test_the_pooled_live_tasks_read_every_live_file() -> None:
    assert get_task("bfcl_live_ast").config.data_source.data_files == (
        "BFCL_v3_live_simple.json",
        "BFCL_v3_live_multiple.json",
        "BFCL_v3_live_parallel.json",
        "BFCL_v3_live_parallel_multiple.json",
    )
    assert len(get_task("bfcl_live").config.data_source.data_files) == 6


def test_only_the_java_and_javascript_tasks_declare_grammar_dependencies() -> None:
    assert get_task("bfcl_java").config.dependencies == [
        "tree-sitter-java>=0.23,<0.24",
        "tree-sitter-javascript>=0.23,<0.24",
    ]
    assert get_task("bfcl_simple").config.dependencies is None


def test_the_overall_suite_averages_the_non_live_summary_with_the_live_one() -> None:
    assert get_suite("bfcl").expand() == (
        "bfcl_simple",
        "bfcl_java",
        "bfcl_javascript",
        "bfcl_multiple",
        "bfcl_parallel",
        "bfcl_parallel_multiple",
        "bfcl_irrelevance",
        "bfcl_live",
    )


def test_the_chat_regimes_sample_the_way_the_leaderboard_does() -> None:
    # BFCL asks for up to 4096 tokens, fitted to the context window, at 0.001.
    for spec in ("bfcl_simple", "bfcl_simple:prompt"):
        params = get_task(spec).config.sampling_params
        assert params.max_tokens == 4096
        assert params.temperature == 0.001
        assert params.fit_max_tokens_to_context


def test_the_completion_regime_keeps_its_own_budget() -> None:
    # The completion regime has no counterpart upstream; its budget and stop
    # sequences are tuned for a base model writing past its answer.
    params = get_task("bfcl_simple:base").config.sampling_params

    assert params.max_tokens == 512
    assert not params.fit_max_tokens_to_context
