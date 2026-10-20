"""Berkeley Function Calling Leaderboard (BFCL) v3, single-turn categories.

BFCL asks a model to pick the right function from a set of documented ones and
fill in its arguments. A prediction is graded by comparing the calls it made
against a list of accepted values per parameter, so an answer is right when it
names the right function and supplies values the benchmark accepts -- not when
it matches one reference string.

Three regimes are registered for every category, differing only in how the
functions reach the model and how its calls are read back:

``bfcl_simple``
    Native function calling. The functions go to the server as tool schemas and
    the calls come back as ``tool_calls``. This is the regime BFCL was built
    for, and the one to use for an instruction-tuned model behind an
    OpenAI-compatible endpoint::

        uv run olmo-eval run -m my-model -t bfcl_simple

    A vLLM server the run starts itself is told to parse tool calls whenever
    a task sends tool schemas, and vLLM picks a ``--tool-call-parser`` from
    the model name; add ``-o provider.kwargs.tool_call_parser=<name>`` to
    choose one. A server that would answer in plain text, or a provider that
    cannot carry tool schemas at all, stops the run with an error rather than
    reporting a score of zero.

``bfcl_simple:prompt``
    BFCL's own prompting mode: the functions are written into a system prompt
    and the model is asked to reply with ``[func(arg=value)]`` text. Works
    against any chat endpoint, with no server-side tool parsing.

``bfcl_simple:base``
    The same task as a plain completion, for a pretrained model with no chat
    template. Curated exemplars carry the answer format; ``:0shot`` through
    ``:5shot`` change how many are shown, and the default is five.

Multi-turn, executable, and REST categories are not registered: the first needs
BFCL's stateful API backend, and the others grade by calling live third-party
APIs.

Paper: https://arxiv.org/abs/2502.17858
Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from olmo_eval.common.formatters import Formatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers.bfcl import (
    GORILLA_TO_OPENAPI,
    BFCLScorer,
    DecodeError,
    decode_text,
    decode_text_lenient,
    decode_tool_calls,
    language_for_category,
)
from olmo_eval.common.scorers.bfcl.constants import Language
from olmo_eval.common.types import (
    Instance,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
    ToolSchema,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

#: Pinned so that a later release of the dataset cannot change what a task
#: hash refers to.
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"

#: Grammar packages the Java and JavaScript categories need to read the calls a
#: model writes. Declared per task so the other categories never install them.
TREE_SITTER_DEPENDENCIES = [
    "tree-sitter-java>=0.23,<0.24",
    "tree-sitter-javascript>=0.23,<0.24",
]

#: The opening of BFCL v3's prompt for prompted models, reproduced verbatim
#: including its wording slips, because the prompt is part of what the
#: benchmark measures.
_PROMPT_OPENING = (
    "You are an expert in composing functions. You are given a question and a set of possible "
    "functions. Based on the question, you will need to make one or more function/tool calls to "
    "achieve the purpose.\n"
    "If none of the function can be used, point it out. If the given question lacks the "
    "parameters required by the function, also point it out.\n"
    "You should only return the function call in tools call sections.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n"
    "You SHOULD NOT include any other text in the response."
)

#: BFCL states how a model should behave across turns even in its single-turn
#: categories, so the chat regimes say it too.
_PROMPT_LATER_TURNS = (
    "At each turn, your should try your best to complete the tasks requested by the user within "
    "the current turn. Continue outputting functions to call until you have fulfilled the user's "
    "request to the best of your ability. Once you have no more functions to call, the system "
    "will consider the current turn complete and proceed to the next turn or task."
)

SYSTEM_PROMPT_WITHOUT_FUNC_DOC = f"{_PROMPT_OPENING}\n\n{_PROMPT_LATER_TURNS}\n"

SYSTEM_PROMPT = (
    SYSTEM_PROMPT_WITHOUT_FUNC_DOC
    + "\nHere is a list of functions in JSON format that you can invoke.\n{functions}\n\n"
)

#: The completion regime drops the paragraph about later turns, which has
#: nothing to say about a single-turn completion, and states the functions
#: per example instead of once in a system message.
BASE_INSTRUCTION = _PROMPT_OPENING


# ---------------------------------------------------------------------------
# Function document preparation
# ---------------------------------------------------------------------------

_LANGUAGE_HINTS = {
    Language.JAVA: " Note that the provided function is in Java 8 SDK syntax.",
    Language.JAVASCRIPT: " Note that the provided function is in JavaScript syntax.",
    Language.PYTHON: " Note that the provided function is in Python 3 syntax.",
}


_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

#: OpenAI's limit on a tool name.
_MAX_TOOL_NAME_LEN = 64


def _annotate_java_parameter(value: dict[str, Any]) -> None:
    """Describe a Java parameter as the string the model should write."""
    if value["type"] == "any":
        value["description"] += (
            " This parameter can be of any type of Java object in string representation."
        )
    else:
        value["description"] += (
            f" This is Java {value['type']} type parameter in string representation."
        )
    if value["type"] in ("ArrayList", "Array"):
        value["description"] += (
            f" The list elements are of type {value['items']['type']}; "
            "they are not in string representation."
        )
        del value["items"]
    value["type"] = "string"


def _annotate_javascript_parameter(value: dict[str, Any]) -> None:
    """Describe a JavaScript parameter as the string the model should write."""
    import json

    if value["type"] == "any":
        value["description"] += (
            " This parameter can be of any type of JavaScript object in string representation."
        )
    else:
        value["description"] += (
            f" This is JavaScript {value['type']} type parameter in string representation."
        )
    if value["type"] == "array":
        value["description"] += (
            f" The list elements are of type {value['items']['type']}; "
            "they are not in string representation."
        )
        del value["items"]
    if value["type"] == "dict" and "properties" in value:
        value["description"] += (
            " The dictionary entries have the following schema; they are not in string "
            f"representation. {json.dumps(value['properties'])}"
        )
        del value["properties"]
    value["type"] = "string"


def prepare_function_docs(functions: list[dict[str, Any]], language: Language) -> list[dict]:
    """Return the function documents as the model should see them.

    The checker compares against the documents as the dataset ships them; the
    model is shown a copy naming the language and, for Java and JavaScript,
    describing each argument as the source text to write rather than as the
    type the call ultimately has.
    """
    prepared = copy.deepcopy(functions)
    for item in prepared:
        item["description"] = (item.get("description") or "") + _LANGUAGE_HINTS[language]
        properties = (item.get("parameters") or {}).get("properties") or {}
        for value in properties.values():
            value.setdefault("description", "")
            if language is Language.JAVA:
                _annotate_java_parameter(value)
            elif language is Language.JAVASCRIPT:
                _annotate_javascript_parameter(value)
    return prepared


def _cast_to_openapi_types(properties: dict[str, Any]) -> dict[str, Any]:
    """Rewrite Gorilla's type names as JSON Schema ones, one level into nesting."""
    for value in properties.values():
        declared = value.get("type")
        if declared is None:
            value["type"] = "string"
        else:
            if declared == "float":
                value["format"] = "float"
                value["description"] = value.get("description", "") + " This is a float type value."
            value["type"] = GORILLA_TO_OPENAPI.get(declared, "string")

        if value["type"] not in ("array", "object"):
            continue
        if "properties" in value:
            value["properties"] = _cast_to_openapi_types(value["properties"])
        elif "items" in value:
            items = value["items"]
            items["type"] = GORILLA_TO_OPENAPI.get(items.get("type"), "string")
            if items["type"] == "array" and "items" in items:
                inner = items["items"]
                inner["type"] = GORILLA_TO_OPENAPI.get(inner.get("type"), "string")
            elif items["type"] == "object" and "properties" in items:
                items["properties"] = _cast_to_openapi_types(items["properties"])
    return properties


def _sanitize_tool_name(name: str, taken: set[str]) -> str:
    """Return a name an OpenAI-style tool schema accepts, unique among ``taken``."""
    candidate = _INVALID_NAME_CHARS.sub("_", name)[:_MAX_TOOL_NAME_LEN] or "function"
    if candidate not in taken:
        return candidate
    for suffix in range(1, 1000):
        marked = f"{candidate[: _MAX_TOOL_NAME_LEN - len(str(suffix)) - 1]}_{suffix}"
        if marked not in taken:
            return marked
    raise ValueError(f"Could not find a unique tool name for {name!r}.")


def build_tool_schemas(
    functions: list[dict[str, Any]],
) -> tuple[tuple[ToolSchema, ...], dict[str, str]]:
    """Convert function documents to tool schemas, and map names back.

    A tool name has to match ``^[a-zA-Z0-9_-]{1,64}$``, which many BFCL function
    names do not, so the returned map restores the dataset's spelling when the
    calls are read back.
    """
    schemas: list[ToolSchema] = []
    name_map: dict[str, str] = {}
    taken: set[str] = set()

    for item in copy.deepcopy(functions):
        parameters = item.get("parameters") or {}
        parameters["type"] = "object"
        parameters["properties"] = _cast_to_openapi_types(parameters.get("properties") or {})
        parameters.setdefault("required", [])

        sanitized = _sanitize_tool_name(item["name"], taken)
        taken.add(sanitized)
        name_map[sanitized] = item["name"]
        schemas.append(
            ToolSchema(
                name=sanitized,
                description=item.get("description") or "",
                parameters=parameters,
            )
        )

    return tuple(schemas), name_map


def render_function_docs(functions: list[dict[str, Any]]) -> str:
    """Render function documents for a text prompt.

    BFCL interpolates the list of documents directly into its prompt, so they
    reach the model as Python's representation of a list of dicts even though
    the surrounding sentence calls it JSON. Reproduced, because changing the
    rendering changes the scores.
    """
    return str(functions)


def render_messages(messages: Sequence[dict[str, Any]]) -> str:
    """Render a question's messages as the text of one completion prompt."""
    if len(messages) == 1:
        return messages[0].get("content") or ""
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content') or ''}" for m in messages)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class CallSource:
    """Where a formatter's regime expects the model's calls to appear."""

    TOOL_CALLS = "tool_calls"
    TEXT = "text"


@dataclass(slots=True)
class BFCLToolFormatter(Formatter):
    """Send the functions as native tool schemas and read back ``tool_calls``."""

    call_source: ClassVar[str] = CallSource.TOOL_CALLS
    lenient: ClassVar[bool] = False

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def format(self, instance: Instance, fewshot: list[Instance] | None = None) -> LMRequest:
        messages = tuple(dict(m) for m in instance.metadata["messages"])
        system_prompt = next((m["content"] for m in messages if m.get("role") == "system"), None)
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=messages,
            tools=instance.tools,
            system_prompt=system_prompt,
        )


@dataclass(slots=True)
class BFCLPromptFormatter(Formatter):
    """Write the functions into a system prompt and read back call text."""

    call_source: ClassVar[str] = CallSource.TEXT
    lenient: ClassVar[bool] = False

    system_prompt_template: str = SYSTEM_PROMPT

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def _system_prompt(self, instance: Instance) -> str:
        return self.system_prompt_template.format(
            functions=render_function_docs(instance.metadata["prepared_functions"])
        )

    def format(self, instance: Instance, fewshot: list[Instance] | None = None) -> LMRequest:
        messages: list[dict[str, Any]] = []
        for example in fewshot or []:
            messages.append({"role": "user", "content": self._exemplar_user(example)})
            messages.append({"role": "assistant", "content": example.gold_answer or ""})

        instance_messages = [dict(m) for m in instance.metadata["messages"]]
        system_prompt = self._system_prompt(instance)
        if instance_messages and instance_messages[0].get("role") == "system":
            # BFCL keeps a question's own system message, appended to its own.
            instance_messages[0]["content"] = (
                system_prompt + "\n\n" + (instance_messages[0].get("content") or "")
            )
            system_prompt = instance_messages[0]["content"]
            messages = [instance_messages[0], *messages, *instance_messages[1:]]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                *messages,
                *instance_messages,
            ]

        return LMRequest(
            request_type=RequestType.CHAT,
            messages=tuple(messages),
            system_prompt=system_prompt,
        )

    def _exemplar_user(self, example: Instance) -> str:
        functions = render_function_docs(example.metadata["prepared_functions"])
        return (
            "Here is a list of functions in JSON format that you can invoke.\n"
            f"{functions}\n\n{example.question}"
        )


@dataclass(slots=True)
class BFCLCompletionFormatter(Formatter):
    """Lay the task out as a plain completion, with exemplars for the format.

    A base model has no chat template and no system role, so the instruction,
    the functions, and the question are one block of text and the model
    continues it with the calls.
    """

    call_source: ClassVar[str] = CallSource.TEXT
    lenient: ClassVar[bool] = True

    instruction: str = BASE_INSTRUCTION
    functions_header: str = "### Functions:"
    query_header: str = "### Query:"
    answer_header: str = "### Answer:"
    separator: str = "\n\n"

    @property
    def request_type(self) -> RequestType:
        return RequestType.COMPLETION

    def _block(self, instance: Instance, answer: str | None) -> str:
        functions = render_function_docs(instance.metadata["prepared_functions"])
        block = (
            f"{self.functions_header}\n{functions}\n\n"
            f"{self.query_header}\n{instance.question}\n\n"
            f"{self.answer_header}\n"
        )
        return block + answer if answer is not None else block

    def format(self, instance: Instance, fewshot: list[Instance] | None = None) -> LMRequest:
        parts = [self.instruction]
        parts.extend(self._block(example, example.gold_answer or "") for example in fewshot or [])
        parts.append(self._block(instance, None))
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=self.separator.join(parts),
        )


# ---------------------------------------------------------------------------
# Curated few-shot exemplars
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Exemplar:
    """One hand-written demonstration of the answer format."""

    question: str
    answer: str
    functions: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_instance(self, language: Language) -> Instance:
        functions = list(copy.deepcopy(self.functions))
        return Instance(
            question=self.question,
            gold_answer=self.answer,
            metadata={
                "prepared_functions": prepare_function_docs(functions, language),
            },
        )


def _parameters(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "dict", "properties": properties, "required": required}


PYTHON_EXEMPLARS: tuple[Exemplar, ...] = (
    Exemplar(
        question=(
            "What is the body mass index of someone who weighs 70.5 kilograms and is "
            "1.75 meters tall?"
        ),
        answer="[calculate_bmi(weight_kg=70.5, height_m=1.75)]",
        functions=(
            {
                "name": "calculate_bmi",
                "description": "Compute the body mass index for a person.",
                "parameters": _parameters(
                    {
                        "weight_kg": {"type": "float", "description": "Body weight in kilograms."},
                        "height_m": {"type": "float", "description": "Height in meters."},
                        "unit_system": {
                            "type": "string",
                            "description": "Unit system used to report the result.",
                            "default": "metric",
                        },
                    },
                    ["weight_kg", "height_m"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Please write me a haiku about the sea.",
        answer="None of the provided functions can be used to write a haiku.",
        functions=(
            {
                "name": "get_stock_price",
                "description": "Look up the most recent trading price of a stock.",
                "parameters": _parameters(
                    {
                        "ticker": {
                            "type": "string",
                            "description": "Ticker symbol of the stock, such as 'AAPL'.",
                        }
                    },
                    ["ticker"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Find me a hotel in Kyoto for two guests checking in on 2023-11-05.",
        answer="[hotel.search(city='Kyoto', check_in='2023-11-05', guests=2)]",
        functions=(
            {
                "name": "flight.search",
                "description": "Search for flights between two airports on a given date.",
                "parameters": _parameters(
                    {
                        "origin": {"type": "string", "description": "Departure airport code."},
                        "destination": {"type": "string", "description": "Arrival airport code."},
                        "date": {"type": "string", "description": "Departure date, as YYYY-MM-DD."},
                    },
                    ["origin", "destination", "date"],
                ),
            },
            {
                "name": "hotel.search",
                "description": "Search for hotel rooms in a city on a given date.",
                "parameters": _parameters(
                    {
                        "city": {"type": "string", "description": "City to search in."},
                        "check_in": {
                            "type": "string",
                            "description": "Check-in date, as YYYY-MM-DD.",
                        },
                        "guests": {
                            "type": "integer",
                            "description": "Number of guests staying in the room.",
                        },
                    },
                    ["city", "check_in", "guests"],
                ),
            },
        ),
    ),
    Exemplar(
        question="What is the three day forecast for Boston and for Denver?",
        answer="[weather.forecast(city='Boston', days=3), weather.forecast(city='Denver', days=3)]",
        functions=(
            {
                "name": "weather.forecast",
                "description": "Get the weather forecast for a city.",
                "parameters": _parameters(
                    {
                        "city": {"type": "string", "description": "City to forecast for."},
                        "days": {"type": "integer", "description": "Number of days to forecast."},
                    },
                    ["city", "days"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Convert 100 US dollars to euros, and 32 degrees Fahrenheit to Celsius.",
        answer=(
            "[convert_currency(amount=100.0, from_currency='USD', to_currency='EUR'), "
            "convert_temperature(value=32.0, from_unit='fahrenheit', to_unit='celsius')]"
        ),
        functions=(
            {
                "name": "convert_currency",
                "description": "Convert an amount of money from one currency to another.",
                "parameters": _parameters(
                    {
                        "amount": {"type": "float", "description": "Amount to convert."},
                        "from_currency": {
                            "type": "string",
                            "description": "Currency code to convert from.",
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "Currency code to convert to.",
                        },
                    },
                    ["amount", "from_currency", "to_currency"],
                ),
            },
            {
                "name": "convert_temperature",
                "description": "Convert a temperature between two units.",
                "parameters": _parameters(
                    {
                        "value": {"type": "float", "description": "Temperature to convert."},
                        "from_unit": {"type": "string", "description": "Unit to convert from."},
                        "to_unit": {"type": "string", "description": "Unit to convert to."},
                    },
                    ["value", "from_unit", "to_unit"],
                ),
            },
        ),
    ),
)


JAVA_EXEMPLARS: tuple[Exemplar, ...] = (
    Exemplar(
        question=(
            "Open a database connection to 'jdbc:mysql://localhost/test' with a thirty "
            "second timeout."
        ),
        answer=(
            '[DatabaseConnector.connect(url="jdbc:mysql://localhost/test", timeoutSeconds=30)]'
        ),
        functions=(
            {
                "name": "DatabaseConnector.connect",
                "description": "Opens a connection to the database at the given URL.",
                "parameters": _parameters(
                    {
                        "url": {"type": "String", "description": "JDBC URL of the database."},
                        "timeoutSeconds": {
                            "type": "integer",
                            "description": "Seconds to wait before giving up.",
                        },
                    },
                    ["url", "timeoutSeconds"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Join the strings 'a', 'b' and 'c' together using a comma as the separator.",
        answer='[StringUtils.join(array=new String[]{"a", "b", "c"}, separator=",")]',
        functions=(
            {
                "name": "StringUtils.join",
                "description": "Joins the elements of the given array into a single String.",
                "parameters": _parameters(
                    {
                        "array": {
                            "type": "Array",
                            "description": "The values to join together.",
                            "items": {"type": "String"},
                        },
                        "separator": {
                            "type": "String",
                            "description": "Separator placed between elements.",
                        },
                    },
                    ["array", "separator"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Apply the settings mapping 'mode' to 'fast', enforcing them strictly.",
        answer=(
            '[ConfigLoader.apply(settings=new HashMap<String, String>() {{ put("mode", "fast"); }},'
            " strict=true)]"
        ),
        functions=(
            {
                "name": "ConfigLoader.apply",
                "description": "Applies a mapping of configuration settings.",
                "parameters": _parameters(
                    {
                        "settings": {
                            "type": "HashMap",
                            "description": "Setting names mapped to their values.",
                        },
                        "strict": {
                            "type": "boolean",
                            "description": "Whether unknown settings are an error.",
                        },
                    },
                    ["settings", "strict"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Round 3.14159 to two decimal places.",
        answer="[MathUtils.round(value=3.14159, places=2)]",
        functions=(
            {
                "name": "MathUtils.round",
                "description": "Rounds a number to a number of decimal places.",
                "parameters": _parameters(
                    {
                        "value": {"type": "double", "description": "The number to round."},
                        "places": {
                            "type": "integer",
                            "description": "How many decimal places to keep.",
                        },
                    },
                    ["value", "places"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Queue the 'build' and 'test' tasks at priority one.",
        answer=(
            '[TaskQueue.enqueue(tasks=new ArrayList<String>(Arrays.asList("build", "test")), '
            "priority=1)]"
        ),
        functions=(
            {
                "name": "TaskQueue.enqueue",
                "description": "Adds tasks to the queue at a given priority.",
                "parameters": _parameters(
                    {
                        "tasks": {
                            "type": "ArrayList",
                            "description": "Names of the tasks to queue.",
                            "items": {"type": "String"},
                        },
                        "priority": {
                            "type": "integer",
                            "description": "Priority to queue the tasks at.",
                        },
                    },
                    ["tasks", "priority"],
                ),
            },
        ),
    ),
)


JAVASCRIPT_EXEMPLARS: tuple[Exemplar, ...] = (
    Exemplar(
        question="Set the text of the element with id 'title' to 'Hello'.",
        answer="[setElementText(elementId='title', text='Hello')]",
        functions=(
            {
                "name": "setElementText",
                "description": "Replaces the text content of a DOM element.",
                "parameters": _parameters(
                    {
                        "elementId": {
                            "type": "String",
                            "description": "Identifier of the element to change.",
                        },
                        "text": {"type": "String", "description": "Text to display."},
                    },
                    ["elementId", "text"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Render an ordered list containing 'one' and 'two'.",
        answer="[renderList(items=['one', 'two'], ordered=true)]",
        functions=(
            {
                "name": "renderList",
                "description": "Renders a list of items into the page.",
                "parameters": _parameters(
                    {
                        "items": {
                            "type": "array",
                            "description": "Items to render.",
                            "items": {"type": "String"},
                        },
                        "ordered": {
                            "type": "Boolean",
                            "description": "Whether the list is numbered.",
                        },
                    },
                    ["items", "ordered"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Configure the chart with its title option set to 'Sales', then redraw it.",
        answer="[configureChart(options={'title': 'Sales'}, redraw=true)]",
        functions=(
            {
                "name": "configureChart",
                "description": "Applies display options to the chart.",
                "parameters": _parameters(
                    {
                        "options": {
                            "type": "dict",
                            "description": "Option names mapped to their values.",
                        },
                        "redraw": {
                            "type": "Boolean",
                            "description": "Whether to redraw the chart afterwards.",
                        },
                    },
                    ["options", "redraw"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Run the job with id 'job-42' after a delay of 1500 milliseconds.",
        answer="[scheduleTask(taskId='job-42', delayMs=1500)]",
        functions=(
            {
                "name": "scheduleTask",
                "description": "Runs a task after a delay.",
                "parameters": _parameters(
                    {
                        "taskId": {"type": "String", "description": "Identifier of the task."},
                        "delayMs": {
                            "type": "integer",
                            "description": "Milliseconds to wait before running.",
                        },
                    },
                    ["taskId", "delayMs"],
                ),
            },
        ),
    ),
    Exemplar(
        question="Turn off the 'beta' and 'dark-mode' features.",
        answer="[toggleFeatures(names=['beta', 'dark-mode'], enabled=false)]",
        functions=(
            {
                "name": "toggleFeatures",
                "description": "Turns a set of feature flags on or off.",
                "parameters": _parameters(
                    {
                        "names": {
                            "type": "array",
                            "description": "Names of the features to change.",
                            "items": {"type": "String"},
                        },
                        "enabled": {
                            "type": "Boolean",
                            "description": "Whether the features should be on.",
                        },
                    },
                    ["names", "enabled"],
                ),
            },
        ),
    ),
)

EXEMPLARS: dict[Language, tuple[Exemplar, ...]] = {
    Language.PYTHON: PYTHON_EXEMPLARS,
    Language.JAVA: JAVA_EXEMPLARS,
    Language.JAVASCRIPT: JAVASCRIPT_EXEMPLARS,
}


def fewshot_source_for(language: Language) -> str:
    """Name the exemplar set a task prompts with, and the revision of it.

    The exemplars are written here rather than drawn from a dataset, so nothing
    about them would otherwise reach :class:`TaskConfig`: a run's stored
    configuration, and the task hash derived from it, would record neither
    which set produced its prompts nor what that set said. Two languages'
    exemplars would be indistinguishable, and editing one would leave the hash
    of every earlier run intact, so results from different prompts would share
    an identity.

    The name follows ``minerva_math``'s fixed set. The digest extends it,
    because these exemplars are expected to be revised: it changes whenever
    their text does, which errs towards giving a run a new identity rather
    than silently reusing one.
    """
    payload = json.dumps(
        [
            {"question": e.question, "answer": e.answer, "functions": e.functions}
            for e in EXEMPLARS[language]
        ],
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"bfcl_fixed_{language}@{digest}"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

BFCL_METRIC = AccuracyMetric(scorer=BFCLScorer)

#: BFCL asks for up to 4096 tokens, shrunk to whatever the context window has
#: left once the prompt is in it, and samples at 0.001 rather than a flat zero.
CHAT_SAMPLING = SamplingParams(
    max_tokens=4096,
    temperature=0.001,
    fit_max_tokens_to_context=True,
)

#: The completion regime stops at the next block header so that a model which
#: keeps writing examples is scored on the answer it gave, not on what follows.
COMPLETION_SAMPLING = SamplingParams(
    max_tokens=512,
    temperature=0.0,
    stop_sequences=("\n###", "\n\n"),
)


def category_from_id(test_id: str) -> str:
    """Return the category an entry belongs to, as BFCL derives it."""
    return test_id.rsplit("_", 1)[0]


class BFCLTask(Task):
    """One or more BFCL categories scored with the leaderboard's own checker.

    A task that names several categories pools their instances, which makes its
    accuracy the instance-weighted mean the leaderboard reports for its live
    summaries; a task naming one category reports that category alone.
    """

    #: Categories this task loads, in the order their files are read.
    categories: tuple[str, ...] = ()

    split = Split.TRAIN
    formatter = BFCLToolFormatter()
    metrics = (BFCL_METRIC,)
    primary_metric = BFCL_METRIC
    sampling_params = CHAT_SAMPLING
    # A reasoning model's trace is not part of its answer; a reply without one
    # is left alone.
    strip_thinking = True

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._answers: dict[str, list[dict[str, Any]]] | None = None

    @property
    def language(self) -> Language:
        """The language this task's calls are written in."""
        return language_for_category(self.categories[0])

    @classmethod
    def data_files(cls) -> tuple[str, ...]:
        return tuple(f"BFCL_v3_{category}.json" for category in cls.categories)

    @classmethod
    def answer_files(cls) -> tuple[str, ...]:
        from olmo_eval.common.scorers.bfcl import RELEVANCE_CATEGORIES

        return tuple(
            f"possible_answer/BFCL_v3_{category}.json"
            for category in cls.categories
            if category not in RELEVANCE_CATEGORIES
        )

    def _load_answers(self) -> dict[str, list[dict[str, Any]]]:
        """Read the possible answers, keyed by entry id.

        The relevance categories ship no answer file: nothing is compared for
        them beyond whether a call was made.
        """
        if self._answers is None:
            files = self.answer_files()
            if not files:
                self._answers = {}
            else:
                loader = DataLoader()
                source = DataSource(
                    path=BFCL_REPO,
                    data_files=files,
                    split="train",
                    revision=BFCL_REVISION,
                )
                self._answers = {row["id"]: row["ground_truth"] for row in loader.load(source)}
        return self._answers

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        test_category = category_from_id(doc["id"])
        language = language_for_category(test_category)
        functions = [dict(function) for function in doc["function"]]
        prepared = prepare_function_docs(functions, language)
        tools, name_map = build_tool_schemas(prepared)

        # Single-turn categories carry exactly one turn of messages.
        messages = [dict(message) for message in doc["question"][0]]
        user_messages = [m for m in messages if m.get("role") == "user"]
        question = render_messages(user_messages or messages)

        return Instance(
            question=question,
            tools=tools,
            metadata={
                "id": doc["id"],
                "test_category": test_category,
                "language": str(language),
                "messages": messages,
                "functions": functions,
                "prepared_functions": prepared,
                "name_map": name_map,
                "ground_truth": self._load_answers().get(doc["id"], []),
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        """Return curated exemplars rather than instances drawn from the data.

        Exemplars taken from the evaluated set would leak its answers, and a
        sample of them would not reliably cover the shapes a category can take
        — one call, one of several functions, several calls, or none. The set
        in use is named by ``config.fewshot_source``.
        """
        count = self.config.num_fewshot
        if count <= 0:
            return []
        pool = EXEMPLARS[self.language]
        return [exemplar.to_instance(self.language) for exemplar in pool[:count]]

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Decode each reply into the calls the checker compares.

        A reply that cannot be read leaves ``extracted_answer`` unset and
        records why, which the AST categories score as a decoding failure and
        the irrelevance categories score as the model declining to call
        anything.

        The completion regime also accepts the JSON tool-call formats models
        pick up during pretraining, so a base model's reply is judged on which
        function it chose rather than on which surface form it reached for.
        """
        formatter = self.config.formatter
        assert formatter is not None
        from_tool_calls = (
            getattr(type(formatter), "call_source", CallSource.TEXT) == CallSource.TOOL_CALLS
        )
        lenient = getattr(type(formatter), "lenient", False)

        for response in responses:
            language = Language(response.instance.metadata["language"])
            name_map = response.instance.metadata.get("name_map") or {}
            for output in response.outputs:
                try:
                    if from_tool_calls:
                        output.extracted_answer = decode_tool_calls(output.tool_calls, name_map)
                        if not output.extracted_answer and output.text.strip():
                            # Usually a server that was not told to parse tool
                            # calls; worth seeing in the saved predictions.
                            output.metadata["bfcl_no_tool_calls_with_text"] = True
                    elif lenient:
                        output.extracted_answer = decode_text_lenient(output.text, language)
                    else:
                        output.extracted_answer = decode_text(output.text, language)
                except DecodeError as exc:
                    output.extracted_answer = None
                    output.metadata["bfcl_decode_error"] = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#: Categories whose calls are written in Python, in leaderboard order.
NON_LIVE_CATEGORIES: tuple[str, ...] = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "java",
    "javascript",
    "irrelevance",
)

LIVE_AST_CATEGORIES: tuple[str, ...] = (
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
)

LIVE_CATEGORIES: tuple[str, ...] = (
    *LIVE_AST_CATEGORIES,
    "live_irrelevance",
    "live_relevance",
)

#: Task name to the categories it pools. The two pooled tasks reproduce the
#: leaderboard's live summaries, which weight their categories by how many
#: instances each contributes rather than treating them as equals.
TASK_CATEGORIES: dict[str, tuple[str, ...]] = {
    **{f"bfcl_{category}": (category,) for category in NON_LIVE_CATEGORIES},
    **{f"bfcl_{category}": (category,) for category in LIVE_CATEGORIES},
    "bfcl_live_ast": LIVE_AST_CATEGORIES,
    "bfcl_live": LIVE_CATEGORIES,
}

#: Exemplar counts offered as variants; every language has at least this many.
SHOT_COUNTS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def _register_bfcl_tasks() -> None:
    """Create and register one task class per entry in ``TASK_CATEGORIES``."""
    import sys

    module = sys.modules[__name__]

    for task_name, categories in TASK_CATEGORIES.items():
        class_name = "BFCL" + "".join(part.title() for part in task_name.split("_")[1:])
        language = language_for_category(categories[0])
        attrs: dict[str, Any] = {
            "categories": categories,
            "data_source": DataSource(
                path=BFCL_REPO,
                data_files=tuple(f"BFCL_v3_{category}.json" for category in categories),
                split="train",
                revision=BFCL_REVISION,
            ),
            "__module__": __name__,
            "__qualname__": class_name,
            "__doc__": f"BFCL v3 categories: {', '.join(categories)}.",
        }
        if any(category in ("java", "javascript") for category in categories):
            attrs["dependencies"] = list(TREE_SITTER_DEPENDENCIES)

        cls = type(class_name, (BFCLTask,), attrs)
        setattr(module, class_name, cls)
        register(task_name)(cls)

        # BFCL's own prompting mode: functions in a system prompt, calls in text.
        register_variant(
            task_name,
            "prompt",
            formatter=BFCLPromptFormatter(),
            fewshot_source=fewshot_source_for(language),
        )

        # The completion regime for a base model, with exemplars for the format.
        register_variant(
            task_name,
            "base",
            formatter=BFCLCompletionFormatter(),
            sampling_params=COMPLETION_SAMPLING,
            num_fewshot=5,
            fewshot_source=fewshot_source_for(language),
        )

        for count in SHOT_COUNTS:
            register_variant(task_name, f"{count}shot", num_fewshot=count)


_register_bfcl_tasks()

# Registered last: the multi-turn tasks build on this module's helpers.
from . import multi_turn  # noqa: E402,F401
