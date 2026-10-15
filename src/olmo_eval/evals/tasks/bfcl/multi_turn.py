"""BFCL v3 multi-turn categories.

A multi-turn entry is a conversation against stateful APIs. The model is graded
on what its calls did -- after each turn the involved instances must hold the
state the ground truth path leaves them in -- so a rollout has to be driven and
executed rather than compared with a reference string. The
:mod:`~olmo_eval.harness.scaffolds.bfcl_multi_turn` scaffold drives it, which
means these tasks need a harness configured with that scaffold::

    uv run olmo-eval run -m my-model --harness bfcl_multi_turn -t bfcl_multi_turn_base

Unlike the single-turn categories, an entry carries no function documents. It
names the API classes it involves, and the documents are assembled from the
dataset's own per-class files. One category holds some of them back until a
turn partway through, so they are split here, before the model sees either
half.

multi_turn_composite is not registered: its ground truth calls an older
signature than the other categories do and some of its calls are not valid
Python, so it cannot be executed against the classes any version of the
reference implementation ships. It is also absent from the v3 multi-turn
summary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from olmo_eval.common.formatters import Formatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers.base import Scorer, set_scorer_result
from olmo_eval.common.scorers.bfcl.constants import Language
from olmo_eval.common.scorers.bfcl.multi_turn import (
    CONTEXT_OVERFLOW_ERROR_TYPE,
    CONTEXT_OVERFLOW_METADATA_KEY,
    FORCE_TERMINATED_ERROR_TYPE,
    STEP_BUDGET_METADATA_KEY,
    calls_from_decoded,
    multi_turn_checker,
)
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.bfcl import (
    BFCL_REPO,
    BFCL_REVISION,
    SYSTEM_PROMPT,
    CallSource,
    build_tool_schemas,
    prepare_function_docs,
    render_function_docs,
)
from olmo_eval.evals.tasks.common import Task, register, register_variant

#: Categories the v3 multi-turn summary averages.
MULTI_TURN_CATEGORIES: tuple[str, ...] = (
    "base",
    "miss_func",
    "miss_param",
    "long_context",
)

#: The API class names an entry involves, to the file documenting their
#: functions. The two do not always share a name.
FUNC_DOC_FILES: dict[str, str] = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

MULTI_TURN_SAMPLING = SamplingParams(
    max_tokens=4096,
    temperature=0.001,
    fit_max_tokens_to_context=True,
)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BFCLMultiTurnFormatter(Formatter):
    """Hand the scaffold everything one rollout needs.

    The request carries no conversation of its own: the turns are played out by
    the scaffold, which reads them, and the entry's starting state, from the
    payload.
    """

    call_source: ClassVar[str] = CallSource.TOOL_CALLS
    system_prompt_template: str | None = None

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def format(self, instance: Instance, fewshot: list[Instance] | None = None) -> LMRequest:
        metadata = instance.metadata
        system_prompt = None
        if self.system_prompt_template is not None:
            system_prompt = self.system_prompt_template.format(
                functions=render_function_docs(metadata["prepared_functions"])
            )

        # A regime that writes the functions into the prompt must not also send
        # them as schemas: the model would be given them twice, and its calls
        # would come back in a form this regime does not read.
        sends_schemas = type(self).call_source == CallSource.TOOL_CALLS

        return LMRequest(
            request_type=RequestType.CHAT,
            messages=(({"role": "system", "content": system_prompt},) if system_prompt else ()),
            tools=instance.tools if sends_schemas else None,
            system_prompt=system_prompt,
            metadata={
                "turns": metadata["turns"],
                "initial_config": metadata["initial_config"],
                "involved_classes": metadata["involved_classes"],
                "missed_function": metadata["missed_function"],
                "missed_function_docs": metadata["missed_function_docs"],
                "name_map": metadata["name_map"],
                "language": metadata["language"],
                "long_context": metadata["long_context"],
                "call_source": type(self).call_source,
            },
        )


@dataclass(slots=True)
class BFCLMultiTurnPromptFormatter(BFCLMultiTurnFormatter):
    """Drive the rollout with the functions written into a system prompt."""

    call_source: ClassVar[str] = CallSource.TEXT
    system_prompt_template: str | None = SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BFCLMultiTurnScorer(Scorer):
    """Grade a rollout by replaying its calls against fresh instances."""

    name: str = "bfcl_multi_turn"

    def score(self, instance: Instance, output: LMOutput) -> float:
        rollout = output.extracted_answer
        if not isinstance(rollout, list):
            raise ValueError(
                "A BFCL multi-turn reply has no recorded rollout. These tasks have to run "
                "under a harness with scaffold=bfcl_multi_turn, for example "
                "`--harness bfcl_multi_turn`."
            )

        flags = output.metadata or {}
        recorded = {"step_budget_exhausted": bool(flags.get(STEP_BUDGET_METADATA_KEY))}

        overflow = flags.get(CONTEXT_OVERFLOW_METADATA_KEY)
        if overflow:
            return self._fail(output, recorded, CONTEXT_OVERFLOW_ERROR_TYPE, str(overflow))

        metadata = instance.metadata
        ground_truth = metadata["ground_truth"]
        if len(rollout) != len(ground_truth):
            return self._fail(
                output,
                recorded,
                FORCE_TERMINATED_ERROR_TYPE,
                f"The rollout stopped after {len(rollout)} of {len(ground_truth)} turns.",
            )

        calls = [[calls_from_decoded(step) for step in turn] for turn in rollout]
        result = multi_turn_checker(
            calls,
            ground_truth,
            metadata["initial_config"],
            metadata["involved_classes"],
            metadata["long_context"],
        )
        if not result["valid"]:
            return self._fail(
                output, recorded, result.get("error_type", ""), result.get("error_message", "")
            )
        set_scorer_result(output, self.name, {"valid": True, **recorded})
        return 1.0

    def _fail(
        self, output: LMOutput, recorded: dict[str, Any], error_type: str, error: str
    ) -> float:
        """Record why an entry failed where the saved predictions keep it."""
        set_scorer_result(
            output,
            self.name,
            {"valid": False, "error_type": error_type, "error": error, **recorded},
        )
        return 0.0


BFCL_MULTI_TURN_METRIC = AccuracyMetric(scorer=BFCLMultiTurnScorer)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class BFCLMultiTurnTask(Task):
    """One BFCL v3 multi-turn category."""

    category: str = "base"

    split = Split.TRAIN
    formatter = BFCLMultiTurnFormatter()
    metrics = (BFCL_MULTI_TURN_METRIC,)
    primary_metric = BFCL_MULTI_TURN_METRIC
    sampling_params = MULTI_TURN_SAMPLING

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._answers: dict[str, list[list[str]]] | None = None
        self._func_docs: dict[str, list[dict[str, Any]]] | None = None

    def _load(self, data_files: str | tuple[str, ...]) -> Iterator[dict[str, Any]]:
        return DataLoader().load(
            DataSource(path=BFCL_REPO, data_files=data_files, split="train", revision=BFCL_REVISION)
        )

    def _load_answers(self) -> dict[str, list[list[str]]]:
        if self._answers is None:
            rows = self._load(f"possible_answer/BFCL_v3_multi_turn_{self.category}.json")
            self._answers = {row["id"]: row["ground_truth"] for row in rows}
        return self._answers

    def _function_docs(self, class_name: str) -> list[dict[str, Any]]:
        """Return the documents for one API class, read once per class.

        A multi-turn entry names the classes it involves rather than carrying
        their functions, so the documents come from the dataset's own per-class
        files.
        """
        if self._func_docs is None:
            self._func_docs = {}
        if class_name not in self._func_docs:
            self._func_docs[class_name] = list(
                self._load(f"multi_turn_func_doc/{FUNC_DOC_FILES[class_name]}")
            )
        return self._func_docs[class_name]

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def _split_held_back(
        self, functions: list[dict[str, Any]], missed: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Take the functions a turn holds back out of the visible set.

        The entry names them; the documents have to be found and removed here,
        so that the model is not shown a function before the turn that offers
        it.
        """
        by_name = {doc["name"]: doc for doc in functions}
        held: dict[str, list[dict[str, Any]]] = {}
        withheld_names: set[str] = set()

        for turn_index, names in (missed or {}).items():
            docs = [by_name[name] for name in names if name in by_name]
            if docs:
                held[str(turn_index)] = docs
                withheld_names.update(doc["name"] for doc in docs)

        visible = [doc for doc in functions if doc["name"] not in withheld_names]
        return visible, held

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        functions: list[dict[str, Any]] = []
        for class_name in doc["involved_classes"]:
            functions.extend(self._function_docs(class_name))

        visible, held_back = self._split_held_back(functions, doc.get("missed_function") or {})
        prepared = prepare_function_docs(visible, Language.PYTHON)
        tools, name_map = build_tool_schemas(prepared)

        # The held back documents are prepared the same way, and their names
        # join the same map, so a call made after they are offered still reads
        # back under the name the dataset gives it.
        held_schemas: dict[str, list[dict[str, Any]]] = {}
        held_docs: dict[str, str] = {}
        for turn_index, docs in held_back.items():
            prepared_held = prepare_function_docs(docs, Language.PYTHON)
            schemas, held_map = build_tool_schemas(prepared_held)
            name_map.update(held_map)
            held_schemas[turn_index] = [schema.to_openai() for schema in schemas]
            # A prompted model reads the functions it is offered in the form the
            # system prompt used for the rest, not in the schema the tool API takes.
            held_docs[turn_index] = render_function_docs(prepared_held)

        turns: list[list[dict[str, Any]]] = [
            [dict(message) for message in turn] for turn in doc["question"]
        ]
        first_user = next(
            (m["content"] for turn in turns for m in turn if m.get("role") == "user"), ""
        )

        return Instance(
            question=first_user,
            tools=tools,
            metadata={
                "id": doc["id"],
                "test_category": f"multi_turn_{self.category}",
                "language": str(Language.PYTHON),
                "turns": turns,
                "initial_config": doc["initial_config"],
                "involved_classes": list(doc["involved_classes"]),
                "missed_function": held_schemas,
                "missed_function_docs": held_docs,
                "prepared_functions": prepared,
                "name_map": name_map,
                "long_context": self.category == "long_context",
                "ground_truth": self._load_answers().get(doc["id"], []),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance)

    def _extract_answers(self, responses: Any) -> None:
        """Leave the rollout the scaffold recorded in place."""
        return None


def _register_multi_turn_tasks() -> None:
    import sys

    module = sys.modules[__name__]

    for category in MULTI_TURN_CATEGORIES:
        task_name = f"bfcl_multi_turn_{category}"
        class_name = "BFCLMultiTurn" + "".join(part.title() for part in category.split("_"))
        cls = type(
            class_name,
            (BFCLMultiTurnTask,),
            {
                "category": category,
                "data_source": DataSource(
                    path=BFCL_REPO,
                    data_files=f"BFCL_v3_multi_turn_{category}.json",
                    split="train",
                    revision=BFCL_REVISION,
                ),
                "__module__": __name__,
                "__qualname__": class_name,
                "__doc__": f"BFCL v3 multi_turn_{category}.",
            },
        )
        setattr(module, class_name, cls)
        register(task_name)(cls)
        register_variant(task_name, "prompt", formatter=BFCLMultiTurnPromptFormatter())


_register_multi_turn_tasks()
