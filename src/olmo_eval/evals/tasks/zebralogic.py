"""ZebraLogic: Logic puzzle evaluation task.

ZebraLogic (https://huggingface.co/blog/yuchenlin/zebra-logic) is a constraint
satisfaction benchmark based on grid puzzles. Part of "ZeroEval"
(https://github.com/WildEval/ZeroEval).
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    Response,
    SamplingParams,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common.base import Task
from olmo_eval.evals.tasks.common.registry import register, register_variant

ZEBRA_GRID = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""

EASY_SIZES = ["2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3"]

_ZEBRA_CACHE_KEY = "_zebralogic_scores"


def extract_last_complete_json(s: str) -> dict | None:
    """Extract the last complete JSON object from a string.

    Source: https://github.com/WildEval/ZeroEval/blob/main/src/evaluation/eval_utils.py#L110
    """
    stack = []
    last_json_start = None
    last_json_str = None
    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    last_json_str = s[last_json_start : i + 1]
                    last_json_start = None
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    return None


def _evaluate_zebralogic(
    output_text: str, solution_table: dict, total_cells: int
) -> dict[str, float]:
    """Parse ZebraLogic output and compute puzzle accuracy, cell accuracy, and parse rate."""
    prediction_table = extract_last_complete_json(output_text)
    if prediction_table is None:
        return {"parsed": 0.0, "puzzle_accuracy": 0.0, "cell_accuracy": 0.0}
    try:
        prediction_table = prediction_table.get("solution", {}) or {}
        if not isinstance(prediction_table, dict) or not all(
            house in prediction_table for house in solution_table
        ):
            return {"parsed": 0.0, "puzzle_accuracy": 0.0, "cell_accuracy": 0.0}
        this_correct_cells = 0
        for house in solution_table:
            for column in solution_table[house]:
                if (
                    isinstance(prediction_table, dict)
                    and house in prediction_table
                    and isinstance(prediction_table[house], dict)
                    and column in prediction_table[house]
                ):
                    truth_cell = solution_table[house][column].lower().strip()
                    predicted = prediction_table[house][column]
                    if predicted is None or (
                        isinstance(predicted, (str, list)) and len(predicted) == 0
                    ):
                        continue
                    # unwrap nested dicts before checking list or str
                    while isinstance(predicted, dict):
                        if not predicted:
                            break
                        predicted = list(predicted.values())[0]
                    if isinstance(predicted, list):
                        if not predicted or not isinstance(predicted[0], str):
                            continue
                        predicted_cell = predicted[0].lower().strip()
                    elif isinstance(predicted, str):
                        predicted_cell = predicted.lower().strip()
                    else:
                        continue
                    if truth_cell == predicted_cell:
                        this_correct_cells += 1
        return {
            "parsed": 1.0,
            "puzzle_accuracy": 1.0 if this_correct_cells == total_cells else 0.0,
            "cell_accuracy": this_correct_cells / total_cells,
        }
    except Exception:
        return {"parsed": 1.0, "puzzle_accuracy": 0.0, "cell_accuracy": 0.0}


def _get_zebralogic_scores(instance: Instance, output: LMOutput) -> dict[str, float]:
    """Evaluate ZebraLogic output, caching the result in output.metadata."""
    if _ZEBRA_CACHE_KEY not in output.metadata:
        solution_table = instance.metadata.get("solution_table", {})
        total_cells = instance.metadata.get("total_cells", 1)
        output.metadata[_ZEBRA_CACHE_KEY] = _evaluate_zebralogic(
            output.text, solution_table, total_cells
        )
    return output.metadata[_ZEBRA_CACHE_KEY]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ZebraLogicPuzzleAccuracyScorer(Scorer):
    """Scores 1.0 if all cells in the predicted solution table are correct."""

    name: str = "puzzle_accuracy"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return _get_zebralogic_scores(instance, output)["puzzle_accuracy"]


@dataclass(frozen=True, slots=True)
class ZebraLogicCellAccuracyScorer(Scorer):
    """Scores the fraction of correctly predicted cells in the solution table."""

    name: str = "cell_accuracy"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return _get_zebralogic_scores(instance, output)["cell_accuracy"]


@dataclass(frozen=True, slots=True)
class ZebraLogicParsedScorer(Scorer):
    """Scores 1.0 if the model output contains a parseable JSON solution."""

    name: str = "parsed"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return _get_zebralogic_scores(instance, output)["parsed"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PuzzleAccuracyMetric(Metric):
    """Mean puzzle accuracy: fraction of puzzles where all cells are correct."""

    name: str = "puzzle_accuracy"
    scorer: type[Scorer] = ZebraLogicPuzzleAccuracyScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        return sum(r.scores.get(scorer_name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True, slots=True)
class CellAccuracyMetric(Metric):
    """Mean cell accuracy: average fraction of correctly predicted cells per puzzle."""

    name: str = "cell_accuracy"
    scorer: type[Scorer] = ZebraLogicCellAccuracyScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        return sum(r.scores.get(scorer_name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True, slots=True)
class ParsedMetric(Metric):
    """Mean parse rate: fraction of outputs that contain valid JSON."""

    name: str = "parsed"
    scorer: type[Scorer] = ZebraLogicParsedScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        return sum(r.scores.get(scorer_name, 0.0) for r in responses) / len(responses)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@register("zebralogic")
class ZebraLogic(Task):
    """ZebraLogic grid puzzle evaluation task."""

    #: When set, keep only docs of this difficulty ("easy" or "hard").
    _difficulty: str | None = None

    data_source = DataSource(
        path="allenai/ZebraLogicBench-private", subset="grid_mode", split="test"
    )
    metrics = (PuzzleAccuracyMetric(), CellAccuracyMetric(), ParsedMetric())
    primary_metric = PuzzleAccuracyMetric()
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        stop_sequences=("Problem:", "\n\n"),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        size = doc["size"]
        puzzle = doc["puzzle"]
        solution = doc["solution"]

        # Build the JSON template for the model to fill in
        json_template: dict = {"reasoning": "___", "solution": {}}
        num_houses = len(solution["rows"])
        columns = solution["header"]
        for i in range(num_houses):
            json_template["solution"][f"House {i + 1}"] = {
                columns[j]: "___" for j in range(1, len(columns))
            }
        json_str = json.dumps(json_template, indent=4)

        prompt_str = ZEBRA_GRID.replace("{puzzle}", puzzle).replace("{json_template}", json_str)

        # Build the ground-truth solution table
        assert columns[0] == "House"
        solution_table: dict[str, dict[str, str]] = {}
        total_cells = 0
        for i in range(num_houses):
            solution_table[f"House {i + 1}"] = {
                columns[j]: solution["rows"][i][j] for j in range(1, len(columns))
            }
            total_cells += len(columns) - 1

        difficulty = "easy" if size in EASY_SIZES else "hard"
        if self._difficulty is not None and difficulty != self._difficulty:
            return None

        return Instance(
            question=prompt_str,
            metadata={
                "id": doc.get("index", index),
                "size": size,
                "difficulty": difficulty,
                "solution_table": solution_table,
                "total_cells": total_cells,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )


@register("zebralogic_easy")
class ZebraLogicEasy(ZebraLogic):
    """ZebraLogic restricted to easy puzzles (grid sizes 2*2 through 3*3)."""

    _difficulty = "easy"


@register("zebralogic_hard")
class ZebraLogicHard(ZebraLogic):
    """ZebraLogic restricted to hard puzzles (grid sizes 3*4 through 6*6)."""

    _difficulty = "hard"


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

# Chat variant for instruct and reasoning models. Mirrors oe-eval's
# `zebralogic::olmo3:adapt` regime: sampled decoding, generation bounded only
# by the model's context (``max_tokens=None``), and no stop sequences (the
# chat template supplies end-of-turn).
for _task_name in ("zebralogic", "zebralogic_easy", "zebralogic_hard"):
    register_variant(
        _task_name,
        "chat",
        formatter=ChatFormatter(),
        sampling_params=SamplingParams(
            max_tokens=None,
            temperature=0.6,
            top_p=0.95,
        ),
    )
