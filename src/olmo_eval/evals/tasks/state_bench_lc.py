"""Long-context state-tracking tasks from StateBench."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register

STATE_BENCH_REPO = "allenai/state-bench"
STATE_BENCH_REVISION = "683d249128a0d387eda27c8cfc8c1f9df0389495"

STATE_BENCH_CONFIGS = (
    "cube-painting--aperiodic",
    "cube-painting--periodic",
    "cube-painting--r-trivial",
    "integer-code--aperiodic",
    "integer-code--periodic",
    "integer-code--r-trivial",
    "people-in-rooms--aperiodic",
    "people-in-rooms--periodic",
    "people-in-rooms--r-trivial",
    "ruler--aperiodic",
    "ruler--periodic",
    "ruler--r-trivial",
    "spreadsheet-cells--aperiodic",
    "spreadsheet-cells--periodic",
    "spreadsheet-cells--r-trivial",
    "status-lights--aperiodic",
    "status-lights--periodic",
    "status-lights--r-trivial",
)

STATE_BENCH_TOKEN_STRATA = (
    "tokens_under_2k",
    "tokens_4k",
    "tokens_8k",
    "tokens_16k",
    "tokens_32k",
    "tokens_64k",
    "tokens_128k",
    "tokens_256k",
    "tokens_512k",
    "tokens_1m",
)

STATE_BENCH_CONFIGS_SET = frozenset(STATE_BENCH_CONFIGS)

# Token strata are half-open [min, max) token ranges, and the dataset omits a stratum from a
# config when that config has no rows inside it. A task pointing at an omitted split fails at
# dataset load, so strata are enumerated per config rather than as a full cross product.
#
# Generation caps the number of extra assignments, and that cap is sized against the most
# token-dense formatter, so sparser formatters top out at shorter contexts. Only the configs
# below reach the `tokens_1m` floor; strata absent from this mapping exist for every config.
STATE_BENCH_CONFIGS_BY_STRATUM: dict[str, frozenset[str]] = {
    "tokens_1m": frozenset(
        {
            "cube-painting--periodic",
            "integer-code--aperiodic",
            "integer-code--periodic",
            "integer-code--r-trivial",
            "ruler--aperiodic",
            "ruler--periodic",
            "ruler--r-trivial",
            "spreadsheet-cells--periodic",
            "status-lights--periodic",
        }
    ),
}


def state_bench_configs_for_stratum(token_stratum: str) -> tuple[str, ...]:
    """Return the dataset configs that have a staged split for a token stratum."""
    available = STATE_BENCH_CONFIGS_BY_STRATUM.get(token_stratum, STATE_BENCH_CONFIGS_SET)
    return tuple(config_name for config_name in STATE_BENCH_CONFIGS if config_name in available)


def state_bench_strata_for_config(config_name: str) -> tuple[str, ...]:
    """Return the token strata that a dataset config has staged splits for."""
    return tuple(
        token_stratum
        for token_stratum in STATE_BENCH_TOKEN_STRATA
        if config_name in STATE_BENCH_CONFIGS_BY_STRATUM.get(token_stratum, STATE_BENCH_CONFIGS_SET)
    )


STATE_BENCH_STRATA_BY_CONFIG = {
    config_name: state_bench_strata_for_config(config_name) for config_name in STATE_BENCH_CONFIGS
}

# (config, stratum) pairs deliberately not registered, for coverage reporting.
STATE_BENCH_OMITTED_SPLITS = tuple(
    (config_name, token_stratum)
    for token_stratum in STATE_BENCH_TOKEN_STRATA
    for config_name in STATE_BENCH_CONFIGS
    if token_stratum not in STATE_BENCH_STRATA_BY_CONFIG[config_name]
)


def state_bench_task_name(config_name: str, token_stratum: str) -> str:
    """Return the task name for a dataset config and token-length stratum."""
    normalized = config_name.replace("--", "_").replace("-", "_")
    return f"state_bench_{normalized}_{token_stratum}"


def state_bench_10pct_task_name(config_name: str, token_stratum: str) -> str:
    """Return the 10%-sample task name for a dataset config and stratum."""
    return f"{state_bench_task_name(config_name, token_stratum)}_10pct"


STATE_BENCH_TASKS = tuple(
    state_bench_task_name(config_name, token_stratum)
    for config_name in STATE_BENCH_CONFIGS
    for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
)

STATE_BENCH_10PCT_TASKS = tuple(
    state_bench_10pct_task_name(config_name, token_stratum)
    for config_name in STATE_BENCH_CONFIGS
    for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
)


class StateBench(Task):
    """Rank candidate final states after a long sequence of assignments.

    The initial state appears at the start of the prompt, so a prompt longer than the
    model's context is unanswerable once the provider left-truncates it. Run only the
    token strata that fit the model's context; logprob providers record the number of
    dropped prompt tokens as ``prompt_truncated_tokens`` in each output's metadata.
    """

    metrics = (LogprobPerTokenMCAccuracyMetric(),)
    dataset_split: str

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached(split=self.dataset_split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        prefix = doc.get("prefix")
        choices = doc.get("choices")
        gold_idx = doc.get("correct_choice_index")
        if not isinstance(prefix, str) or not isinstance(choices, list) or not choices:
            return None
        if not isinstance(gold_idx, int) or not 0 <= gold_idx < len(choices):
            return None
        if not all(isinstance(choice, str) for choice in choices):
            return None

        gold_answer = choices[gold_idx]
        return Instance(
            question=prefix,
            gold_answer=gold_answer,
            choices=tuple(choices),
            metadata={
                "id": doc.get("example_id", index),
                "instance_id": doc.get("instance_id"),
                "index": index,
                "dataset": "state_bench",
                "formatter": doc.get("formatter"),
                "complexity": doc.get("complexity"),
                "gold_idx": gold_idx,
                "gold_text": gold_answer,
                "num_variables": doc.get("num_variables"),
                "num_values": doc.get("num_values"),
                "num_assignments": doc.get("num_assignments"),
                "num_extra_assignments": doc.get("num_extra_assignments"),
                "target_extra_assignments": doc.get("target_extra_assignments"),
                "filler_proportion": doc.get("filler_proportion"),
                "actual_filler_proportion": doc.get("actual_filler_proportion"),
                "filler_distraction": doc.get("filler_distraction"),
                "num_filler_lines": doc.get("num_filler_lines"),
                "token_length": doc.get("token_length"),
                "token_length_is_estimate": doc.get("token_length_is_estimate"),
                "token_stratum": doc.get("token_stratum"),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=tuple(f" {choice}" for choice in (instance.choices or ())),
            max_length=self.config.max_length,
        )


class StateBench10Percent(StateBench):
    """StateBench task using a deterministic 10% sample of the evaluation split."""

    @property
    def instances(self) -> Iterator[Instance]:
        instances = list(self._load_instances_cached(split=self.dataset_split))
        sample_size = math.ceil(len(instances) / 10)
        yield from random.Random(42).sample(instances, sample_size)


for _config_name in STATE_BENCH_CONFIGS:
    for _token_stratum in STATE_BENCH_STRATA_BY_CONFIG[_config_name]:
        _name = state_bench_task_name(_config_name, _token_stratum)
        _class_name = (
            "StateBench_" + _config_name.title().replace("-", "_") + "_" + _token_stratum.title()
        )
        _dataset_split = f"test__{_token_stratum}"
        _class = type(
            _class_name,
            (StateBench,),
            {
                "data_source": DataSource(
                    STATE_BENCH_REPO,
                    subset=_config_name,
                    split=_dataset_split,
                    revision=STATE_BENCH_REVISION,
                ),
                "dataset_split": _dataset_split,
                "__module__": __name__,
                "__qualname__": _class_name,
            },
        )
        globals()[_class_name] = register(_name)(_class)

        _sample_name = state_bench_10pct_task_name(_config_name, _token_stratum)
        _sample_class_name = f"{_class_name}10Percent"
        _sample_class = type(
            _sample_class_name,
            (StateBench10Percent,),
            {
                "data_source": DataSource(
                    STATE_BENCH_REPO,
                    subset=_config_name,
                    split=_dataset_split,
                    revision=STATE_BENCH_REVISION,
                ),
                "dataset_split": _dataset_split,
                "__module__": __name__,
                "__qualname__": _sample_class_name,
            },
        )
        globals()[_sample_class_name] = register(_sample_name)(_sample_class)
