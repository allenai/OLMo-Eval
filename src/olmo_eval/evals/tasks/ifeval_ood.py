"""IFEval OOD: out-of-distribution instruction-following slice of IFBench.

Dataset: ``allenai/IFBench_test`` (300 prompts), each carrying a list of
instruction IDs and per-instruction kwargs. Verifiers come from the vendored
registry in :mod:`olmo_eval.common.scorers.ifeval_deps`.

Defaults follow the post-training reasoning regime shared by this repo's
other reasoning tasks, and reasoning inside ``<think>`` tags is removed
before verification. The ``:tulu`` variant keeps the ``ifeval_ood::tulu``
regime from oe-eval-internal.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import (
    IFEvalInstLooseAccuracy,
    IFEvalInstStrictAccuracy,
    IFEvalPromptLooseAccuracy,
    IFEvalPromptStrictAccuracy,
)
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

_PRIMARY_METRIC = IFEvalPromptLooseAccuracy()


@register("ifeval_ood")
class IFEvalOOD(Task):
    data_source = DataSource(path="allenai/IFBench_test", split="train")
    split = Split.TRAIN
    metrics = (
        IFEvalPromptStrictAccuracy(),
        IFEvalPromptLooseAccuracy(),
        IFEvalInstStrictAccuracy(),
        IFEvalInstLooseAccuracy(),
    )
    primary_metric = _PRIMARY_METRIC
    strip_thinking = True
    # max_tokens=None: generate up to the context limit.
    sampling_params = SamplingParams(
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        prompt = doc["prompt"]
        instruction_id_list = list(doc.get("instruction_id_list") or [])
        raw_kwargs = doc.get("kwargs") or []
        kwargs_list = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in raw_kwargs]
        return Instance(
            question=prompt,
            gold_answer=None,
            metadata={
                "id": doc.get("key", index),
                "key": doc.get("key", index),
                "prompt": prompt,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs_list,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def extract_answer(self, output: LMOutput) -> str:
        return output.text


register_variant(
    "ifeval_ood",
    "tulu",
    sampling_params=SamplingParams(max_tokens=2048, temperature=0.0, do_sample=False),
)
