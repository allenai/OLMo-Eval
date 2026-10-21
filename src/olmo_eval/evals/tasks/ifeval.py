"""IFEval: verifiable instruction-following prompts.

Dataset: ``google/IFEval`` (541 prompts), each carrying a list of instruction
IDs and per-instruction kwargs. Verifiers come from the vendored registry in
:mod:`olmo_eval.common.scorers.ifeval_deps`, which covers every instruction ID
used by this dataset.

Mirrors the ``ifeval::hamish_zs_reasoning_deepseek`` configuration in
oe-eval-internal, the regime used for Olmo 3 post-training evaluation: chat
format, sampled decoding (temperature 0.6, top-p 0.95) with generation bounded
only by the model context, primary metric ``prompt_level_loose_acc``.
Reasoning inside ``<think>`` tags is removed before verification, as the
reference harness does for reasoning models.
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
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.extract import extract_think_answer
from olmo_eval.evals.tasks.common import Task, register

_PRIMARY_METRIC = IFEvalPromptLooseAccuracy()


@register("ifeval")
class IFEval(Task):
    data_source = DataSource(path="google/IFEval", split="train")
    split = Split.TRAIN
    metrics = (
        IFEvalPromptStrictAccuracy(),
        IFEvalPromptLooseAccuracy(),
        IFEvalInstStrictAccuracy(),
        IFEvalInstLooseAccuracy(),
    )
    primary_metric = _PRIMARY_METRIC
    sampling_params = SamplingParams(
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
    )
    answer_extractor = extract_think_answer
    strip_thinking = True

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
