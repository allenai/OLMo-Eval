"""IFEval-MT: multi-turn instruction following from ``VGraf/ifeval_mt``.

Each row of the dataset is a multi-turn conversation that ends in a user turn
asking the assistant to rewrite/repeat its prior reply under a list of
instructions. ``instruction_id_list`` and ``kwargs`` are scored against the
final assistant reply using the same IFEval verifiers as ``ifeval_ood``.

Two registered variants follow ``IFBENCH_MT_TASKS`` in oe-eval-internal:

- ``ifeval_mt_wildchat_unused_withRewrite`` (HF subset
  ``wildchat_unused_withRewrite``)
- ``ifeval_mt_ood_wildchat_unused_withRewrite`` (HF subset
  ``ood_wildchat_unused_withRewrite``)

Defaults follow the post-training reasoning regime shared by this repo's
other reasoning tasks. Each task's ``:tulu`` variant keeps the ``::tulu``
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
_DATASET_PATH = "VGraf/ifeval_mt"
# max_tokens=None: generate up to the context limit.
_SAMPLING_PARAMS = SamplingParams(
    max_tokens=None,
    temperature=0.6,
    top_p=0.95,
)
_TULU_SAMPLING_PARAMS = SamplingParams(
    max_tokens=2048,
    temperature=0.0,
    do_sample=False,
)


class IFEvalMTBase(Task):
    split = Split.TEST
    metrics = (
        IFEvalPromptStrictAccuracy(),
        IFEvalPromptLooseAccuracy(),
        IFEvalInstStrictAccuracy(),
        IFEvalInstLooseAccuracy(),
    )
    primary_metric = _PRIMARY_METRIC
    strip_thinking = True
    sampling_params = _SAMPLING_PARAMS

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
        messages = tuple({"role": m["role"], "content": m["content"]} for m in doc["messages"])
        return Instance(
            question=prompt,
            gold_answer=None,
            metadata={
                "id": doc.get("id", doc.get("key", index)),
                "key": doc.get("key", doc.get("id", index)),
                "prompt": prompt,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs_list,
                "messages": messages,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=tuple(instance.metadata["messages"]),
        )

    def extract_answer(self, output: LMOutput) -> str:
        return output.text


@register("ifeval_mt_wildchat_unused_withRewrite")
class IFEvalMTWildchatUnusedWithRewrite(IFEvalMTBase):
    data_source = DataSource(path=_DATASET_PATH, subset="wildchat_unused_withRewrite", split="test")


@register("ifeval_mt_ood_wildchat_unused_withRewrite")
class IFEvalMTOODWildchatUnusedWithRewrite(IFEvalMTBase):
    data_source = DataSource(
        path=_DATASET_PATH, subset="ood_wildchat_unused_withRewrite", split="test"
    )


for _task_name in (
    "ifeval_mt_wildchat_unused_withRewrite",
    "ifeval_mt_ood_wildchat_unused_withRewrite",
):
    register_variant(_task_name, "tulu", sampling_params=_TULU_SAMPLING_PARAMS)
