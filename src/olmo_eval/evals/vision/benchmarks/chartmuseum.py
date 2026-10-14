"""ChartMuseum (https://arxiv.org/abs/2505.13444): chart QA that needs visual reasoning.

Real-world charts with expert-written questions and short answers. Each question carries the
reasoning skill it primarily needs: ``visual``, ``synthesis``, ``visual/text`` or ``text``.

* ``chartmuseum_visual`` — the 510 ``visual`` questions of the ``test`` split, the
  "Visual" column of the leaderboard (https://chartmuseum-leaderboard.github.io).
* ``chartmuseum`` — the full 1,000-question ``test`` split; ``overall`` is the leaderboard's
  "Overall", reported with one accuracy per reasoning type.

The protocol is the official repository's (https://github.com/Liyan06/ChartMuseum): the
chain-of-thought QA prompt from ``prompt.py`` in a single user turn with the chart, the short
answer taken from the ``<answer>`` span (an unclosed span still counts; no span is an empty
answer), and ``gpt-4.1-mini-2025-04-14`` at temperature 0 judging it against the reference with
the official equivalence prompt. Grading needs ``OPENAI_API_KEY``; judge replies are cached
under ``CHARTMUSEUM_JUDGE_CACHE_DIR`` (or a temp dir). Metrics are 0-1 (the leaderboard
reports x100).

Deviations and choices the official code leaves open:

* Decoding is greedy with up to 8,192 new tokens. The paper uses temperature 0 for
  non-reasoning models and each model's default token limit.
* A judge call that fails after every retry scores 0 and is counted by ``n_invalid``; the
  official script (via ``curator``) has no such accounting.
* ``answer_tag_rate`` reports how often the response contains an ``<answer>`` tag, since a
  model that ignores the format scores 0 under the official extraction.

Data comes from the Hub (``lytang/ChartMuseum``) at a pinned revision, images included; set
``CHARTMUSEUM_DIR`` to a local copy of the dataset repository to read it from disk instead.

Parity: rescoring the official repository's Claude-3.7-Sonnet dev outputs gives 0.6296,
the README's expected number. ``gpt-4o-2024-11-20`` (via LiteLLM) scores 0.349 on the
visual split against 0.318 in the paper (about 1.5 standard errors). Qwen3-VL-4B / 8B at
their default sampling settings score 0.237 / 0.261 against the leaderboard's 0.253 / 0.278;
greedy decoding scores 0.216 / 0.259.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.chartmuseum import (
    ChartMuseumJudgeScorer,
    build_qa_prompt,
    extract_answer,
)
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

HF_REPO = "lytang/ChartMuseum"
HF_REVISION = "462d46deb187d8a40c5a9de4e69e14f1df982e58"

#: Reasoning types in the leaderboard's column order.
REASONING_TYPES: tuple[str, ...] = ("visual", "synthesis", "visual/text", "text")

_SPLITS = {Split.TEST: "test", Split.VALIDATION: "dev"}

_SCORER = ChartMuseumJudgeScorer()


def _result_for(response: Response) -> dict | None:
    for output in response.outputs:
        if output.metadata and "chartmuseum_result" in output.metadata:
            return output.metadata["chartmuseum_result"]
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class ChartMuseumAccuracyMetric(Metric):
    """Judge accuracy, optionally over one reasoning type (a failed grade counts as wrong)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    reasoning_type: str | None = None

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        return _mean(values)

    def compute_instance(self, response: Response) -> float | None:
        """The example's graded score, or ``None`` outside this reasoning type."""
        if (
            self.reasoning_type is not None
            and response.instance.metadata["reasoning_type"] != self.reasoning_type
        ):
            return None
        result = _result_for(response)
        if result is None:
            return None
        return float(result["equal"] or 0)

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class ChartMuseumInvalidCountMetric(Metric):
    """Number of examples the judge never graded."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        return float(sum(v for r in responses if (v := self.compute_instance(r)) is not None))

    def compute_instance(self, response: Response) -> float | None:
        """1.0 when this example's grading failed, 0.0 when it graded."""
        result = _result_for(response)
        if result is None:
            return None
        return 1.0 if result["equal"] is None else 0.0

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True)
class ChartMuseumAnswerTagRateMetric(Metric):
    """Fraction of responses that contain an ``<answer>`` tag."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        return _mean(values)

    def compute_instance(self, response: Response) -> float | None:
        result = _result_for(response)
        if result is None:
            return None
        return 1.0 if result["has_answer_tag"] else 0.0

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


def _type_metric_name(reasoning_type: str) -> str:
    return reasoning_type.replace("/", "_")


_OVERALL = ChartMuseumAccuracyMetric(name="overall", scorer=_SCORER)
_PER_TYPE = tuple(
    ChartMuseumAccuracyMetric(name=_type_metric_name(t), scorer=_SCORER, reasoning_type=t)
    for t in REASONING_TYPES
)
_DIAGNOSTICS = (
    ChartMuseumInvalidCountMetric(name="n_invalid", scorer=_SCORER),
    ChartMuseumAnswerTagRateMetric(name="answer_tag_rate", scorer=_SCORER),
)


def _data_dir() -> Path:
    local = os.environ.get("CHARTMUSEUM_DIR")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            revision=HF_REVISION,
            allow_patterns=["data/*", "images/*"],
        )
    )


@register("chartmuseum")
class ChartMuseumTask(ImageQATask):
    #: The judge calls OpenAI; a grading failure is counted by `n_invalid` rather than
    #: silently scored as a wrong answer.
    dependencies = ["pillow", "openai", "huggingface-hub", "datasets"]
    required_secrets = ("OPENAI_API_KEY",)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    metrics = (_OVERALL, *_PER_TYPE, *_DIAGNOSTICS)
    primary_metric = _OVERALL
    split = Split.TEST
    #: The reasoning types this task keeps; ``None`` keeps the whole split.
    reasoning_type: str | None = None

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        split = _SPLITS.get(self.config.split)
        if split is None:
            raise ValueError(f"ChartMuseum has no {self.config.split.value} split")
        root = _data_dir()
        ds = datasets.load_dataset(
            "parquet", data_files={split: str(root / "data" / f"{split}-*.parquet")}, split=split
        )
        for idx in range(len(ds)):
            ex = ds[idx]
            if self.reasoning_type is not None and ex["reasoning_type"] != self.reasoning_type:
                continue
            image_path = root / ex["image"]
            if not image_path.is_file():
                raise FileNotFoundError(f"ChartMuseum image {image_path} is missing")
            yield Instance(
                question=build_qa_prompt(ex["question"]),
                gold_answer=ex["answer"],
                metadata={
                    "example_id": ex["hash"],
                    "raw_question": ex["question"],
                    "answer": ex["answer"],
                    "reasoning_type": ex["reasoning_type"],
                    "source": ex["source"],
                    "image_path": str(image_path),
                },
            )

    def extract_answer(self, output: LMOutput) -> str:
        return extract_answer(output.text or "")


_VISUAL = next(m for m in _PER_TYPE if m.reasoning_type == "visual")


@register("chartmuseum_visual")
class ChartMuseumVisualTask(ChartMuseumTask):
    metrics = (_VISUAL, *_DIAGNOSTICS)
    primary_metric = _VISUAL
    reasoning_type = "visual"
