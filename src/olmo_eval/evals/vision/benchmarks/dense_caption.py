"""Pixmo-cap dense-caption evaluation task.

Loads the pixmo-cap test split (2730 images), joins judge reference data from
two on-disk sources, runs a GPT-4o recall+consistency judge, and aggregates
into recall / recall_at_10 / consistency / num_statements / avg metrics.

Environment variables (optional — defaults point to Weka paths):
    DENSE_CAPTION_EVAL_DIR   root of dense_caption_eval/ (contains final-data.json,
                             mturk-eval-statements/, gpt4-cache/)
    MOLMO_DATA_DIR           parent of torch_datasets/ (same convention as the 11 image-QA
                             tasks); images live at torch_datasets/pixmo_images/ and the split
                             manifest at torch_datasets/pixmo_datasets/dense-caption-eval/test.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, Response, SamplingParams
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.paths import torch_datasets_dir
from olmo_eval.evals.vision.scoring.judges import DenseCaptionJudgeScorer
from olmo_eval.evals.vision.scoring.prompt_templates import dense_caption_question
from olmo_eval.evals.vision.scoring.prompts import (
    NO_PREFIX_STYLES,
    STYLE_PREFIX_STYLES,
)
from olmo_eval.evals.vision.tasks.base import VisionTask

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_DIR = "/weka/oe-training-default/mm-olmo/dense_caption_eval"

# Shared scorer instance — all 5 metrics hold a reference so _get_scorers()
# deduplicates to a single GPT-judge call per example (via Scorer.__call__).
_JUDGE = DenseCaptionJudgeScorer()


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


def _judge_result(response: Response) -> dict | None:
    """The judge result for a response's single output (multi-sample runs are rejected)."""
    for output in response.outputs:
        if output.metadata and "dense_caption_result" in output.metadata:
            return output.metadata["dense_caption_result"]
    return None


@dataclass(frozen=True, slots=True)
class DenseCaptionMetric(Metric):
    """Mean of one judge field over the examples where that side of the judge is valid.

    Values keep mm_olmo's legacy scale (recall/consistency/recall-at-10 x100,
    ``num_statements`` raw), so the viewer must not re-scale them: the display
    format is pinned to raw units. Per-instance storage returns the same field on
    the same scale, and ``None`` for examples the aggregate excludes as invalid.
    """

    name: str  # type: ignore[misc]
    scorer: type[Scorer] | Scorer = _JUDGE
    field_name: str = "recall"
    valid_key: str = "recall_valid"
    scale: float = 100.0

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [v for v in (self.compute_instance(r) for r in responses) if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        result = _judge_result(response)
        if result is None or not result.get(self.valid_key):
            return None
        return float(result[self.field_name]) * self.scale

    def supports_scorer_fallback(self) -> bool:
        # The scorer channel is recall; it is not this metric's per-instance value.
        return False

    def display_format(self) -> str:
        return "raw"


@dataclass(frozen=True, slots=True)
class DenseCaptionAvgMetric(Metric):
    """Primary metric: (mean_recall + mean_consistency) / 2 x100.

    The two means run over different valid subsets, so the aggregate has no exact
    per-instance decomposition; per-instance storage is disabled rather than
    approximated.
    """

    name: str = "avg"
    scorer: type[Scorer] | Scorer = _JUDGE

    def compute(self, responses: Sequence[Response]) -> float:
        results = [r for r in (_judge_result(response) for response in responses) if r is not None]
        recall_vals = [r["recall"] for r in results if r.get("recall_valid")]
        cons_vals = [r["consistency"] for r in results if r.get("consistency_valid")]
        mean_recall = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0
        mean_cons = sum(cons_vals) / len(cons_vals) if cons_vals else 0.0
        return (mean_recall + mean_cons) / 2.0 * 100

    def compute_instance(self, response: Response) -> float | None:
        return None

    def supports_scorer_fallback(self) -> bool:
        return False

    def display_format(self) -> str:
        return "raw"


_DEFAULT_METRICS: tuple[Metric, ...] = (
    DenseCaptionMetric(name="recall", field_name="recall", valid_key="recall_valid"),
    DenseCaptionMetric(name="consistency", field_name="consistency", valid_key="consistency_valid"),
    DenseCaptionMetric(name="recall_at_10", field_name="recall_at_10", valid_key="recall_valid"),
    # mm_olmo's reported num_statements: canonical statements GPT derives from the
    # *model caption* during the consistency judge, not the recall-side mturk count.
    DenseCaptionMetric(
        name="num_statements",
        field_name="consistency_num_statements",
        valid_key="consistency_valid",
        scale=1.0,
    ),
    DenseCaptionAvgMetric(),
)
_AVG_METRIC = DenseCaptionAvgMetric()


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@register("dense_caption")
class DenseCaptionEval(VisionTask):
    """Pixmo-cap dense-caption GPT-judge evaluation.

    Data comes from three on-disk sources:
    * ``final-data.json`` — whisper transcripts (consistency reference)
    * ``mturk-eval-statements/{sha256(url)}.json`` — canonical statements
      (recall reference)
    * ``torch_datasets/pixmo_datasets/dense-caption-eval/test.jsonl`` — image
      paths and URLs

    The model inference request is a CHAT message whose text reproduces mm_olmo's
    per-example seeded ``long_caption`` template (the released Molmo2-4B uses
    ``demo_or_style_v2`` + ``uber_model_v2``: no style prefix, template sampled by line
    index). The local image path is stored in ``instance.metadata["image_path"]``.
    """

    sampling_params = SamplingParams(temperature=0.0, max_tokens=448)
    metrics = _DEFAULT_METRICS
    primary_metric = _AVG_METRIC
    #: Image decoding builds the instances; the GPT judge needs an OpenAI client.
    #: The judge needs the OpenAI client on top of the base image decoding.
    dependencies = [*VisionTask.dependencies, "openai"]
    #: The judge calls OpenAI; beaker mounts this as the user-scoped secret.
    required_secrets = ("OPENAI_API_KEY",)
    # None => decide from the checkpoint's prompt family (see `_question`). Set to a fixed
    # string on a subclass to force one prompt regardless of family.
    caption_prompt: str | None = None
    #: Prompt family assumed when the run does not say. Matches the instruction-tuned
    #: checkpoints, mirroring `ModelPromptPointingTask`'s defaults.
    default_system_prompt_style = "demo_or_style_v2"

    def _question(self, idx: int) -> str:
        """The caption prompt for example ``idx``, following the checkpoint's family.

        The prompt has to follow the checkpoint rather than the task, exactly as for the
        `_mp` pointing tasks: a stage-1 checkpoint (OLMo-core's `Molmo2-Stage1.py`,
        `system_prompt="style_and_length_v2"`) was trained with the caption's style tag as
        its whole user turn, `long_caption:`, with no instruction and no length number
        (OLMo-core #875). Handing such a checkpoint a natural-language instruction instead
        measures it out of distribution: on a 4B stage-1 run it cost 7.1 recall points
        (38.18 -> 45.28) and made 3.4% of answers come back as `<points .../>` markup
        rather than prose.

        So `-o system_prompt_style=style_and_length_v2` selects the bare tag here, the same
        override that selects the stage-1 prompt for the pointing and OCR tasks.
        """
        if self.caption_prompt is not None:
            return self.caption_prompt
        style = self.config.system_prompt_style or self.default_system_prompt_style
        if style in STYLE_PREFIX_STYLES:
            return "long_caption:"
        if style in NO_PREFIX_STYLES:
            return dense_caption_question(idx)
        raise ValueError(
            f"Unsupported system_prompt_style {style!r} for dense_caption; "
            f"expected one of {NO_PREFIX_STYLES + STYLE_PREFIX_STYLES}"
        )

    def _build_instances(self) -> Iterator[Instance]:
        # `config.limit` is applied here rather than left to `VisionTask.instances`
        # so the raw-line index driving the seeded prompt never shifts; the base
        # class's slice then has nothing left to remove.
        eval_dir = Path(os.environ.get("DENSE_CAPTION_EVAL_DIR", _DEFAULT_EVAL_DIR))
        if not eval_dir.is_dir():
            raise FileNotFoundError(
                f"Dense-caption reference data {eval_dir} does not exist; set "
                "DENSE_CAPTION_EVAL_DIR to the directory holding final-data.json and "
                "mturk-eval-statements/ (Ai2 default tracked in allenai/olmo-eval#381)."
            )
        # Same MOLMO_DATA_DIR convention as the 11 image-QA tasks (the mm-olmo parent;
        # `torch_datasets_dir()` appends `torch_datasets`), so one env var covers all tasks.
        data_home = torch_datasets_dir()
        test_jsonl = data_home / "pixmo_datasets" / "dense-caption-eval" / "test.jsonl"

        with open(eval_dir / "final-data.json") as f:
            final_data = json.load(f)
        url_to_transcripts: dict[str, list[dict]] = {
            ex["image"]: ex["transcripts"] for ex in final_data
        }

        limit = self.config.limit
        count = 0
        with open(test_jsonl) as f:
            # `idx` is the raw line position in test.jsonl (matches mm_olmo's
            # DeterministicDataset index over the full readlines()); it drives the seeded
            # prompt and must NOT be affected by the transcript/mturk skips below.
            for idx, line in enumerate(f):
                if limit is not None and count >= limit:
                    break
                rec = json.loads(line)
                url: str = rec["url"]
                image_id: str = rec.get("image_id", _sha256(url))
                image_name: str = rec.get("image", image_id)
                image_path = data_home / "pixmo_images" / image_name

                transcripts = url_to_transcripts.get(url)
                if transcripts is None:
                    logger.warning("No transcripts for %s — skipping", url)
                    continue

                mturk_file = eval_dir / "mturk-eval-statements" / f"{_sha256(url)}.json"
                if not mturk_file.exists():
                    logger.warning("No mturk file for %s — skipping", url)
                    continue
                with open(mturk_file) as f2:
                    mturk_data = json.load(f2)
                mturk_statements: str = mturk_data["canonical_statements"]

                question = self._question(idx)
                yield Instance(
                    question=question,
                    gold_answer=None,
                    metadata={
                        "id": image_id,
                        "url": url,
                        "image_path": str(image_path),
                        "transcripts": transcripts,
                        "mturk_statements": mturk_statements,
                    },
                )
                count += 1


@register("dense_caption_captioner")
class DenseCaptionCaptionerEval(DenseCaptionEval):
    """Dense-caption eval prompt for stage-1 (captioner-family) checkpoints.

    Checkpoints trained with ``prompt_templates="none"`` + ``system_prompt=
    "style_and_length_v2"`` see a constant eval prompt, the caption's style tag with
    nothing after it: ``"long_caption:"``. That is exactly what OLMo-core's
    ``Molmo2-Stage1.py`` trains on (OLMo-core #875), so train and test share one prompt.
    Use ``dense_caption`` for released-Molmo2-family (``uber_model_v2``) checkpoints,
    which sample a seeded natural-language instruction instead.

    mm_olmo's captioner-family checkpoints were trained on ``"long_caption 65:"`` instead.
    They are no longer evaluated, so this task follows OLMo-core's stage 1 and no longer
    reproduces their training prompt.
    """

    caption_prompt: str | None = "long_caption:"
