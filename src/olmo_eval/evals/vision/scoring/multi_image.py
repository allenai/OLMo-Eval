"""Multi-image multiple-choice response scoring.

Vendored from mm_olmo: ``muir_bench_mc`` (``olmo/eval/vqa.py``) plus the response
pre-processing shared by the ``MuirBenchEval`` / ``BLINKEval`` / ``MMIUEval`` /
``MMSIBenchEval`` evaluators (``olmo/eval/molmo_prediction_evaluators.py``).

mm_olmo's ``muir_bench_parse_multi_choice_response`` is a copy of its MMMU parser,
so the letter parsing reuses :func:`olmo_eval.evals.vision.scoring.multiple_choice.
parse_multi_choice_response` — including its deterministic stable-id fallback in
place of the original's order-dependent ``random.choice``.
"""

from __future__ import annotations

import string
from collections import Counter
from dataclasses import dataclass

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.vision.scoring.common import response_text
from olmo_eval.evals.vision.scoring.multiple_choice import parse_multi_choice_response


def strip_multi_image_response(prediction: str) -> str:
    """The response cleanup all mm_olmo multi-image MC evaluators apply before parsing.

    Text after the first ``"Answer:"`` wins; otherwise multi-line responses collapse
    to their most frequent line (first on ties); otherwise whitespace is normalized.
    """
    pred = prediction.strip()
    if "Answer:" in pred:
        pred = pred.split("Answer:")[1].strip()
    elif "\n" in pred:
        preds = [" ".join(x.strip().split()) for x in pred.split("\n")]
        counts = Counter(preds)
        max_count = max(counts.values())
        pred = [x for x in preds if counts[x] == max_count][0]
    else:
        pred = " ".join(pred.strip().split())
    return pred


def multi_image_mc_score(
    target: str,
    prediction: str,
    options: list[str],
    stable_id: str | None = None,
) -> float:
    """mm_olmo ``muir_bench_mc``: parse the predicted letter and match the gold letter.

    ``target`` is the gold option letter (may fall outside ``options`` — MMIU has
    gold answers not listed among the options, which then simply score 0).
    """
    pred = strip_multi_image_response(prediction)
    all_choices = list(string.ascii_uppercase[: len(options)])
    index2ans = {choice: opt.strip() for choice, opt in zip(all_choices, options, strict=True)}
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans, stable_id=stable_id)
    return float(target == parsed_pred)


@dataclass(frozen=True, slots=True)
class MultiImageMcScorer(Scorer):
    """Accuracy of the parsed option letter against the gold letter."""

    name: str = "multi_image_mc"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        return multi_image_mc_score(
            meta["answer"],
            response_text(output),
            list(meta.get("options") or []),
            stable_id=str(meta.get("example_id", "")),
        )
