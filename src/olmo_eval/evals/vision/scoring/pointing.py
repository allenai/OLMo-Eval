"""Point-in-mask scoring for the pointing benchmarks.

Mask decode (COCO RLE) and maximum-bipartite point matching, mirroring mm_olmo's
``SegmentationPointingScorer``; heavy dependencies (``pycocotools``, ``scipy``)
are imported lazily inside the functions so the module imports without them.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.vision.scoring.common import response_text
from olmo_eval.evals.vision.scoring.count_parsing import extract_image_points

logger = logging.getLogger(__name__)


def require_scoring_dependencies() -> None:
    """Import the mask-decode and matching backends, raising if either is missing.

    Called when a pointing task builds its instances so an environment without
    ``pycocotools``/``scipy`` fails before a model is loaded, rather than at the
    first scored example.
    """
    from pycocotools import mask as mask_utils  # noqa: F401
    from scipy.sparse.csgraph import maximum_bipartite_matching  # noqa: F401


def decode_segmentation(seg: Any, height: int, width: int) -> np.ndarray | None:
    """Decode a COCO segmentation (RLE dict or polygon list) to a binary ``HxW`` array.

    Faithful to mm_olmo's ``_decode_segmentation`` but first normalizes the JSON-serialized
    RLE form (string ``counts`` / string ``size``) emitted by the SACoGold subset.
    ``height``/``width`` are only used for polygon / uncompressed-RLE inputs; a compressed RLE
    carries its own ``size`` and decodes at that resolution.
    """
    from pycocotools import mask as mask_utils

    if isinstance(seg, dict):
        rle = dict(seg)
        size = rle.get("size")
        if isinstance(size, str):
            rle["size"] = [int(v) for v in ast.literal_eval(size)]
        counts = rle.get("counts")
        if isinstance(counts, str):
            rle["counts"] = counts.encode("utf-8")
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        m = mask_utils.decode(rle)
    elif isinstance(seg, list):
        rle = mask_utils.frPyObjects(seg, height, width)
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m.max(axis=2)
    else:
        return None
    return m  # uint8 0/1


def score_pointing_example(
    points: Any,
    annotators: list[list[np.ndarray]],
) -> tuple[float, float, float]:
    """Point-in-mask ``(precision, recall, f1)`` for one example, best over annotators.

    ``points`` is an ``Nx2`` (or ``Nx3`` with a leading id) array of predicted points in the
    mask's pixel coordinates. ``annotators`` is a list of annotator mask-lists (one inner list
    per annotator; each entry an ``HxW`` binary mask). For each annotator, predicted points are
    matched to masks by maximum bipartite matching (a point matches a mask iff it lies inside
    it); precision = matches / #points, recall = matches / #masks. The maximum over annotators is
    returned. Empty-mask / empty-point edge cases mirror mm_olmo exactly.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = points.reshape(0, 2)
    if len(points) > 0:
        points = points[:, -2:]

    max_pr = max_rec = max_f1 = 0.0
    for masks in annotators:
        if len(masks) == 0:
            pr, rec = (1.0, 1.0) if len(points) == 0 else (0.0, 0.0)
        elif len(points) == 0:
            pr, rec = 0.0, 0.0
        else:
            stacked = np.stack(masks, axis=-1)  # H x W x M
            mask_h, mask_w = stacked.shape[:2]
            rounded = np.minimum(np.round(points), np.array([mask_w, mask_h]) - 1).astype(np.int64)
            distances = np.zeros((len(points), len(masks)), dtype=bool)
            for p_idx, (x, y) in enumerate(rounded):
                distances[p_idx] = stacked[y, x].astype(bool)
            col_ind = maximum_bipartite_matching(csr_matrix(distances))
            num_matches = int((col_ind != -1).sum())
            pr = num_matches / len(points)
            rec = num_matches / stacked.shape[-1]
        f1 = 2 * pr * rec / (pr + rec) if (pr > 0 and rec > 0) else 0.0
        max_pr, max_rec, max_f1 = max(max_pr, pr), max(max_rec, rec), max(max_f1, f1)
    return max_pr, max_rec, max_f1


@dataclass(frozen=True, slots=True)
class PointingScorer(Scorer):
    """Point-in-mask precision/recall/f1 for one pointing example.

    Stores ``{precision, recall, f1, count, weight}`` in
    ``output.metadata["pointing_result"]`` and returns ``f1``.
    """

    name: str = "pointing"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        if meta.get("image_size") is None:
            # Every pointing task records this; its absence is a task bug, not bad
            # data, so it must fail the run rather than score the example zero.
            raise KeyError(
                f"pointing instance {meta.get('example_id', '?')!r} has no image_size; "
                "the task must record (width, height) in instance.metadata"
            )
        try:
            return self._score(instance, output)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            # Malformed ground truth scores that example zero with the error
            # recorded. Everything else -- a missing pycocotools/scipy, or a bug
            # in the scorer -- propagates so the runner records a scoring error
            # instead of reporting a run of zeros as Success.
            logger.warning(
                "Pointing scoring failed for %s: %s",
                instance.metadata.get("example_id", "?"),
                exc,
            )
            meta = instance.metadata
            rle_counts = [len(rles) for rles in meta.get("pointing_annotators", [])]
            if output.metadata is None:
                output.metadata = {}
            output.metadata["pointing_result"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "count": max(rle_counts, default=0),
                "weight": float(meta.get("weight", 1.0)),
                "no_gt": all(c == 0 for c in rle_counts),
                "has_gt": bool(rle_counts) and all(c > 0 for c in rle_counts),
                "made_prediction": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return 0.0

    def _score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        image_w, image_h = (int(v) for v in meta["image_size"])

        # Decode the ground-truth masks for each annotator.
        annotator_masks: list[list[np.ndarray]] = []
        for rles in meta["pointing_annotators"]:
            masks: list[np.ndarray] = []
            for seg in rles:
                m = decode_segmentation(seg, image_h, image_w)
                if m is not None:
                    masks.append(np.asarray(m, dtype=bool))
            annotator_masks.append(masks)

        # Predicted points in image-pixel coordinates.
        points = np.asarray(
            extract_image_points(response_text(output).strip(), image_w, image_h), dtype=np.float64
        ).reshape(-1, 2)

        # Masks rarely live at a higher resolution than the image; rescale the predicted points
        # into mask coordinates when so (no-op when mask resolution == image resolution).
        mask_res = next((m[0].shape for m in annotator_masks if m), None)
        if mask_res is not None and len(points) > 0:
            mask_h, mask_w = int(mask_res[0]), int(mask_res[1])
            if (mask_h, mask_w) != (image_h, image_w):
                points = points * np.array([mask_w / image_w, mask_h / image_h])[None, :]

        precision, recall, f1 = score_pointing_example(points, annotator_masks)
        count = max((len(m) for m in annotator_masks), default=0)

        if output.metadata is None:
            output.metadata = {}
        output.metadata["pointing_result"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": count,
            "weight": float(meta.get("weight", 1.0)),
            # Abstention accounting: whether the phrase is absent for every annotator,
            # present for every annotator, and whether the model pointed at anything.
            "no_gt": all(len(masks) == 0 for masks in annotator_masks),
            "has_gt": all(len(masks) > 0 for masks in annotator_masks),
            "made_prediction": len(points) > 0,
        }
        return f1
