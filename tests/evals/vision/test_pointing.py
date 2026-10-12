"""Unit tests for the image-pointing family (geometry, metrics, prompts).

Pure CPU — no GPU, no model, no dataset.  ``decode_segmentation`` needs pycocotools and is
skipped when it is not installed; everything else uses scipy (a declared dep) only.
"""

from __future__ import annotations

import numpy as np
import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.vision.benchmarks.pixmo_points_eval import _format_query as pixmo_query
from olmo_eval.evals.vision.benchmarks.sa_co_gold import _format_query as saco_query
from olmo_eval.evals.vision.scoring.pointing import (
    PointingScorer,
    score_pointing_example,
)
from olmo_eval.evals.vision.tasks.pointing import PointingMetric


def _mask(region: tuple[slice, slice], shape: tuple[int, int] = (10, 10)) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[region] = True
    return m


# ---------------------------------------------------------------------------
# score_pointing_example
# ---------------------------------------------------------------------------


class TestScorePointingExample:
    # one 3x3 mask block at rows/cols 2..4 -> mask[y, x] True for x,y in [2,4]
    BLOCK = (slice(2, 5), slice(2, 5))

    def test_point_inside(self) -> None:
        pr, rec, f1 = score_pointing_example([[3, 3]], [[_mask(self.BLOCK)]])
        assert (pr, rec, f1) == (1.0, 1.0, 1.0)

    def test_point_outside(self) -> None:
        assert score_pointing_example([[8, 8]], [[_mask(self.BLOCK)]]) == (0.0, 0.0, 0.0)

    def test_two_points_one_mask(self) -> None:
        pr, rec, f1 = score_pointing_example([[3, 3], [8, 8]], [[_mask(self.BLOCK)]])
        assert pr == pytest.approx(0.5)
        assert rec == pytest.approx(1.0)
        assert f1 == pytest.approx(2 / 3)

    def test_two_points_two_masks(self) -> None:
        masks = [_mask((slice(2, 5), slice(2, 5))), _mask((slice(6, 9), slice(6, 9)))]
        pr, rec, f1 = score_pointing_example([[3, 3], [7, 7]], [masks])
        assert (pr, rec, f1) == (1.0, 1.0, 1.0)

    def test_empty_masks_empty_points(self) -> None:
        assert score_pointing_example([], [[]]) == (1.0, 1.0, 1.0)

    def test_empty_masks_with_points(self) -> None:
        assert score_pointing_example([[3, 3]], [[]]) == (0.0, 0.0, 0.0)

    def test_max_over_annotators(self) -> None:
        miss = [_mask((slice(6, 9), slice(6, 9)))]  # (3,3) not inside
        hit = [_mask(self.BLOCK)]  # (3,3) inside
        assert score_pointing_example([[3, 3]], [miss, hit]) == (1.0, 1.0, 1.0)

    def test_strips_leading_id_column(self) -> None:
        # Nx3 input (leading object id) -> only last two cols are coordinates.
        assert score_pointing_example([[0, 3, 3]], [[_mask(self.BLOCK)]]) == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# PointingMetric
# ---------------------------------------------------------------------------


def _pointing_response(count: int, f1: float, weight: float = 1.0) -> Response:
    instance = Instance(question="q", metadata={})
    result = {"precision": f1, "recall": f1, "f1": f1, "count": count, "weight": weight}
    output = LMOutput(text="", metadata={"pointing_result": result})
    return Response(instance=instance, request=None, outputs=[output])


class TestPointingMetric:
    SCORER = PointingScorer()

    def _responses(self) -> list[Response]:
        # (count, f1, weight)
        rows = [(1, 1.0, 1.0), (2, 0.0, 3.0), (11, 0.5, 1.0)]
        return [_pointing_response(c, f, w) for c, f, w in rows]

    def test_simple_mean_all(self) -> None:
        m = PointingMetric(name="f1", scorer=self.SCORER, kind="f1", bucket="all")
        assert m.compute(self._responses()) == pytest.approx((1.0 + 0.0 + 0.5) / 3)

    def test_weighted_mean_all(self) -> None:
        m = PointingMetric(name="f1", scorer=self.SCORER, kind="f1", bucket="all", weighted=True)
        # (1*1 + 0*3 + 0.5*1) / (1 + 3 + 1)
        assert m.compute(self._responses()) == pytest.approx(1.5 / 5)

    def test_bucket_single(self) -> None:
        m = PointingMetric(name="single_f1", scorer=self.SCORER, kind="f1", bucket="single")
        assert m.compute(self._responses()) == pytest.approx(1.0)

    def test_bucket_high_freq(self) -> None:
        m = PointingMetric(name="high_freq_f1", scorer=self.SCORER, kind="f1", bucket="high_freq")
        assert m.compute(self._responses()) == pytest.approx(0.5)

    def test_empty(self) -> None:
        assert PointingMetric(name="f1", scorer=self.SCORER).compute([]) == 0.0


# ---------------------------------------------------------------------------
# prompt builders
# ---------------------------------------------------------------------------


class TestFormatQuery:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Earring", "Point to earring."),
            ("a cat", "Point to cat."),
            ("The Dogs", "Point to dogs."),
        ],
    )
    def test_pixmo(self, label: str, expected: str) -> None:
        assert pixmo_query(label) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("cotton t-shirt", "Point to the cotton t-shirt."),
            ("a cat", "Point to a cat."),
            ("the dog", "Point to the dog."),
        ],
    )
    def test_saco(self, text: str, expected: str) -> None:
        assert saco_query(text) == expected


# ---------------------------------------------------------------------------
# decode_segmentation (needs pycocotools)
# ---------------------------------------------------------------------------


class TestDecodeSegmentation:
    def test_roundtrip_and_str_normalization(self) -> None:
        mask_utils = pytest.importorskip("pycocotools.mask")
        from olmo_eval.evals.vision.scoring.pointing import decode_segmentation

        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 2:4] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))  # {counts: bytes, size: [4, 5]}

        decoded = decode_segmentation(rle, 4, 5)
        assert decoded is not None
        np.testing.assert_array_equal(decoded.astype(bool), mask.astype(bool))

        # SACoGold JSON form: counts as str, size as the string "[h, w]".
        str_rle = {"counts": rle["counts"].decode("ascii"), "size": str(list(rle["size"]))}
        decoded_str = decode_segmentation(str_rle, 4, 5)
        assert decoded_str is not None
        np.testing.assert_array_equal(decoded_str.astype(bool), mask.astype(bool))


class TestPointingPerInstanceMetrics:
    """Persisted per-instance values are each metric's own field, never the scorer's f1."""

    RESULT = {
        "precision": 0.5,
        "recall": 0.25,
        "f1": 1 / 3,
        "count": 4,
        "weight": 2.0,
        "no_gt": False,
        "has_gt": True,
        "made_prediction": True,
    }

    def _response(self, result: dict) -> Response:
        output = LMOutput(text="<points/>")
        output.metadata = {"pointing_result": result}
        return Response(
            instance=Instance(question="q", gold_answer=None, metadata={}),
            request=LMRequest(request_type=RequestType.CHAT, prompt="q"),
            outputs=[output],
            scores={"pointing": result["f1"]},  # poisoned fallback channel
        )

    def test_each_kind_stores_its_own_field(self):
        from olmo_eval.evals.vision.tasks.pointing import pointing_metrics

        metrics = {m.name: m for m in pointing_metrics(PointingScorer(), buckets=("single",))}
        response = self._response(self.RESULT)
        assert metrics["precision"].compute_instance(response) == pytest.approx(0.5)
        assert metrics["recall"].compute_instance(response) == pytest.approx(0.25)
        assert metrics["f1"].compute_instance(response) == pytest.approx(1 / 3)
        # count=4 is outside the `single` bucket
        assert metrics["single_f1"].compute_instance(response) is None
        assert metrics["precision"].supports_scorer_fallback() is False

    def test_weighted_metrics_store_none(self):
        from olmo_eval.evals.vision.tasks.pointing import PointingMetric, pointing_metrics

        metrics = pointing_metrics(PointingScorer(), buckets=(), weighted_primary=True)
        response = self._response(self.RESULT)
        weighted = [m for m in metrics if isinstance(m, PointingMetric)]
        assert weighted, "expected the weighted precision/recall/f1 family"
        # a weighted mean is not a mean of per-instance values, so it stores nothing
        assert all(m.compute_instance(response) is None for m in weighted)

    def test_scoring_error_count_is_reported(self):
        from olmo_eval.evals.vision.tasks.pointing import pointing_metrics

        metric = next(
            m
            for m in pointing_metrics(PointingScorer(), buckets=())
            if m.name == "n_scoring_errors"
        )
        ok = self._response(self.RESULT)
        failed = self._response({**self.RESULT, "error": "ValueError: bad rle"})
        assert metric.compute_instance(ok) == 0.0
        assert metric.compute_instance(failed) == 1.0
        assert metric.compute([ok, failed, failed]) == 2.0

    def test_presence_metrics_store_membership_values(self):
        from olmo_eval.evals.vision.tasks.pointing import presence_metrics

        absent_acc, present_acc = presence_metrics(PointingScorer())
        response = self._response(self.RESULT)  # has_gt example that pointed: correct
        assert absent_acc.compute_instance(response) is None
        assert present_acc.compute_instance(response) == pytest.approx(1.0)


class TestPointingScorerFailureContract:
    def test_decode_failure_scores_zero_and_records_error(self):
        pytest.importorskip("pycocotools")
        scorer = PointingScorer()
        instance = Instance(
            question="Point to the cat.",
            gold_answer=None,
            metadata={
                "example_id": "broken",
                "image_size": (10, 10),
                # malformed RLE: decoding raises inside the scorer
                "pointing_annotators": [[{"size": "not-a-size", "counts": 12345}]],
            },
        )
        output = LMOutput(text='<point x="1" y="1" alt="cat">cat</point>')
        score = scorer.score(instance, output)
        assert score == 0.0
        result = output.metadata["pointing_result"]
        assert result["f1"] == 0.0
        assert result["count"] == 1
        assert "error" in result


class TestPointCountPerInstanceMetrics:
    def _response(self, result: dict, count: int) -> Response:
        output = LMOutput(text="3")
        output.metadata = {"point_count_result": result}
        return Response(
            instance=Instance(question="q", gold_answer=str(count), metadata={"count": count}),
            request=LMRequest(request_type=RequestType.CHAT, prompt="q"),
            outputs=[output],
            scores={"point_count": result["correct"]},  # poisoned fallback channel
        )

    def test_fields_and_buckets(self):
        from olmo_eval.evals.vision.scoring.counting import PointCountScorer
        from olmo_eval.evals.vision.tasks.image_qa import (
            PointCountCategoryAverageMetric,
            PointCountMetric,
            PointCountPerCountMetric,
        )

        scorer = PointCountScorer()
        result = {"correct": 0.0, "close": 1.0, "valid": 1.0, "pred_count": 4}
        response = self._response(result, count=3)
        assert PointCountMetric(name="close", scorer=scorer, kind="close").compute_instance(
            response
        ) == pytest.approx(1.0)
        assert PointCountMetric(name="correct", scorer=scorer, kind="correct").compute_instance(
            response
        ) == pytest.approx(0.0)
        per3 = PointCountPerCountMetric(name="correct_count_3", scorer=scorer, k=3)
        per4 = PointCountPerCountMetric(name="correct_count_4", scorer=scorer, k=4)
        assert per3.compute_instance(response) == pytest.approx(0.0)
        assert per4.compute_instance(response) is None
        cat = PointCountCategoryAverageMetric(name="correct_category_average", scorer=scorer)
        assert cat.compute_instance(response) is None
        assert cat.supports_scorer_fallback() is False


class TestCountingPromptFamilies:
    def test_pixmo_count_follows_the_checkpoint(self):
        from dataclasses import replace

        import olmo_eval.evals  # noqa: F401
        from olmo_eval.evals.tasks.common.registry import get_task

        task = get_task("pixmo_count")
        # instruction-tuned default: seeded template, no prefix
        templated = task._question_for("Cats", 3)
        assert "Cats" in templated or "cats" in templated
        assert not templated.startswith("point_count:")
        # pretrain family: bare lowercased label with the point_count prefix
        task.config = replace(
            task.config, prompt_templates="none", system_prompt_style="style_and_length_v2"
        )
        assert task._question_for("Cats", 3) == "point_count: cats"


class TestPointingScoringDependencies:
    def test_missing_backend_propagates_instead_of_scoring_zero(self, monkeypatch):
        """A worker without pycocotools/scipy must fail, not score every instance zero."""
        import builtins

        from olmo_eval.evals.vision.scoring import pointing as pointing_scoring

        real_import = builtins.__import__

        def no_pycocotools(name, *args, **kwargs):
            if name.startswith("pycocotools"):
                raise ImportError("No module named 'pycocotools'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pycocotools)
        with pytest.raises(ImportError, match="pycocotools"):
            pointing_scoring.require_scoring_dependencies()

        instance = Instance(
            question="Point to the cat.",
            gold_answer=None,
            metadata={
                "example_id": "x",
                "image_size": (10, 10),
                "pointing_annotators": [[{"size": [10, 10], "counts": "abc"}]],
            },
        )
        output = LMOutput(text='<point x="1" y="1" alt="cat">cat</point>')
        with pytest.raises(ImportError, match="pycocotools"):
            PointingScorer().score(instance, output)

    def test_missing_image_size_is_a_hard_error(self):
        instance = Instance(
            question="Point to the cat.",
            gold_answer=None,
            metadata={"example_id": "y", "pointing_annotators": [[]]},
        )
        with pytest.raises(KeyError, match="image_size"):
            PointingScorer().score(instance, LMOutput(text="none"))


class TestPixmoImageSize:
    """An absent phrase has no mask, so its size must come from the image itself."""

    def test_absent_phrase_reads_the_image_header(self, tmp_path, monkeypatch):
        PIL_Image = pytest.importorskip("PIL.Image")
        datasets = pytest.importorskip("datasets")
        from olmo_eval.evals.vision.benchmarks import pixmo_points_eval as pixmo

        image = tmp_path / "scene.png"
        PIL_Image.new("RGB", (40, 30)).save(image)
        rows = [
            {
                "label": "cat",
                "segmentation_rles": [{"size": [20, 10], "counts": "x"}],
                "image": str(image),
                "example_id": "present",
            },
            {"label": "dog", "segmentation_rles": [], "image": str(image), "example_id": "absent"},
        ]
        monkeypatch.setattr(datasets, "load_from_disk", lambda path: rows)
        monkeypatch.setattr(pixmo, "torch_datasets_dir", lambda: tmp_path)
        monkeypatch.setattr(pixmo, "rebase_data_path", lambda path: path)

        sizes = {
            inst.metadata["example_id"]: inst.metadata["image_size"]
            for inst in pixmo._build_instances(lambda label, idx: label)
        }
        assert sizes == {"present": (10, 20), "absent": (40, 30)}
