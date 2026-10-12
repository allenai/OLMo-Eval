"""Cross-cutting contracts for the vision tasks: per-instance metric storage,
task-identity hashing of the prompt-family fields, config validation, and the
lazy image path through requests."""

from __future__ import annotations

from dataclasses import replace

import pytest

import olmo_eval.evals  # noqa: F401  (registration side effect)
from olmo_eval.common.images import resolve_images
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.vision.benchmarks.dense_caption import _DEFAULT_METRICS, DenseCaptionAvgMetric
from olmo_eval.runners.processing.utils import compute_task_hash


def _metric(name: str):
    return next(m for m in _DEFAULT_METRICS if m.name == name)


def _response(result: dict | None, scores: dict | None = None) -> Response:
    instance = Instance(question="q", gold_answer=None, metadata={})
    output = LMOutput(text="caption")
    if result is not None:
        output.metadata = {"dense_caption_result": result}
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, prompt="q"),
        outputs=[output],
        scores=scores or {},
    )


class TestDenseCaptionPerInstanceMetrics:
    """Persisted per-instance values must be each metric's own field, never the
    scorer channel (which is recall)."""

    RESULT = {
        "recall": 0.4,
        "recall_at_10": 0.5,
        "num_statements": 10,
        "num_covered": 4,
        "recall_valid": True,
        "consistency": 0.75,
        "num_consistent": 3,
        "consistency_num_statements": 4,
        "consistency_valid": True,
    }

    def test_each_metric_stores_its_own_field(self):
        # poison the scorer channel: nothing may fall back to it
        response = _response(self.RESULT, scores={"dense_caption_judge": 0.123})
        assert _metric("recall").compute_instance(response) == pytest.approx(40.0)
        assert _metric("consistency").compute_instance(response) == pytest.approx(75.0)
        assert _metric("recall_at_10").compute_instance(response) == pytest.approx(50.0)
        assert _metric("num_statements").compute_instance(response) == pytest.approx(4.0)

    def test_invalid_side_stores_none(self):
        result = dict(self.RESULT, recall_valid=False)
        response = _response(result, scores={"dense_caption_judge": 0.123})
        assert _metric("recall").compute_instance(response) is None
        assert _metric("recall_at_10").compute_instance(response) is None
        assert _metric("consistency").compute_instance(response) == pytest.approx(75.0)

    def test_avg_has_no_per_instance_value(self):
        response = _response(self.RESULT, scores={"dense_caption_judge": 0.123})
        assert DenseCaptionAvgMetric().compute_instance(response) is None
        assert DenseCaptionAvgMetric().supports_scorer_fallback() is False

    def test_scorer_fallback_disabled(self):
        # no judge result at all: the poisoned scorer channel must NOT leak through
        response = _response(None, scores={"dense_caption_judge": 0.123})
        for name in ("recall", "consistency", "recall_at_10", "num_statements"):
            assert _metric(name).compute_instance(response) is None

    def test_aggregate_matches_mean_of_instances(self):
        responses = [
            _response(self.RESULT),
            _response(dict(self.RESULT, recall=0.6, recall_valid=True)),
            _response(dict(self.RESULT, recall_valid=False)),
        ]
        vals = [_metric("recall").compute_instance(r) for r in responses]
        expected = sum(v for v in vals if v is not None) / 2
        assert _metric("recall").compute(responses) == pytest.approx(expected)

    def test_legacy_scale_displays_as_raw(self):
        # 0-100 legacy values must not be re-scaled by the percentage renderer
        assert _metric("recall").display_format() == "raw"
        assert DenseCaptionAvgMetric().display_format() == "raw"


class TestPromptFamilyFieldsAreOptIn:
    """The prompt-family fields must not perturb the hash of tasks that never set them,
    or every stored text-task result stops matching new runs of the same config."""

    @pytest.mark.parametrize("name", ["arc_easy", "gsm8k", "hellaswag"])
    def test_text_task_config_omits_unset_prompt_fields(self, name):
        config = get_task(name).config.to_dict()
        assert "prompt_templates" not in config
        assert "system_prompt_style" not in config

    def test_setting_a_field_adds_it_back(self):
        # the omission must be conditional, not a removal: opting in still changes identity
        task = get_task("arc_easy")
        opted_in = replace(task.config, system_prompt_style="style_and_length_v2").to_dict()
        assert opted_in["system_prompt_style"] == "style_and_length_v2"
        assert compute_task_hash(opted_in) != compute_task_hash(task.config.to_dict())


class TestPromptFamilyTaskIdentity:
    """The prompt-family fields change the request, so they must change the task hash."""

    def test_system_prompt_style_changes_hash(self):
        task = get_task("dense_caption")
        base = compute_task_hash(task.config.to_dict())
        styled = replace(task.config, system_prompt_style="style_and_length_v2")
        assert compute_task_hash(styled.to_dict()) != base

    def test_prompt_templates_changes_hash(self):
        task = get_task("dense_caption")
        base = compute_task_hash(task.config.to_dict())
        templated = replace(task.config, prompt_templates="none")
        assert compute_task_hash(templated.to_dict()) != base


class TestDenseCaptionConfigValidation:
    def test_unknown_system_prompt_style_raises(self):
        task = get_task("dense_caption")
        task.config = replace(task.config, system_prompt_style="style_and_lenght_v2")
        with pytest.raises(ValueError, match="Unsupported system_prompt_style"):
            task._question(0)

    def test_multi_sample_runs_rejected(self):
        task = get_task("dense_caption")
        params = replace(task.config.sampling_params, num_samples=2)
        task.config = replace(task.config, sampling_params=params)
        with pytest.raises(ValueError, match="num_samples"):
            list(task.instances)


class TestLazyImageRequests:
    """Requests carry lazy image references; providers resolve them."""

    def test_format_request_attaches_path_not_pixels(self):
        task = get_task("dense_caption")
        instance = Instance(
            question="q", gold_answer=None, metadata={"image_path": "/nonexistent/img.png"}
        )
        request = task.format_request(instance)
        # a path string — building the request must not open the file
        assert request.images == ("/nonexistent/img.png",)

    def test_resolve_images_opens_paths_and_calls_callables(self, tmp_path):
        PIL_Image = pytest.importorskip("PIL.Image")
        path = tmp_path / "img.png"
        PIL_Image.new("RGB", (3, 2)).save(path)
        resolved = resolve_images((str(path), lambda: PIL_Image.new("RGB", (5, 4))))
        assert resolved is not None
        assert resolved[0].size == (3, 2)
        assert resolved[1].size == (5, 4)
        assert resolve_images(None) is None


class TestJudgeCacheSettingsOutsideTaskHash:
    """Cache location and mode are machine-local; only the judge model is output-affecting."""

    def test_hash_independent_of_cache_settings(self, monkeypatch, tmp_path):
        base = compute_task_hash(get_task("dense_caption").config.to_dict())
        monkeypatch.setenv("DENSE_CAPTION_EVAL_DIR", str(tmp_path))
        moved = compute_task_hash(get_task("dense_caption").config.to_dict())
        assert moved == base
        from olmo_eval.evals.vision.scoring.judges import DenseCaptionJudgeScorer

        a = DenseCaptionJudgeScorer(cache_dir="/a", cache_only=True).to_dict()
        b = DenseCaptionJudgeScorer(cache_dir="/b", recompute=True).to_dict()
        assert a == b
        assert set(a) == {"type", "name", "model"}


class TestLimitLeftToRunner:
    def test_vision_task_does_not_head_slice(self):
        task = get_task("dense_caption")
        task.config = replace(task.config, limit=3)
        # dense_caption applies its own index-stable limit while building; the
        # base class must not add a second, first-N slice on top for the family.
        from olmo_eval.evals.vision.tasks.base import VisionTask

        assert "limit" not in VisionTask.instances.fget.__code__.co_names


class TestJudgeScorersOutsideTaskHash:
    """A per-process cache dir must not make every run of a task a different config."""

    @pytest.mark.parametrize("name", ["charxiv_descriptive", "charxiv_reasoning", "math_vista"])
    def test_task_hash_is_stable_across_constructions(self, name):
        first = compute_task_hash(get_task(name).config.to_dict())
        second = compute_task_hash(get_task(name).config.to_dict())
        assert first == second
        serialized = get_task(name).config.to_dict()
        assert "mkdtemp" not in repr(serialized)
        assert "/tmp" not in repr(serialized)

    def test_scorer_to_dict_carries_only_output_affecting_settings(self):
        from olmo_eval.evals.vision.scoring.image_qa import MathVistaGptScorer
        from olmo_eval.evals.vision.scoring.judges import CharxivJudgeScorer

        for cls in (CharxivJudgeScorer, MathVistaGptScorer):
            a = cls(cache_dir="/a", cache_only=True).to_dict()
            b = cls(cache_dir="/b", recompute=True).to_dict()
            assert a == b, cls.__name__
            assert set(a) == {"type", "name", "model"}, cls.__name__


class TestSubsetMetricsPerInstance:
    """Subset metrics must not persist the overall scorer value for out-of-scope rows."""

    def _response(self, metadata: dict, scorer_name: str, result: dict | None = None):
        output = LMOutput(text="A")
        if result is not None:
            output.metadata = result
        return Response(
            instance=Instance(question="q", gold_answer=None, metadata=metadata),
            request=LMRequest(request_type=RequestType.CHAT, prompt="q"),
            outputs=[output],
            scores={scorer_name: 1.0},  # poisoned channel
        )

    def test_chartqa_subset(self):
        from olmo_eval.evals.vision.scoring.image_qa import RelaxedCorrectnessScorer
        from olmo_eval.evals.vision.tasks.image_qa import ChartQaSubsetMetric

        scorer = RelaxedCorrectnessScorer()
        human = ChartQaSubsetMetric(name="rc_human", scorer=scorer, subset="human")
        aug = ChartQaSubsetMetric(name="rc_aug", scorer=scorer, subset="augmented")
        r = self._response({"is_human": False}, scorer.name)
        assert human.compute_instance(r) is None  # augmented row, human metric
        assert aug.compute_instance(r) == 1.0
        assert human.supports_scorer_fallback() is False

    def test_mmmu_pro_setting(self):
        from olmo_eval.evals.vision.benchmarks.mmmu_pro import MmmuProSettingMetric
        from olmo_eval.evals.vision.scoring.mmmu_pro import MmmuProScorer

        scorer = MmmuProScorer()
        std = MmmuProSettingMetric(name="standard_10", scorer=scorer, setting="standard10")
        r = self._response({"mmmu_pro_setting": "vision"}, scorer.name)
        assert std.compute_instance(r) is None
        assert std.supports_scorer_fallback() is False

    def test_charxiv_category_and_invalid_flag(self):
        from olmo_eval.evals.vision.benchmarks.charxiv import (
            CharxivInvalidCountMetric,
            CharxivScoreMetric,
        )
        from olmo_eval.evals.vision.scoring.judges import CharxivJudgeScorer

        scorer = CharxivJudgeScorer()
        graded = self._response({}, scorer.name, {"charxiv_result": {"qid": 1, "score": 1}})
        invalid = self._response({}, scorer.name, {"charxiv_result": {"qid": 1, "score": -1}})
        n_invalid = CharxivInvalidCountMetric(name="n_invalid", scorer=scorer)
        assert n_invalid.compute_instance(graded) == 0.0
        assert n_invalid.compute_instance(invalid) == 1.0
        overall = CharxivScoreMetric(name="score", scorer=scorer, category=None)
        assert overall.compute_instance(invalid) == 0.0
        assert overall.supports_scorer_fallback() is False


class TestLazyMultiImageRequests:
    def test_list_entries_flatten_and_cap(self, tmp_path):
        PIL_Image = pytest.importorskip("PIL.Image")
        from olmo_eval.evals.vision.data.images import capped_image_list

        paths = []
        for i in range(3):
            path = tmp_path / f"{i}.png"
            PIL_Image.new("RGB", (2 + i, 2)).save(path)
            paths.append(str(path))
        entry = capped_image_list(paths, max_images=2)
        resolved = resolve_images((entry,))
        assert resolved is not None
        assert [img.size for img in resolved] == [(2, 2), (3, 2)]


class TestMultiImageSubsetMetricsPerInstance:
    """MuirBench categories, BLINK subtasks and MMIU buckets must not persist the
    overall scorer value for instances outside their scope."""

    def _response(self, metadata: dict, scorer_name: str):
        return Response(
            instance=Instance(question="q", gold_answer=None, metadata=metadata),
            request=LMRequest(request_type=RequestType.CHAT, prompt="q"),
            outputs=[LMOutput(text="A")],
            scores={scorer_name: 1.0},  # poisoned channel
        )

    def test_category_metric_scopes_to_its_category(self):
        from olmo_eval.evals.vision.scoring.multi_image import MultiImageMcScorer
        from olmo_eval.evals.vision.tasks.multi_image import MultiImageCategoryMetric

        scorer = MultiImageMcScorer()
        ordering = MultiImageCategoryMetric(
            name="ordering", scorer=scorer, field="task", category="Ordering"
        )
        overall = MultiImageCategoryMetric(name="all", scorer=scorer, field="task", category=None)
        counting = self._response({"task": "Counting"}, scorer.name)
        assert ordering.compute_instance(counting) is None
        assert overall.compute_instance(counting) == 1.0
        assert ordering.supports_scorer_fallback() is False

    def test_count_bucket_metric_scopes_to_its_bucket(self):
        from olmo_eval.evals.vision.scoring.multi_image import MultiImageMcScorer
        from olmo_eval.evals.vision.tasks.multi_image import MultiImageCountBucketMetric

        scorer = MultiImageMcScorer()
        small = MultiImageCountBucketMetric(name="le10", scorer=scorer, max_images=10)
        large = MultiImageCountBucketMetric(name="gt20", scorer=scorer, min_images=20)
        r = self._response({"num_images": 4}, scorer.name)
        assert small.compute_instance(r) == 1.0
        assert large.compute_instance(r) is None
        assert large.supports_scorer_fallback() is False
