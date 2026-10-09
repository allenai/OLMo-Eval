"""Scoring for the document-OCR benchmarks: the vendored scorers, their metrics, the
per-instance values saved with the predictions, and OmniDocBench's evaluation outcome."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

import olmo_eval.evals  # noqa: F401  (registration side effect)
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.vision.benchmarks import omnidocbench as omnidocbench_task
from olmo_eval.evals.vision.benchmarks.cc_ocr import _subset_of
from olmo_eval.evals.vision.scoring import olmocr_bench as olmocr_scoring
from olmo_eval.evals.vision.scoring.cc_ocr import score_ocr_sample
from olmo_eval.evals.vision.scoring.omnidocbench import (
    OfficialEvaluation,
    _count_failures,
    _page_means,
    _summary,
)
from olmo_eval.runners.asynq.preparation import finalize_task
from olmo_eval.runners.asynq.types import TaskTracker
from olmo_eval.runners.io.builders import build_predictions


def _response(metadata: dict, text: str, gold: str | None = None) -> Response:
    return Response(
        instance=Instance(question="q", gold_answer=gold, metadata=metadata),
        request=LMRequest(request_type=RequestType.CHAT, prompt="q"),
        outputs=[LMOutput(text=text)],
    )


def _score(task_name: str, responses: list[Response]):
    task = get_task(task_name)
    asyncio.run(task.score_responses(responses))
    return task


def _metric(task, name: str):
    return next(m for m in task.config.metrics if m.name == name)


# ---------------------------------------------------------------------------
# CC-OCR
# ---------------------------------------------------------------------------


class TestCcOcrSample:
    def test_word_level_is_a_lowercase_alphanumeric_multiset(self):
        stats = score_ocr_sample("Hello, World! hello ###", "hello world foo", "TotalText")
        assert (stats["right_num"], stats["gt_num"], stats["pred_num"]) == (2, 3, 3)
        assert stats["f1"] == pytest.approx(2 / 3, abs=1e-6)

    def test_chinese_sub_datasets_compare_characters(self):
        stats = score_ocr_sample("中文 字", "中文***", "zh_scene")
        assert (stats["right_num"], stats["gt_num"], stats["pred_num"]) == (2, 2, 3)
        assert stats["f1"] == pytest.approx(0.8, abs=1e-6)

    def test_empty_prediction_scores_zero(self):
        stats = score_ocr_sample("", "a b", "IC15")
        assert stats["pred_num"] == 0
        assert stats["f1"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        ("stem", "subset"),
        [("zh_scene_450", "zh_scene"), ("TotalText_300", "TotalText"), ("IC15", "IC15")],
    )
    def test_file_stem_names_its_sub_dataset(self, stem: str, subset: str):
        assert _subset_of(stem) == subset


class TestCcOcrMetrics:
    @pytest.fixture
    def scored(self):
        rows = [
            ("TotalText", "a b", "a b"),  # f1 1.0
            ("TotalText", "c d e f", "c"),  # recall 1/4, precision 1, f1 0.4
            ("IC15", "x y", "x"),  # recall 1/2, precision 1, f1 2/3
        ]
        responses = [
            _response({"id": f"{dataset}/{i}", "dataset": dataset, "answer": answer}, text)
            for i, (dataset, answer, text) in enumerate(rows)
        ]
        return _score("cc_ocr_multi_scene", responses), responses

    def test_dataset_and_track_scores(self, scored):
        task, responses = scored
        metrics = task.compute_metrics(responses)
        assert metrics["TotalText"]["cc_ocr"] == pytest.approx(0.7, abs=1e-6)
        assert metrics["IC15"]["cc_ocr"] == pytest.approx(2 / 3, abs=1e-6)
        assert metrics["zh_scene"]["cc_ocr"] == 0.0
        # Unweighted over the sub-datasets that ran, not over images.
        assert metrics["macro_f1"]["cc_ocr"] == pytest.approx((0.7 + 2 / 3) / 2, abs=1e-6)
        # TotalText pools 3 right of 6 reference and 3 predicted units: f1 2/3.
        assert metrics["micro_f1"]["cc_ocr"] == pytest.approx(2 / 3, abs=1e-6)

    def test_predictions_keep_the_scorer_result(self, scored):
        _, responses = scored
        prediction = build_predictions(responses)[1]
        result = prediction["model_output"][0]["scorer_results"]["cc_ocr"]
        assert result["dataset"] == "TotalText"
        assert (result["right_num"], result["gt_num"], result["pred_num"]) == (1, 4, 1)

    def test_instance_metrics_are_scoped_to_the_sub_dataset(self, scored):
        task, responses = scored
        predictions = build_predictions(responses, task.config.metrics)
        first, third = predictions[0]["instance_metrics"], predictions[2]["instance_metrics"]
        assert first["TotalText"]["cc_ocr"] == pytest.approx(1.0, abs=1e-6)
        assert "IC15" not in first
        assert third["IC15"]["cc_ocr"] == pytest.approx(2 / 3, abs=1e-6)
        assert "TotalText" not in third
        assert third["macro_f1"]["cc_ocr"] == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# olmOCR-bench
# ---------------------------------------------------------------------------


class _FakeTest:
    def __init__(self, row: dict):
        self.row = row

    def run(self, markdown: str) -> tuple[bool, str]:
        if self.row.get("raises"):
            raise ValueError("render failed")
        return self.row["text"] in markdown, ""


@pytest.fixture
def fake_olmocr(monkeypatch):
    """The official test runner, reduced to substring checks."""
    tests_module = types.ModuleType("olmocr.bench.tests")
    tests_module.load_single_test = _FakeTest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "olmocr", types.ModuleType("olmocr"))
    monkeypatch.setitem(sys.modules, "olmocr.bench", types.ModuleType("olmocr.bench"))
    monkeypatch.setitem(sys.modules, "olmocr.bench.tests", tests_module)
    monkeypatch.setattr(olmocr_scoring, "ensure_olmocr_bench_runtime", lambda: None)


def _test(category: str, test_id: str, test_type: str, text: str, **extra) -> dict:
    return {"category": category, "test": {"id": test_id, "type": test_type, "text": text, **extra}}


class TestOlmocrBench:
    @pytest.fixture
    def scored(self, fake_olmocr):
        math_page = [
            _test("arxiv_math", "t1", "present", "alpha"),
            _test("arxiv_math", "t2", "math", "beta", raises=True),
            _test("baseline", "b1", "baseline", ""),
        ]
        table_page = [
            _test("table_tests", "t3", "table", "gamma"),
            _test("baseline", "b2", "baseline", ""),
        ]
        responses = [
            _response({"id": "math.pdf", "tests": math_page}, "alpha"),
            _response({"id": "table.pdf", "tests": table_page}, "nothing"),
        ]
        return _score("olmocr_bench", responses), responses

    def test_page_score_is_its_pass_rate(self, scored):
        _, responses = scored
        assert responses[0].scores["olmocr_bench"] == pytest.approx(2 / 3)
        assert responses[1].scores["olmocr_bench"] == pytest.approx(1 / 2)

    def test_a_raising_test_fails_and_is_flagged(self, scored):
        _, responses = scored
        prediction = build_predictions(responses)[0]
        tests = prediction["model_output"][0]["scorer_results"]["olmocr_bench"]["tests"]
        raised = next(t for t in tests if t["id"] == "t2")
        assert raised["passed"] is False
        assert raised["raised"] is True
        assert raised["explanation"] == "ValueError: render failed"
        assert not any(t["raised"] for t in tests if t["id"] != "t2")

    def test_overall_is_the_mean_over_categories(self, scored):
        task, responses = scored
        metrics = task.compute_metrics(responses)
        assert metrics["arxiv_math"]["olmocr_bench"] == pytest.approx(0.5)
        assert metrics["table_tests"]["olmocr_bench"] == pytest.approx(0.0)
        assert metrics["baseline"]["olmocr_bench"] == pytest.approx(1.0)
        assert metrics["overall"]["olmocr_bench"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)
        assert metrics["type_present"]["olmocr_bench"] == pytest.approx(1.0)
        assert metrics["type_math"]["olmocr_bench"] == pytest.approx(0.0)
        assert metrics["n_test_errors"]["olmocr_bench"] == 1.0

    def test_instance_metrics_cover_the_page_own_tests(self, scored):
        task, responses = scored
        predictions = build_predictions(responses, task.config.metrics)
        math_page, table_page = (p["instance_metrics"] for p in predictions)
        assert math_page["overall"]["olmocr_bench"] == pytest.approx((0.5 + 1.0) / 2)
        assert math_page["arxiv_math"]["olmocr_bench"] == pytest.approx(0.5)
        assert math_page["type_math"]["olmocr_bench"] == 0.0
        assert math_page["n_test_errors"]["olmocr_bench"] == 1.0
        assert "table_tests" not in math_page
        assert "type_table" not in math_page
        assert table_page["overall"]["olmocr_bench"] == pytest.approx(0.5)
        assert table_page["n_test_errors"]["olmocr_bench"] == 0.0
        assert "arxiv_math" not in table_page

    def test_error_count_is_lower_is_better(self, scored):
        task, _ = scored
        assert _metric(task, "n_test_errors").pairwise_higher_is_better() is False
        assert _metric(task, "overall").pairwise_higher_is_better() is True


# ---------------------------------------------------------------------------
# OmniDocBench
# ---------------------------------------------------------------------------


class TestOmniDocBenchEvaluatorOutput:
    def test_page_means_average_a_page_samples(self):
        per_sample = {
            "p1.jpg_[0]": {"Edit_dist": 0.2},
            "p1.jpg_[1]": {"Edit_dist": 0.4},
            "p2.png_[0, 1]": {"Edit_dist": 1.0},
            "p3.png_[0]": {"Edit_dist": None},
            "unkeyed": {"Edit_dist": 0.5},
        }
        assert _page_means(per_sample, "Edit_dist") == pytest.approx({"p1.jpg": 0.3, "p2.png": 1.0})
        assert _page_means({"p1.jpg_[0]": 0.7}, None) == {"p1.jpg": 0.7}
        assert _page_means(None, "Edit_dist") == {}

    def test_summary_reads_the_leaderboard_numbers(self):
        metric_result = {
            "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.2}}},
            "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.3}}},
            "table": {"page": {"TEDS": {"ALL": 0.8}, "TEDS_structure_only": {"ALL": 0.9}}},
            "display_formula": {"page": {"CDM": {"ALL": 0.5}}},
        }
        summary = _summary(metric_result)
        assert summary["text_edit"] == 0.2
        assert summary["table_teds_s"] == 0.9
        assert summary["overall"] == pytest.approx((80.0 + 80.0 + 50.0) / 3)
        del metric_result["display_formula"]
        assert "overall" not in _summary(metric_result)

    def test_failures_come_from_the_debug_counters(self):
        metric_result = {
            "table": {"metric_debug": {"TEDS": {"error_case_count": 1, "timeout_case_count": 2}}},
            "display_formula": {"metric_debug": {"CDM": {"exception_case_count": 1}}},
            "match_debug": {"text_match_fallback_counts": {"timeout": 2}},
        }
        assert _count_failures(metric_result, "score is set to 0") == 6

    def test_failures_fall_back_to_the_log(self):
        log = "page a score is set to 0\npage b score is set to 0\n"
        assert _count_failures({}, log) == 2


#: Three pages: an English page with a table and a formula, a Chinese page whose table the
#: evaluator matched nothing to, and an English page with neither.
_PAGES = {
    "p1.jpg": (
        {"language": "english", "has_table": True, "has_formula": True},
        {
            "text_edit": 0.2,
            "read_order_edit": 0.1,
            "formula_edit": 0.3,
            "table_teds": 0.8,
            "table_teds_s": 0.9,
            "formula_cdm": 0.6,
        },
    ),
    "p2.jpg": (
        {"language": "simplified_chinese", "has_table": True, "has_formula": False},
        {"text_edit": 0.4},
    ),
    "p3.jpg": (
        {"language": "english", "has_table": False, "has_formula": False},
        {"text_edit": 0.0},
    ),
}


def _omnidocbench_responses(task_name: str, monkeypatch, evaluate) -> tuple:
    """The task, with its data and evaluator stubbed, and one generated response per page."""
    task = get_task(task_name)
    gt_pages = [{"page_info": {"image_path": f"images/{name}"}} for name in _PAGES]
    monkeypatch.setattr(omnidocbench_task, "_data_dir", lambda version: None)
    monkeypatch.setattr(omnidocbench_task, "_load_pages", lambda data_dir: gt_pages)
    monkeypatch.setattr(omnidocbench_task, "run_official_evaluation", evaluate)
    responses = [
        _response({"id": name, "image_name": name, **attributes}, f"page {name}")
        for name, (attributes, _) in _PAGES.items()
    ]
    return task, responses


def _omnidocbench(task_name: str, monkeypatch, evaluate) -> tuple:
    task, responses = _omnidocbench_responses(task_name, monkeypatch, evaluate)
    asyncio.run(task.score_responses(responses))
    return task, responses, task.compute_metrics(responses)


def _evaluation(predictions, gt_pages, *, with_cdm, version) -> OfficialEvaluation:
    assert sorted(predictions) == sorted(_PAGES)
    assert with_cdm
    return OfficialEvaluation(
        per_page={name: dict(result) for name, (_, result) in _PAGES.items()},
        num_scorer_failures=3,
        workers=16,
    )


class TestOmniDocBenchTask:
    def test_page_averages_zero_fill_on_v16(self, monkeypatch):
        _, _, metrics = _omnidocbench("omnidocbench", monkeypatch, _evaluation)
        assert metrics["text_edit"]["omnidocbench"] == pytest.approx(0.2)
        # p2's unmatched table counts as zero.
        assert metrics["table_teds"]["omnidocbench"] == pytest.approx(40.0)
        assert metrics["formula_cdm"]["omnidocbench"] == pytest.approx(60.0)
        assert metrics["text_edit_english"]["omnidocbench"] == pytest.approx(0.1)
        assert metrics["overall"]["omnidocbench"] == pytest.approx((80.0 + 40.0 + 60.0) / 3)

    def test_page_averages_skip_missing_pages_on_v15(self, monkeypatch):
        _, _, metrics = _omnidocbench("omnidocbench_v15", monkeypatch, _evaluation)
        assert metrics["table_teds"]["omnidocbench"] == pytest.approx(80.0)
        assert metrics["overall"]["omnidocbench"] == pytest.approx((80.0 + 80.0 + 60.0) / 3)

    def test_run_level_facts_are_reported(self, monkeypatch):
        _, _, metrics = _omnidocbench("omnidocbench", monkeypatch, _evaluation)
        assert metrics["n_scorer_failures"] == {"omnidocbench": 3.0}
        assert metrics["eval_workers"] == {"omnidocbench": 16.0}

    def test_predictions_keep_each_page_result(self, monkeypatch):
        _, responses, _ = _omnidocbench("omnidocbench", monkeypatch, _evaluation)
        predictions = build_predictions(responses)
        for prediction, (_, result) in zip(predictions, _PAGES.values(), strict=True):
            assert prediction["model_output"][0]["scorer_results"]["omnidocbench"] == result

    def test_instance_metrics_are_each_metric_own_value(self, monkeypatch):
        task, responses, _ = _omnidocbench("omnidocbench", monkeypatch, _evaluation)
        p1, p2, p3 = (
            p["instance_metrics"] for p in build_predictions(responses, task.config.metrics)
        )
        # The edit distance itself, not the scorer channel's 1 - edit distance.
        assert p1["text_edit"]["omnidocbench"] == pytest.approx(0.2)
        assert p1["omnidocbench"]["omnidocbench"] == pytest.approx(0.8)
        assert p1["table_teds"]["omnidocbench"] == pytest.approx(80.0)
        assert p1["overall"]["omnidocbench"] == pytest.approx((80.0 + 80.0 + 60.0) / 3)
        assert p2["table_teds"]["omnidocbench"] == 0.0
        assert p2["overall"]["omnidocbench"] == pytest.approx((60.0 + 0.0) / 2)
        assert "text_edit_english" not in p2
        assert "table_teds" not in p3
        assert p3["overall"]["omnidocbench"] == pytest.approx(100.0)

    def test_instance_metrics_skip_missing_pages_on_v15(self, monkeypatch):
        task, responses, _ = _omnidocbench("omnidocbench_v15", monkeypatch, _evaluation)
        p2 = build_predictions(responses, task.config.metrics)[1]["instance_metrics"]
        assert "table_teds" not in p2
        assert p2["overall"]["omnidocbench"] == pytest.approx(60.0)

    def test_edit_distances_are_lower_is_better(self):
        task = get_task("omnidocbench")
        for name in ("text_edit", "formula_edit", "read_order_edit", "text_edit_english"):
            assert _metric(task, name).pairwise_higher_is_better() is False
        for name in ("overall", "table_teds", "formula_cdm"):
            assert _metric(task, name).pairwise_higher_is_better() is True

    def test_evaluator_failure_is_an_infrastructure_error(self, monkeypatch):
        def fail(*args, **kwargs):
            raise RuntimeError("evaluator crashed")

        task, responses = _omnidocbench_responses("omnidocbench", monkeypatch, fail)
        tracker = TaskTracker(
            model_name="m",
            spec="omnidocbench",
            task=task,
            total_instances=len(responses),
            completed_count=len(responses),
            responses=dict(enumerate(responses)),
        )
        result = asyncio.run(finalize_task(tracker))

        assert result.error is not None
        assert result.metrics == {}
        # Every page is still saved, each with the failure that cost it its score.
        assert len(result.predictions) == len(_PAGES)
        for prediction in result.predictions:
            output = prediction["model_output"][0]
            assert "scorer_results" not in output
            error = output["scoring_errors"]["omnidocbench"]
            assert error["infrastructure"] == "true"
            assert error["type"] == "RuntimeError"
            assert error["message"] == "evaluator crashed"
