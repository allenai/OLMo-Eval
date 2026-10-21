"""Tests for the LongBench v2 tasks."""

import pytest

from olmo_eval.common.types import LMOutput, RequestType, Response
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.longbench_v2 import (
    DEV_DOMAINS,
    DEV_GROUP,
    HELDOUT_GROUP,
    LONGBENCH_V2_REVISION,
    SPLIT_GROUP_KEY,
    SplitGroupAccuracyMetric,
    extract_answer,
)


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


def _doc(domain: str, answer: str = "B") -> dict:
    return {
        "_id": f"id-{domain}",
        "domain": domain,
        "sub_domain": "sub",
        "difficulty": "hard",
        "length": "short",
        "question": " Which one? ",
        "choice_A": "alpha",
        "choice_B": "beta ",
        "choice_C": "gamma",
        "choice_D": "delta",
        "answer": answer,
        "context": "Context with {braces} and $DOC$ markers.",
    }


def _response(task, domain: str, text: str) -> Response:
    instance = task.process_doc(_doc(domain))
    assert instance is not None
    return Response(
        instance=instance, request=task.format_request(instance), outputs=[LMOutput(text=text)]
    )


class TestRegistration:
    @pytest.mark.parametrize("task_name", ["longbench_v2", "longbench_v2_dev"])
    def test_registered_as_chat(self, task_name):
        assert task_name in list_tasks()
        task = get_task(task_name)
        assert task.request_type == RequestType.CHAT
        assert task.config.split.value == "train"

    def test_sampling_matches_reference_direct_answer_setting(self):
        params = get_task("longbench_v2").config.sampling_params
        assert params.max_tokens == 128
        assert params.temperature == 0.1
        assert params.truncate_prompt_tokens == -1
        assert params.truncation_side == "left"

    def test_full_task_reports_three_scores(self):
        task = get_task("longbench_v2")
        assert [m.name for m in task.config.metrics] == [
            "accuracy",
            "heldout_accuracy",
            "dev_accuracy",
        ]
        assert task.config.primary_metric.name == "accuracy"

    def test_dev_task_reports_accuracy_only(self):
        assert [m.name for m in get_task("longbench_v2_dev").config.metrics] == ["accuracy"]

    def test_dataset_revision_is_pinned(self):
        assert get_task("longbench_v2").config.to_dict()["data_source"]["revision"] == (
            LONGBENCH_V2_REVISION
        )


class TestProcessDoc:
    def test_prompt_follows_reference_template(self):
        instance = get_task("longbench_v2").process_doc(_doc("Single-Document QA"), index=3)
        assert instance is not None
        assert instance.question.startswith(
            "Please read the following text and answer the question below.\n\n<text>\n"
            "Context with {braces} and $DOC$ markers.\n</text>\n\n"
            "What is the correct answer to this question: Which one?\nChoices:\n"
            "(A) alpha\n(B) beta\n(C) gamma\n(D) delta\n\n"
        )
        assert instance.question.endswith(
            'Format your response as follows: "The correct answer is (insert answer here)".'
        )
        assert instance.gold_answer == "B"
        assert instance.choices == ("alpha", "beta", "gamma", "delta")
        assert instance.metadata["id"] == "id-Single-Document QA"
        assert instance.metadata["index"] == 3

    @pytest.mark.parametrize("domain", sorted(DEV_DOMAINS))
    def test_dev_domains_tagged_dev(self, domain):
        instance = get_task("longbench_v2").process_doc(_doc(domain))
        assert instance is not None
        assert instance.metadata[SPLIT_GROUP_KEY] == DEV_GROUP

    @pytest.mark.parametrize(
        "domain",
        [
            "Single-Document QA",
            "Multi-Document QA",
            "Long In-context Learning",
            "Long-dialogue History Understanding",
        ],
    )
    def test_other_domains_tagged_heldout(self, domain):
        instance = get_task("longbench_v2").process_doc(_doc(domain))
        assert instance is not None
        assert instance.metadata[SPLIT_GROUP_KEY] == HELDOUT_GROUP

    def test_dev_task_keeps_only_dev_domains(self):
        task = get_task("longbench_v2_dev")
        assert task.process_doc(_doc("Single-Document QA")) is None
        assert task.process_doc(_doc("Code Repository Understanding")) is not None
        assert task.process_doc(_doc("Long Structured Data Understanding")) is not None

    @pytest.mark.parametrize(
        "patch",
        [{"question": ""}, {"choice_C": ""}, {"answer": "E"}, {"answer": ""}],
    )
    def test_skips_malformed_docs(self, patch):
        doc = {**_doc("Single-Document QA"), **patch}
        assert get_task("longbench_v2").process_doc(doc) is None

    def test_single_user_message(self):
        task = get_task("longbench_v2")
        instance = task.process_doc(_doc("Single-Document QA"))
        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert len(request.messages) == 1
        assert request.messages[0]["role"] == "user"
        assert request.messages[0]["content"] == instance.question


class TestExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The correct answer is (B)", "B"),
            ("**The correct answer is (C)**", "C"),
            ("The correct answer is D.", "D"),
            ("The correct answer is (A). On reflection, The correct answer is (B)", "A"),
            ("Because of X, the correct answer is (A). The correct answer is (B)", "B"),
            ("I think B", None),
            ("", None),
        ],
    )
    def test_reference_extraction(self, text, expected):
        assert extract_answer(text) == expected


class TestMetrics:
    def test_scores_split_by_group_from_one_run(self):
        task = get_task("longbench_v2")
        responses = [
            _response(task, "Single-Document QA", "The correct answer is (B)"),
            _response(task, "Multi-Document QA", "The correct answer is (A)"),
            _response(task, "Code Repository Understanding", "The correct answer is (B)"),
            _response(task, "Long Structured Data Understanding", "The correct answer is (B)"),
        ]
        import asyncio

        asyncio.run(task.score_responses(responses))
        metrics = task.compute_metrics(responses)
        assert metrics["accuracy"]["multiple_choice"] == 0.75
        assert metrics["heldout_accuracy"]["multiple_choice"] == 0.5
        assert metrics["dev_accuracy"]["multiple_choice"] == 1.0

    def test_group_metric_instance_values_outside_group_are_none(self):
        task = get_task("longbench_v2")
        heldout_metric = task.config.metrics[1]
        response = _response(task, "Code Repository Understanding", "The correct answer is (B)")
        assert heldout_metric.compute_instance(response) is None


class TestSplitGroupSerialization:
    def test_split_group_distinguishes_identically_named_metrics(self):
        """Two groups under one metric name must not serialize, and so hash, alike."""
        dev = SplitGroupAccuracyMetric(name="subset_accuracy", split_group=DEV_GROUP)
        heldout = SplitGroupAccuracyMetric(name="subset_accuracy", split_group=HELDOUT_GROUP)
        assert dev.to_dict() != heldout.to_dict()

    def test_split_group_reaches_the_task_config(self):
        serialized = get_task("longbench_v2").config.to_dict()["metrics"]
        groups = {m["name"]: m.get("split_group") for m in serialized}
        assert groups == {
            "accuracy": None,
            "heldout_accuracy": HELDOUT_GROUP,
            "dev_accuracy": DEV_GROUP,
        }
