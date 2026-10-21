"""Tests for IFBench tasks (ifeval_ood + ifeval_mt) and the IFEval scoring stack."""

from __future__ import annotations

import unittest
from typing import Any

from olmo_eval.common.metrics import (
    IFEvalInstLooseAccuracy,
    IFEvalInstStrictAccuracy,
    IFEvalPromptLooseAccuracy,
    IFEvalPromptStrictAccuracy,
)
from olmo_eval.common.scorers import IFEvalScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common import get_task


def _make_instance(
    prompt: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
) -> Instance:
    return Instance(
        question=prompt,
        gold_answer=None,
        metadata={
            "id": "test",
            "key": "test",
            "prompt": prompt,
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs_list,
        },
    )


def _make_response(instance: Instance, response_text: str) -> Response:
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question),
        outputs=[LMOutput(text=response_text)],
    )


class TestIFEvalOODTask(unittest.TestCase):
    def test_registered(self) -> None:
        task = get_task("ifeval_ood")
        self.assertIsNotNone(task)
        metric_names = {m.name for m in task.config.metrics}
        self.assertEqual(
            metric_names,
            {
                "prompt_level_strict_acc",
                "prompt_level_loose_acc",
                "inst_level_strict_acc",
                "inst_level_loose_acc",
            },
        )
        self.assertEqual(task.request_type, RequestType.CHAT)

    def test_process_doc_strips_none_kwargs(self) -> None:
        task = get_task("ifeval_ood")
        doc = {
            "key": 0,
            "prompt": "hi",
            "instruction_id_list": ["count:numbers"],
            "kwargs": [{"N": 2, "keyword": None, "frequency": None}],
        }
        instance = task.process_doc(doc, index=0)
        self.assertIsNotNone(instance)
        assert instance is not None
        self.assertEqual(instance.metadata["instruction_id_list"], ["count:numbers"])
        self.assertEqual(instance.metadata["kwargs"], [{"N": 2}])
        self.assertEqual(instance.metadata["key"], 0)


class TestIFEvalMTTask(unittest.TestCase):
    def test_registered_variants(self) -> None:
        for name in (
            "ifeval_mt_wildchat_unused_withRewrite",
            "ifeval_mt_ood_wildchat_unused_withRewrite",
        ):
            task = get_task(name)
            self.assertIsNotNone(task)
            self.assertEqual(task.request_type, RequestType.CHAT)

    def test_process_doc_preserves_messages(self) -> None:
        task = get_task("ifeval_mt_wildchat_unused_withRewrite")
        messages = [
            {"role": "user", "content": "say hi"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "Rewrite with exactly 2 numbers."},
        ]
        doc = {
            "id": "x",
            "key": 0,
            "prompt": "Rewrite with exactly 2 numbers.",
            "instruction_id_list": ["count:numbers"],
            "kwargs": [{"N": 2}],
            "messages": messages,
        }
        instance = task.process_doc(doc, index=0)
        assert instance is not None
        self.assertEqual(len(instance.metadata["messages"]), 3)
        self.assertEqual(instance.metadata["messages"][-1]["content"], messages[-1]["content"])
        request = task.format_request(instance)
        self.assertEqual(request.request_type, RequestType.CHAT)
        self.assertEqual(len(request.messages), 3)


_IFBENCH_TASKS = (
    "ifeval_ood",
    "ifeval_mt_wildchat_unused_withRewrite",
    "ifeval_mt_ood_wildchat_unused_withRewrite",
)


class TestIFBenchSamplingRegime(unittest.TestCase):
    def test_default_budget_is_bounded_only_by_context(self) -> None:
        for name in _IFBENCH_TASKS:
            with self.subTest(task=name):
                config = get_task(name).config
                params = config.sampling_params
                assert params is not None
                self.assertIsNone(params.max_tokens)
                self.assertTrue(params.do_sample)
                self.assertEqual(params.temperature, 0.6)
                self.assertEqual(params.top_p, 0.95)
                self.assertTrue(config.strip_thinking)

    def test_tulu_variant_keeps_greedy_2048_regime(self) -> None:
        for name in _IFBENCH_TASKS:
            with self.subTest(task=name):
                params = get_task(f"{name}:tulu").config.sampling_params
                assert params is not None
                self.assertEqual(params.max_tokens, 2048)
                self.assertFalse(params.do_sample)
                self.assertEqual(params.temperature, 0.0)


class TestIFBenchSuite(unittest.TestCase):
    def test_suite_registered(self) -> None:
        suite = get_suite("ifbench")
        self.assertEqual(
            tuple(suite.tasks),
            (
                "ifeval_mt_wildchat_unused_withRewrite",
                "ifeval_mt_ood_wildchat_unused_withRewrite",
                "ifeval_ood",
            ),
        )


class TestIFEvalScorer(unittest.TestCase):
    """Use ``count:numbers`` (must include exactly N digits) — a deterministic verifier."""

    INSTRUCTION_ID = "count:numbers"
    KWARGS: dict[str, Any] = {"N": 2}

    def _score(self, response_text: str) -> dict[str, list[bool]]:
        instance = _make_instance(
            "Include exactly 2 numbers.", [self.INSTRUCTION_ID], [self.KWARGS]
        )
        output = LMOutput(text=response_text)
        IFEvalScorer().score(instance, output)
        return output.metadata["ifeval"]

    def test_satisfied(self) -> None:
        result = self._score("I have 3 apples and 4 oranges.")
        self.assertEqual(result["strict"], [True])
        self.assertEqual(result["loose"], [True])

    def test_violated_too_few(self) -> None:
        result = self._score("I have apples and oranges.")
        self.assertEqual(result["strict"], [False])
        self.assertEqual(result["loose"], [False])

    def test_violated_too_many(self) -> None:
        result = self._score("I have 1 apple, 2 oranges, and 3 grapes.")
        self.assertEqual(result["strict"], [False])
        self.assertEqual(result["loose"], [False])

    def test_loose_strips_leading_line(self) -> None:
        # Strict fails: leading "Sure!" line adds no number, body has 2.
        # But loose strips first line, leaving exactly 2 — passes.
        result = self._score(
            "Sure! Here's a response with 5 trailing extras at the end.\n"
            "I have 3 apples and 4 oranges."
        )
        self.assertEqual(result["strict"], [False])
        self.assertEqual(result["loose"], [True])

    def test_unbuildable_instruction_scores_false(self) -> None:
        """Dataset kwargs the verifier does not accept must not fail the response.

        ``VGraf/ifeval_mt`` carries one row whose kwargs use ``keyword`` where
        ``length_constraints:nth_paragraph_first_word`` expects ``first_word``;
        the instruction scores as not followed, matching the reference
        implementation.
        """
        instance = _make_instance(
            "Write 3 paragraphs; paragraph 2 starts with a given word.",
            ["length_constraints:nth_paragraph_first_word"],
            [{"keyword": "schedule", "num_paragraphs": 3}],
        )
        output = LMOutput(text="one\n\ntwo\n\nthree")
        score = IFEvalScorer().score(instance, output)
        self.assertEqual(score, 0.0)
        result = output.metadata["ifeval"]
        self.assertEqual(result["strict"], [False])
        self.assertEqual(result["loose"], [False])
        self.assertIn("length_constraints:nth_paragraph_first_word", result["errors"])

    def test_unknown_instruction_id_scores_false(self) -> None:
        """An id absent from the verifier registry is a dataset problem too."""
        instance = _make_instance("Do something.", ["nonexistent:instruction"], [{}])
        output = LMOutput(text="anything")
        self.assertEqual(IFEvalScorer().score(instance, output), 0.0)
        result = output.metadata["ifeval"]
        self.assertEqual(result["strict"], [False])
        self.assertIn("nonexistent:instruction", result["errors"])

    def test_check_time_failure_propagates(self) -> None:
        """A verifier that builds but fails while checking is not a model error.

        Missing NLTK corpora on an offline worker raise ``LookupError`` from
        ``check_following``. Scoring that as an unfollowed instruction would
        silently depress metrics, so it must surface instead.
        """
        from ifbench import instructions_registry

        instruction_id = "words:start_verb"
        registry = instructions_registry.INSTRUCTION_DICT
        original = registry[instruction_id]

        class _FailsWhileChecking(original):  # type: ignore[misc, valid-type]
            def check_following(self, value: str) -> bool:
                raise LookupError("Resource punkt_tab not found.")

        registry[instruction_id] = _FailsWhileChecking
        try:
            instance = _make_instance("Start with a verb.", [instruction_id], [{}])
            with self.assertRaises(LookupError):
                IFEvalScorer().score(instance, LMOutput(text="Run the script."))
        finally:
            registry[instruction_id] = original

    def test_loose_never_stricter_than_strict_with_random_kwargs(self) -> None:
        """``count:word_count_range`` draws random min/max words when unset.

        The verifier must be built once and reused across the strict check and
        all loose variants; rebuilding per check redraws the bounds, letting
        loose (a superset of strict) score below strict. Flip probability is
        ~0.04 per trial under rebuild-per-check, hence 200 trials.
        """
        response_text = " ".join(["word"] * 250)
        for _ in range(200):
            instance = _make_instance(
                "Write between X and Y words.", ["count:word_count_range"], [{}]
            )
            output = LMOutput(text=response_text)
            IFEvalScorer().score(instance, output)
            result = output.metadata["ifeval"]
            if result["strict"][0]:
                self.assertTrue(
                    result["loose"][0],
                    "loose scored False while strict scored True on the same response",
                )


class TestIFEvalMetrics(unittest.TestCase):
    INSTRUCTION_ID = "count:numbers"
    KWARGS: dict[str, Any] = {"N": 2}

    def _scored_response(self, response_text: str) -> Response:
        instance = _make_instance("count.", [self.INSTRUCTION_ID], [self.KWARGS])
        response = _make_response(instance, response_text)
        IFEvalScorer().score(instance, response.outputs[0])
        return response

    def test_aggregation(self) -> None:
        responses = [
            self._scored_response("I have 3 apples and 4 oranges."),  # passes
            self._scored_response("Just words, no digits."),  # fails
        ]
        self.assertAlmostEqual(IFEvalPromptStrictAccuracy().compute(responses), 0.5)
        self.assertAlmostEqual(IFEvalPromptLooseAccuracy().compute(responses), 0.5)
        self.assertAlmostEqual(IFEvalInstStrictAccuracy().compute(responses), 0.5)
        self.assertAlmostEqual(IFEvalInstLooseAccuracy().compute(responses), 0.5)

    def test_multi_instruction_prompt_requires_all(self) -> None:
        instance = _make_instance(
            "Two numbers and an emoji-terminated sentence.",
            ["count:numbers", "format:emoji"],
            [{"N": 2}, {}],
        )
        # Has 2 numbers (passes count) but no emoji at end of sentence (fails emoji).
        response = _make_response(instance, "I have 3 apples and 4 oranges.")
        IFEvalScorer().score(instance, response.outputs[0])
        prompt_acc = IFEvalPromptStrictAccuracy().compute([response])
        inst_acc = IFEvalInstStrictAccuracy().compute([response])
        self.assertEqual(prompt_acc, 0.0)  # not all instructions satisfied
        self.assertEqual(inst_acc, 0.5)  # 1 of 2 satisfied


if __name__ == "__main__":
    unittest.main()
