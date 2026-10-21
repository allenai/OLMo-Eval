"""Tests for the PopQA task."""

from __future__ import annotations

import unittest

from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.popqa import PopQAContainsScorer


def _make_instance(aliases: list[str]) -> Instance:
    return Instance(
        question="Q: What is X? A:",
        gold_answer=aliases[0],
        metadata={"id": "test", "aliases": aliases},
    )


class TestPopQATask(unittest.TestCase):
    def test_registered(self) -> None:
        task = get_task("popqa")
        self.assertEqual(task.request_type, RequestType.COMPLETION)
        self.assertEqual(task.config.num_fewshot, 15)
        self.assertEqual(task.config.sampling_params.max_tokens, 15)
        self.assertEqual(task.config.sampling_params.stop_sequences, ("\n\n",))

    def test_metrics(self) -> None:
        """First-line containment is primary; whole-output containment stays."""
        task = get_task("popqa")
        self.assertEqual(
            {metric.name for metric in task.config.metrics},
            {"exact_match_first_line", "exact_match_containment"},
        )
        primary = task.config.get_primary_metric()
        assert primary is not None
        self.assertEqual(primary.name, "exact_match_first_line")

    def test_chat_variant(self) -> None:
        task = get_task("popqa:chat")
        self.assertEqual(task.request_type, RequestType.CHAT)
        self.assertIsNone(task.config.sampling_params.stop_sequences)
        self.assertEqual(task.config.sampling_params.temperature, 0.6)

    def test_both_variants_strip_reasoning(self) -> None:
        """A completed trace fits even the base task's 15-token budget."""
        self.assertTrue(get_task("popqa").config.strip_thinking)
        self.assertTrue(get_task("popqa:chat").config.strip_thinking)

    def test_process_doc(self) -> None:
        task = get_task("popqa")
        doc = {
            "id": 1,
            "prop_id": 22,
            "question": "What is George Rankin's occupation?",
            "obj": "politician",
            "possible_answers": '["politician", "political leader"]',
            "s_pop": 100,
        }
        instance = task.process_doc(doc, index=0)
        assert instance is not None
        self.assertEqual(instance.question, "Q: What is George Rankin's occupation? A:")
        self.assertEqual(instance.gold_answer, "politician")
        self.assertEqual(instance.metadata["aliases"], ["politician", "political leader"])
        self.assertEqual(instance.metadata["subject_popularity"], 100)

    def test_fewshot_prompt_format(self) -> None:
        task = get_task("popqa")
        instance = _make_instance(["answer"])
        request = task.format_request(instance)
        self.assertEqual(request.request_type, RequestType.COMPLETION)
        examples = request.prompt.split("\n\n")
        self.assertEqual(len(examples), 16)  # 15 shots + the question
        self.assertTrue(examples[0].startswith("Q: "))
        self.assertIn(" A: ", examples[0])
        self.assertTrue(request.prompt.endswith("Q: What is X? A:"))

    def _task_with_fewshot(self, num_fewshot: int):
        """``get_task`` hands back a shared config, so restore it afterwards."""
        task = get_task("popqa")
        original = task.config.num_fewshot
        self.addCleanup(setattr, task.config, "num_fewshot", original)
        task.config.num_fewshot = num_fewshot
        return task

    def test_zero_shot_override_drops_all_examples(self) -> None:
        """A zero-shot override must not fall back to the full fixed set."""
        task = self._task_with_fewshot(0)
        self.assertEqual(task._build_fewshot(), [])
        request = task.format_request(_make_instance(["answer"]))
        self.assertEqual(request.prompt, "Q: What is X? A:")

    def test_fewshot_truncates_to_requested_count(self) -> None:
        self.assertEqual(len(self._task_with_fewshot(3)._build_fewshot()), 3)

    def test_chat_fewshot_in_single_message(self) -> None:
        task = get_task("popqa:chat")
        request = task.format_request(_make_instance(["answer"]))
        self.assertEqual(request.request_type, RequestType.CHAT)
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0]["role"], "user")
        self.assertEqual(len(request.messages[0]["content"].split("\n\n")), 16)


class TestPopQAContainsScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = PopQAContainsScorer()
        self.first_line = PopQAContainsScorer(name="popqa_contains_first_line", first_line=True)
        self.instance = _make_instance(["United States of America", "USA"])

    def _score(self, text: str) -> float:
        return self.scorer.score(self.instance, LMOutput(text=text))

    def test_first_line_ignores_rambled_hits(self) -> None:
        """An alias appearing only in invented follow-up Q/A must not count."""
        rambled = "France\n\nQ: Where is NASA? A: USA\n\nQ: Who wrote it?"
        self.assertEqual(self.scorer.score(self.instance, LMOutput(text=rambled)), 1.0)
        self.assertEqual(self.first_line.score(self.instance, LMOutput(text=rambled)), 0.0)

    def test_first_line_accepts_answer_on_first_line(self) -> None:
        text = "The USA, of course.\nSome elaboration follows."
        self.assertEqual(self.first_line.score(self.instance, LMOutput(text=text)), 1.0)

    def test_first_line_skips_leading_blank_lines(self) -> None:
        """What a stripped reasoning trace leaves behind: blank lines, then the answer."""
        text = "\n\nThe USA.\n\nQ: Where is NASA? A: USA"
        self.assertEqual(self.first_line.score(self.instance, LMOutput(text=text)), 1.0)
        self.assertEqual(self.first_line.score(self.instance, LMOutput(text="\n\nFrance")), 0.0)

    def test_verbatim_match(self) -> None:
        self.assertEqual(self._score(" United States of America"), 1.0)

    def test_any_alias_matches(self) -> None:
        self.assertEqual(self._score("I believe it is the USA."), 1.0)

    def test_lowercase_alias_in_response(self) -> None:
        # Response is checked against the lowered alias too.
        self.assertEqual(self._score("the united states of america"), 1.0)

    def test_miss(self) -> None:
        self.assertEqual(self._score("Canada"), 0.0)

    def test_empty_response(self) -> None:
        self.assertEqual(self._score(""), 0.0)


if __name__ == "__main__":
    unittest.main()
