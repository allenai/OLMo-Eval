"""Tests for the SciCode external evaluation."""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from parameterized import parameterized

from olmo_eval.common.types import LMOutput, SamplingParams
from olmo_eval.evals.external.benchmarks.scicode import eval as scicode_eval
from olmo_eval.evals.external.benchmarks.scicode import loader as scicode_loader
from olmo_eval.evals.external.benchmarks.scicode import prompts as scicode_prompts
from olmo_eval.evals.external.benchmarks.scicode import verifier as scicode_verifier


def _make_sub_step(
    step_number: str,
    description: str = "Do a thing.",
    header: str = "def foo(x):",
    return_line: str = "    return x",
    background: str | None = None,
    test_cases: tuple[str, ...] = ("assert foo(1) == 1",),
) -> dict[str, Any]:
    return {
        "step_number": step_number,
        "step_description_prompt": description,
        "function_header": header,
        "return_line": return_line,
        "step_background": background,
        "test_cases": test_cases,
    }


def _make_problem(
    problem_id: str = "99",
    num_steps: int = 2,
    required_dependencies: str = "import numpy as np",
) -> scicode_loader.SciCodeProblem:
    sub_steps = [
        _make_sub_step(f"{problem_id}.{i + 1}", description=f"Step {i + 1}")
        for i in range(num_steps)
    ]
    return scicode_loader.SciCodeProblem(
        problem_id=problem_id,
        problem_name=f"problem_{problem_id}",
        required_dependencies=required_dependencies,
        sub_steps=sub_steps,
    )


class TestPrompts(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_build_step_prompt_contains_required_sections(self, with_background: bool) -> None:
        problem = _make_problem(num_steps=3)
        previous = ["def step1():\n    return 1", None, None]
        prompt = scicode_prompts.build_step_prompt(
            sub_steps=problem.sub_steps,
            required_dependencies=problem.required_dependencies,
            step_idx=1,
            previous_llm_code=previous,
            with_background=with_background,
        )
        self.assertIn("PROBLEM DESCRIPTION", prompt)
        self.assertIn("def step1()", prompt)
        self.assertIn("import numpy as np", prompt)
        self.assertIn(problem.sub_steps[1]["function_header"], prompt)

    def test_extract_step_code_strips_imports(self) -> None:
        text = (
            "```python\nimport numpy as np\nfrom scipy import linalg\n\n"
            "def foo():\n    return 1\n```"
        )
        code = scicode_prompts.extract_step_code(text)
        self.assertNotIn("import", code)
        self.assertIn("def foo()", code)

    def test_extract_step_code_returns_last_python_block(self) -> None:
        text = (
            "Here is some scratch work:\n"
            "```python\ndef scratch():\n    return 0\n```\n"
            "And the final answer:\n"
            "```python\ndef final():\n    return 1\n```\n"
        )
        code = scicode_prompts.extract_step_code(text)
        self.assertIn("def final()", code)
        self.assertNotIn("def scratch()", code)

    def test_extract_step_code_returns_empty_when_no_python_fence(self) -> None:
        self.assertEqual(scicode_prompts.extract_step_code(""), "")
        self.assertEqual(
            scicode_prompts.extract_step_code("```\ndef bar():\n    return 2\n```"), ""
        )


class TestVerifierScript(unittest.TestCase):
    def test_build_step_script_includes_all_sections(self) -> None:
        step = _make_sub_step(
            step_number="99.1",
            test_cases=("assert foo(2) == 2", "assert foo(3) == 3"),
        )
        script = scicode_verifier.build_step_script(
            step=step,
            required_dependencies="import numpy as np",
            full_code="def foo(x):\n    return x",
            hardcoded_prelude="# hardcoded",
            h5py_file="/tmp/test.h5",
        )
        self.assertIn("process_hdf5_to_tuple('99.1', 2, '/tmp/test.h5')", script)
        self.assertIn("target = targets[0]", script)
        self.assertIn("target = targets[1]", script)
        self.assertIn("assert foo(2) == 2", script)
        self.assertIn("# hardcoded", script)
        self.assertIn("def foo(x)", script)


class FakeProvider:
    """Minimal async provider that returns fixed code blocks per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, SamplingParams | None]] = []
        self.model_name = "fake-model"

    async def agenerate(
        self, requests: Any, sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        text = self._responses.pop(0) if self._responses else ""
        self.calls.append((requests[0].messages[0]["content"], sampling_params))
        return [[LMOutput(text=text)]]


class TestRunProblemCascade(unittest.IsolatedAsyncioTestCase):
    async def test_cascade_generates_one_call_per_scorable_step(self) -> None:
        problem = _make_problem(problem_id="99", num_steps=3)
        responses = [
            "```python\ndef a():\n    return 1\n```",
            "```python\ndef b():\n    return 2\n```",
            "```python\ndef c():\n    return 3\n```",
        ]
        provider = FakeProvider(responses)
        sc_args = scicode_eval.SciCodeConfig(max_concurrency=1, with_background=True)
        evaluator = scicode_eval.SciCodeExternalEval()

        async def fake_verify(**_kwargs: Any) -> list[bool]:
            return [True, True, True]

        with mock.patch.object(evaluator, "_verify", side_effect=fake_verify):
            result = await evaluator._run_problem(
                problem=problem,
                provider=provider,
                sampling_params=SamplingParams(
                    max_tokens=sc_args.max_tokens, temperature=sc_args.temperature
                ),
                sc_args=sc_args,
                container_runtime="podman",
            )

        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["passed"], 3)
        self.assertTrue(result["all_passed"])

    async def test_hardcoded_snippet_is_not_generated(self) -> None:
        problem = _make_problem(problem_id="62", num_steps=3)
        responses = [
            "```python\ndef a():\n    return 1\n```",
            "```python\ndef b():\n    return 2\n```",
        ]
        provider = FakeProvider(responses)
        sc_args = scicode_eval.SciCodeConfig(max_concurrency=1, with_background=True)
        evaluator = scicode_eval.SciCodeExternalEval()

        async def fake_verify(**_kwargs: Any) -> list[bool]:
            return [False, True]

        with mock.patch.object(evaluator, "_verify", side_effect=fake_verify):
            result = await evaluator._run_problem(
                problem=problem,
                provider=provider,
                sampling_params=SamplingParams(
                    max_tokens=sc_args.max_tokens, temperature=sc_args.temperature
                ),
                sc_args=sc_args,
                container_runtime="podman",
            )

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["passed"], 1)
        self.assertNotIn(0, result["step_codes"])

    async def test_cascade_embeds_previous_step_code_in_later_prompts(self) -> None:
        problem = _make_problem(problem_id="99", num_steps=3)
        responses = [
            "```python\ndef step_one():\n    return 'ALPHA_MARKER'\n```",
            "```python\ndef step_two():\n    return 'BETA_MARKER'\n```",
            "```python\ndef step_three():\n    return 'GAMMA_MARKER'\n```",
        ]
        provider = FakeProvider(responses)
        sc_args = scicode_eval.SciCodeConfig(max_concurrency=1, with_background=True)
        evaluator = scicode_eval.SciCodeExternalEval()

        async def fake_verify(**_kwargs: Any) -> list[bool]:
            return [True, True, True]

        with mock.patch.object(evaluator, "_verify", side_effect=fake_verify):
            await evaluator._run_problem(
                problem=problem,
                provider=provider,
                sampling_params=SamplingParams(
                    max_tokens=sc_args.max_tokens, temperature=sc_args.temperature
                ),
                sc_args=sc_args,
                container_runtime="podman",
            )

        first_prompt = provider.calls[0][0]
        self.assertNotIn("ALPHA_MARKER", first_prompt)
        self.assertNotIn("BETA_MARKER", first_prompt)

        second_prompt = provider.calls[1][0]
        self.assertIn("def step_one()", second_prompt)
        self.assertIn("ALPHA_MARKER", second_prompt)
        self.assertNotIn("BETA_MARKER", second_prompt)

        third_prompt = provider.calls[2][0]
        self.assertIn("ALPHA_MARKER", third_prompt)
        self.assertIn("BETA_MARKER", third_prompt)
        self.assertIn("def step_two()", third_prompt)


if __name__ == "__main__":
    unittest.main()


class TestRunProblemContextOverflow(unittest.IsolatedAsyncioTestCase):
    async def test_a_context_overflow_fails_the_problem_instead_of_scoring_empty_code(
        self,
    ) -> None:
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        # A server that rejects the prompt as too long. The provider hands that back
        # as a marked reply, which a caller reading the text directly must not take
        # for an empty answer.
        provider = object.__new__(VLLMServerProvider)
        provider._tool_calls_parsed = None
        provider.max_concurrency = 1
        provider.max_retries = 0

        async def overflow(*_args: Any, **_kwargs: Any) -> list[LMOutput]:
            raise ValueError("This model's maximum context length is 40960 tokens.")

        provider._generate_single_async = overflow  # ty: ignore[invalid-assignment]
        evaluator = scicode_eval.SciCodeExternalEval()
        sc_args = scicode_eval.SciCodeConfig(max_concurrency=1, with_background=True)

        with self.assertRaisesRegex(RuntimeError, "maximum context length"):
            await evaluator._run_problem(
                problem=_make_problem(problem_id="99", num_steps=1),
                provider=provider,
                sampling_params=SamplingParams(max_tokens=16),
                sc_args=sc_args,
                container_runtime="podman",
            )
