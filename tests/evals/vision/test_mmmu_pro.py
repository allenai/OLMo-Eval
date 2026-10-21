"""Unit tests for MMMU-Pro (parser, prompts, image-order, metrics, request).

Pure CPU — no GPU, no model, no dataset. The parser/prompt logic mirrors the official
``MMMU-Benchmark/MMMU`` ``mmmu-pro`` repo.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from olmo_eval.common.images import resolve_images
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.vision.benchmarks.mmmu_pro import (
    _SCORER,
    MmmuProOverallMetric,
    MmmuProSettingMetric,
)
from olmo_eval.evals.vision.scoring.mmmu_pro import (
    MMMU_PRO_STANDARD_DIRECT,
    MMMU_PRO_VISION_DIRECT,
    construct_standard_prompt,
    get_multi_choice_info,
    mmmu_pro_score,
    parse_multi_choice_response,
    replace_images_tokens,
)

_OPTS10 = [f"opt{i}" for i in range(10)]
_I2A, _CHOICES = get_multi_choice_info(_OPTS10)  # A..J


class TestParseMultiChoiceResponse:
    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("blah Answer: J", "J"),  # rfind("Answer:") priority
            ("I think (J) is right", "J"),  # (X)
            ("the answer is J here", "J"),  # "X "
            ("J.", "J"),  # "X."
            ("the correct choice is opt3 obviously here", "D"),  # option text (>5 tokens)
            ("(A) no wait (C)", "C"),  # multiple -> last occurrence
            ("(A) ... Answer: D", "D"),  # Answer: overrides an earlier bracket letter
        ],
    )
    def test_ladder(self, response: str, expected: str) -> None:
        assert parse_multi_choice_response(response, _CHOICES, _I2A, seed=1) == expected

    def test_ten_options_reach_j(self) -> None:
        assert "J" in _CHOICES and len(_CHOICES) == 10

    def test_fallback_is_seeded_deterministic(self) -> None:
        a = parse_multi_choice_response("xyzzy", _CHOICES, _I2A, seed=123)
        b = parse_multi_choice_response("xyzzy", _CHOICES, _I2A, seed=123)
        assert a == b and a in _CHOICES

    def test_score_match(self) -> None:
        assert mmmu_pro_score("Answer: B", _OPTS10, "B", example_id="x") == 1.0
        assert mmmu_pro_score("Answer: A", _OPTS10, "B", example_id="x") == 0.0


class TestPromptsAndImageOrder:
    def test_replace_images_tokens_shuffled(self) -> None:
        text, order = replace_images_tokens("Compare <image 2> and <image 1>.\nA. <image 3>\nB. x")
        assert order == [2, 1, 3]  # token-appearance order, not key order
        assert "<image 2>" not in text and text.count("<image>") == 3

    def test_construct_standard_prompt(self) -> None:
        prompt = construct_standard_prompt("Q?", ["a", "b"])
        assert prompt == f"Q?\nA. a\nB. b\n{MMMU_PRO_STANDARD_DIRECT}"

    def test_prompt_constants_verbatim(self) -> None:
        assert MMMU_PRO_STANDARD_DIRECT == (
            "Answer with the option letter from the given choices directly."
        )
        assert MMMU_PRO_VISION_DIRECT.startswith(
            "Answer with the option letter from the given choices directly. The last line"
        )


def _resp(setting: str, score: float) -> Response:
    instance = Instance(question="q", metadata={"mmmu_pro_setting": setting})
    output = LMOutput(text="", metadata={})
    response = Response(instance=instance, request=None, outputs=[output])
    response.scores["mmmu_pro"] = score
    return response


class TestMetrics:
    RESPONSES = [
        _resp("standard10", 1.0),
        _resp("standard10", 0.0),
        _resp("standard4", 1.0),
        _resp("vision", 1.0),
        _resp("vision", 0.0),
    ]

    def test_per_setting(self) -> None:
        assert MmmuProSettingMetric(name="s", scorer=_SCORER, setting="standard10").compute(
            self.RESPONSES
        ) == pytest.approx(0.5)
        assert MmmuProSettingMetric(name="s", scorer=_SCORER, setting="standard4").compute(
            self.RESPONSES
        ) == pytest.approx(1.0)
        assert MmmuProSettingMetric(name="s", scorer=_SCORER, setting="vision").compute(
            self.RESPONSES
        ) == pytest.approx(0.5)

    def test_overall_ignores_standard4(self) -> None:
        # (mean(standard10)=0.5 + mean(vision)=0.5) / 2
        assert MmmuProOverallMetric(name="overall", scorer=_SCORER).compute(
            self.RESPONSES
        ) == pytest.approx(0.5)

    def test_empty(self) -> None:
        assert MmmuProOverallMetric(name="overall", scorer=_SCORER).compute([]) == 0.0


# Stand-ins for decoded images: the resolver treats strings as paths, so use objects.
IMG1 = SimpleNamespace(mode="RGB", tag=1)
IMG2 = SimpleNamespace(mode="RGB", tag=2)


class TestFormatRequest:
    def test_standard_multi_image(self) -> None:
        task = get_task("mmmu_pro")
        inst = Instance(
            question="Q <image>",
            metadata={"images": [lambda: IMG1, lambda: IMG2, lambda: None]},
        )
        req = task.format_request(inst)
        # lazy entries travel on the request; the provider resolves them, dropping None
        assert resolve_images(req.images) == (IMG1, IMG2)
        assert req.messages[0]["content"] == "Q <image>"

    def test_vision_single_image(self) -> None:
        task = get_task("mmmu_pro")
        inst = Instance(question=MMMU_PRO_VISION_DIRECT, metadata={"image": IMG1})
        req = task.format_request(inst)
        assert req.images == (IMG1,)
