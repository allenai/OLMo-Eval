"""Multi-image scoring, prompt construction and MMIU parsing."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.vision.benchmarks.mmiu import _METRICS, _extract_options, _split_image_path
from olmo_eval.evals.vision.scoring.multi_image import (
    multi_image_mc_score,
    strip_multi_image_response,
)
from olmo_eval.evals.vision.tasks.multi_image import replace_images


class TestStripMultiImageResponse:
    def test_text_after_answer_marker_wins(self) -> None:
        assert strip_multi_image_response("Reasoning here. Answer: B. the dog") == "B. the dog"

    def test_multiline_takes_the_most_frequent_line(self) -> None:
        assert strip_multi_image_response("A\nB\n  B  ") == "B"

    def test_multiline_tie_takes_the_first(self) -> None:
        assert strip_multi_image_response("C\nD") == "C"

    def test_single_line_normalizes_whitespace(self) -> None:
        assert strip_multi_image_response("  the   second  image ") == "the second image"


class TestMultiImageMcScore:
    OPTIONS = ["the first image", "the second image", "the third image"]

    def test_letter_matches_gold(self) -> None:
        assert multi_image_mc_score("B", "B. the second image", self.OPTIONS) == 1.0
        assert multi_image_mc_score("A", "B. the second image", self.OPTIONS) == 0.0

    def test_option_text_matches_gold(self) -> None:
        response = "I think it is clearly the third image in the set."
        assert multi_image_mc_score("C", response, self.OPTIONS) == 1.0

    def test_gold_outside_the_options_scores_zero(self) -> None:
        # MMIU has gold letters that are not among the listed options.
        assert multi_image_mc_score("E", "E", self.OPTIONS) == 0.0

    def test_unparseable_response_guesses_deterministically(self) -> None:
        scores = {
            multi_image_mc_score("A", "no idea", self.OPTIONS, stable_id="row-7") for _ in range(5)
        }
        assert len(scores) == 1


class TestReplaceImages:
    def test_numbers_through_the_question_then_the_options(self) -> None:
        question, options = replace_images("<image> or <image>?", ["<image>", "neither"])
        assert question == "Image 1 or Image 2?"
        assert options == ["Image 3", "neither"]

    def test_max_images_leaves_later_placeholders(self) -> None:
        question, options = replace_images("<image> <image>", ["<image>"], max_images=2)
        assert question == "Image 1 Image 2"
        assert options == ["<image>"]


class TestMmiuParsing:
    def test_extract_options(self) -> None:
        assert _extract_options("A: a cat\nB: a dog\nC: a bird") == ["a cat", "a dog", "a bird"]

    def test_extract_options_rejects_out_of_order_letters(self) -> None:
        with pytest.raises(ValueError, match="letter order"):
            _extract_options("A: a cat\nC: a dog")

    def test_split_image_path(self) -> None:
        path = "./Low-level-semantic/forensic/img_1.jpg"
        assert _split_image_path(path, 0) == (
            "Low-level-semantic/forensic/img_1.jpg",
            "Low-level-semantic",
        )

    @pytest.mark.parametrize("path", ["Low-level-semantic/forensic/img.jpg", "./img.jpg"])
    def test_split_image_path_rejects_other_layouts(self, path: str) -> None:
        with pytest.raises(ValueError, match="relationship"):
            _split_image_path(path, 3)


class TestMmiuImageCountBuckets:
    """mm_olmo buckets are <=10, 11-20 and >20 images."""

    @pytest.mark.parametrize(
        ("num_images", "bucket"),
        [
            (10, "num_images<=10"),
            (11, "num_images<=20"),
            (20, "num_images<=20"),
            (21, "num_images>20"),
        ],
    )
    def test_boundaries(self, num_images: int, bucket: str) -> None:
        response = Response(
            instance=Instance(question="q", gold_answer=None, metadata={"num_images": num_images}),
            request=LMRequest(request_type=RequestType.CHAT, prompt="q"),
            outputs=[LMOutput(text="A")],
            scores={"multi_image_mc": 1.0},
        )
        buckets = {m.name: m for m in _METRICS if m.name.startswith("num_images")}
        assert {name: m.compute_instance(response) for name, m in buckets.items()} == {
            name: (1.0 if name == bucket else None) for name in buckets
        }
