"""Unit tests for the vision prompt families (the mm_olmo model-prompt builder).

Pure CPU. The parity tests compare against the mm_olmo checkout when present and
skip otherwise.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from olmo_eval.evals.vision.scoring.prompts import POINTING_TEMPLATES, build_pointing_prompt

# model-prompt (_mp) builder
# ---------------------------------------------------------------------------


class TestBuildPointingPrompt:
    def test_none_templates_lowercase_the_label(self) -> None:
        assert (
            build_pointing_prompt(
                "Earring", 0, prompt_templates="none", system_prompt="demo_or_style_v2"
            )
            == "earring"
        )

    def test_style_and_length_prefixes_the_style(self) -> None:
        assert (
            build_pointing_prompt(
                "Earring", 0, prompt_templates="none", system_prompt="style_and_length_v2"
            )
            == "pointing: earring"
        )

    def test_none_templates_ignore_the_index(self) -> None:
        prompts = {build_pointing_prompt("Earring", i, prompt_templates="none") for i in range(20)}
        assert prompts == {"earring"}

    def test_templated_prompts_are_seeded_by_index(self) -> None:
        first = [build_pointing_prompt("Earring", i) for i in range(20)]
        assert first == [build_pointing_prompt("Earring", i) for i in range(20)]
        assert len(set(first)) > 1, "template choice should vary with the index"

    def test_templated_prompt_contains_the_label(self) -> None:
        for i in range(50):
            assert "earring" in build_pointing_prompt("Earring", i).lower()

    def test_rejects_unknown_styles(self) -> None:
        with pytest.raises(ValueError, match="prompt_templates"):
            build_pointing_prompt("cat", 0, prompt_templates="nope")
        with pytest.raises(ValueError, match="system_prompt"):
            build_pointing_prompt("cat", 0, system_prompt="nope")


# ---------------------------------------------------------------------------
# parity with mm_olmo (skipped when the reference checkout is absent)
# ---------------------------------------------------------------------------

_MM_OLMO_FORMATTER = (
    Path(os.environ.get("MM_OLMO_ROOT", "/root/mm_olmo")) / "olmo" / "data" / "data_formatter.py"
)


def _mm_olmo_available() -> bool:
    """Whether the reference checkout is readable; an unreadable ancestor (CI runners
    cannot stat under ``/root``) counts as absent rather than erroring at collection."""
    try:
        return _MM_OLMO_FORMATTER.is_file()
    except OSError:
        return False


def _mm_olmo_pointing_templates() -> list[str]:
    """Read ``GENERAL_PROMPTS_V1["pointing"]`` out of mm_olmo without importing it.

    mm_olmo pulls in heavy optional deps at import time, so parse the literal instead.
    """
    tree = ast.parse(_MM_OLMO_FORMATTER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "GENERAL_PROMPTS_V1" for target in node.targets
        ):
            return list(ast.literal_eval(node.value)["pointing"])
    raise AssertionError("GENERAL_PROMPTS_V1 not found in mm_olmo")


@pytest.mark.skipif(not _mm_olmo_available(), reason="mm_olmo checkout not available")
class TestMmOlmoParity:
    def test_templates_match_verbatim(self) -> None:
        assert list(POINTING_TEMPLATES) == _mm_olmo_pointing_templates()

    def test_prompts_match_mm_olmo(self) -> None:
        """Reproduce mm_olmo's two draws and its first-occurrence keyword substitution."""
        templates = _mm_olmo_pointing_templates()

        def reference(label: str, index: int) -> str:
            rng = np.random.RandomState((691203 * 195172 + index) % (2**32 - 1))
            text = label.lower() if rng.random() > 0.5 else label
            head, tail = templates[rng.randint(0, len(templates))].split("{label}", maxsplit=2)
            return head + text + tail

        labels = ["Earring", "a red car", "Traffic Light", "the White House", "dog"]
        for index in range(200):
            label = labels[index % len(labels)]
            assert build_pointing_prompt(label, index) == reference(label, index)


# ---------------------------------------------------------------------------
