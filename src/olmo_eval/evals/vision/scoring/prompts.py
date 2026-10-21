"""mm_olmo's "model prompt" (``_mp``) pointing prompts.

The ``_mp`` pointing datasets hand the model a bare ``label`` rather than a
pre-built question, and mm_olmo's ``DataFormatter`` turns that label into the
prompt. Two independent settings from the checkpoint's own config decide the
result:

``prompt_templates``
    ``"none"`` uses the lowercased label verbatim. ``"uber_model_v2"`` lowercases
    the label half the time and wraps it in one of :data:`POINTING_TEMPLATES`.

``system_prompt``
    ``"demo_or_style_v2"``/``"v3"`` add nothing. ``"style_and_length"``/``"_v2"``
    prefix the style name.

Template choice is seeded from the example's index in the dataset, so prompts are
reproducible but differ per example. Instruction-tuned checkpoints were trained on
the templated form and pretrain checkpoints on the terse prefixed form, so the
setting has to follow the checkpoint rather than the task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

#: mm_olmo's ``GENERAL_PROMPTS_V1["pointing"]``. Two entries are identical, which
#: doubles their sampling weight; keep them.
POINTING_TEMPLATES: tuple[str, ...] = (
    "Point to {label}\nPlease say 'There are none.' if it is not in the image.",
    'Point to all occurrences of "{label}"',
    "Point to any {label} in the image",
    "Point to any {label} in the image.",
    "Point: Where are the {label}",
    "Show me where the {label} are",
    "Can you show me where the {label} are?",
    "Show me where the {label} are",
    "Show me where a {label} is",
    "Show me where a {label} is.",
    "If there are any {label} in the image? Show me where they are.",
    "Where are the {label}?",
    "Generate a list of points showing where the {label} are.",
    'Find the "{label}".',
    'Find a "{label}".',
    "Locate all {label}.",
    "Locate an {label}.",
    "Locate a {label}.",
    "Locate every {label}.",
    "Locate {label}.",
    "Locate the {label}.",
    "Object: {label}\nInstruction: Point to the object.",
    "find {label}",
    "find {label}.",
    "Point to every {label}",
    "find any {label} in the picture",
    "Find the {label}",
    "Find any {label}",
    "Point to a {label}",
    "Point to an {label}",
    "Look for {label} in the image and show me where they are.",
    "Help me find an object in the image by pointing to them.\nObject: {label}.",
    "I am looking for {label}, where can they be found in the image?",
    "Can you see any {label} in the image? Point to them.",
    "Point out each {label} in the image.",
    "Point out every {label} in the image.",
    "Point to the {label} in the image.",
    "Locate each {label} in the image.",
    "Can you point out all {label} in this image?",
    "Please find {label} and show me where they are.",
    "If there are any {label} present, indicate their positions.",
    "If there is a {label} present, indicate its positions.",
    "show me all visible {label}",
)

#: The eval dataloader seed and index multiplier mm_olmo derives per-example RNG from.
EVAL_DATA_SEED = 691203
_SEED_MULTIPLIER = 195172

_TEMPLATED_STYLES = ("uber_model", "uber_model_v2")
NO_PREFIX_STYLES = ("demo_or_style_v2", "demo_or_style_v3")
STYLE_PREFIX_STYLES = ("style_and_length", "style_and_length_v2")

#: The formatter's style name for these tasks, used as the system prefix.
POINTING_STYLE = "pointing"
#: The formatter's style name for the counting benchmarks (PixMo-Count, CountBenchQA).
POINT_COUNT_STYLE = "point_count"


def pointing_rng(index: int) -> np.random.RandomState:
    """RNG for the example at ``index``, matching mm_olmo's per-example seeding."""
    import numpy as np

    return np.random.RandomState((EVAL_DATA_SEED * _SEED_MULTIPLIER + index) % (2**32 - 1))


def system_prefix(system_prompt: str, style: str = POINTING_STYLE) -> str:
    """Prefix that ``system_prompt`` puts in front of a question of ``style``.

    mm_olmo's ``DataFormatter`` prefixes ``"<style>:"`` for the pointing/counting style
    family under ``style_and_length``/``_v2`` and adds nothing under ``demo_or_style_v*``
    (``olmo/data/data_formatter.py``). It keys off the *style*, not off whether the
    benchmark supplied a ready-made question, so the pre-built-question benchmarks need the
    prefix just as much as the ``_mp`` ones.
    """
    if system_prompt in NO_PREFIX_STYLES:
        return ""
    if system_prompt in STYLE_PREFIX_STYLES:
        return f"{style}:"
    raise ValueError(
        f"Unsupported system_prompt {system_prompt!r} for {style}; "
        f"expected one of {NO_PREFIX_STYLES + STYLE_PREFIX_STYLES}"
    )


def apply_style_prefix(question: str, system_prompt: str, style: str) -> str:
    """Return ``question`` with the family's ``"<style>:"`` prefix, if the family adds one."""
    prefix = system_prefix(system_prompt, style)
    if not prefix:
        return question
    return f"{prefix} {question}" if question else prefix


def build_pointing_prompt(
    label: str,
    index: int,
    *,
    prompt_templates: str = "uber_model_v2",
    system_prompt: str = "demo_or_style_v2",
) -> str:
    """Render the prompt mm_olmo would send for ``label`` at dataset position ``index``.

    :param index: Position in the dataset build order. Template choice is seeded
        from it, so the order has to match mm_olmo's or every prompt shifts.
    """
    if prompt_templates == "none":
        question = label.lower()
    elif prompt_templates in _TEMPLATED_STYLES:
        rng = pointing_rng(index)
        text = label.lower() if rng.random() > 0.5 else label
        template = POINTING_TEMPLATES[rng.randint(0, len(POINTING_TEMPLATES))]
        # mm_olmo's `apply_keywords` splits on the placeholder, replacing only the first.
        question = template.replace("{label}", text, 1)
    else:
        raise ValueError(
            f"Unsupported prompt_templates {prompt_templates!r} for pointing; "
            f"expected 'none' or one of {_TEMPLATED_STYLES}"
        )

    prefix = system_prefix(system_prompt)
    if prefix:
        return f"{prefix} {question}" if question else prefix
    return question
