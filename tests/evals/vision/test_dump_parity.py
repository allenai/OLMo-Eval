"""Parity tests against the released mm_olmo Molmo2-4B prediction dumps.

For every benchmark with a saved mm_olmo evaluation (``predictions-ck2000-*``) this
re-scores those predictions with the new task/scorer/metric stack and asserts:

1. **Prompt parity** — the user-turn text of each saved prompt equals the
   ``instance.question`` produced by the new task (style prefixes, MC
   formatting, and the seeded prompt templates must all match exactly).
2. **Metric parity** — the recomputed metrics equal the reference
   ``metrics.json`` values within a small tolerance.

The dumps are reference ground truth and are opened **read-only**; nothing
in this test writes to them. The dense-caption test re-judges through the shared
GPT judge cache with the API key removed, so it never calls the API.

Opt-in:

    RUN_DUMP_PARITY_TESTS=1 \
    HF_DATASETS_CACHE=/weka/oe-training-default/mm-olmo/hf_datasets \
    HF_DATASETS_OFFLINE=1 \
    pytest tests/evals/vision/test_dump_parity.py -v

``MOLMO2_PREDICTIONS_ROOT`` overrides the dump location (default: the
released Molmo2-4B directory).
"""

from __future__ import annotations

import asyncio
import json
import os
import string
from pathlib import Path

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, Response

if not os.environ.get("RUN_DUMP_PARITY_TESTS"):
    pytest.skip(
        "Set RUN_DUMP_PARITY_TESTS=1 (and HF_DATASETS_CACHE for the HF-hub tasks) "
        "to run dump-parity tests",
        allow_module_level=True,
    )

from olmo_eval.evals.tasks.common.registry import get_task  # noqa: E402
from olmo_eval.evals.vision.scoring.judges import DenseCaptionJudgeScorer  # noqa: E402
from olmo_eval.evals.vision.scoring.multi_image import (  # noqa: E402
    strip_multi_image_response,
)
from olmo_eval.evals.vision.scoring.multiple_choice import (  # noqa: E402
    parse_multi_choice_response,
)
from olmo_eval.evals.vision.scoring.prompt_templates import (  # noqa: E402
    DENSE_CAPTION_LOADER_SEED,
    EVAL_LOADER_SEED,
    dense_caption_question,
)

DEFAULT_PREDICTIONS_ROOT = "/weka/oe-training-default/mm-olmo/released-models-molmo2-1225/Molmo2-4B"

# Per-task plumbing: (task spec, dump dir name, join key fn, metric tolerance)
TOLERANCE_DEFAULT = 2e-4
TOLERANCE_MMMU = 2e-3
TOLERANCE_EXACT = 1e-6


def _root() -> Path:
    return Path(os.environ.get("MOLMO2_PREDICTIONS_ROOT", DEFAULT_PREDICTIONS_ROOT))


def _load_dump(dump_name: str) -> tuple[list[dict], dict[str, float]]:
    dump_dir = _root() / f"predictions-ck2000-{dump_name}"
    if not dump_dir.exists():
        pytest.skip(f"reference dump not found: {dump_dir}")
    with open(dump_dir / "predictions.json") as f:
        rows = json.load(f)
    with open(dump_dir / "metrics.json") as f:
        metrics = json.load(f)["metrics"]
    return rows, {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


def _user_text(prompt: str) -> str:
    """Extract the user-turn text from a decoded native prompt."""
    text = prompt.split("<|im_start|>user\n", 1)[1]
    return text.split("<|im_end|>", 1)[0]


def _score_against_dump(task, joined: list[tuple[Instance, str]]) -> dict[str, float]:
    responses = [
        Response(
            instance=instance,
            request=task.format_request(instance),
            outputs=[LMOutput(text=prediction)],
        )
        for instance, prediction in joined
    ]
    responses = asyncio.run(task.score_responses(responses))
    nested = task.compute_metrics(responses)
    return {name: next(iter(by_scorer.values())) for name, by_scorer in nested.items()}


def _assert_metrics(mine: dict[str, float], ref: dict[str, float], tol: float) -> None:
    compared = 0
    for name, value in mine.items():
        if name not in ref:
            continue
        assert value == pytest.approx(ref[name], abs=tol), (
            f"{name}: recomputed {value:.6f} != reference {ref[name]:.6f}"
        )
        compared += 1
    assert compared > 0, "no overlapping metric names with the reference"


#: Questions the dataset's maintainers edited upstream after the reference dump was
#: made; both loaders read the Hub's current revision, so these cannot match the dump.
#: MMMU dropped "on the right"/"on the left" from three Art_Theory questions.
UPSTREAM_EDITS = frozenset(
    {"validation_Art_Theory_9", "validation_Art_Theory_21", "validation_Art_Theory_25"}
)


def _assert_prompt_parity(joined: list[tuple[Instance, dict]]) -> None:
    mismatches = [
        (instance.metadata.get("example_id"), _user_text(row["prompt"]), instance.question)
        for instance, row in joined
        if _user_text(row["prompt"]) != instance.question
        and instance.metadata.get("example_id") not in UPSTREAM_EDITS
    ]
    assert not mismatches, (
        f"{len(mismatches)}/{len(joined)} prompt mismatches; first: {mismatches[0]}"
    )


# ---------------------------------------------------------------------------
# Simple joined tasks: example_id-keyed, full prompt parity
# ---------------------------------------------------------------------------


def _join_by(instances, rows, instance_key, row_key):
    by_key = {instance_key(inst): inst for inst in instances}
    assert len(by_key) == len(instances), "join keys are not unique"
    joined = []
    for row in rows:
        key = row_key(row)
        assert key in by_key, f"dump row {key!r} has no matching instance"
        joined.append((by_key[key], row))
    assert len(joined) == len(rows)
    return joined


@pytest.mark.parametrize(
    ("spec", "dump_name", "tol"),
    [
        ("chart_qa", "chart_qa-validation", TOLERANCE_DEFAULT),
        ("vqa2", "coco_2014_vqa_8192-validation", TOLERANCE_DEFAULT),
        ("doc_qa", "doc_qa-validation", TOLERANCE_DEFAULT),
        ("info_qa", "info_qa-validation", TOLERANCE_DEFAULT),
        ("text_vqa", "text_vqa-validation", TOLERANCE_DEFAULT),
        ("mmmu", "mmmu_test-validation", TOLERANCE_MMMU),
        ("ai2d", "ai2_diagram_v2_mix_transparent-validation", TOLERANCE_DEFAULT),
        ("countbench_qa", "countbench_qa-huggingface", TOLERANCE_DEFAULT),
        ("pixmo_count", "pixmo_count_counting-validation", TOLERANCE_DEFAULT),
    ],
)
def test_dump_parity(spec: str, dump_name: str, tol: float) -> None:
    rows, ref = _load_dump(dump_name)
    task = get_task(spec)
    instances = list(task.instances)
    assert len(instances) == len(rows)

    if spec == "chart_qa":
        joined = _join_by(
            instances,
            rows,
            lambda inst: (inst.metadata["example_id"], inst.metadata["is_human"]),
            lambda row: (row["example_id"], row["is_human"]),
        )
    elif spec == "pixmo_count":
        joined = _join_by(
            instances,
            rows,
            lambda inst: inst.metadata["image_url"],
            lambda row: row["image_url"],
        )
    elif spec == "countbench_qa":
        # image_url is not unique in CountBench; the dump saves the integer
        # example_id under "image_id".
        joined = _join_by(
            instances,
            rows,
            lambda inst: inst.metadata["example_id"],
            lambda row: row["image_id"],
        )
    else:
        joined = _join_by(
            instances,
            rows,
            lambda inst: inst.metadata["example_id"],
            lambda row: row["example_id"],
        )

    # 1. Prompt parity
    _assert_prompt_parity(joined)

    # 2. Metric parity
    mine = _score_against_dump(task, [(inst, row["prediction"]) for inst, row in joined])
    _assert_metrics(mine, ref, tol)


# ---------------------------------------------------------------------------
# Unlabeled test-split variants (eval-server submissions): answers are not
# public, so only prompt parity is asserted against the native test dumps.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "dump_name"),
    [
        ("doc_qa:test", "doc_qa-test-base_native_test"),
        ("info_qa:test", "info_qa-test-base_native_test"),
    ],
)
def test_dump_prompt_parity_unlabeled_test_split(spec: str, dump_name: str) -> None:
    rows, _ = _load_dump(dump_name)
    task = get_task(spec)
    instances = list(task.instances)
    assert len(instances) == len(rows)

    joined = _join_by(
        instances,
        rows,
        lambda inst: inst.metadata["example_id"],
        lambda row: row["example_id"],
    )
    _assert_prompt_parity(joined)


# ---------------------------------------------------------------------------
# RealWorldQA: the dump's `prompt` field is the *original* HF question (it is
# overwritten by metadata["prompt"] in SavePredictions), so prompt parity is
# checked against the documented derivation instead of the decoded input.
# ---------------------------------------------------------------------------


def test_dump_parity_real_world_qa() -> None:
    rows, ref = _load_dump("real_world_qa_no_instruction-test")
    task = get_task("real_world_qa")
    instances = list(task.instances)
    assert len(instances) == len(rows)

    # RealWorldQA has duplicate question texts, so join as a multiset keyed by
    # (question, answer, question_type) — duplicates beyond that are
    # interchangeable for scoring purposes.
    pools: dict[tuple, list[Instance]] = {}
    for inst in instances:
        key = (
            inst.metadata["original_question"],
            inst.metadata["answer"],
            inst.metadata["question_type"],
        )
        pools.setdefault(key, []).append(inst)
    joined = []
    for row in rows:
        key = (row["prompt"], row["answer"], row["question_type"])
        assert pools.get(key), f"dump row has no matching instance: {key[0][:80]!r}"
        joined.append((pools[key].pop(), row))
    assert len(joined) == len(rows)

    for instance, row in joined:
        original = row["prompt"]
        if row["question_type"] == "short_answer":
            expected = f"vqa2: {original.split(chr(10))[0]}"
        else:
            expected = original
        assert instance.question == expected, instance.metadata["example_id"]

    mine = _score_against_dump(task, [(inst, row["prediction"]) for inst, row in joined])
    _assert_metrics(mine, ref, TOLERANCE_DEFAULT)


# ---------------------------------------------------------------------------
# MathVista: prompt parity is exact; the reference `score` (0.5670) used GPT-4
# answer extraction, so the offline score is only asserted as a sanity band.
# The `math_vista:gpt` variant can be asserted against the reference with
# RUN_MATHVISTA_GPT_PARITY=1 + OPENAI_API_KEY (fresh API calls, own cache).
# ---------------------------------------------------------------------------


def test_dump_parity_math_vista_offline() -> None:
    rows, ref = _load_dump("math_vista_v2-validation")
    task = get_task("math_vista:offline")
    instances = list(task.instances)
    assert len(instances) == len(rows)

    joined = _join_by(
        instances,
        rows,
        lambda inst: inst.metadata["example_id"],
        lambda row: row["example_id"],
    )

    _assert_prompt_parity(joined)

    mine = _score_against_dump(task, [(inst, row["prediction"]) for inst, row in joined])
    # Offline extraction is not the GPT protocol that produced ref["score"];
    # assert a sanity band and report the delta.
    assert mine["score"] >= 0.50, f"offline MathVista score suspiciously low: {mine['score']}"
    print(f"math_vista offline={mine['score']:.4f} vs GPT reference={ref['score']:.4f}")


@pytest.mark.skipif(
    not os.environ.get("RUN_MATHVISTA_GPT_PARITY"),
    reason="Set RUN_MATHVISTA_GPT_PARITY=1 + OPENAI_API_KEY for GPT parity (~1000 API calls)",
)
def test_dump_parity_math_vista_gpt() -> None:
    rows, ref = _load_dump("math_vista_v2-validation")
    task = get_task("math_vista:gpt")
    instances = list(task.instances)

    joined = _join_by(
        instances,
        rows,
        lambda inst: inst.metadata["example_id"],
        lambda row: row["example_id"],
    )

    responses = [
        Response(
            instance=instance,
            request=task.format_request(instance),
            outputs=[LMOutput(text=row["prediction"])],
        )
        for instance, row in joined
    ]
    responses = asyncio.run(task.score_responses(responses, ScoringContext()))
    nested = task.compute_metrics(responses)
    score = next(iter(nested["score"].values()))
    assert score == pytest.approx(ref["score"], abs=0.01), (
        f"GPT-extraction score {score:.4f} vs reference {ref['score']:.4f}"
    )


# ---------------------------------------------------------------------------
# Multi-image multiple choice. When a response names no option, mm_olmo guesses
# from the global `random` stream while the port seeds the guess per instance, so
# guessed rows may differ. Each metric must match exactly on the other rows and
# stay within the number of guesses; with no guesses (MuirBench) that is exact.
# ---------------------------------------------------------------------------


def _is_guess(instance: Instance, prediction: str) -> bool:
    """Whether the option parser falls back to a guess (its pick depends on the seed)."""
    options = [option.strip() for option in instance.metadata["options"]]
    choices = list(string.ascii_uppercase[: len(options)])
    index2ans = dict(zip(choices, options, strict=True))
    response = strip_multi_image_response(prediction)
    picks = {
        parse_multi_choice_response(response, choices, index2ans, stable_id=str(seed))
        for seed in range(32)
    }
    return len(picks) > 1


@pytest.mark.parametrize(
    ("spec", "dump_name"),
    [
        ("muir_bench", "muir_bench-test"),
        ("blink", "blink-validation"),
        ("mmiu", "mmiu-test"),
    ],
)
def test_dump_parity_multi_image(spec: str, dump_name: str) -> None:
    rows, ref = _load_dump(dump_name)
    task = get_task(spec)
    instances = list(task.instances)
    assert len(instances) == len(rows)
    joined = _join_by(
        instances,
        rows,
        lambda inst: str(inst.metadata["example_id"]),
        lambda row: str(row["example_id"]),
    )
    _assert_prompt_parity(joined)

    responses = [
        Response(
            instance=instance,
            request=task.format_request(instance),
            outputs=[LMOutput(text=row["prediction"])],
        )
        for instance, row in joined
    ]
    responses = asyncio.run(task.score_responses(responses))
    guessed = [_is_guess(instance, row["prediction"]) for instance, row in joined]

    compared = 0
    for metric in task.metrics:
        if metric.name not in ref:
            continue
        scored = [
            (value, guess)
            for value, guess in zip(
                (metric.compute_instance(r) for r in responses), guessed, strict=True
            )
            if value is not None
        ]
        certain = sum(value for value, guess in scored if not guess)
        guesses = sum(guess for _, guess in scored)
        ref_correct = ref[metric.name] * len(scored)
        # The reference is a float32 mean; allow its rounding in the correct count.
        assert certain - 0.01 <= ref_correct <= certain + guesses + 0.01, (
            f"{metric.name}: reference {ref_correct:.2f} correct is outside "
            f"[{certain}, {certain + guesses}] (non-guessed correct, plus {guesses} guesses)"
        )
        if guesses == 0:
            assert certain / len(scored) == pytest.approx(ref[metric.name], abs=TOLERANCE_EXACT)
        compared += 1
    assert compared == len(ref), f"compared {compared} of {len(ref)} reference metrics"


# ---------------------------------------------------------------------------
# Pointing. The dumps keep the parsed points rather than the raw text, so each
# row's points are re-encoded in the model's per-mille ``<points>`` format and
# scored from that text. Rows join by ``_idx``, the dataset index that also seeds
# the ``_mp`` prompt; SA-Co example ids repeat across phrasings.
# ---------------------------------------------------------------------------


def _dump_points(row: dict) -> list:
    points = row["points"]
    return json.loads(points) if isinstance(points, str) else points


def _points_text(points: list, image_size: tuple[int, int]) -> str:
    """Re-encode a dump's pixel-space points as the model's per-mille ``<points>`` text."""
    if not points:
        return "There are none."
    width, height = image_size
    triplets = " ".join(
        f"{k} {round(point[-2] / width * 1000):03d} {round(point[-1] / height * 1000):03d}"
        for k, point in enumerate(points, start=1)
    )
    return f'<points coords="1 {triplets}">object</points>'


def _join_by_position(instances: list[Instance], rows: list[dict]):
    positions = [int(row["_idx"]) for row in rows]
    assert len(set(positions)) == len(rows) == len(instances), "dump rows do not cover the task"
    return [(instances[position], row) for position, row in zip(positions, rows, strict=True)]


@pytest.mark.parametrize(
    ("spec", "dump_name", "check_prompts", "tol"),
    [
        ("pixmo_points_eval", "pixmo_point_eval_v3.1-test", True, TOLERANCE_EXACT),
        ("pixmo_points_eval_mp", "pixmo_point_eval_v3.1-test-mp0816", True, TOLERANCE_EXACT),
        ("sa_co_gold_point_4k_mp", "sa-co-gold-v4-test-mp0816", True, TOLERANCE_EXACT),
        # A few SA-Co images carry an EXIF rotation. mm_olmo's loader turns them upright
        # and scales the model's points by the rotated size; this stack (like the vision
        # branch and OLMo-core) keeps the stored orientation the masks are drawn in.
        ("sa_co_gold_subset", "sa-co-gold-subset-v3-test", True, 1e-3),
        # The only dump of the v2 subset was made after mm_olmo changed its plain
        # "Point to ..." wording, so it checks this task's data and scoring; the `_mp`
        # wording is the same one checked against the full gold set above.
        ("sa_co_gold_subset_mp", "sa-co-gold-subset-v4-test", False, 1e-3),
    ],
)
def test_dump_parity_pointing(spec: str, dump_name: str, check_prompts: bool, tol: float) -> None:
    rows, ref = _load_dump(dump_name)
    task = get_task(spec)
    joined = _join_by_position(list(task.instances), rows)
    assert all(
        str(instance.metadata["example_id"]) == str(row.get("example_id", row.get("id")))
        for instance, row in joined
    )
    if check_prompts:
        _assert_prompt_parity(joined)

    texts = [
        (instance, _points_text(_dump_points(row), instance.metadata["image_size"]))
        for instance, row in joined
    ]
    mine = _score_against_dump(task, texts)
    assert mine["n_scoring_errors"] == 0
    _assert_metrics(mine, ref, tol)


# ---------------------------------------------------------------------------
# Dense caption. The mp0816 dump was judged by mm_olmo's gpt_dense_caption_eval.py
# into the shared gpt4-cache/, so the judge runs cache-only here and never calls
# the API. Every recall-side judgment is in the cache, so recall must reproduce
# exactly. The consistency side cannot fully: mm_olmo's recall and consistency
# threads both request the canonical statements, race on the cache, and for some
# captions the entry left behind is not the one the consistency call used, so
# only the rest can be re-judged.
#
# That dump came from eval_molmo2.py, whose loader seed picks the caption
# template; the task follows eval_captioner.sh's seed, so the template choice is
# checked at both seeds against the same dataset index.
# ---------------------------------------------------------------------------


async def _judge_all(scorer: DenseCaptionJudgeScorer, task, joined) -> tuple[list[Response], int]:
    """Judge every dump caption; returns the responses and how many missed the cache."""
    context = ScoringContext()
    misses = 0

    async def judge(instance: Instance, prediction: str) -> Response:
        nonlocal misses
        output = LMOutput(text=prediction)
        try:
            await scorer.ascore_with_context(instance, output, context)
        except ValueError as exc:
            if "Cache miss" not in str(exc):
                raise
            misses += 1
        return Response(instance=instance, request=task.format_request(instance), outputs=[output])

    responses = await asyncio.gather(*(judge(inst, row["prediction"]) for inst, row in joined))
    return list(responses), misses


def test_dump_parity_dense_caption() -> None:
    dump_dir = _root() / "predictions-ck2000-dense_caption_eval-test-mp0816"
    if not (dump_dir / "gpt4o_judge_metrics.json").exists():
        pytest.skip(f"reference dump not found: {dump_dir}")
    with open(dump_dir / "predictions.json") as f:
        rows = json.load(f)
    with open(dump_dir / "gpt4o_judge_metrics.json") as f:
        ref = json.load(f)

    task = get_task("dense_caption")
    instances = list(task.instances)
    assert len(instances) == len(rows) == ref["n"]
    joined = _join_by(
        instances, rows, lambda inst: inst.metadata["url"], lambda row: row["image_url"]
    )
    for instance, row in joined:
        idx = int(row["_idx"])
        assert _user_text(row["prompt"]) == dense_caption_question(idx, seed=EVAL_LOADER_SEED)
        assert instance.question == dense_caption_question(idx, seed=DENSE_CAPTION_LOADER_SEED)

    def metrics(responses: list[Response]) -> dict[str, float]:
        nested = task.compute_metrics(responses)
        return {name: next(iter(by_scorer.values())) for name, by_scorer in nested.items()}

    # The reference was saved rounded to two decimals.
    recall_scorer = DenseCaptionJudgeScorer(cache_only=True, target_metrics=("recall",))
    responses, misses = asyncio.run(_judge_all(recall_scorer, task, joined))
    assert misses == 0, f"{misses} recall judgments missed the cache"
    mine = metrics(responses)
    for name in ("recall", "recall_at_10"):
        assert mine[name] == pytest.approx(ref[name], abs=0.006), (
            f"{name}: recomputed {mine[name]:.4f} != reference {ref[name]}"
        )

    full_scorer = DenseCaptionJudgeScorer(cache_only=True)
    responses, misses = asyncio.run(_judge_all(full_scorer, task, joined))
    judged = len(joined) - misses
    assert judged >= 0.8 * len(joined), f"only {judged} captions fully re-judged from the cache"
    consistency = metrics(responses)["consistency"]
    assert consistency == pytest.approx(ref["consistency"], abs=0.5), (
        f"consistency over the {judged} re-judged captions {consistency:.2f} "
        f"vs reference {ref['consistency']}"
    )
