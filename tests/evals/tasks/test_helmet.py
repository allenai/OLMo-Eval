"""Offline tests for the HELMET tasks, scorers, and suite structure.

Everything here runs without network or GPU: scorers and parsers are pure
functions, task/suite structure is import-time registration, and the one
loader test writes its own fixture file. Reference expectations were derived
from HELMET's own implementations (utils.py, eval_alce.py,
scripts/eval_gpt4_*.py) -- where a case is labeled "HELMET gives X", it was
checked against upstream code.
"""

import json

import pytest

from olmo_eval.common.scorers import (
    NDCGScorer,
    RougeLF1Scorer,
    SubstringExactMatchScorer,
    SubstringRecallScorer,
    ndcg_at_k,
)
from olmo_eval.common.scorers.alce import AlceQampariRecTop5Scorer, AlceStrEmScorer
from olmo_eval.common.scorers.helmet_judge import (
    HelmetLongQAJudgeScorer,
    HelmetSummJudgeScorer,
    parse_judge_json,
)
from olmo_eval.common.scorers.helmet_summ_prompts import (
    FLUENCY_PROMPT,
    FLUENCY_PROMPT_BOOK,
    PRECISION_PROMPT,
    PRECISION_PROMPT_BOOK,
    RECALL_PROMPT,
    RECALL_PROMPT_BOOK,
)
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.data.helmet_icl_loader import _balance_labels
from olmo_eval.data.helmet_kilt_loader import _load_jsonl, _question_key
from olmo_eval.data.helmet_loader import sample_jsonl_by_key
from olmo_eval.data.helmet_msmarco_loader import parse_rankings
from olmo_eval.data.helmet_tasks import HELMET_TASKS, STANDARD_CONTEXT_SIZES
from olmo_eval.evals.tasks.helmet import (
    HelmetExactMatchScorer,
    InfbenchChoiceScorer,
    _parse_labeled_output,
)


def _instance(gold=None, **metadata) -> Instance:
    return Instance(question="q", gold_answer=gold, metadata=metadata)


def _output(text: str, extracted=None) -> LMOutput:
    out = LMOutput(text=text)
    out.extracted_answer = extracted if extracted is not None else text
    return out


# ---------------------------------------------------------------------------
# answer parsing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", "42"),
        (" 42\n", "42"),
        ("label: 42", "42"),
        ("Label: 42\nfoo", "42"),
        ("label:42", "42"),
        # the prefix match wins over the first line, as in HELMET's parse_output
        ("7\nlabel: 9", "9"),
        ("", ""),
        ("LABEL:  13  ", "13"),
    ],
)
def test_parse_labeled_output_matches_helmet(raw, expected):
    assert _parse_labeled_output(raw, prefix="label:") == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ranking: ID3 > ID1 > ID2", ["3", "1", "2"]),
        ("[ID: 8305152] > [ID: 12] > [ID: 7]", ["8305152", "12", "7"]),
        # duplicates keep their first (best) position
        ("Ranking: 5 > 3 > 5 > 9", ["5", "3", "9"]),
        ("The ranking is: 100 > 200 > 300", ["100", "200", "300"]),
        ("Ranking: ID10>ID20>ID30", ["10", "20", "30"]),
    ],
)
def test_parse_rankings_matches_helmet(raw, expected):
    assert parse_rankings(raw) == expected


def test_parse_rankings_unparseable_output_scores_zero():
    ranking = parse_rankings("no ranking here")
    assert ndcg_at_k(ranking, {"1": 3, "2": 1}, 10) == 0.0


# ---------------------------------------------------------------------------
# NDCG (reference values computed with pytrec_eval, trec_eval's linear gain)


def test_ndcg_matches_pytrec_eval_reference():
    relevance = {"d1": 3, "d2": 0, "d3": 2, "d4": 1, "d5": 0}
    ranking = ["d3", "d1", "d5", "d2", "d4"]
    assert ndcg_at_k(ranking, relevance, 3) == pytest.approx(0.817494, abs=1e-6)
    assert ndcg_at_k(ranking, relevance, 10) == pytest.approx(0.898733, abs=1e-6)


def test_ndcg_edge_cases():
    relevance = {"d1": 3, "d2": 1}
    # ideal ranking comes from all judged docs, so omissions cost score
    assert ndcg_at_k(["d2"], relevance, 10) < 1.0
    assert ndcg_at_k(["d1", "d2"], relevance, 10) == 1.0
    assert ndcg_at_k([], relevance, 10) == 0.0
    assert ndcg_at_k(["unjudged"], relevance, 10) == 0.0
    assert ndcg_at_k(["d1"], {}, 10) == 0.0


def test_ndcg_scorer_reads_qrel_pair_list():
    inst = _instance(qrel=[["d1", "3"], ["d2", "0"], ["d3", "2"]])
    out = LMOutput(text="")
    out.extracted_answer = ["d1", "d3", "d2"]
    assert NDCGScorer().score(inst, out) == 1.0
    out.extracted_answer = "not a list"
    assert NDCGScorer().score(inst, out) == 0.0


# ---------------------------------------------------------------------------
# substring / exact match


def test_substring_exact_match_is_max_over_aliases_not_fraction():
    golds = ["Barack Obama", "Obama", "President Obama"]
    inst = _instance(gold=golds, all_gold_answers=golds)
    out = _output("Answer: Barack Obama")
    # one alias found = fully correct; the recall scorer would give 1/3
    assert SubstringExactMatchScorer().score(inst, out) == 1.0
    assert SubstringRecallScorer().score(inst, out) == pytest.approx(2 / 3)


def test_substring_exact_match_uses_helmet_normalization():
    inst = _instance(gold=["United States"], all_gold_answers=["United States"])
    assert (
        SubstringExactMatchScorer().score(inst, _output("the answer is THE united states")) == 1.0
    )
    assert SubstringExactMatchScorer().score(inst, _output("I do not know")) == 0.0


def test_helmet_exact_match_normalizes_punctuation_and_articles():
    # HELMET's drqa-normalized EM scores these 1; the generic scorer scored 0
    inst = _instance(gold="42")
    assert HelmetExactMatchScorer().score(inst, _output("42.")) == 1.0
    assert HelmetExactMatchScorer().score(inst, _output('"42"')) == 1.0
    assert HelmetExactMatchScorer().score(inst, _output("42")) == 1.0
    assert HelmetExactMatchScorer().score(inst, _output("label 7")) == 0.0


# ---------------------------------------------------------------------------
# InfiniteBench multiple choice


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("B", 1.0),
        ("b", 1.0),
        ("B.", 1.0),
        ("B. Paris", 1.0),
        ("Answer: B", 1.0),
        ("The answer is B. Paris", 1.0),  # embedded long form counts
        ("B. Paris is correct", 1.0),
        ("C", 0.0),
        ("I think the answer is B", 0.0),  # bare letter in prose does NOT count
        ("", 0.0),
        ("Answer: C", 0.0),
    ],
)
def test_infbench_choice_scorer_matches_helmet(prediction, expected):
    golds = ["B", "B. Paris"]
    inst = _instance(gold=golds, all_gold_answers=golds)
    assert InfbenchChoiceScorer().score(inst, _output(prediction)) == expected


# ---------------------------------------------------------------------------
# ALCE


def test_alce_str_em_partial_credit_and_citation_stripping():
    qa_pairs = [
        {"short_answers": ["Daei", "Ali Daei"]},
        {"short_answers": ["Bican"]},
        {"short_answers": ["Pele"]},
    ]
    inst = _instance(qa_pairs=qa_pairs)
    scorer = AlceStrEmScorer()
    # citations must be stripped before matching; 2 of 3 sub-answers present
    assert scorer.score(
        inst, _output("Ali Daei scored most [1]. Bican also [2].")
    ) == pytest.approx(2 / 3)
    assert scorer.score(inst, _output("Nobody knows.")) == 0.0
    assert scorer.score(inst, _output("Daei, Bican, and Pele [1][2][3].")) == 1.0


def test_alce_qampari_rec_top5_caps_at_five():
    answers = [
        ["Heat"],
        ["Mai, the Psychic Girl"],
        ["Akira"],
        ["Nausicaa"],
        ["Ghost"],
        ["Lone Wolf"],
    ]
    inst = _instance(qampari_answers=answers)
    scorer = AlceQampariRecTop5Scorer()
    # 5 of 6 found, denominator capped at 5 -> full credit
    assert scorer.score(inst, _output("Heat, Akira, Ghost, Nausicaa, Lone Wolf, Extra")) == 1.0
    assert scorer.score(inst, _output("Heat, Akira")) == pytest.approx(2 / 5)
    assert scorer.score(inst, _output("nothing relevant")) == 0.0


# ---------------------------------------------------------------------------
# LLM-judge parsing and combination


def test_longqa_judge_parses_and_normalizes():
    scorer = HelmetLongQAJudgeScorer()
    # fluency x correctness / 3
    assert scorer.parse_judge_response('Reasoning. {"fluency": 1, "correctness": 3}') == 1.0
    assert scorer.parse_judge_response('x {"fluency": 1, "correctness": 2}') == pytest.approx(2 / 3)
    # the fluency gate zeroes a correct but disfluent answer
    assert scorer.parse_judge_response('x {"fluency": 0, "correctness": 3}') == 0.0
    # the LAST json object is the verdict
    assert scorer.parse_judge_response('{"a": 1} then {"fluency": 1, "correctness": 3}') == 1.0
    # malformed output scores zero rather than raising
    assert scorer.parse_judge_response("no json at all") == 0.0
    assert scorer.parse_judge_response('{"fluency": 1}') == 0.0
    assert parse_judge_json("{fluency: 1}") is None


def test_summ_judge_combination_matches_helmet_formula():
    combine = HelmetSummJudgeScorer.combine
    # fluency * 2*rec*prec/(rec+prec), rec = 4/7, prec = 3/4
    rec, prec = 4 / 7, 3 / 4
    expected = 2 * rec * prec / (rec + prec)
    assert combine(
        {"fluency": 1}, {"recall": 4}, {"precision": 3, "sentence_count": 4}, 7
    ) == pytest.approx(expected)
    # disfluency gates everything
    assert combine({"fluency": 0}, {"recall": 7}, {"precision": 4, "sentence_count": 4}, 7) == 0.0
    # zero denominators and missing/malformed verdicts degrade to 0, not raise
    assert combine({"fluency": 1}, {"recall": 0}, {"precision": 3, "sentence_count": 5}, 10) == 0.0
    assert combine(None, {"recall": 1}, {"precision": 1, "sentence_count": 1}, 5) == 0.0
    assert combine({"x": 1}, {"recall": 1}, {"precision": 1, "sentence_count": 1}, 5) == 0.0


def test_summ_prompts_format_cleanly():
    """The rubrics contain literal JSON braces; a bad escape would raise here."""
    for prompt in (FLUENCY_PROMPT, FLUENCY_PROMPT_BOOK):
        assert "{text}" in prompt and prompt.format(text="s")
    for prompt in (RECALL_PROMPT, RECALL_PROMPT_BOOK):
        assert prompt.format(keypoints="1. k", summary="s")
    for prompt in (PRECISION_PROMPT, PRECISION_PROMPT_BOOK):
        assert prompt.format(expert_summary="e", summary="s")


def _prompts_third(inst, out):
    return HelmetSummJudgeScorer(is_book=True)._prompts(inst, out)[2]


def test_summ_judge_selects_rubric_variant_by_task():
    inst = _instance(keypoints=["k1", "k2"], expert_summary="expert", judge_question="")
    out = _output("A summary.")
    fluency_book, recall_book, _ = HelmetSummJudgeScorer(is_book=True)._prompts(inst, out)
    fluency_law, recall_law, _ = HelmetSummJudgeScorer(is_book=False)._prompts(inst, out)
    # the recall rubric names its domain; fluency variants differ only in examples
    assert "novel" in recall_book.lower() and "lawsuit" not in recall_book.lower()
    assert "lawsuit" in recall_law.lower() and "novel" not in recall_law.lower()
    assert fluency_book != fluency_law
    # the judge inputs are embedded where the rubric expects them
    assert "1. k1" in recall_book and "expert" in _prompts_third(inst, out)


# ---------------------------------------------------------------------------
# ROUGE takes the best of raw and extracted, per HELMET's default_post_process


def test_rouge_scores_max_of_raw_and_extracted():
    inst = _instance(gold=["the cat sat on the mat"], all_gold_answers=["the cat sat on the mat"])
    out = LMOutput(text="the cat sat on the mat")
    out.extracted_answer = "unrelated words entirely"  # a bad parse must not zero the score
    assert RougeLF1Scorer().score(inst, out) == 1.0


# ---------------------------------------------------------------------------
# ICL demo balancing (pure function)


def test_balance_labels_is_deterministic_and_covers_labels():
    records = [{"label": i % 5, "text": f"t{i}"} for i in range(50)]
    first = _balance_labels(records, "label", 10, seed=7)
    second = _balance_labels(records, "label", 10, seed=7)
    assert first == second
    assert len(first) == 10
    # 10 shots over 5 labels -> every label appears exactly twice
    counts = {label: 0 for label in range(5)}
    for record in first:
        counts[record["label"]] += 1
    assert all(count == 2 for count in counts.values())


# ---------------------------------------------------------------------------
# KILT streaming sampler (fixture file; no network)


def test_sampled_rows_identical_to_full_load(tmp_path):
    rows = []
    for question in range(20):
        for depth in (0.0, 0.5, 0.95):  # dep-style repeats per question
            rows.append({"question": f"q{question}", "depth": depth, "s_pop": 10 ** (question % 6)})
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    import math
    import random

    def old_path(max_samples, keep=None):
        loaded = _load_jsonl(str(path))
        if keep is not None:
            loaded = [r for r in loaded if keep(r)]
        if max_samples is not None:
            keys = sorted({r["question"] for r in loaded})
            kept = set(random.Random(42).sample(keys, min(max_samples, len(keys))))
            loaded = [r for r in loaded if r["question"] in kept]
        return loaded

    def popularity(r):
        return math.log10(r["s_pop"]) < 3.0

    for cap in (3, 10, None):
        assert sample_jsonl_by_key(str(path), cap, 42, key=_question_key) == old_path(cap)
        assert sample_jsonl_by_key(
            str(path), cap, 42, key=_question_key, keep=popularity
        ) == old_path(cap, keep=popularity)
    # sampling caps questions, not rows: each question carries its 3 depths
    sampled = sample_jsonl_by_key(str(path), 3, 42, key=_question_key)
    assert len(sampled) == 9
    assert len({r["question"] for r in sampled}) == 3


def test_kilt_runner_keeps_every_depth_of_each_question(tmp_path, monkeypatch):
    """The runner's instance cap must not break up a question's depth sweep.

    Runs the real `prepare_task_items` against a stub 300-question x 6-depth
    file. The loader caps questions; if the task also set an instance
    `limit`, the runner would re-sample rows and queue a question at only
    some of its depths.
    """
    import olmo_eval.data.helmet_kilt_loader as kilt
    from olmo_eval.runners.asynq.preparation import prepare_task_items

    depths = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
    rows = [
        {
            "id": f"q{question}",
            "question": f"question {question}?",
            "answers": [f"answer {question}"],
            "depth": depth,
            "ctxs": [{"title": f"t{question}", "text": f"depth {depth}"}],
        }
        for question in range(300)
        for depth in depths
    ]
    test_file = tmp_path / "test.jsonl"
    test_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    demo_file = tmp_path / "demo.jsonl"
    demo_file.write_text("\n".join(json.dumps(r) for r in rows[:12]) + "\n")

    manifest = {"kilt_nq": {"4k": {"test_file": str(test_file), "demo_file": str(demo_file)}}}
    monkeypatch.setattr(kilt, "load_kilt_manifest", lambda: manifest)
    monkeypatch.setattr(kilt, "download_helmet_plus_file", lambda path: path)
    monkeypatch.setattr(kilt, "_load_reference_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(kilt, "_truncate_prompt_context", lambda row, *args: row)

    task, items = prepare_task_items("helmet_kilt_nq__4096", "stub-model", None)

    max_questions = HELMET_TASKS["kilt_nq__4096"]["max_questions"]
    assert task.config.limit is None
    assert len(items) == max_questions * len(depths)
    depths_by_question: dict[str, set[str]] = {}
    for item in items:
        question = item.instance.metadata["judge_question"]
        depths_by_question.setdefault(question, set()).add(item.instance.question)
    assert len(depths_by_question) == max_questions
    assert all(len(seen) == len(depths) for seen in depths_by_question.values())


# ---------------------------------------------------------------------------
# task registry and suite structure


def test_helmet_task_inventory():
    by_tag: dict[str, int] = {}
    for config in HELMET_TASKS.values():
        by_tag[config["tag"]] = by_tag.get(config["tag"], 0) + 1
    assert by_tag == {
        "recall": 24,
        "rag": 24,
        "rerank": 6,
        "longqa": 18,
        "summ": 12,
        "icl": 30,
        "cite": 24,
    }
    assert len(HELMET_TASKS) == 138


def test_helmet_context_budgets_match_helmet():
    # HELMET truncates to `size - reserve - generation`; these are its numbers
    assert HELMET_TASKS["narrativeqa__4096"]["max_context_tokens"] == 4096 - 200 - 100
    assert HELMET_TASKS["infbench_qa_eng__4096"]["max_context_tokens"] == 4096 - 200 - 10
    assert HELMET_TASKS["infbench_sum_eng__4096"]["max_context_tokens"] == 4096 - 200 - 1200
    # multi_lexsum reserves 300 where the others reserve 200
    assert HELMET_TASKS["multi_lexsum__4096"]["max_context_tokens"] == 4096 - 300 - 400


def test_helmet_task_config_details():
    # PopQA is restricted to long-tail entities (kilt_popqa_3 upstream)
    assert HELMET_TASKS["kilt_popqa__4096"]["popularity_threshold"] == 3.0
    assert "popularity_threshold" not in HELMET_TASKS["kilt_nq__4096"]
    # nocite ALCE variants run zero-shot with a larger budget, per HELMET
    assert HELMET_TASKS["alce_asqa_nocite__4096"]["shots"] == 0
    assert HELMET_TASKS["alce_asqa_nocite__4096"]["max_gen_toks"] == 600
    assert HELMET_TASKS["alce_asqa__4096"]["shots"] == 2
    # ICL shot counts are HELMET's own (comparability over recalibration)
    assert [HELMET_TASKS[f"icl_banking77__{size}"]["shots"] for size in STANDARD_CONTEXT_SIZES] == [
        180,
        360,
        720,
        1450,
        2900,
        5900,
    ]
    # judged tasks are flagged, so the nojudge suites can exclude them
    judged = {name for name, config in HELMET_TASKS.items() if config.get("judged")}
    assert judged == {
        f"{task}__{size}"
        for task in ("narrativeqa", "infbench_sum_eng", "multi_lexsum")
        for size in STANDARD_CONTEXT_SIZES
    }


def test_helmet_suite_structure():
    import olmo_eval.evals.suites.helmet  # noqa: F401 - triggers registration
    from olmo_eval.evals.suites.registry import get_suite

    assert len(get_suite("helmet_all__4096").expanded_tasks) == 21
    # HELMET's Recall average: json_kv plus three RULER NIAH tasks
    assert set(get_suite("helmet_recall__4096").expanded_tasks) == {
        "helmet_json_kv__4096",
        "helmet_ruler_niah_mk_2__4096",
        "helmet_ruler_niah_mk_3__4096",
        "helmet_ruler_niah_mv__4096",
    }
    # HELMET's Cite average leaves out the nocite ablations, which stay runnable
    assert set(get_suite("helmet_cite__4096").expanded_tasks) == {
        "helmet_alce_asqa__4096",
        "helmet_alce_qampari__4096",
    }
    assert HELMET_TASKS["alce_asqa_nocite__4096"]["ablation"] is True
    # nojudge excludes exactly the three judged tasks
    all_tasks = set(get_suite("helmet_all__4096").expanded_tasks)
    nojudge = set(get_suite("helmet_nojudge__4096").expanded_tasks)
    assert all_tasks - nojudge == {
        "helmet_narrativeqa__4096",
        "helmet_infbench_sum_eng__4096",
        "helmet_multi_lexsum__4096",
    }


# ---------------------------------------------------------------------------
# config serialization (regression: judge_fn closure crashed results
# aggregation after all instances were already scored)


def test_every_helmet_task_config_is_json_serializable():
    """`compute_task_hash` json.dumps's each task's config at aggregation time.

    The judged tasks' scorers carry a `judge_fn` closure, which leaked into
    the serialized config via dataclasses.asdict and crashed a full demo run
    at 100% scored. Instantiating a task touches no data, so this sweeps all
    registered helmet tasks cheaply.
    """
    from olmo_eval.evals.tasks.common.registry import get_task

    for task_name in HELMET_TASKS:
        task = get_task(f"helmet_{task_name}", {"limit": 1, "seed": 42})
        config = task.config.to_dict()
        serialized = json.dumps(config, sort_keys=True)
        assert serialized == json.dumps(config, sort_keys=True)


def test_judge_scorer_to_dict_is_stable_and_serializable():
    from olmo_eval.common.scorers.helmet_judge import (
        HelmetLongQAJudgeScorer,
        HelmetSummJudgeScorer,
    )

    for scorer in (HelmetLongQAJudgeScorer(), HelmetSummJudgeScorer(is_book=True)):
        data = scorer.to_dict()
        # a live closure must never appear in the serialized form; the judge
        # is described by its settings instead
        assert data["judge_fn"]["model"] == scorer.judge_model
        assert data["judge_fn"]["temperature"] == scorer.judge_temperature
        assert data["judge_fn"]["max_tokens"] == scorer.judge_max_tokens
        assert json.dumps(data, sort_keys=True)
    # two independent constructions carry distinct closures but must
    # serialize identically, or task hashes would differ run to run
    assert HelmetLongQAJudgeScorer().to_dict() == HelmetLongQAJudgeScorer().to_dict()


# ---------------------------------------------------------------------------
# judge settings reach the task hash, and judge failures are recorded


def _stub_judge(*replies: str):
    pending = list(replies)

    async def judge(prompt: str, **kwargs) -> str:
        return pending.pop(0)

    return judge


@pytest.mark.parametrize(
    "change",
    [{"judge_model": "gpt-4o-mini"}, {"judge_temperature": 1.0}, {"judge_max_tokens": 64}],
)
def test_judge_settings_change_the_serialized_scorer(change):
    for scorer_cls in (HelmetLongQAJudgeScorer, HelmetSummJudgeScorer):
        assert scorer_cls().to_dict() != scorer_cls(**change).to_dict()


def test_provider_path_uses_the_judge_settings(monkeypatch):
    import asyncio

    seen = {}

    async def fake_provider(self, prompt, context, **kwargs):
        seen.update(kwargs)
        return '{"fluency": 1, "correctness": 3}'

    monkeypatch.setattr(HelmetLongQAJudgeScorer, "_score_with_provider", fake_provider)
    monkeypatch.setattr(HelmetLongQAJudgeScorer, "_validate_provider", lambda self, ctx: None)
    scorer = HelmetLongQAJudgeScorer(provider_name="judge", judge_temperature=0.3)
    score = asyncio.run(scorer.ascore_with_context(_instance(gold="x"), _output("x"), None))
    assert score == 1.0
    assert seen == {"temperature": 0.3, "max_tokens": scorer.judge_max_tokens}


def test_longqa_judge_records_parse_errors_in_predictions():
    import asyncio

    from olmo_eval.common.types import LMRequest, RequestType, Response
    from olmo_eval.runners.io.builders import build_predictions

    instance = _instance(gold="x")
    good, bad = _output("x"), _output("x")
    scorer = HelmetLongQAJudgeScorer(
        judge_fn=_stub_judge('{"fluency": 1, "correctness": 3}', "no json here")
    )
    assert asyncio.run(scorer.ascore_with_context(instance, good, None)) == 1.0
    assert asyncio.run(scorer.ascore_with_context(instance, bad, None)) == 0.0
    assert good.metadata["judge_result"]["parse_error"] is False
    assert bad.metadata["judge_result"] == {
        "raw_judge_response": "no json here",
        "parse_error": True,
    }

    request = LMRequest(request_type=RequestType.COMPLETION, prompt="q")
    (prediction,) = build_predictions([Response(instance=instance, request=request, outputs=[bad])])
    assert prediction["model_output"][0]["judge_result"]["parse_error"] is True


def test_summ_judge_records_parse_errors_and_missing_keypoints():
    import asyncio

    scorer = HelmetSummJudgeScorer(
        judge_fn=_stub_judge('{"fluency": 1}', '{"recall": 2}', "unparseable")
    )
    output = _output("A summary.")
    instance = _instance(gold="ref", keypoints=["a", "b"], expert_summary="ref")
    assert asyncio.run(scorer.ascore_with_context(instance, output, None)) == 0.0
    result = output.metadata["judge_result"]
    assert result["parse_error"] is True
    assert result["raw_judge_responses"]["precision"] == "unparseable"

    output = _output("A summary.")
    assert asyncio.run(scorer.ascore_with_context(_instance(gold="ref"), output, None)) == 0.0
    assert output.metadata["judge_result"] == {"missing_keypoints": True, "parse_error": False}


# ---------------------------------------------------------------------------
# HELMET's own settings and source revisions reach the task hash


def test_helmet_settings_reach_the_task_hash(monkeypatch):
    from olmo_eval.common.types.base import compute_task_hash
    from olmo_eval.data import helmet_icl_loader
    from olmo_eval.evals.tasks import helmet
    from olmo_eval.evals.tasks.common.registry import get_task

    config = get_task("helmet_icl_trec_coarse__4096").config
    settings = config.to_dict()["task_settings"]
    assert settings["helmet"]["shots"] == 200
    assert settings["sources"]["icl"]["revision"] == "65752bf53af25bc935a0dce92fb5b6c930728450"
    assert settings["templates"]["_SYSTEM_TEMPLATE"] == "label:"

    base = helmet._task_settings(HELMET_TASKS["icl_trec_coarse__4096"])
    assert helmet._task_settings({**HELMET_TASKS["icl_trec_coarse__4096"], "shots": 100}) != base
    monkeypatch.setattr(helmet_icl_loader, "_SYSTEM_TEMPLATE", "Label:")
    assert helmet._task_settings(HELMET_TASKS["icl_trec_coarse__4096"]) != base
    monkeypatch.setattr(helmet, "HELMET_PLUS_REVISION", "0" * 40)
    kilt_cfg = HELMET_TASKS["kilt_nq__4096"]
    assert helmet._task_settings(kilt_cfg)["sources"]["helmet_plus"].endswith("0" * 40)

    def task_hash(settings):
        from dataclasses import replace

        return compute_task_hash(replace(config, task_settings=settings).to_dict())

    assert task_hash(base) != task_hash({**base, "helmet": {**base["helmet"], "shots": 100}})


def test_task_settings_are_omitted_when_unset():
    from olmo_eval.evals.tasks.common.registry import get_task

    assert "task_settings" not in get_task("hellaswag").config.to_dict()


def test_nlu_csv_is_parsed_like_its_original_loading_script(tmp_path, monkeypatch):
    from olmo_eval.data import helmet_icl_loader

    header = (
        "userid;answerid;scenario;intent;status;answer_annotation;notes;"
        "suggested_entities;answer_normalised;answer;question"
    )
    lines = [
        header,
        '1;1;alarm;set;;"wake me up at [time : five am] this week";;;x;x;q',
        '1;2;alarm;set;;"null";;;x;x;q',
        '1;3;weather;query;;"is it [weather_descriptor : raining]";;;x;x;q',
        '1;4;alarm;query;;"what alarms do i have";;;x;x;q',
    ]
    path = tmp_path / "nlu.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(helmet_icl_loader.DownloadManager, "download", lambda self, url: str(path))

    train = helmet_icl_loader._load_nlu_csv("unused")["train"]
    # the null row is dropped, entity markup stripped, and file order kept
    assert train["text"] == [
        "wake me up at five am this week",
        "is it raining",
        "what alarms do i have",
    ]
    # labels are `{scenario}_{intent}`, indexed in sorted order
    assert train.features["label"].names == ["alarm_query", "alarm_set", "weather_query"]
    assert train["label"] == [1, 2, 0]


def test_helmet_ruler_tasks_render_helmets_prompt(tmp_path, monkeypatch):
    """HELMET renders RULER from its templates with the answer prefix on its own
    line, not from the files' prebuilt `input`, and keeps HELMET's budgets."""
    from olmo_eval.data.ruler_tasks import RULER_TASKS
    from olmo_eval.evals.tasks import helmet
    from olmo_eval.evals.tasks.common.registry import get_task

    row = {
        "index": 0,
        "input": "prebuilt prompt that HELMET does not use",
        "outputs": ["111", "222"],
        "answer": ["111", "222"],
        "context": "HAYSTACK",
        "query": "blue-tick",
        "type_needle_v": "numbers",
        "length": 10,
    }
    path = tmp_path / RULER_TASKS["niah_mv__4096"]["data"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(helmet, "download_ruler_data", lambda: str(tmp_path))

    task = get_task("helmet_ruler_niah_mv__4096")
    (instance,) = list(task.instances)
    prompt = task.format_request(instance).prompt
    assert prompt.endswith(
        "What are all the special magic numbers for blue-tick mentioned in the provided text?"
        "\nThe special magic numbers for blue-tick mentioned in the provided text are"
    )
    assert "HAYSTACK" in prompt and "prebuilt" not in prompt
    assert instance.gold_answer == ["111", "222"]
    # ruler_recall is the fraction of gold values found
    assert task.config.get_primary_metric().scorer().score(instance, _output("111 and 333")) == 0.5

    budgets = {
        name: HELMET_TASKS[f"ruler_{name}__4096"]["max_gen_toks"]
        for name in ("niah_mk_2", "niah_mk_3", "niah_mv")
    }
    assert budgets == {"niah_mk_2": 50, "niah_mk_3": 100, "niah_mv": 50}
    sources = task.config.to_dict()["task_settings"]["sources"]
    assert sources["ruler_data"].startswith("allenai/ruler_data@")
