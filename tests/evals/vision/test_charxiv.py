"""Unit tests for CharXiv (question construction, grading protocol, metrics, judge).

Pure CPU — no GPU, no dataset, no network (the GPT judge is mocked). Expected strings and
hash pins mirror the official ``princeton-nlp/CharXiv`` evaluation code; the hash pins guard
the vendored constants against accidental edits (incl. invisible trailing whitespace, which
is part of the official prompts).
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.vision.benchmarks.charxiv import (
    _SCORER,
    CharxivInvalidCountMetric,
    CharxivScoreMetric,
)
from olmo_eval.evals.vision.scoring import charxiv as cx
from olmo_eval.evals.vision.scoring import judges as charxiv_judge

# ---------------------------------------------------------------------------
# vendored constants: hash pins (byte-verified against the official constants.py)
# ---------------------------------------------------------------------------


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class TestConstantPins:
    def test_constants_byte_identical_to_official(self) -> None:
        assert (
            _h(json.dumps({k: v for k, v in sorted(cx.DESCRIPTIVE_RESP_INST.items())}))
            == "4a22037683ecffb9"
        )
        assert _h(cx.DESCRIPTIVE_GRADING_PREFIX) == "f04d1536f0280544"
        assert (
            _h(json.dumps(dict(sorted(cx.DESCRIPTIVE_GRADING_ICL.items())))) == "cba037872834ade7"
        )
        assert _h(cx.REASONING_GRADING_PREFIX) == "700f2784e5f6b947"
        assert (
            _h(json.dumps({k: v for k, v in sorted(cx.REASONING_GRADING_INST.items())}))
            == "b824810cf907eb43"
        )
        assert (
            _h(json.dumps({k: v for k, v in sorted(cx.REASONING_RESP_INST.items())}))
            == "dd788d9551812137"
        )


# ---------------------------------------------------------------------------
# question construction (official descriptive_query_helper / reasoning builders)
# ---------------------------------------------------------------------------


class TestQuestionConstruction:
    def test_subplot_grid_prefix(self) -> None:
        q = cx.descriptive_query_helper(7, [1, 2])
        assert q.startswith(
            "For the subplot at row 1 and column 2, what is the spatially highest labeled tick"
        )

    def test_single_plot_prefix(self) -> None:
        q = cx.descriptive_query_helper(1, [0, 0])
        assert q.startswith("For the current plot, what is its title?")

    def test_string_subplot_loc(self) -> None:
        q = cx.descriptive_query_helper(2, "the left subplot")
        assert q.startswith("For the left subplot, what is the label of the x-axis?")

    @pytest.mark.parametrize("qid", [18, 19])
    def test_layout_questions_have_no_prefix(self, qid: int) -> None:
        assert cx.descriptive_query_helper(qid, [3, 4]) == cx.DESCRIPTIVE_RESP_INST[qid]

    def test_reasoning_text_categories(self) -> None:
        for cat in (1, 2, 3):
            q = cx.build_reasoning_question("Which model?", cat)
            assert q.startswith("Which model?\n")

    def test_reasoning_number_in_general_decimals(self) -> None:
        q = cx.build_reasoning_question("What value?", 4, "3.25")
        assert "* Your final answer must be a number with 2 decimal places." in q
        q = cx.build_reasoning_question("What value?", 4, "7")
        assert "* Your final answer must be an exact integer." in q

    def test_number_instruction_requires_answer(self) -> None:
        with pytest.raises(AssertionError):
            cx.build_reasoning_question("What value?", 4, None)


# ---------------------------------------------------------------------------
# grading query construction (official prefix/rubric/triplet assembly)
# ---------------------------------------------------------------------------


class TestGradingQueries:
    def test_rubric_routing_all_qids(self) -> None:
        expected = {
            1: "title",
            2: "ocr", 3: "ocr", 4: "ocr", 5: "ocr", 6: "ocr", 7: "ocr",
            8: "quant", 9: "quant", 10: "quant", 12: "quant", 14: "quant",
            15: "quant", 17: "quant", 19: "quant",
            11: "bool", 13: "enum", 16: "trend", 18: "layout",
        }  # fmt: skip
        for qid, key in expected.items():
            assert cx.get_rubric(qid) == cx.DESCRIPTIVE_GRADING_ICL[key]

    def test_build_json_keys(self) -> None:
        assert cx.build_json_keys(2) == (
            "['extract_answer_T1', 'score_T1', 'extract_answer_T2', 'score_T2']"
        )

    def test_populate_grading_inputs(self) -> None:
        out = cx.populate_grading_inputs([("k1", "resp A", "gt A"), ("k2", "resp B", "gt B")])
        assert out == (
            "T1:\nResponse 1: resp A\nGround Truth 1: gt A\n\n"
            "T2:\nResponse 2: resp B\nGround Truth 2: gt B\n\n"
        )

    def test_descriptive_prompt_substitutions(self) -> None:
        prompt = cx.build_descriptive_grading_prompt(7, [("k1", "60", "60")])
        assert "<|NUM_TRIPLETS|>" not in prompt and "<|JSON_KEYS|>" not in prompt
        assert "You will be given 1 pairs of ground truth answers" in prompt
        assert cx.DESCRIPTIVE_GRADING_QMAP[7] in prompt
        assert cx.DESCRIPTIVE_GRADING_ICL["ocr"] in prompt
        assert prompt.endswith("T1:\nResponse 1: 60\nGround Truth 1: 60\n\n")

    def test_batching_groups_by_qid_five_per_call(self) -> None:
        items = [(f"k{i}", f"r{i}", f"a{i}", 7) for i in range(7)]
        items += [("x0", "rx", "ax", 13)]
        batches = cx.build_descriptive_grading_batches(items)
        sizes = [(len(b["resp_keys"])) for b in batches]
        assert sizes == [5, 2, 1]  # qid7 -> 5+2, qid13 -> 1
        assert batches[0]["resp_keys"] == [f"k{i}" for i in range(5)]
        assert batches[2]["resp_keys"] == ["x0"]

    def test_reasoning_prompt_substitutions(self) -> None:
        prompt = cx.build_reasoning_grading_prompt("Which model?", 1, "Joint-CNN", "resp")
        assert prompt.startswith(cx.REASONING_GRADING_PREFIX)
        assert "* Question: Which model?" in prompt
        assert "* Ground Truth: Joint-CNN" in prompt
        assert "* Response: resp" in prompt
        assert "<|question|>" not in prompt


# ---------------------------------------------------------------------------
# grading output validation
# ---------------------------------------------------------------------------


class TestGradingOutput:
    def test_verify_accepts_valid(self) -> None:
        data = {"extract_answer_T1": "x", "score_T1": 1, "extract_answer_T2": "y", "score_T2": 0}
        assert cx.verify_grading_output(data, 2)

    def test_verify_rejects_missing_or_nonbinary(self) -> None:
        with pytest.raises(AssertionError):
            cx.verify_grading_output({"extract_answer_T1": "x"}, 1)
        with pytest.raises(AssertionError):
            cx.verify_grading_output({"extract_answer_T1": "x", "score_T1": 2}, 1)

    def test_dummy_output(self) -> None:
        dummy = cx.build_dummy_output(2)
        assert dummy["score_T1"] == -1 and dummy["score_T2"] == -1


# ---------------------------------------------------------------------------
# metrics (official get_stats semantics: -1 counts as 0)
# ---------------------------------------------------------------------------


def _resp(score: int, qid: int | None = None, inst_category: int | None = None) -> Response:
    result: dict = {"extracted_answer": "x", "score": score}
    if qid is not None:
        result["qid"] = qid
    if inst_category is not None:
        result["inst_category"] = inst_category
    output = LMOutput(text="", metadata={"charxiv_result": result})
    response = Response(
        instance=Instance(question="q", metadata={}), request=None, outputs=[output]
    )
    response.scores["charxiv"] = float(score) if score in (0, 1) else 0.0
    return response


class TestMetrics:
    def test_invalid_counts_as_zero(self) -> None:
        responses = [_resp(1, qid=1), _resp(-1, qid=1)]
        m = CharxivScoreMetric(name="descriptive_overall", scorer=_SCORER)
        assert m.compute(responses) == pytest.approx(0.5)

    def test_category_filter(self) -> None:
        responses = [_resp(1, qid=1), _resp(0, qid=13), _resp(1, qid=17)]
        ie = CharxivScoreMetric(name="x", scorer=_SCORER, category="Information Extraction")
        enum = CharxivScoreMetric(name="x", scorer=_SCORER, category="Enumeration")
        comp = CharxivScoreMetric(name="x", scorer=_SCORER, category="Compositionality")
        assert ie.compute(responses) == 1.0
        assert enum.compute(responses) == 0.0
        assert comp.compute(responses) == 1.0

    def test_inst_category_filter(self) -> None:
        responses = [_resp(1, inst_category=1), _resp(0, inst_category=4)]
        m1 = CharxivScoreMetric(name="x", scorer=_SCORER, inst_category=1)
        m4 = CharxivScoreMetric(name="x", scorer=_SCORER, inst_category=4)
        assert m1.compute(responses) == 1.0 and m4.compute(responses) == 0.0

    def test_n_invalid(self) -> None:
        responses = [_resp(1, qid=1), _resp(-1, qid=1), _resp(-1, qid=2)]
        assert CharxivInvalidCountMetric(name="n_invalid", scorer=_SCORER).compute(responses) == 2.0

    def test_empty(self) -> None:
        assert CharxivScoreMetric(name="x", scorer=_SCORER).compute([]) == 0.0


# ---------------------------------------------------------------------------
# judge call (mocked client): retry ladder, dummy fallback, cache
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                reply = outer.replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                message = type("M", (), {"content": reply})
                choice = type("C", (), {"message": message})
                return type("R", (), {"choices": [choice]})

        self.chat = type("Chat", (), {"completions": _Completions()})


def _patch_client(monkeypatch, client: _FakeClient) -> None:
    monkeypatch.setattr(charxiv_judge, "_get_client", lambda model: client)


class TestJudgeCall:
    def test_valid_reply_and_cache(self, monkeypatch, tmp_path) -> None:
        reply = json.dumps({"extract_answer_T1": "60", "score_T1": 1})
        client = _FakeClient([reply])
        _patch_client(monkeypatch, client)
        result = asyncio.run(
            charxiv_judge.grade_descriptive_batch("prompt-a", 1, cache_dir=str(tmp_path))
        )
        assert result["score_T1"] == 1
        assert client.calls[0]["model"] == "gpt-4o-2024-05-13"
        assert client.calls[0]["response_format"] == {"type": "json_object"}
        assert client.calls[0]["seed"] == 42 and client.calls[0]["max_tokens"] == 256
        # second call is served from cache (no new API call)
        result2 = asyncio.run(
            charxiv_judge.grade_descriptive_batch("prompt-a", 1, cache_dir=str(tmp_path))
        )
        assert result2 == result and len(client.calls) == 1

    def test_truncated_json_doubles_max_tokens(self, monkeypatch, tmp_path) -> None:
        good = json.dumps({"extracted_answer": "x", "score": 1})
        client = _FakeClient(['{"extracted_answer": "trunc', good])
        _patch_client(monkeypatch, client)
        ext, score = asyncio.run(charxiv_judge.grade_reasoning("p", cache_dir=str(tmp_path)))
        assert (ext, score) == ("x", 1)
        assert [c["max_tokens"] for c in client.calls] == [256, 512]

    def test_retry_cap_returns_dummy(self, monkeypatch, tmp_path) -> None:
        client = _FakeClient([RuntimeError("boom")] * 10)
        _patch_client(monkeypatch, client)
        ext, score = asyncio.run(charxiv_judge.grade_reasoning("p2", cache_dir=str(tmp_path)))
        assert score == -1 and ext == "Failed to parse response"
        assert len(client.calls) == 10

    def test_invalid_scores_retry(self, monkeypatch, tmp_path) -> None:
        bad = json.dumps({"extract_answer_T1": "x", "score_T1": 2})
        good = json.dumps({"extract_answer_T1": "x", "score_T1": 0})
        client = _FakeClient([bad, good])
        _patch_client(monkeypatch, client)
        result = asyncio.run(
            charxiv_judge.grade_descriptive_batch("p3", 1, cache_dir=str(tmp_path))
        )
        assert result["score_T1"] == 0 and len(client.calls) == 2
