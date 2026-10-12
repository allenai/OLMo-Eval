"""Instance extraction from predictions and requests JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path

from olmo_eval.upload.extract import (
    MAX_NATIVE_ID,
    batched,
    count_instances,
    find_task_file,
    index_requests,
    iter_instances,
)
from olmo_eval.upload.validation import PayloadValidator
from tests.upload.fixtures import (
    CONTRACT_SCHEMA,
    MODEL_DIR,
    chat_requests,
    generative_predictions,
    judge_predictions,
    mc_predictions,
    mc_requests,
    multi_sample_predictions,
    write_jsonl,
)

VALIDATOR = PayloadValidator(CONTRACT_SCHEMA)


def _assert_valid(rows: list[dict]) -> None:
    errors = VALIDATOR.errors("InstanceBatchRequest", {"batch_index": 0, "instances": rows})
    assert errors == []


def _line_at(path: Path, offset: int, length: int) -> dict:
    with path.open("rb") as f:
        f.seek(offset)
        return json.loads(f.read(length))


def test_multiple_choice_rows(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    reqs = tmp_path / "r.jsonl"
    write_jsonl(preds, mc_predictions(4))
    write_jsonl(reqs, mc_requests(4))

    rows = list(iter_instances(preds, reqs, "accuracy:LogprobScorer"))

    assert [r["native_id"] for r in rows] == [f"Mercury_{i}" for i in range(4)]
    assert [r["primary_score"] for r in rows] == [1.0, 0.0, 1.0, 0.0]
    assert rows[1]["label"] == "1"
    assert rows[0]["num_outputs"] == 4
    assert rows[0]["completion_tokens"] == 3  # falls back to num_tokens
    assert rows[0]["prompt_preview"].startswith("Question 0:")
    assert list(rows[0]["metrics"])[0] == "accuracy:LogprobScorer"
    _assert_valid(rows)


def test_byte_offsets_point_at_the_records(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    reqs = tmp_path / "r.jsonl"
    write_jsonl(preds, generative_predictions(3))
    write_jsonl(reqs, chat_requests(3))

    for row in iter_instances(preds, reqs, "exact_match:exact_match"):
        pred = _line_at(preds, row["pred_offset"], row["pred_length"])
        req = _line_at(reqs, row["req_offset"], row["req_length"])
        assert str(pred["native_id"]) == row["native_id"]
        assert str(req["native_id"]) == row["native_id"]


def test_crlf_line_lengths_exclude_the_line_ending(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    write_jsonl(preds, generative_predictions(2), newline="\r\n")

    rows = list(iter_instances(preds, None, None))

    for row in rows:
        assert _line_at(preds, row["pred_offset"], row["pred_length"])["doc_id"] is not None
    assert rows[1]["pred_offset"] == rows[0]["pred_length"] + 2


def test_generative_rows_and_previews(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    reqs = tmp_path / "r.jsonl"
    write_jsonl(preds, generative_predictions(3))
    write_jsonl(reqs, chat_requests(3))

    rows = list(iter_instances(preds, reqs, "exact_match:exact_match"))

    assert rows[0]["native_id"] == "1"  # int native_id stored as text
    assert rows[0]["finish_reason"] == "length"
    assert rows[1]["finish_reason"] == "stop"
    assert rows[0]["completion_tokens"] == 40
    assert rows[2]["extracted_answer"] == "4"
    assert len(rows[0]["output_preview"]) == 300
    assert rows[1]["prompt_preview"] == "What is 1 times two?"  # last user message
    _assert_valid(rows)


def test_judge_rows(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    write_jsonl(preds, judge_predictions(3))

    rows = list(iter_instances(preds, None, "accuracy:judge"))

    assert [r["judge_verdict"] for r in rows] == ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
    assert all(r["has_judge_result"] for r in rows)
    assert rows[0]["label"] == json.dumps({"answer": "gold 0"})
    assert rows[0]["req_offset"] is None  # no requests file
    _assert_valid(rows)


def test_multi_sample_flags_and_metric_filtering(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    write_jsonl(preds, multi_sample_predictions(2))

    rows = list(iter_instances(preds, None, "pass_at_1:code"))

    assert rows[0]["num_outputs"] == 3
    assert rows[0]["has_scoring_error"] is True
    assert rows[1]["has_scoring_error"] is False
    assert rows[0]["has_execution_result"] is True
    assert rows[1]["has_trajectory"] is True
    assert rows[0]["label"] is None
    # Booleans become 0/1 and NaN is dropped.
    assert rows[0]["metrics"] == {"pass_at_1:code": 1.0 / 3, "flag:code": 1.0}
    _assert_valid(rows)


def test_multi_sample_counts_every_sample(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    records = multi_sample_predictions(2)
    # Only the second sample of the first instance was truncated.
    records[0]["model_output"][1]["finish_reason"] = "length"
    records[0]["model_output"][2]["completion_tokens"] = 30
    write_jsonl(preds, records)

    rows = list(iter_instances(preds, None, "pass_at_1:code"))

    assert rows[0]["completion_tokens"] == 12 + 12 + 30
    assert rows[1]["completion_tokens"] == 3 * 12
    assert rows[0]["finish_reason"] == "length"
    assert rows[1]["finish_reason"] == "stop"
    _assert_valid(rows)


def test_multiple_choice_tokens_are_not_summed_over_choices(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    write_jsonl(preds, mc_predictions(1))

    rows = list(iter_instances(preds, None, None))

    assert rows[0]["num_outputs"] == 4
    assert rows[0]["completion_tokens"] == 3
    assert rows[0]["finish_reason"] is None


def test_duplicate_native_ids_get_suffixes(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    reqs = tmp_path / "r.jsonl"
    records = mc_predictions(3)
    for rec in records:
        rec["native_id"] = "same"
    write_jsonl(preds, records)
    write_jsonl(reqs, [{**r, "native_id": "same"} for r in mc_requests(1)])

    rows = list(iter_instances(preds, reqs, "accuracy:LogprobScorer"))

    assert [r["native_id"] for r in rows] == ["same", "same#2", "same#3"]
    # All three match the request by their original native_id.
    assert {r["req_offset"] for r in rows} == {0}


def test_long_duplicate_native_ids_stay_unique_and_within_limit(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    long_a = "a" * 600
    long_b = "a" * 600 + "b"  # same first 512 characters as long_a
    ids = [long_a, long_a, long_b, "x", "x", "x#2", "a" * 510 + "#2"]
    records = mc_predictions(len(ids))
    for rec, native_id in zip(records, ids, strict=True):
        rec["native_id"] = native_id
    write_jsonl(preds, records)

    rows = list(iter_instances(preds, None, None))
    native_ids = [r["native_id"] for r in rows]

    assert len(set(native_ids)) == len(native_ids)
    assert all(1 <= len(n) <= MAX_NATIVE_ID for n in native_ids)
    assert native_ids[0] == "a" * MAX_NATIVE_ID
    assert native_ids[1] == "a" * (MAX_NATIVE_ID - 2) + "#2"
    assert native_ids[2] == "a" * (MAX_NATIVE_ID - 2) + "#3"
    assert native_ids[3:6] == ["x", "x#2", "x#2#2"]
    _assert_valid(rows)


def test_missing_native_id_falls_back_to_doc_id(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    reqs = tmp_path / "r.jsonl"
    record = mc_predictions(1)[0]
    del record["native_id"]
    request = mc_requests(1)[0]
    del request["native_id"]
    write_jsonl(preds, [record])
    write_jsonl(reqs, [request])

    rows = list(iter_instances(preds, reqs, None))

    assert rows[0]["native_id"] == "doc_0"
    assert rows[0]["req_offset"] == 0  # matched by doc_id
    assert rows[0]["primary_score"] is None


def test_count_matches_rows_and_tolerates_bad_lines(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    write_jsonl(preds, [*mc_predictions(2), "not json", "", json.dumps([1, 2])])

    rows = list(iter_instances(preds, None, None))

    assert count_instances(preds) == len(rows) == 4
    assert rows[2]["native_id"] == "line_3"


def test_index_requests_keeps_first_occurrence(tmp_path: Path) -> None:
    reqs = tmp_path / "r.jsonl"
    write_jsonl(reqs, [*mc_requests(2), mc_requests(1)[0]])

    by_native, by_doc = index_requests(reqs)

    assert by_native["Mercury_0"].offset == 0
    assert set(by_doc) == {0, 1}


def test_find_task_file(tmp_path: Path) -> None:
    target = tmp_path / "predictions" / MODEL_DIR / "arc_easy_mc_abcdef-predictions.jsonl"
    write_jsonl(target, [])
    write_jsonl(tmp_path / "predictions" / "other" / "gsm8k-predictions.jsonl", [])

    assert find_task_file(tmp_path, "predictions", "arc_easy:mc", "0123456789abcdef") == target
    assert find_task_file(tmp_path, "predictions", "arc_easy:mc", "000000") is None
    unhashed = find_task_file(tmp_path, "predictions", "gsm8k", None)
    assert unhashed is not None and unhashed.name == "gsm8k-predictions.jsonl"
    assert find_task_file(tmp_path, "requests", "arc_easy:mc", "0123456789abcdef") is None


def test_find_task_file_prefers_the_run_model_directory(tmp_path: Path) -> None:
    name = "gsm8k_abcdef-predictions.jsonl"
    other = tmp_path / "predictions" / "aaa-model" / name
    ours = tmp_path / "predictions" / "zzz-model" / name
    write_jsonl(other, [])
    write_jsonl(ours, [])
    task_hash = "0123456789abcdef"

    assert find_task_file(tmp_path, "predictions", "gsm8k", task_hash, "zzz-model") == ours
    assert find_task_file(tmp_path, "predictions", "gsm8k", task_hash, "org/missing") is None
    assert find_task_file(tmp_path, "predictions", "gsm8k", task_hash, "aaa-model") == other


def test_find_task_file_without_model_ignores_stale_and_ambiguous_files(tmp_path: Path) -> None:
    name = "gsm8k_abcdef-predictions.jsonl"
    old = tmp_path / "predictions" / "aaa-model" / name
    new = tmp_path / "predictions" / "zzz-model" / name
    write_jsonl(old, [])
    write_jsonl(new, [])
    os.utime(old, (1000, 1000))
    task_hash = "0123456789abcdef"

    assert find_task_file(tmp_path, "predictions", "gsm8k", task_hash, since=2000) == new
    warnings: list[str] = []
    assert find_task_file(tmp_path, "predictions", "gsm8k", task_hash, warnings=warnings) is None
    assert "several candidate files" in warnings[0]


def test_batched() -> None:
    batches = list(batched(iter([{"i": i} for i in range(5)]), size=2))
    assert [len(b) for b in batches] == [2, 2, 1]


def test_values_too_large_for_32_bit_fields_are_dropped(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    records = mc_predictions(1)
    records[0]["doc_id"] = 2**40
    records[0]["model_output"][0]["num_tokens"] = 2**33
    write_jsonl(preds, records)

    rows = list(iter_instances(preds, None, "accuracy:LogprobScorer"))

    assert rows[0]["doc_id"] is None
    assert rows[0]["completion_tokens"] is None
    _assert_valid(rows)


def test_stale_file_in_the_run_model_directory_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "predictions" / MODEL_DIR / "gsm8k-predictions.jsonl"
    write_jsonl(path, [{}])
    os.utime(path, (1_600_000_000, 1_600_000_000))

    found = find_task_file(tmp_path, "predictions", "gsm8k", None, MODEL_DIR, since=1_700_000_000)
    assert found is None
    assert find_task_file(tmp_path, "predictions", "gsm8k", None, MODEL_DIR) == path
