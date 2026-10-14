"""Tests for the IFEval verifiers adapted to non-English responses."""

from __future__ import annotations

import pytest
from ifbench import instructions_registry

from olmo_eval.common.scorers.multilingual_instructions import (
    MULTI_IF_INSTRUCTIONS,
    NumberOfSentences,
    NumberOfWords,
    QuotationChecker,
    count_sentences,
    count_words,
)

CHINESE = "我们今天去公园散步。天气很好，阳光明媚！你想一起来吗？"
HINDI = "आज मौसम बहुत अच्छा है। हम पार्क में टहलने गए। क्या आप आना चाहेंगे?"
ENGLISH = "The weather is lovely today. We walked to the park. Would you like to come?"


def test_chinese_words_are_counted_by_character() -> None:
    assert count_words(CHINESE) == 23


def test_mixed_japanese_counts_characters_words_and_emoji() -> None:
    # The example from the reference implementation's docstring.
    assert count_words("こんにちは、ジェイソンさん、Jason? Nice to meet you☺ ❤") == 19


def test_chinese_sentences_split_on_cjk_punctuation() -> None:
    assert count_sentences(CHINESE) == 3


def test_hindi_sentences_split_on_the_danda() -> None:
    assert count_sentences(HINDI) == 3


def test_english_counts_match_ifbench() -> None:
    assert count_words(ENGLISH) == 15
    assert count_sentences(ENGLISH) == 3


def test_language_detection_is_deterministic() -> None:
    # Unseeded ``langdetect`` calls this Japanese about one time in seven,
    # which would count it as 7 words instead of 4.
    text = "data テスト ok 测试"
    assert {count_words(text) for _ in range(100)} == {4}


def test_undetectable_text_is_counted_as_english() -> None:
    assert count_words("") == 0
    assert count_words("1234 5678") == 2


@pytest.mark.parametrize(
    ("relation", "threshold", "expected"),
    [("at least", 23, True), ("at least", 24, False), ("less than", 24, True)],
)
def test_word_count_instruction_uses_the_responses_language(
    relation: str, threshold: int, expected: bool
) -> None:
    instruction = NumberOfWords("length_constraints:number_words")
    instruction.build_description(num_words=threshold, relation=relation)
    assert instruction.check_following(CHINESE) is expected


@pytest.mark.parametrize(
    ("relation", "threshold", "expected"),
    [("at least", 3, True), ("less than", 3, False), ("less than", 4, True)],
)
def test_sentence_count_instruction_uses_the_responses_language(
    relation: str, threshold: int, expected: bool
) -> None:
    instruction = NumberOfSentences("length_constraints:number_sentences")
    instruction.build_description(num_sentences=threshold, relation=relation)
    assert instruction.check_following(HINDI) is expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('"Quoted."', True),
        ("“Quoted.”", True),
        ("「引用」", True),
        ('  "Padded"  ', True),
        ('“Mismatched"', False),
        ("Unquoted.", False),
        ('"', False),
    ],
)
def test_quotation_accepts_curly_and_corner_marks(response: str, expected: bool) -> None:
    instruction = QuotationChecker("startend:quotation")
    instruction.build_description()
    assert instruction.check_following(response) is expected


def test_registry_replaces_only_the_adapted_verifiers() -> None:
    adapted = {
        "length_constraints:number_words": NumberOfWords,
        "length_constraints:number_sentences": NumberOfSentences,
        "startend:quotation": QuotationChecker,
    }
    assert MULTI_IF_INSTRUCTIONS.keys() == instructions_registry.INSTRUCTION_DICT.keys()
    for instruction_id, checker in MULTI_IF_INSTRUCTIONS.items():
        assert checker is adapted.get(
            instruction_id, instructions_registry.INSTRUCTION_DICT[instruction_id]
        )
