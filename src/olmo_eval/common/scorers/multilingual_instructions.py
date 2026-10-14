"""IFEval verifiers adapted for responses in languages other than English.

Multi-IF (He et al., 2024, https://github.com/facebookresearch/Multi-IF) asks
for IFEval's instructions in eight languages. Its reference implementation
changes a few of the original verifiers so they read non-English responses
correctly. This module applies those changes on top of the ``ifbench``
verifiers:

- Word and sentence counts follow the response's detected language. Chinese
  and Japanese count each character as a word and split sentences on CJK
  punctuation, and Hindi splits sentences on the danda.
- A response wrapped in curly or corner quotation marks counts as quoted.

Language detection is seeded here, unlike the reference, so a rerun scores the
same reply the same way.

The reference also counts Thai words and sentences with ``pythainlp``. No
Multi-IF conversation combines a Thai response with a length instruction, so
that path is left out.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import emoji
from ifbench import classic_instructions, instructions_registry, instructions_util
from langdetect.detector_factory import PROFILES_DIRECTORY, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

#: ``langdetect`` codes whose words and sentences are counted by character and
#: CJK punctuation.
_CHINESE_JAPANESE = frozenset({"zh", "zh-cn", "zh-tw", "ja"})
_HINDI = "hi"

_CJK_CHARACTER = re.compile(
    r"[一-鿿㐀-䶿"  # Chinese
    r"぀-ゟ゠-ヿ"  # Japanese kana
    r"가-힯]"  # Korean Hangul
)
_CJK_SEPARATORS = re.compile(
    r"[\\.\!\?\．\/_,\{\}<>:;$%^&*(+\"\'+——！，。？、`~@#￥……（）：；《）《》“”()\[\]»〔〕\-「」]+"
)
_CJK_SENTENCE = re.compile(r"[^!?。\.\!\?\！\？\．\n]+[!?。\.\!\?\！\？\．\n]?")
_HINDI_SENTENCE_END = re.compile(r"(?<=[।!?])\s*")

_QUOTATION_MARKS = (('"', '"'), ("“", "”"), ("「", "」"))


@functools.cache
def _language_detector_factory() -> DetectorFactory:
    """A seeded ``langdetect`` factory, kept apart from the library's shared one."""
    factory = DetectorFactory()
    factory.load_profile(PROFILES_DIRECTORY)
    factory.set_seed(0)
    return factory


def _detect_language(text: str) -> str:
    """The text's ``langdetect`` code, or English when it cannot be told.

    Detection is seeded, so the same reply is always counted the same way.
    """
    detector = _language_detector_factory().create()
    detector.append(text)
    try:
        return detector.detect()
    except LangDetectException:
        return "en"


def _count_cjk_words(text: str) -> int:
    """Count each CJK character, emoji, and other whitespace-separated word once."""
    text = _CJK_SEPARATORS.sub("", text)
    emoji_count = emoji.emoji_count(text)
    text = emoji.replace_emoji(text, "")
    cjk_count = len(_CJK_CHARACTER.findall(text))
    other_count = len(_CJK_CHARACTER.sub(" ", text).split())
    return cjk_count + emoji_count + other_count


def count_words(text: str) -> int:
    """Count the words in a response, in the way its language divides them."""
    if _detect_language(text) in _CHINESE_JAPANESE:
        return _count_cjk_words(text)
    return instructions_util.count_words(text)


def count_sentences(text: str) -> int:
    """Count the sentences in a response, in the way its language ends them."""
    language = _detect_language(text)
    if language == _HINDI:
        return sum(1 for sentence in _HINDI_SENTENCE_END.split(text) if sentence.strip())
    if language in _CHINESE_JAPANESE:
        return sum(len(_CJK_SENTENCE.findall(line.strip())) for line in text.splitlines())
    return instructions_util.count_sentences(text)


def _compare(count: int, threshold: int, relation: str) -> bool:
    """Apply an IFEval length relation, ``less than`` or ``at least``."""
    return count < threshold if relation == "less than" else count >= threshold


class NumberOfWords(classic_instructions.NumberOfWords):
    """Check the number of words, counted for the response's language."""

    _num_words: int
    _comparison_relation: str

    def check_following(self, value: str) -> bool:
        return _compare(count_words(value), self._num_words, self._comparison_relation)


class NumberOfSentences(classic_instructions.NumberOfSentences):
    """Check the number of sentences, counted for the response's language."""

    _num_sentences_threshold: int
    _comparison_relation: str

    def check_following(self, value: str) -> bool:
        return _compare(
            count_sentences(value), self._num_sentences_threshold, self._comparison_relation
        )


class QuotationChecker(classic_instructions.QuotationChecker):
    """Check that the response is wrapped in straight, curly, or corner quotation marks."""

    def check_following(self, value: str) -> bool:
        value = value.strip()
        return len(value) > 1 and any(
            value.startswith(opening) and value.endswith(closing)
            for opening, closing in _QUOTATION_MARKS
        )


#: The ``ifbench`` registry with the multilingual verifiers in place.
MULTI_IF_INSTRUCTIONS: Mapping[str, Any] = MappingProxyType(
    {
        **instructions_registry.INSTRUCTION_DICT,
        "length_constraints:number_words": NumberOfWords,
        "length_constraints:number_sentences": NumberOfSentences,
        "startend:quotation": QuotationChecker,
    }
)
