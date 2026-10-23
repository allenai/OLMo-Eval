"""Grading for the PixelRAG reader benchmark (https://arxiv.org/abs/2606.28344).

Vendored from ``eval/lib/grader.py`` and ``eval/repro_assets/`` of
https://github.com/StarTrail-org/PixelRAG (commit c4a0f530). The two judge prompts are the
official files byte for byte, and the judge call, the verdict parsing and the score follow the
grader:

* ``simpleqa`` is graded with OpenAI's SimpleQA template, whose reply is a letter (A/B/C);
* every other subset is graded with WorldVQA's judge prompt, whose reply ends in
  ``Label: Correct|Incorrect|Unattempted``. Its ground truth reads ``Any of: a | b`` when a
  question has several references;
* NQ and NQ-Tables are also scored by the grader's normalized exact match, its default mode
  for them (the paper's numbers use the judge).

The judge is ``gpt-4.1-2025-04-14`` at temperature 0 with seed 42 and a 1000-token reply, under
the system message ``You are a helpful assistant.``. A response's ``<think>`` block is removed
before it is judged. The score is the fraction graded correct among the responses the judge
graded. Unlike the official grader, a failed call is retried with backoff before it counts as
an error.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import string
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from olmo_eval.common.scorers.base import Scorer, get_scorer_result
from olmo_eval.common.types import Instance, LMOutput

logger = logging.getLogger(__name__)

SCORER_NAME = "pixelrag"

Grader = Literal["simpleqa", "worldvqa"]
#: A judge verdict; ``unattempted`` is SimpleQA's NOT_ATTEMPTED and WorldVQA's Unattempted.
Verdict = Literal["correct", "incorrect", "unattempted"]

JUDGE_MODEL = "gpt-4.1-2025-04-14"
JUDGE_SYSTEM_MESSAGE = "You are a helpful assistant."
JUDGE_MAX_TOKENS = 1000
JUDGE_SEED = 42

#: ``eval/repro_assets/judge_worldvqa_prompt.txt`` (MoonshotAI/WorldVQA's
#: ``JUDGE_WORLDQA_PROMPT_EN``), verbatim. The ``\u`` escapes are literal text in the file.
JUDGE_WORLDVQA_PROMPT = r"""
### Role
You are an expert judge specialized in evaluating the correctness of answers. Your task is to assess whether a model-generated answer is correct based on a given question, the model's response, and the ground truth answer.

### Task: Evaluate Answer Correctness
Please classify the model's response into one of the following three categories. Ignore differences in formatting, punctuation, language (Chinese vs. English), or abbreviations/full names. Focus strictly on the **core semantics** and the **level of detail (granularity)**:

1. **Correct**:
    - The model answer contains the core information of the ground truth.
    - The model answer is semantically consistent with the ground truth and contains no contradictions.
    - **The granularity of the model answer is equal to or finer than the ground truth.**
    - Extra irrelevant information is allowed as long as it does not conflict with the ground truth.

2. **Incorrect**:
    - The model answer provides information that contradicts the ground truth.
    - The model answer provides the wrong specific entity, value, or description.
    - **The granularity of the model answer is coarser than the ground truth**, leading to incomplete or insufficiently specific information.
    - Even if the model expresses uncertainty but follows up with a wrong answer (e.g., "I'm not sure, maybe it's B" when the truth is A), it is considered Incorrect.

3. **Unattempted**:
    - The model explicitly states it does not know the answer (e.g., "I don't know," "I cannot answer this question").
    - The model suggests the user search elsewhere (e.g., "Please search the internet").
    - The model answer contains no information from the ground truth but provides no incorrect or contradictory information.

### Output Format
Please strictly follow this two-line format for your output:
1. **Evaluation**: [A brief explanation of your reasoning]
2. **Label**: [Final classification: "Correct", "Incorrect", or "Unattempted"]

---
### Examples

**Example 1 (Incorrect - Granularity Mismatch/Too Coarse)**
Input:
'''
Question: \u56fe\u7247\u4e2d\u5c5e\u4e8e\u4ec0\u4e48\u7c7b\u578b\u7684\u7530\u5730\uff1f
Model Answer: \u56fe\u7247\u4e2d\u5c55\u793a\u7684\u662f\u68af\u7530\u3002\u68af\u7530\u662f\u5728\u5c71\u5761\u5730\u4e0a\u5f00\u5782\u5e76\u4fee\u7b51\u7684\u9636\u68af\u72b6\u519c\u7530\u3002
Ground Truth Answer: \u9f99\u810a\u68af\u7530
'''
Evaluation: \u6807\u51c6\u7b54\u6848\u7279\u6307\u201c\u9f99\u810a\u68af\u7530\u201d\uff0c\u6a21\u578b\u53ea\u56de\u7b54\u4e86\u901a\u7528\u7684\u201c\u68af\u7530\u201d\u3002\u6a21\u578b\u7b54\u6848\u5c42\u7ea7\u6bd4\u7b54\u6848\u5c42\u7ea7\u66f4\u7c97\u7565\uff0c\u672a\u80fd\u63d0\u4f9b\u6807\u51c6\u7b54\u6848\u6240\u9700\u7684\u7279\u6307\u4fe1\u606f\uff0c\u5c5e\u4e8e\u5c42\u7ea7\u4e0d\u4e00\u81f4\u5bfc\u81f4\u7684\u56de\u7b54\u9519\u8bef\u3002
Label: Incorrect

**Example 2 (Correct - Finer Granularity)**
Input:
'''
Question: What weather phenomenon is in the image?
Model Answer: Based on the visual evidence in the image, the weather phenomenon shown is a **severe storm with extremely high winds**, most likely a **tornado** or a very powerful **hurricane/typhoon**.
Ground Truth Answer: High winds
'''
Evaluation: The ground truth is "high winds," and a "tornado" is a more specific and granular type of high wind. The semantics are correct and the detail is finer.
Label: Correct

**Example 3 (Correct)**
Input:
'''
Question: \u56fe\u4e2d\u5185\u5bb9\u662f\u4ec0\u4e48\u54c1\u724c\u7684logo\uff1f
Model Answer: via\u6d4f\u89c8\u5668
Ground Truth Answer: via
'''
Evaluation: \u6a21\u578b\u7b54\u6848\u201cvia\u6d4f\u89c8\u5668\u201d\u5305\u542b\u4e86\u6807\u51c6\u7b54\u6848\u201cvia\u201d\uff0c\u6838\u5fc3\u8bed\u4e49\u4e00\u81f4\uff0c\u4e14\u201cvia\u6d4f\u89c8\u5668\u201d\u662f\u66f4\u5177\u4f53\u7684\u63cf\u8ff0\uff0c\u5c42\u7ea7\u4e0a\u662f\u5339\u914d\u7684\u3002
Label: Correct

**Example 4 (Unattempted)**
Input:
'''
Question: Which athlete is in the image?
Model Answer: I cannot answer this question as I do not have relevant sports data.
Ground Truth Answer: Wout Weghorst
'''
Evaluation: The model explicitly states its inability to answer and provides no incorrect information.
Label: Unattempted

**Example 5 (Incorrect)**
Input:
'''
Question: \u56fe\u7247\u4e2d\u5c55\u793a\u7684\u662f\u4ec0\u4e48\u82f9\u679c\u54c1\u79cd\uff1f
Model Answer: \u6211\u89c9\u5f97\u53ef\u80fd\u662f\u963f\u514b\u82cf\u82f9\u679c\u3002
Ground Truth Answer: \u70df\u53f0\u82f9\u679c
'''
Evaluation: \u867d\u7136\u6a21\u578b\u7528\u4e86\u201c\u53ef\u80fd\u201d\u7b49\u8bcd\u6c47\uff0c\u4f46\u5b83\u7ed9\u51fa\u7684\u5177\u4f53\u7b54\u6848\u201c\u963f\u514b\u82cf\u82f9\u679c\u201d\u4e0e\u6807\u51c6\u7b54\u6848\u201c\u70df\u53f0\u82f9\u679c\u201d\u4e0d\u7b26\uff0c\u63d0\u4f9b\u4e86\u9519\u8bef\u4fe1\u606f\u3002
Label: Incorrect

**Example 6 (Unattempted)**
Input:
'''
Question: What is the name of the insect in this image?
Model Answer: This is a photo of an insect. To find the species, consult an entomologist or use recognition software.
Ground Truth Answer: Japanese rhinoceros beetle
'''
Evaluation: The model does not attempt to name the insect and suggests the user search elsewhere, providing no incorrect information.
Label: Unattempted

---
### Current Task
Input:
'''
Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {ground_truth_answer}
'''

Evaluation:
"""

#: ``eval/repro_assets/simpleqa_grader_template.txt`` (OpenAI simple-evals' SimpleQA grader),
#: verbatim.
SIMPLEQA_GRADER_TEMPLATE = r"""Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it."""


def strip_think(text: str | None) -> str:
    """The answer after a ``<think>`` block (verbatim from the grader's ``strip_think``)."""
    if text is None:
        return ""
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    elif "think>" in text:
        return text.split("think>")[-1].strip()
    return text


def worldvqa_prompt(question: str, model_answer: str, ground_truth: str) -> str:
    return JUDGE_WORLDVQA_PROMPT.format(
        question=question, model_answer=model_answer, ground_truth_answer=ground_truth
    )


def simpleqa_prompt(question: str, target: str, predicted_answer: str) -> str:
    return SIMPLEQA_GRADER_TEMPLATE.format(
        question=question, target=target, predicted_answer=predicted_answer
    )


_WORLDVQA_LABELS: dict[str, Verdict] = {
    "correct": "correct",
    "incorrect": "incorrect",
    "unattempted": "unattempted",
}
_SIMPLEQA_GRADES: dict[str, Verdict] = {"A": "correct", "B": "incorrect", "C": "unattempted"}


def parse_worldvqa_label(judge_text: str) -> Verdict:
    """The verdict of a WorldVQA judge reply (the grader's ``parse_label``)."""
    m = re.search(r"Label:\s*(Correct|Incorrect|Unattempted)", judge_text, re.IGNORECASE)
    if m:
        return _WORLDVQA_LABELS[m.group(1).lower()]
    tl = judge_text.lower()
    if "incorrect" in tl:
        return "incorrect"
    if "unattempted" in tl:
        return "unattempted"
    if "correct" in tl:
        return "correct"
    return "incorrect"


def parse_simpleqa_grade(judge_text: str | None) -> Verdict:
    """The verdict of a SimpleQA grader reply: its first A/B/C, ``C`` when there is none."""
    m = re.search(r"(A|B|C)", judge_text or "")
    letter = m.group(0) if m else "C"
    return _SIMPLEQA_GRADES[letter]


def _normalize_text(s: str) -> str:
    s = re.sub(
        r"\b(a|an|the)\b",
        " ",
        s.lower().translate(str.maketrans("", "", string.punctuation)),
    )
    return " ".join(s.split())


def is_exact_match(prediction: str | None, golds: Sequence[Any]) -> bool:
    """The grader's NQ exact match: ``Exact Answer:`` dropped, then SQuAD normalization."""
    prediction = (prediction or "").replace("Exact Answer: ", "").strip()
    pred_norm = _normalize_text(prediction)
    return any(_normalize_text(str(g)) == pred_norm for g in golds)


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

_ASYNC_CLIENT: list[Any] = []
_PROCESS_CACHE_DIR: list[str] = []
_MAX_ATTEMPTS = 8
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0


class JudgeError(RuntimeError):
    """The judge could not grade a response."""


def default_judge_cache_dir() -> str:
    """Judge-reply cache: ``$PIXELRAG_JUDGE_CACHE_DIR``, else a process-local temp dir."""
    env_dir = os.environ.get("PIXELRAG_JUDGE_CACHE_DIR")
    if env_dir:
        return env_dir
    if not _PROCESS_CACHE_DIR:
        _PROCESS_CACHE_DIR.append(tempfile.mkdtemp(prefix="pixelrag-judge-cache-"))
    return _PROCESS_CACHE_DIR[0]


def _client() -> Any:
    if not _ASYNC_CLIENT:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise JudgeError("OPENAI_API_KEY is required to grade PixelRAG responses.")
        from openai import AsyncOpenAI

        _ASYNC_CLIENT.append(AsyncOpenAI(api_key=api_key))
    return _ASYNC_CLIENT[0]


def _is_auth_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in (401, 403) or type(exc).__name__ in (
        "AuthenticationError",
        "PermissionDeniedError",
    )


def _cache_key(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "system": JUDGE_SYSTEM_MESSAGE,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": JUDGE_MAX_TOKENS,
        "seed": JUDGE_SEED,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


async def judge_reply(prompt: str, *, model: str = JUDGE_MODEL, cache_dir: str) -> str:
    """The judge's reply to ``prompt``, cached on disk by model and prompt.

    Raises :class:`JudgeError` when every attempt fails, and at once on unusable credentials.
    """
    cache_file = Path(cache_dir) / f"{_cache_key(model, prompt)}.json"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)["reply"]

    client = _client()
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=JUDGE_MAX_TOKENS,
                seed=JUDGE_SEED,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
            reply = response.choices[0].message.content
            if reply is None:
                raise JudgeError("the judge returned no content")
            break
        except Exception as exc:
            if _is_auth_error(exc):
                raise JudgeError(f"the judge rejected the credentials: {exc}") from exc
            last_error = exc
            logger.warning("PixelRAG judge call failed (attempt %d): %s", attempt, exc)
            delay = min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * 2 ** (attempt - 1))
            await asyncio.sleep(delay * (0.5 + random.random()))
    else:
        raise JudgeError(f"the judge failed {_MAX_ATTEMPTS} times: {last_error}")

    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(".tmp", dir=cache_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"model": model, "prompt": prompt, "reply": reply}, f)
    os.replace(tmp, cache_file)
    return reply


async def judge_verdict(
    grader: Grader,
    *,
    question: str,
    ground_truth: str,
    response: str | None,
    cache_dir: str,
    model: str = JUDGE_MODEL,
) -> tuple[Verdict, str]:
    """Grade one response as the official grader does: ``(verdict, judge reply)``."""
    answer = strip_think(response)
    if grader == "simpleqa":
        reply = await judge_reply(
            simpleqa_prompt(question, ground_truth, answer), model=model, cache_dir=cache_dir
        )
        return parse_simpleqa_grade(reply), reply
    reply = await judge_reply(
        worldvqa_prompt(question, answer, ground_truth), model=model, cache_dir=cache_dir
    )
    return parse_worldvqa_label(reply), reply


@dataclass(frozen=True)
class PixelRagJudgeScorer(Scorer):
    """Score channel for the PixelRAG judge: 1.0 when graded correct.

    Grading runs task-level, asynchronously, and stores each verdict as the scorer result;
    this scorer reads it back so the metric plumbing stays uniform.
    """

    name: str = SCORER_NAME

    def to_dict(self) -> dict[str, Any]:
        """The output-affecting settings only; the cache location is machine-local."""
        return {"type": self.__class__.__name__, "name": self.name, "model": JUDGE_MODEL}

    def score(self, instance: Instance, output: LMOutput) -> float:
        result = get_scorer_result(output, self.name) or {}
        return 1.0 if result.get("verdict") == "correct" else 0.0
