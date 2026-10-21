"""APTBench: agentic potential of base LLMs during pre-training (arXiv 2510.24397).

APTBench probes base (pre-trained, non-instruction-tuned) models for the
abilities an agent later relies on -- planning, choosing the next action, and
summarizing or attributing retrieved evidence -- by turning agent trajectories
into few-shot completion problems. Subtasks are grouped into five categories:

``env_setup`` and ``issue_fix``
    Software-engineering scenarios (APTBench-SWE): choosing a setup plan, writing
    the next shell command, diagnosing setup errors, locating buggy code, picking
    fix and test patches, and choosing the next step of a repair trajectory.
``deepresearch``
    Deep-research scenarios (APTBench-DR), in English and Chinese: next-step
    planning, answer summarization, report structuring, citation attribution,
    and report-quality selection.
``tool`` and ``agentic_math``
    Function selection and argument filling (ACEBench, BFCL v4) and math-agent
    atomic abilities. These ship with the upstream repository but are not part
    of the paper's headline SWE/DR scores.

Every prompt comes from the upstream template for its subtask. Templates embed
their fixed in-context examples, so ``num_fewshot`` does not apply. Each
template is split into that fixed prefix and the instance section, whose
placeholders are filled exactly as the reference ``predict.py`` does, so the
concatenated prompt is identical to the upstream prompt.

Decoding is greedy with the per-subtask generation budget from the reference
implementation, and answers are pulled out with ports of its regex extractors.
Scoring is case-sensitive exact match, except citation subtasks, which compare
the set of selected labels.

Several subtasks have prompts far longer than typical base-model contexts. The
``ctx<N>k`` variants cap the prompt at N*1024 tokens (minus a small reserve for
the answer) by dropping tokens from the start of the prompt, which preserves
the instance being asked about. Truncation is applied by the ``vllm_server``
provider.

Data and templates are fetched from a pinned commit of the upstream repository.
"""

from __future__ import annotations

import functools
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric, Metric
from olmo_eval.common.scorers import ExactMatchScorer, LabelSetMatchScorer, Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

APTBENCH_REPO = "TencentYoutuResearch/APTBench"
# Pinned so an upstream update cannot silently change prompts or gold answers.
APTBENCH_REVISION = "9aab1301ab2c743be6bf11d1a2f3c3c6bc123ffc"
_BASE_URL = f"https://raw.githubusercontent.com/{APTBENCH_REPO}/{APTBENCH_REVISION}"

#: Context budgets (in thousands of tokens) exposed as ``ctx<N>k`` variants.
APTBENCH_CONTEXT_BUDGETS = (8, 32, 64, 128)
#: Tokens held back from the context budget for the generated answer.
_ANSWER_RESERVE_TOKENS = 64


# -- Answer extraction (ported from the reference predict.py) ------------------


def extract_letter(response: str) -> str | None:
    """Return the first standalone letter followed by ``)`` or a newline."""
    response = response.replace("*", "")
    match = re.search(r"\b([A-Za-z])[\)\n]", response)
    return match.group(1) if match else None


def _extract_choice(response: str, letters: str) -> str | None:
    response = response.replace("*", "")
    match = re.search(rf"\b([{letters}])[\)\n\s:]", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(rf"\b([{letters}])\s*$", response, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_choice_a_to_e(response: str) -> str | None:
    """Return an upper-cased choice label in A-E."""
    return _extract_choice(response, "A-E")


def extract_choice_a_to_d(response: str) -> str | None:
    """Return an upper-cased choice label in A-D."""
    return _extract_choice(response, "A-D")


def extract_correct_wrong(response: str) -> str | None:
    """Return ``correct`` or ``wrong``, whichever appears first."""
    response = response.lower().replace("*", "")
    match = re.search(r"\b(correct|wrong)\b", response)
    return match.group(1) if match else None


def extract_first_command(response: str) -> str:
    """Return the first line, cut at the first ``;``."""
    if "\n" in response:
        return response.split("\n")[0].split(";")[0]
    return response


def extract_bracketed(response: str) -> str | None:
    """Return the text before the closing ``]`` of a bracketed answer."""
    return response.split("]")[0] if "]" in response else None


def extract_parenthesized(response: str) -> str | None:
    """Return the text before the closing ``)`` of a parenthesized answer."""
    return response.split(")")[0] if ")" in response else None


# -- Scoring --------------------------------------------------------------------


def _is_empty_generation(response: Response) -> bool:
    """Whether the model returned output with empty text.

    A response with no outputs is a failed request, not an empty generation.
    Only the first output is inspected, since every subtask draws one sample.
    """
    return bool(response.outputs) and not response.outputs[0].text


@dataclass(frozen=True, slots=True)
class NonEmptyAccuracyMetric(AccuracyMetric):
    """Mean accuracy over responses, excluding empty generations.

    The reference implementation discards empty generations before scoring,
    so they count toward neither the numerator nor the denominator. Failed
    requests (no outputs) still count as incorrect.
    """

    def compute(self, responses: Sequence[Response]) -> float:
        return AccuracyMetric.compute(self, [r for r in responses if not _is_empty_generation(r)])

    def compute_instance(self, response: Response) -> float | None:
        if _is_empty_generation(response):
            return None
        return float(response.scores.get(self.scorer().name, 0.0))

    def supports_scorer_fallback(self) -> bool:
        # Excluded instances carry a scorer value that is not part of the mean.
        return False


@dataclass(frozen=True, slots=True)
class EmptyGenerationRateMetric(Metric):
    """Fraction of responses excluded from accuracy as empty generations."""

    name: str = "empty_generation_rate"
    scorer: type[Scorer] | Scorer = ExactMatchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(_is_empty_generation(r) for r in responses) / len(responses)

    def compute_instance(self, response: Response) -> float | None:
        return 1.0 if _is_empty_generation(response) else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False

    def higher_is_better(self) -> bool:
        return False

    def display_format(self) -> str:
        return "percentage"


_EXACT_MATCH_SCORER = ExactMatchScorer(case_sensitive=True)
_LABEL_SET_SCORER = LabelSetMatchScorer()


# -- Prompt templates -----------------------------------------------------------


@functools.cache
def _fetch_text(relative_path: str) -> str:
    """Fetch a text file from the pinned upstream revision (cached on disk)."""
    from datasets.utils.file_utils import cached_path

    local_path = cached_path(f"{_BASE_URL}/{relative_path}")
    with open(local_path, encoding="utf-8") as f:
        return f.read()


def split_template(template: str, placeholders: tuple[str, ...]) -> tuple[str, str]:
    """Split a template into its fixed prefix and the instance section.

    The instance section starts at the paragraph holding the first placeholder,
    so the prefix contains only the fixed in-context examples.
    """
    positions = [template.find(p) for p in placeholders if p in template]
    if not positions:
        raise ValueError("Template contains none of the expected placeholders")
    cut = template.rfind("\n\n", 0, min(positions))
    cut = 0 if cut < 0 else cut + 2
    return template[:cut], template[cut:]


# -- Subtask definitions --------------------------------------------------------


@dataclass(frozen=True)
class APTBenchSubtask:
    """Static definition of one APTBench subtask."""

    category: str
    name: str
    template_file: str
    fields: tuple[tuple[str, str], ...]
    answer_key: str
    extractor: Callable[[str], str | None]
    max_tokens: int
    set_match: bool = False

    @property
    def task_name(self) -> str:
        return f"aptbench_{self.category}_{self.name}"

    @property
    def data_url(self) -> str:
        return f"{_BASE_URL}/data/{self.category}/{self.name}/input_data.jsonl"


_CHOICES = ("$CHOICES$", "choices")

APTBENCH_SUBTASKS: tuple[APTBenchSubtask, ...] = (
    # -- SWE: environment setup
    APTBenchSubtask(
        category="env_setup",
        name="plan",
        template_file="env_setup_plan_3shot.txt",
        fields=(("$REPO_INFO$", "repo_info"), _CHOICES),
        answer_key="answer",
        extractor=extract_letter,
        max_tokens=10,
    ),
    APTBenchSubtask(
        category="env_setup",
        name="action",
        template_file="env_setup_action_3shot.txt",
        fields=(("$EXE_PLAN$", "execution_plan"), ("$EXE_CMDS$", "executed_cmds")),
        answer_key="target_cmd",
        extractor=extract_first_command,
        max_tokens=30,
    ),
    APTBenchSubtask(
        category="env_setup",
        name="error",
        template_file="env_setup_error_3shot.txt",
        fields=(
            ("$SETUP$", "setup_instruct"),
            ("$ISSUE_TITLE$", "issue_title"),
            ("$ISSUE_BODY$", "issue_body"),
            _CHOICES,
        ),
        answer_key="answer",
        extractor=extract_letter,
        max_tokens=10,
    ),
    # -- SWE: issue fixing
    APTBenchSubtask(
        category="issue_fix",
        name="locate",
        template_file="issue_fix_locate_3shot.txt",
        fields=(("$ISSUE$", "issue_statement"), _CHOICES),
        answer_key="answer",
        extractor=extract_letter,
        max_tokens=10,
    ),
    APTBenchSubtask(
        category="issue_fix",
        name="fix_patch",
        template_file="issue_fix_fix_patch_3shot.txt",
        fields=(("$ISSUE$", "issue_statement"), _CHOICES),
        answer_key="answer",
        extractor=extract_letter,
        max_tokens=10,
    ),
    APTBenchSubtask(
        category="issue_fix",
        name="plan",
        template_file="issue_fix_plan_3shot.txt",
        fields=(("$TRAJ$", "trajs"), _CHOICES),
        answer_key="answer",
        extractor=extract_letter,
        max_tokens=10,
    ),
    APTBenchSubtask(
        category="issue_fix",
        name="action",
        template_file="issue_fix_action_3shot.txt",
        fields=(("$TRAJ$", "trajs"),),
        answer_key="answer",
        extractor=extract_first_command,
        max_tokens=10,
    ),
    APTBenchSubtask(
        category="issue_fix",
        name="test_patch",
        template_file="issue_fix_test_patch_3shot.txt",
        fields=(("$PROBLEM$", "problem_statement"), _CHOICES),
        answer_key="answer",
        extractor=extract_letter,
        max_tokens=10,
    ),
    # -- Deep research: closed-ended
    *(
        APTBenchSubtask(
            category="deepresearch",
            name=f"plan_{lang}",
            template_file=f"deepresearch_plan_{lang}_3shot.txt",
            fields=(("$QUERY$", "query"), ("$TRAJ$", "trajectory"), _CHOICES),
            answer_key="answer",
            extractor=extract_letter,
            max_tokens=10,
        )
        for lang in ("en", "zh")
    ),
    *(
        APTBenchSubtask(
            category="deepresearch",
            name=f"summ_ans_{lang}",
            template_file=f"deepresearch_summ_ans_{lang}_3shot.txt",
            fields=(("$QUERY$", "query"), ("$TRAJ$", "trajectory")),
            answer_key="answer",
            extractor=extract_bracketed,
            max_tokens=30,
        )
        for lang in ("en", "zh")
    ),
    # -- Deep research: open-ended
    APTBenchSubtask(
        category="deepresearch",
        name="openend_plan_en",
        template_file="deepresearch_openend_plan_en_3shot.txt",
        fields=(("$QUERY$", "question"), _CHOICES),
        answer_key="answer",
        extractor=extract_letter,
        max_tokens=10,
    ),
    *(
        APTBenchSubtask(
            category="deepresearch",
            name=f"openend_citation_{lang}",
            template_file=f"deepresearch_openend_citation_{lang}_3shot.txt",
            fields=(("$ARTICLE$", "article"), _CHOICES, ("$WEB_PAGE$", "url_content")),
            answer_key="answer",
            extractor=extract_parenthesized,
            max_tokens=10,
            set_match=True,
        )
        for lang in ("en", "zh")
    ),
    *(
        APTBenchSubtask(
            category="deepresearch",
            name=f"openend_quality_{lang}",
            template_file=f"deepresearch_openend_quality_{lang}_2shot.txt",
            fields=(("$QUERY$", "query"), _CHOICES),
            answer_key="answer",
            extractor=extract_letter,
            max_tokens=10,
        )
        for lang in ("en", "zh")
    ),
    # -- Tool use
    *(
        APTBenchSubtask(
            category="tool",
            name=f"{source}_api_select",
            template_file=f"tool_{source}_api_select_3shot.txt",
            fields=(("$USER_PROMPT$", "user_prompt"), ("$FUNCS$", "function")),
            answer_key="answer4select",
            extractor=extract_first_command,
            max_tokens=30,
        )
        for source in ("acebench", "bfcl_v4")
    ),
    *(
        APTBenchSubtask(
            category="tool",
            name=f"{source}_api_param",
            template_file=f"tool_{source}_api_param_3shot.txt",
            fields=(
                ("$USER_PROMPT$", "user_prompt"),
                ("$FUNCS$", "function"),
                ("$ANSWER_SELECT$", "answer4select"),
                ("$PARA_NAME$", "param_name"),
            ),
            answer_key="answer4param",
            extractor=extract_first_command,
            max_tokens=30,
        )
        for source in ("acebench", "bfcl_v4")
    ),
    # -- Agentic math
    APTBenchSubtask(
        category="agentic_math",
        name="planning_single",
        template_file="math_planning_single_3shot.txt",
        fields=(("$REPO_INFO$", "repo_info"), _CHOICES),
        answer_key="answer",
        extractor=extract_choice_a_to_e,
        max_tokens=10,
    ),
    APTBenchSubtask(
        category="agentic_math",
        name="feedback_tf",
        template_file="math_feedback_tf.txt",
        fields=(("$REPO_INFO$", "repo_info"), _CHOICES),
        answer_key="answer",
        extractor=extract_correct_wrong,
        max_tokens=10,
    ),
    APTBenchSubtask(
        category="agentic_math",
        name="action_cal",
        template_file="math_action_calculation.txt",
        fields=(("$REPO_INFO$", "repo_info"), _CHOICES),
        answer_key="answer",
        extractor=extract_choice_a_to_d,
        max_tokens=10,
    ),
)


def aptbench_task_names(category: str | None = None) -> tuple[str, ...]:
    """Registered APTBench task names, optionally restricted to one category."""
    return tuple(
        s.task_name for s in APTBENCH_SUBTASKS if category is None or s.category == category
    )


# -- Task -----------------------------------------------------------------------


class APTBenchTask(Task):
    """Base class for APTBench subtasks; each subclass binds one subtask."""

    split = Split.TRAIN  # JSON data files load as a single "train" split
    num_fewshot = 0
    subtask: APTBenchSubtask

    @functools.cached_property
    def _template_parts(self) -> tuple[str, str]:
        template = _fetch_text(f"code/prompts/{self.subtask.template_file}")
        return split_template(template, tuple(p for p, _ in self.subtask.fields))

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = self._template_parts[1]
        for placeholder, key in self.subtask.fields:
            question = question.replace(placeholder, str(doc[key]).strip())

        gold = doc[self.subtask.answer_key]
        if isinstance(gold, list):
            gold = ",".join(str(label) for label in gold)

        return Instance(
            question=question,
            gold_answer=str(gold),
            metadata={
                "id": f"{self.subtask.task_name}_{index}",
                "index": index,
                "uuid": doc.get("uuid"),
                "category": self.subtask.category,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=self._template_parts[0] + instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        pred = self.subtask.extractor(output.text.strip())
        return pred.strip() if pred is not None else None


def _register_subtask(subtask: APTBenchSubtask) -> None:
    class_name = "APTBench_" + "".join(
        part.title() for part in f"{subtask.category}_{subtask.name}".split("_")
    )
    sampling_params = SamplingParams(max_tokens=subtask.max_tokens, temperature=0.0)
    scorer = _LABEL_SET_SCORER if subtask.set_match else _EXACT_MATCH_SCORER
    accuracy = NonEmptyAccuracyMetric(scorer=scorer)
    cls = type(
        class_name,
        (APTBenchTask,),
        {
            "subtask": subtask,
            "data_source": DataSource(path="json", data_files=subtask.data_url, split="train"),
            "metrics": (accuracy, EmptyGenerationRateMetric(scorer=_EXACT_MATCH_SCORER)),
            "primary_metric": accuracy,
            "sampling_params": sampling_params,
            "__module__": __name__,
            "__qualname__": class_name,
        },
    )
    setattr(sys.modules[__name__], class_name, cls)
    register(subtask.task_name)(cls)

    for budget in APTBENCH_CONTEXT_BUDGETS:
        register_variant(
            subtask.task_name,
            f"ctx{budget}k",
            sampling_params=replace(
                sampling_params,
                truncate_prompt_tokens=budget * 1024 - _ANSWER_RESERVE_TOKENS,
                truncation_side="left",
            ),
        )


for _subtask in APTBENCH_SUBTASKS:
    _register_subtask(_subtask)
