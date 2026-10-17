"""ExpertQA: agentic, grounded attributed long-form QA.

2,177 expert-written questions across 32 fields (Malaviya et al., NAACL 2024,
arXiv 2309.07852). The model generates a well-cited long-form answer, which is
graded for attribution and on-topic precision.

This implementation is AGENTIC: the model is expected to use web search/fetch
tools, and citation snippets are graded only when grounded in content the agent
actually retrieved. Before judging, each quoted snippet is checked verbatim
against `response.trajectory` tool output (with normalization), and fabricated
snippets are removed from the judge input.

Requirements: this task only measures grounded attribution when run through a
tool-providing agentic harness with web search/fetch available, e.g. the
`web_search_agent` preset. Run without tools, every snippet is ungrounded and
citation scores collapse toward zero; that is by design, not a bug.

ExpertQA's human annotations grade the dataset's own pre-generated answers, so
they cannot grade a fresh model answer directly. We therefore score a model
under test on citation_precision, citation_recall, answer_precision, and a
separate snippet_grounding_rate that reports how much quoted evidence was found
in retrieved tool text.

Answer formats: `expertqa` is the primary/official JSON-schema task.
`expertqa:cite` is prose with cite tags. Both feed the same metrics, but numbers
are not comparable across modes; pick one mode for any comparison table.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.scorers.citation import (
    extract_json_from_response,
    ground_citations_by_url,
    ground_citations_in_sources,
    parse_cite_tag_response,
    score_citations_for_sections,
)
from olmo_eval.common.scorers.llm_judge import JudgeFn, build_default_judge_fn
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
from olmo_eval.evals.tasks.astabench_sqa import (
    PRECISION_EVAL_PROMPT,
    AnswerPrecisionMetric,
    CitationPrecisionMetric,
    CitationRecallMetric,
    GlobalAvgMetric,
    compute_precision_score,
    format_report,
    normalize_agent_response_dict,
)
from olmo_eval.evals.tasks.common import Task, register

logger = logging.getLogger(__name__)

EXPERTQA_REPO = "cmalaviya/expertqa"
EXPERTQA_WEB_SEARCH_TOOL = "serper_google_webpage_search"
EXPERTQA_WEB_FETCH_TOOL = "serper_fetch_webpage_content"
EXPERTQA_CRAWL4AI_FETCH_TOOL = "browse_webpage"
EXPERTQA_WEB_FETCH_TOOLS = frozenset({EXPERTQA_WEB_FETCH_TOOL, EXPERTQA_CRAWL4AI_FETCH_TOOL})
EXPERTQA_FETCH_EMPTY_CONTENT = "No content extracted from webpage."
_URL_RE = re.compile(r"https?://[^\s<>()\]\"']+")

EXPERTQA_GENERATION_PROMPT = """Answer the following expert question as a well-cited report.

You have a web search tool and one page-fetch tool available. The possible tool names are:
- serper_google_webpage_search
- serper_fetch_webpage_content
- browse_webpage

Use the available tool names verbatim. Search first with serper_google_webpage_search,
then fetch promising pages with serper_fetch_webpage_content or browse_webpage before
answering.

Return valid JSON with a single key `sections`, whose value is a list of section
objects. Each section has keys `title`, `text`, and `citations`. Each citation
has keys `id`, `url`, `title`, and `snippets`:
- `id` is the inline marker exactly as it appears in `text`.
- `url` is the source page.
- `title` is the source page title when available.
- `snippets` is a list of VERBATIM excerpts copied exactly from fetched content.
Do not paraphrase snippets and do not invent quotes.

Your final answer must consist of ONLY a single JSON object.
The first character of your final answer must be '{' and the last must be '}'.
Do not use Markdown, code fences, headings, or any text outside the JSON.
If no supporting source was found for a claim, do not attach a fabricated citation.
Do not create a References section.

Workflow (follow in order):
1. Call serper_google_webpage_search to find relevant pages.
2. Call serper_fetch_webpage_content or browse_webpage to read the most promising pages.
3. Only after you have gathered evidence from fetched pages, write your final answer.
Do not answer from memory. You must search before answering and read pages before answering.
When you answer, output only the single JSON object described above.

Question: """

EXPERTQA_CITE_PROMPT = """Answer the following expert question in plain prose. Markdown is allowed.

You have a web search tool and one page-fetch tool available. The possible tool names are:
- serper_google_webpage_search
- serper_fetch_webpage_content
- browse_webpage

Use the available tool names verbatim. Search first with serper_google_webpage_search,
then fetch promising pages with serper_fetch_webpage_content or browse_webpage before
answering.

Do not return JSON.
For every source-based claim, wrap the exact claim text in a cite tag:
<cite url="https://example.com/page">claim text</cite>.
Use the exact URL of a page you fetched or a search result you relied on. Do not
invent URLs. Every factual claim needs a citation.
Do not create a References section.

Workflow (follow in order):
1. Call serper_google_webpage_search to find relevant pages.
2. Call serper_fetch_webpage_content or browse_webpage to read the most promising pages.
3. Only after you have gathered evidence from fetched pages, write your final answer.
Do not answer from memory. You must search before answering and read pages before answering.
When you answer, write prose with <cite url="..."> tags as described above.

Question: """

ZERO_GROUNDING_STATS = {
    "n_snippets": 0.0,
    "n_grounded": 0.0,
    "snippet_grounding_rate": 0.0,
}

# Metrics averaged into global_avg (ingredient_recall excluded, no rubric here).
EXPERTQA_METRIC_LABELS = ["citation_precision", "citation_recall", "answer_precision"]
EXPERTQA_OUTPUT_LABELS = EXPERTQA_METRIC_LABELS + ["snippet_grounding_rate"]


@dataclass(frozen=True)
class SnippetGroundingMetric(AnswerPrecisionMetric):
    """Mean fraction of model citation snippets grounded in trajectory tool text."""

    name: str = "snippet_grounding_rate"


EXPERTQA_METRICS = (
    GlobalAvgMetric(),
    CitationPrecisionMetric(),
    CitationRecallMetric(),
    AnswerPrecisionMetric(),
    SnippetGroundingMetric(),
)


def _build_judge_fn() -> JudgeFn:
    """Build the ExpertQA judge function; kept as a small test hook."""
    return build_default_judge_fn(scorer_name="ExpertQA")


def _trajectory_source_text(response: Response) -> str:
    """Concatenate fetched-page content visible to the agent."""
    if response.trajectory is None:
        return ""

    source_chunks: list[str] = []
    for tool_call, result in _iter_trajectory_tool_results(response):
        content = result.content or ""
        tool_name = tool_call.function.name
        if (
            tool_name in EXPERTQA_WEB_FETCH_TOOLS
            and content
            and not _fetch_result_is_error(result, content)
        ):
            source_chunks.append(content)

    if source_chunks:
        return "\n\n".join(source_chunks)

    # Older tests and saved traces may contain tool results without tool-call
    # metadata; keep those usable when there is no call sequence to pair.
    if not response.trajectory.tool_call_sequence:
        return "\n\n".join(
            result.content or "" for result in response.trajectory.tool_result_sequence
        )
    return ""


def _strip_think_block(text: str) -> str:
    """Strip leading chain-of-thought block emitted by some reasoning models."""
    think_end = text.find("</think>")
    if think_end >= 0:
        return text[think_end + len("</think>") :]
    return text


def _extract_urls(text: str) -> list[str]:
    """Extract URL-like strings from search result text."""
    return [url.strip().rstrip("/.,;:)]}'\"") for url in _URL_RE.findall(text or "")]


def _tool_call_arguments(tool_call: Any) -> dict[str, Any]:
    """Decode a trajectory tool call's JSON arguments."""
    try:
        parsed = json.loads(tool_call.function.arguments)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _fetch_result_is_error(result: Any, content: str) -> bool:
    """Return whether a fetch result should be treated as unfetched."""
    return bool(
        getattr(result, "is_error", False)
        or content.startswith("Error")
        or content.strip() == EXPERTQA_FETCH_EMPTY_CONTENT
    )


def _iter_trajectory_tool_results(response: Response) -> Iterator[tuple[Any, Any]]:
    """Yield trajectory tool results paired with their originating tool call."""
    if response.trajectory is None:
        return

    pending_tool_calls: list[Any] = []
    pending_tool_calls_by_id: dict[str, Any] = {}

    def pop_tool_call(result: Any) -> Any | None:
        result_tool_call_id = str(getattr(result, "tool_call_id", "") or "")
        if result_tool_call_id:
            tool_call = pending_tool_calls_by_id.pop(result_tool_call_id, None)
            if tool_call is not None:
                for idx, pending_tool_call in enumerate(pending_tool_calls):
                    if pending_tool_call is tool_call:
                        del pending_tool_calls[idx]
                        break
                return tool_call

        # Saved trajectories carry empty tool_call_id on results; fall back to order.
        tool_call = pending_tool_calls.pop(0) if pending_tool_calls else None
        if tool_call is not None:
            pending_tool_calls_by_id.pop(getattr(tool_call, "id", ""), None)
        return tool_call

    for turn in response.trajectory.turns:
        if turn.role == "assistant":
            for tool_call in turn.tool_calls:
                pending_tool_calls.append(tool_call)
                if tool_call.id:
                    pending_tool_calls_by_id[tool_call.id] = tool_call
            continue
        if turn.role != "tool":
            continue

        for result in turn.tool_results:
            tool_call = pop_tool_call(result)
            if tool_call is not None:
                yield tool_call, result


def _trajectory_url_content(response: Response) -> dict[str, str]:
    """Map trajectory-observed URLs to fetched content, or empty text if only seen."""
    if response.trajectory is None:
        return {}

    url_to_content: dict[str, str] = {}
    for tool_call, result in _iter_trajectory_tool_results(response):
        content = result.content or ""
        tool_name = tool_call.function.name
        if tool_name == EXPERTQA_WEB_SEARCH_TOOL:
            for url in _extract_urls(content):
                url_to_content.setdefault(url, "")
        elif tool_name in EXPERTQA_WEB_FETCH_TOOLS:
            url = str(_tool_call_arguments(tool_call).get("url", "")).strip()
            if url:
                if _fetch_result_is_error(result, content):
                    url_to_content.setdefault(url, "")
                elif content:
                    url_to_content[url] = content
                else:
                    url_to_content.setdefault(url, "")
    return url_to_content


@register("expertqa")
class ExpertQA(Task):
    """ExpertQA attributed long-form QA, graded for attribution and precision."""

    split = Split.TRAIN  # `main` config exposes a single train split
    data_source = DataSource(path=EXPERTQA_REPO, subset="main", split="train")
    metrics = EXPERTQA_METRICS
    # citation_recall is the core attribution signal; report tiers separately
    # rather than hillclimbing on the global_avg aggregate.
    primary_metric = CitationRecallMetric()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    # The LLM-judge scorer needs an OpenAI key at scoring time; the launcher
    # mounts {user}_OPENAI_API_KEY, or {user}_OPENROUTER_API_KEY in its place.
    required_secrets = ("OPENAI_API_KEY",)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        if not question:
            return None

        metadata = doc.get("metadata") or {}
        return Instance(
            question=question,
            metadata={
                "case_id": f"expertqa_{index}",
                "field": metadata.get("field", ""),
                "specific_field": metadata.get("specific_field", ""),
                "question_type": metadata.get("question_type", ""),
                "index": index,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        prompt_text = EXPERTQA_GENERATION_PROMPT + instance.question
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": prompt_text},),
        )

    def extract_answer(self, output: LMOutput) -> Any:
        """Parse JSON from model output and store the structured response."""
        # Strip <think>...</think> blocks so JSON extraction ignores reasoning braces.
        text = _strip_think_block(output.text)
        parsed = extract_json_from_response(text)
        if parsed is not None:
            parsed = normalize_agent_response_dict(parsed)
        output.metadata["parsed_response"] = parsed
        return parsed

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Score responses for citation precision/recall and answer precision."""
        self._extract_answers(responses)

        judge_fn = _build_judge_fn()
        missing_trajectory = 0

        for response in responses:
            if response.trajectory is None:
                missing_trajectory += 1
            per_label: dict[str, dict[int, float]] = {label: {} for label in EXPERTQA_OUTPUT_LABELS}

            for out_idx, output in enumerate(response.outputs):
                scores = await self._score_output(
                    response=response,
                    output=output,
                    judge_fn=judge_fn,
                )
                for label in EXPERTQA_OUTPUT_LABELS:
                    score = scores.get(label, 0.0)
                    output.metadata[f"score:{label}"] = score
                    per_label[label][out_idx] = score

            for label, output_scores in per_label.items():
                response.scores[label] = self._aggregate_output_scores(output_scores)

            response.scores["global_avg"] = sum(
                response.scores.get(label, 0.0) for label in EXPERTQA_METRIC_LABELS
            ) / len(EXPERTQA_METRIC_LABELS)

        if missing_trajectory:
            logger.warning(
                "ExpertQA scored %d/%d responses with no trajectory; this task needs an "
                "agentic harness exposing web search/fetch tools, e.g. web_search_agent. "
                "Without trajectory tool text, every snippet is ungrounded and citation "
                "scores collapse toward zero.",
                missing_trajectory,
                len(responses),
            )
        return responses

    async def _score_output(
        self,
        response: Response,
        output: LMOutput,
        judge_fn: JudgeFn,
    ) -> dict[str, float]:
        zeros = {k: 0.0 for k in EXPERTQA_OUTPUT_LABELS}
        parsed = output.metadata.get("parsed_response")
        if not parsed or not parsed.get("sections"):
            output.metadata["grounding_stats"] = dict(ZERO_GROUNDING_STATS)
            return zeros

        grounded, grounding_stats = self._ground(parsed, response)
        output.metadata["grounding_stats"] = grounding_stats

        answer_text = format_report(grounded)
        precision_score = await self._score_precision(
            judge_fn, response.instance.question, answer_text
        )
        citation_scores = await score_citations_for_sections(judge_fn, grounded)

        return {
            "citation_precision": citation_scores.get("citation_precision", 0.0),
            "citation_recall": citation_scores.get("citation_recall", 0.0),
            "answer_precision": precision_score,
            "snippet_grounding_rate": grounding_stats["snippet_grounding_rate"],
        }

    def _ground(
        self, parsed: dict[str, Any], response: Response
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Ground parsed JSON citation snippets against trajectory tool text."""
        return ground_citations_in_sources(parsed, _trajectory_source_text(response))

    async def _score_precision(self, judge_fn: JudgeFn, question: str, answer: str) -> float:
        """Answer precision via the irrelevant-paragraph judge (astabench precision_eval)."""
        prompt = PRECISION_EVAL_PROMPT.format(query=question, answer=answer)
        raw = await judge_fn(prompt)
        parsed = extract_json_from_response(raw)
        if not parsed:
            return 1.0  # No irrelevant paragraphs identified = perfect precision

        score, _ = compute_precision_score(parsed, answer)
        return score


@register("expertqa:cite")
class ExpertQACite(ExpertQA):
    """ExpertQA prose cite-tag variant, sharing ExpertQA scoring."""

    def format_request(self, instance: Instance) -> LMRequest:
        prompt_text = EXPERTQA_CITE_PROMPT + instance.question
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": prompt_text},),
        )

    def extract_answer(self, output: LMOutput) -> Any:
        """Parse cite-tag prose into the structured response shape."""
        text = _strip_think_block(output.text)
        parsed = parse_cite_tag_response(text)
        output.metadata["parsed_response"] = parsed
        return parsed

    def _ground(
        self, parsed: dict[str, Any], response: Response
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Ground parsed cite-tag citations by trajectory-observed URL."""
        return ground_citations_by_url(parsed, _trajectory_url_content(response))
