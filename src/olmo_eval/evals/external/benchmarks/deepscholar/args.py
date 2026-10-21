"""Arguments for DeepScholar-Bench evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The seven metrics DeepScholar-Bench defines (paper Table 1), grouped by
# dimension: knowledge synthesis, retrieval quality, verifiability. The paper's
# headline score is the geometric mean over all seven (Table 2 "Geo. Mean"; no
# system exceeds ~31%). These strings are the upstream EvaluationFunction values,
# which are also the per-metric output subdir and aggregated_results.csv column.
PRIMARY_METRICS = (
    "organization",
    "nugget_coverage",
    "coverage_relevance_rate",
    "document_importance",
    "reference_coverage",
    "cite_p",
    "claim_coverage",
)


def _parse_optional(data: dict[str, Any], key: str, type_fn: type) -> Any:
    value = data.get(key)
    return type_fn(value) if value is not None else None


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


@dataclass
class DeepScholarArgs:
    """Arguments for the deepscholar_bench evaluation.

    The model under test drives the generation phase via the LOTUS ``lm`` config
    block; the judge model (``judge_model``) scores the generated related-work
    sections in the eval phase.
    """

    # Generation phase (model under test)
    limit: int | None = None  # -> generation --end-idx (smoke runs)
    start_idx: int = 0  # -> generation --start-idx
    search_mode: str | None = None  # "agentic" | "recursive"; None keeps the YAML default
    # Retrieval backend. "arxiv" is upstream's recursive default (keyless, but the
    # export.arxiv.org API rate-limits/hangs at volume). "s2" routes recursive search
    # to the Semantic Scholar API via a runtime shim (keyed, request timeouts, no
    # arXiv hangs; needs S2_API_KEY). "tavily" skips the hardwired arXiv corpus and
    # uses only the TAVILY web corpus (needs TAVILY_API_KEY).
    search_backend: str = "arxiv"
    # Web search corpus for retrieval (arxiv backend only). ARXIV is keyless;
    # TAVILY/GOOGLE/GOOGLE_SCHOLAR/BING need their own API keys.
    web_corpuses: list[str] = field(default_factory=lambda: ["ARXIV"])
    # Recursive-search intensity. Total search requests scale with
    # steps * queries_per_step * papers; lowering these reduces backend load. None
    # keeps the upstream config value.
    search_steps: int | None = None  # -> num_search_steps
    search_queries_per_step: int | None = None  # -> num_search_queries_per_step_per_corpus
    temperature: float | None = None
    max_tokens: int = 10000
    # Token budget for upstream's stage LMs (filter/search/taxonomize/generation),
    # which otherwise default to 512 and truncate structured outputs. None uses the
    # eval's DEFAULT_STAGE_MAX_TOKENS (kept below max_model_len, since LOTUS sends it
    # as max_completion_tokens and the server rejects prompt + budget > context).
    stage_max_tokens: int | None = None
    # litellm provider prefix for a local OpenAI-compatible (vLLM) server.
    # "openai" routes via litellm's OpenAI handler against api_base; an alternative
    # is "hosted_vllm". Ignored for external API models.
    local_model_prefix: str = "openai"
    # Per-request timeout passed to LOTUS/litellm.
    lm_timeout: int = 240

    # Eval phase (judge). Default to all seven metrics (the geomean inputs and how
    # both the paper's Table 2 and the leaderboard report results). Pass a
    # comma-separated subset to run fewer; the geomean is reported only when every
    # metric in PRIMARY_METRICS is present.
    judge_model: str = "gpt-4o"
    evals: list[str] = field(default_factory=lambda: list(PRIMARY_METRICS))

    # Strict by default: a partial generation (some queries failed) would score
    # only the succeeded subset, which misrepresents the benchmark. Opt in to
    # score whatever generated.
    allow_partial_generation: bool = False

    # Escape hatches for validation iterations
    extra_gen_args: list[str] = field(default_factory=list)
    extra_eval_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeepScholarArgs:
        evals = data.get("evals")
        if isinstance(evals, str):
            evals = [e.strip() for e in evals.split(",") if e.strip()]
        if evals == ["all"]:
            evals = list(PRIMARY_METRICS)

        def _as_list(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [a.strip() for a in value.split(",") if a.strip()]
            return list(value)

        return cls(
            limit=_parse_optional(data, "limit", int),
            start_idx=int(data.get("start_idx", 0)),
            search_mode=data.get("search_mode"),
            search_backend=data.get("search_backend", "arxiv"),
            web_corpuses=_as_list(data.get("web_corpuses")) or ["ARXIV"],
            search_steps=_parse_optional(data, "search_steps", int),
            search_queries_per_step=_parse_optional(data, "search_queries_per_step", int),
            temperature=_parse_optional(data, "temperature", float),
            max_tokens=int(data.get("max_tokens", 10000)),
            stage_max_tokens=_parse_optional(data, "stage_max_tokens", int),
            local_model_prefix=data.get("local_model_prefix", "openai"),
            lm_timeout=int(data.get("lm_timeout", 240)),
            judge_model=data.get("judge_model", "gpt-4o"),
            evals=evals or list(PRIMARY_METRICS),
            allow_partial_generation=_parse_bool(data.get("allow_partial_generation")),
            extra_gen_args=_as_list(data.get("extra_gen_args")),
            extra_eval_args=_as_list(data.get("extra_eval_args")),
        )
