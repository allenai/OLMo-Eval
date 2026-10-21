"""RULER: What's the Real Context Size of Your Long-Context Language Models?

This task implements the RULER benchmark for evaluating long-context language models.
RULER generates synthetic examples to evaluate models across 4 task categories:
- NIAH (Needle in a Haystack): Single/multi-key/multi-value/multi-query variants
- Multi-hop tracing: Variable tracking (VT)
- Aggregation: Common word extraction (CWE), Frequency word extraction (FWE)
- Question Answering: QA with long context

Paper: https://arxiv.org/abs/2404.06654
Original implementation: https://github.com/hsiehjackson/RULER
"""

import os
import re
import string
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import RecallMetric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data.ruler_loader import download_ruler_data, load_ruler_dataset
from olmo_eval.data.ruler_tasks import RULER_TASKS
from olmo_eval.evals.tasks.common.base import Task, TaskConfig
from olmo_eval.evals.tasks.common.registry import register


def _normalize_answer(s: str) -> str:
    """Normalize answer text for QA scoring (matches HELMET/old framework)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


@dataclass(frozen=True, slots=True)
class RulerQAScorer(Scorer):
    """Substring scorer for RULER QA tasks with HELMET-style normalization.

    Matches old framework behavior: normalizes both gold and prediction
    (lowercase, remove punctuation, remove articles) then checks if any
    normalized gold answer is a substring of the normalized prediction.
    Takes max over all gold answers.
    """

    name: str = "substring_recall"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.text is None:
            return 0.0

        gold_answers = (
            instance.gold_answer
            if isinstance(instance.gold_answer, list)
            else [str(instance.gold_answer)]
        )
        if not gold_answers:
            return 0.0

        pred_norm = _normalize_answer(output.text)

        # max over ground truths (1.0 if any gold answer is found)
        for answer in gold_answers:
            if _normalize_answer(str(answer)) in pred_norm:
                return 1.0
        return 0.0


class RulerTask(Task):
    """Base RULER task implementation.

    Each RULER task variant (e.g., niah_s_1__4096) is registered as a separate task
    with specific configuration from RULER_TASKS.

    Subclasses point at a different task registry and name prefix by overriding
    ``_tasks_registry`` and ``_name_prefix``, and can customize data loading by
    overriding ``_load_data`` (e.g. RulerPlusTask, where each task needs only
    one shard of a much larger dataset repo).
    """

    _tasks_registry: dict[str, dict[str, Any]] = RULER_TASKS
    _name_prefix: str = "ruler_"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        task_name = config.name.removeprefix(self._name_prefix)
        self.task_name = task_name
        self.ruler_config = self._tasks_registry[task_name]

        # Extract context size from task name
        task_type, context_size_str = task_name.rsplit("__", 1)
        self.task_type = task_type
        self.context_size = int(context_size_str)

        # Load dataset during initialization
        self._dataset = None
        self._templates = None

    def _load_data(self) -> None:
        """Load RULER dataset if not already loaded.

        RULER uses a custom data loader rather than the standard HuggingFace pipeline
        because its data is pre-generated offline at specific context lengths (e.g. 4096,
        8192 tokens) and stored as task-specific JSONL files.
        Each file pairs synthetic context (needle-in-haystack padding, variable tracking
        chains, etc.) with gold answers and formatting templates that vary by task type.
        """
        if self._dataset is not None:
            return

        # Download RULER data if needed
        root_dir = download_ruler_data()

        # Get data path from config
        data_path = os.path.join(root_dir, self.ruler_config["data"])

        # Load dataset
        loaded = load_ruler_dataset(
            task_name=self.task_name,
            data_path=data_path,
            max_samples=self.config.limit,
            seed=42,
        )

        self._dataset = loaded["data"]
        self._templates = {
            "prompt": loaded["prompt_template"],
            "user": loaded["user_template"],
            "system": loaded["system_template"],
        }

    @property
    def instances(self) -> Iterator[Instance]:
        """Generate Instance objects from the dataset."""
        # Load data if needed
        self._load_data()

        # Check cache first
        if self._instances_cache is not None:
            yield from self._instances_cache
            return

        # Generate instances
        self._instances_cache = []
        for idx, doc in enumerate(self._dataset):  # type: ignore
            instance = self.process_doc(cast(dict[str, Any], doc), index=idx)
            if instance is not None:
                self._instances_cache.append(instance)
                yield instance

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a raw document to an Instance.

        Args:
            doc: Raw document from dataset
            index: Index of the document

        Returns:
            Instance object or None if document should be skipped
        """
        # Build the context by formatting the user template
        context_fields = dict(doc)
        if "context" not in context_fields:
            context_fields["context"] = ""

        # Prefer the preformatted RULER input when present. The released JSONL
        # already includes the benchmark's exact prompt plus answer prefix.
        question = doc.get("input")
        prepend_text = ""
        if not isinstance(question, str) or not question:
            # Format the question using the user template
            if self._templates is None:
                raise RuntimeError("Templates not loaded. Call _load_data() first.")
            question = self._templates["user"].format(**context_fields)

            # Add system template as prepend text for non-chat format
            if not self.ruler_config.get("use_chat_template", False):
                prepend_text = self._templates["system"].format(**context_fields)

        # Get answer (handle both "answer" and "outputs" fields)
        answer = doc.get("answer") or doc.get("outputs")

        # QA tasks have list gold answers. Keep the list for SubstringRecallScorer
        # (which handles lists natively) and store it in metadata for other scorers.
        metadata: dict = {
            "id": doc.get("index", index),
            "task_type": self.task_type,
            "context_size": self.context_size,
            "prepend_text": prepend_text,
            "tag": self.ruler_config["tag"],
        }
        if isinstance(answer, list):
            metadata["all_gold_answers"] = answer

        return Instance(
            question=question,
            gold_answer=answer,
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        """Return the request type for this task."""
        return RequestType.COMPLETION

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance as an LMRequest.

        Args:
            instance: Instance to format

        Returns:
            LMRequest for model inference
        """
        # Use config formatter if provided
        if self.config.formatter is not None:
            # For PPL formatters (BPB variant), convert list answers to strings
            if isinstance(self.config.formatter, PPLFormatter) and isinstance(
                instance.gold_answer, list
            ):
                # Create a modified instance with string answer
                instance = Instance(
                    question=instance.question,
                    gold_answer=", ".join(str(a) for a in instance.gold_answer),
                    metadata=instance.metadata,
                )
            return self.config.formatter.format(instance, self.get_fewshot())

        # Build prompt: append system template as completion prefix for non-chat format.
        # For base models, the prefix anchors the expected output format.
        prompt = instance.question
        prepend_text = (instance.metadata or {}).get("prepend_text", "")
        if prepend_text:
            prompt = prompt + "\n" + prepend_text

        return LMRequest(
            request_type=self.request_type,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        """Extract answer from model output.

        For RULER tasks, we don't parse the output - we use the raw text
        for substring matching in the scorer.

        Args:
            output: Model output

        Returns:
            Raw output text
        """
        return output.text


def make_ruler_task_class(
    task_name: str,
    task_cfg: dict,
    *,
    base_cls: type[RulerTask] = RulerTask,
    class_prefix: str = "Ruler",
    limit: int = 100,
) -> type[RulerTask]:
    """Create a task subclass for a RULER task variant.

    Subclasses carry only class-level attributes (metrics, sampling_params, limit);
    all runtime state is derived from config.name inside RulerTask.__init__.

    Args:
        task_name: Task variant key (e.g., "niah_s_1__4096"), matching a key in
            base_cls's ``_tasks_registry``.
        task_cfg: Task configuration dict (see RULER_TASKS/RULER_PLUS_TASKS).
        base_cls: RulerTask subclass to build on (allows other data sources/prefixes).
        class_prefix: Prefix for the generated class name.
        limit: Number of samples to draw per task/context-size condition.
    """
    # QA tasks use RulerQAScorer which applies HELMET-style normalization
    # (removes articles/punctuation, normalizes whitespace) matching the old framework.
    # All other tasks use the standard SubstringRecallScorer via RecallMetric.
    stop_sequences: tuple[str, ...] | None = (
        ("\n", "Ċ", "ĊĊ", "<0x0A>") if task_cfg.get("stop_new_line", False) else None
    )
    return type(
        f"{class_prefix}_{task_name}",
        (base_cls,),
        {
            "__module__": base_cls.__module__,
            "metrics": (RecallMetric(scorer=RulerQAScorer),)
            if task_cfg["tag"] == "qa"
            else (RecallMetric(),),
            "primary_metric": "recall",
            "sampling_params": SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=task_cfg.get("max_gen_toks", 50),
                stop_sequences=stop_sequences,
            ),
            "limit": limit,
        },
    )


# Dynamically register all RULER tasks
for _task_name, _task_config in RULER_TASKS.items():
    _cls = make_ruler_task_class(_task_name, _task_config)
    # Inject into module globals so pickle can find the class by name
    globals()[_cls.__name__] = _cls
    register(f"ruler_{_task_name}")(_cls)
