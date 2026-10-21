"""RULER-plus (ruler-plus) tasks.

Long-context companion to ruler.py: same task types, scorers, and prompting
behavior, but downloads pre-generated data from the allenai/ruler-plus
HuggingFace dataset repo (see ruler_plus_loader.py) instead of the
allenai/ruler_data HuggingFace release, and extends context sizes beyond the
original 131072-token cap. Data is generated with https://github.com/jopetty/RULER
via scripts/generate-data.sh.

Generation budgets follow upstream RULER's per-task ``tokens_to_generate``,
and each condition has 512 samples, whereas ruler.py follows HELMET's budgets
with 100 samples. Scores are therefore not directly comparable between
``ruler_plus_*`` and ``ruler_*`` tasks at the same context size.

These tasks are internal-only: allenai/ruler-plus is private because its
haystacks include Paul Graham essay excerpts that we can't relicense (see
ruler_plus_loader.py).
"""

from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data.ruler_plus_loader import (
    ROW_FIELD,
    get_ruler_plus_data_file,
    load_ruler_plus_shard,
)
from olmo_eval.data.ruler_plus_tasks import RULER_PLUS_TASKS
from olmo_eval.evals.tasks.common.registry import register
from olmo_eval.evals.tasks.ruler import RulerTask, make_ruler_task_class


class RulerPlusTask(RulerTask):
    """RULER task variant backed by the ruler-plus dataset.

    ruler-plus records always carry a preformatted ``input`` plus a trailing
    ``answer_prefix`` that must be concatenated with no separator, unlike
    ruler.py's chat/system-template fallback path for records without a
    preformatted ``input``.
    """

    _tasks_registry: dict[str, dict[str, Any]] = RULER_PLUS_TASKS
    _name_prefix: str = "ruler_plus_"

    _data_source: str | None = None

    def _load_data(self) -> None:
        """Download only the requested shard and reservoir-sample it.

        Bypasses RulerTask._load_data's root-dir-plus-full-file-load path:
        ruler-plus shards can be many GB (contexts run up to 2097152 tokens),
        and every task only needs the one shard matching its task type and
        context size, not the rest of the dataset repo. ruler-plus records
        never need the system/user templates that RulerTask._load_data also
        prepares (process_doc below always expects a preformatted ``input``).
        """
        if self._dataset is not None:
            return

        data_file = get_ruler_plus_data_file(self.ruler_config["data"])
        self._data_source = data_file.source
        self._dataset = load_ruler_plus_shard(
            data_path=data_file.path,
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("input")
        if not isinstance(question, str) or not question:
            return None

        answer = doc.get("outputs")
        metadata: dict = {
            # The shard row, not the generator's `index` field, which repeats
            # within a shard (for NIAH it is the needle's character offset)
            "id": doc.get(ROW_FIELD, index),
            "task_type": self.task_type,
            "context_size": self.context_size,
            "tag": self.ruler_config["tag"],
            # The task hash doesn't cover the data, so record what was read
            "data_source": self._data_source,
        }
        if isinstance(answer, list):
            metadata["all_gold_answers"] = answer

        return Instance(
            question=f"{question}{doc.get('answer_prefix', '')}",
            gold_answer=answer,
            metadata=metadata,
        )


# Number of samples to draw per task/context-size condition.
_RULER_PLUS_LIMIT = 512

# Dynamically register all ruler-plus tasks
for _task_name, _task_config in RULER_PLUS_TASKS.items():
    _cls = make_ruler_task_class(
        _task_name,
        _task_config,
        base_cls=RulerPlusTask,
        class_prefix="RulerPlus",
        limit=_RULER_PLUS_LIMIT,
    )
    # Inject into module globals so pickle can find the class by name
    globals()[_cls.__name__] = _cls
    register(f"ruler_plus_{_task_name}")(_cls)
